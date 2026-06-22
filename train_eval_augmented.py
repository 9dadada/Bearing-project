"""증강 강건 모델 학습 + production 비교 평가 (2~4단계).

[2] data/normal_aug 의 증강 정상 36개로 AE 학습(입력=목표=있는 그대로 복원) → 별도 저장
[3] 임계값 = 증강 정상 재구성오차 평균+3σ
[4] production(클린 학습) vs 증강 모델 비교:
    ① 노이즈 낀 정상 오탐률  ② 고장 탐지율  ③ 약한 고장(0.007") 탐지

실행:
    .venv\\Scripts\\python.exe train_eval_augmented.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

from src.autoencoder_detector import (build_autoencoder, fit_threshold_b, load_model,
                                       recon_errors, save_model, train_ae)
from src.data_loader import load_fault_signals, load_normal_signals
from src.evaluator import add_white_noise
from src.preprocessing import make_windows, make_windows_from_signals
from src.spectrogram import (apply_spec_scaler, fit_spec_scaler, load_spec_scaler,
                             save_spec_scaler, to_spectrograms)
from src.utils import get_device, load_config, resolve_path, set_seed

FS = 12000


def impulse_random(sig, snr_db, rng, fs=FS, rate_per_sec=10):
    sig = np.asarray(sig, float).copy()
    n = len(sig); k = max(1, int(rate_per_sec * n / fs))
    train = np.zeros(n)
    train[rng.integers(0, n, k)] = rng.uniform(0.5, 1.5, k) * rng.choice([-1.0, 1.0], k)
    t = np.sqrt(np.mean(train ** 2))
    if t > 0:
        train *= (np.sqrt(np.mean(sig ** 2)) / 10 ** (snr_db / 20.0)) / t
    return sig + train


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config(); device = get_device(cfg)
    L, ov = cfg["window"]["length"], cfg["window"]["overlap"]
    md = cfg["artifacts"]["models"]

    # ---------- [2] 증강 학습 ----------
    aug = {}
    for f in sorted((_ROOT / "data" / "normal_aug").glob("*.mat")):
        d = sio.loadmat(f); k = [x for x in d if x.endswith("DE_time")][0]
        aug[f.stem] = np.asarray(d[k], float).ravel()
    print(f"증강 정상 {len(aug)}개 로드")

    specs = to_spectrograms(make_windows_from_signals(aug, L, ov), cfg)
    sc_aug = fit_spec_scaler(specs)
    tr = apply_spec_scaler(specs, sc_aug)
    set_seed(cfg["seed"])
    m_aug = build_autoencoder(cfg)
    print("증강 모델 학습...")
    train_ae(m_aug, tr, cfg, device=device, noise_std=0.0)   # 데이터가 이미 노이즈 → 추가 노이즈 X
    T_aug = fit_threshold_b(recon_errors(m_aug, tr, device), cfg)["threshold"]
    save_model(m_aug, f"{md}/autoencoder_aug.pth")
    save_spec_scaler(sc_aug, f"{md}/spec_scaler_aug.pkl")
    print(f"증강 모델 저장 (임계값 {T_aug:.4f})")

    # ---------- production 로드 ----------
    import json
    m_prod = load_model(cfg, f"{md}/autoencoder.pth", device)
    sc_prod = load_spec_scaler(f"{md}/spec_scaler.pkl")
    T_prod = json.loads(resolve_path(cfg["artifacts"]["thresholds"]).read_text("utf-8"))["path_b"]["threshold"]

    models = {"production(클린 학습)": (m_prod, sc_prod, T_prod),
              "증강 모델(노이즈 학습)": (m_aug, sc_aug, T_aug)}

    # ---------- [4] 평가 ----------
    normal = list(load_normal_signals(cfg).values())
    fa = load_fault_signals(cfg)
    fault = []
    for loc in ("IR", "OR", "B"):
        fault += [v["signal"] for v in fa.values() if v["location"] == loc][:2]
    weak = next(v["signal"] for n, v in fa.items() if "007" in n)   # 약한 고장(0.007")

    conds = [("노이즈\n없음", lambda s, r: s),
             ("화이트\n+10dB", lambda s, r: add_white_noise(s, 10, r)),
             ("화이트\n3dB", lambda s, r: add_white_noise(s, 3, r)),
             ("화이트\n-3dB", lambda s, r: add_white_noise(s, -3, r)),
             ("화이트\n-5dB", lambda s, r: add_white_noise(s, -5, r)),
             ("화이트\n-10dB", lambda s, r: add_white_noise(s, -10, r)),
             ("임펄스\n랜덤", lambda s, r: impulse_random(s, 0, r))]

    def err(model, scaler, sig):
        w = make_windows(np.asarray(sig, float), L, ov)
        return recon_errors(model, apply_spec_scaler(to_spectrograms(w, cfg), scaler), device) if len(w) else np.empty(0)

    fpr = {k: [] for k in models}; det = {k: [] for k in models}
    for name, fn in conds:
        rng = np.random.default_rng(cfg["seed"])
        nz_n = [fn(np.asarray(s, float), rng) for s in normal]
        nz_f = [fn(np.asarray(s, float), rng) for s in fault]
        for tag, (m, sc, T) in models.items():
            ne = np.concatenate([err(m, sc, s) for s in nz_n])
            fe = np.concatenate([err(m, sc, s) for s in nz_f])
            fpr[tag].append(float((ne > T).mean())); det[tag].append(float((fe > T).mean()))

    # ③ 약한 고장(클린) 탐지
    weak_det = {}
    for tag, (m, sc, T) in models.items():
        e = err(m, sc, weak)
        weak_det[tag] = float((e > T).mean())

    # ---------- 그래프 ----------
    x = np.arange(len(conds)); names = [c[0] for c in conds]
    col = {"production(클린 학습)": "#e74c3c", "증강 모델(노이즈 학습)": "#2980b9"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.2))
    for k in models:
        a1.plot(x, [100 * v for v in fpr[k]], "o-", color=col[k], lw=2.2, label=k)
        a2.plot(x, [100 * v for v in det[k]], "o-", color=col[k], lw=2.2, label=k)
    a1.set_title("① 노이즈 낀 정상 오탐률 (낮을수록 좋음)", fontweight="bold"); a1.set_ylabel("오탐률 %")
    a2.set_title("② 고장 탐지율 (높을수록 좋음)", fontweight="bold"); a2.set_ylabel("탐지율 %")
    for a in (a1, a2):
        a.set_xticks(x); a.set_xticklabels(names, fontsize=8); a.set_ylim(0, 105); a.grid(alpha=0.3); a.legend()
    fig.suptitle("노이즈 증강 학습 효과 — production(클린) vs 증강 모델", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = resolve_path("outputs/figures/augmented_robustness.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120); plt.close(fig)

    # ---------- 출력 ----------
    print("\n=== ① 노이즈 낀 정상 오탐률 ===")
    print(f"{'조건':<10} | {'production':>11} | {'증강':>8}")
    for i, (nm, _) in enumerate(conds):
        n = nm.replace(chr(10), "")
        print(f"{n:<10} | {100*fpr['production(클린 학습)'][i]:>10.1f}% | {100*fpr['증강 모델(노이즈 학습)'][i]:>7.1f}%")
    print("\n=== ③ 약한 고장(0.007') 탐지율 ===")
    for k in models:
        print(f"  {k}: {100*weak_det[k]:.1f}%")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
