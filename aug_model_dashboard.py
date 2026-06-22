"""노이즈 증강 오토인코더 — 신뢰도 종합 (5개 지표).

①분리도 ②탐지율 ③오탐율 ④정밀도 ⑤노이즈 강건성(입력 시점 노이즈)
모델: 클린 + 화이트(+6/+3/0) + 임펄스(중간 랜덤) 증강 학습.
⑤는 '학습 노이즈'가 아니라 '입력(테스트)에 노이즈가 낀 상황'에서의 탐지율·오탐율.

실행:
    .venv\\Scripts\\python.exe aug_model_dashboard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

from src.autoencoder_detector import build_autoencoder, fit_threshold_b, recon_errors, train_ae
from src.data_loader import load_fault_signals, load_normal_signals
from src.evaluator import add_white_noise
from src.preprocessing import make_windows, make_windows_from_signals
from src.reliability import precision_at
from src.spectrogram import apply_spec_scaler, fit_spec_scaler, to_spectrograms
from src.utils import get_device, load_config, resolve_path, set_seed

FS = 12000


def impulse_random(sig, snr_db, rng, fs=FS, rate=10):
    sig = np.asarray(sig, float).copy(); n = len(sig); k = max(1, int(rate * n / fs))
    tr = np.zeros(n); tr[rng.integers(0, n, k)] = rng.uniform(0.5, 1.5, k) * rng.choice([-1.0, 1.0], k)
    t = np.sqrt(np.mean(tr ** 2))
    if t > 0:
        tr *= (np.sqrt(np.mean(sig ** 2)) / 10 ** (snr_db / 20.0)) / t
    return sig + tr


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config(); device = get_device(cfg)
    L, ov = cfg["window"]["length"], cfg["window"]["overlap"]
    normal = load_normal_signals(cfg)
    fa = load_fault_signals(cfg)
    fault = []
    for loc in ("IR", "OR", "B"):
        fault += [v["signal"] for v in fa.values() if v["location"] == loc][:2]
    fault_nz = [fault[0], fault[2], fault[4]]                 # 노이즈 패널용(위치별 1)

    # ---- 노이즈 증강 학습 (+6/+3/0) ----
    rng = np.random.default_rng(cfg["seed"])
    aug = {}
    for n, s in normal.items():
        s = np.asarray(s, float); aug[f"{n}_c"] = s
        for db in (6, 3, 0):
            aug[f"{n}_w{db}"] = add_white_noise(s, db, rng)
        for j in range(3):
            aug[f"{n}_i{j}"] = impulse_random(s, rng.uniform(0, 6), rng)
    specs = to_spectrograms(make_windows_from_signals(aug, L, ov), cfg)
    sc = fit_spec_scaler(specs); tr = apply_spec_scaler(specs, sc)
    set_seed(cfg["seed"]); m = build_autoencoder(cfg)
    print("노이즈 증강 모델 학습...")
    train_ae(m, tr, cfg, device=device, noise_std=0.0)
    T = fit_threshold_b(recon_errors(m, tr, device), cfg)["threshold"]

    def score(sig):
        w = make_windows(np.asarray(sig, float), L, ov)
        e = recon_errors(m, apply_spec_scaler(to_spectrograms(w, cfg), sc), device) if len(w) else np.empty(0)
        return e / T                                          # >1 이면 이상

    # ---- 클린 점수 ----
    ns = np.concatenate([score(s) for s in normal.values()])
    fs_ = np.concatenate([score(s) for s in fault])
    fpr = float((ns > 1).mean()); det = float((fs_ > 1).mean())

    # ---- ⑤ 입력 노이즈 강건성 ----
    conds = [("없음", lambda s, r: s),
             ("화이트\n+6", lambda s, r: add_white_noise(s, 6, r)),
             ("화이트\n+3", lambda s, r: add_white_noise(s, 3, r)),
             ("화이트\n0", lambda s, r: add_white_noise(s, 0, r)),
             ("화이트\n-3", lambda s, r: add_white_noise(s, -3, r)),
             ("화이트\n-5", lambda s, r: add_white_noise(s, -5, r)),
             ("임펄스\n랜덤", lambda s, r: impulse_random(s, 0, r))]
    nz_det, nz_fpr = [], []
    for _, fn in conds:
        rr = np.random.default_rng(cfg["seed"])
        ne = np.concatenate([score(fn(np.asarray(s, float), rr)) for s in normal.values()])
        fe = np.concatenate([score(fn(np.asarray(s, float), rr)) for s in fault_nz])
        nz_fpr.append(float((ne > 1).mean())); nz_det.append(float((fe > 1).mean()))

    # ===== 시각화 =====
    fig, ax = plt.subplots(2, 3, figsize=(16, 9.5))
    fig.suptitle("노이즈 증강 오토인코더 — 신뢰도 종합 (5개 지표)", fontsize=15, fontweight="bold")
    B, G, R = "#2980b9", "#27ae60", "#e74c3c"

    a = ax[0, 0]                                              # ① 분리도
    bins = np.linspace(0, max(3, float(fs_.max())), 40)
    a.hist(np.clip(ns, 0, bins[-1]), bins=bins, color=B, alpha=0.7, label="정상")
    a.hist(np.clip(fs_, 0, bins[-1]), bins=bins, color=R, alpha=0.7, label="고장")
    a.axvline(1.0, color="k", ls="--", lw=1.5, label="임계값")
    a.set_title("① 분리도 (정상 vs 고장 점수)", fontweight="bold")
    a.set_xlabel("이상점수 (1=임계값)"); a.set_ylabel("윈도우 수"); a.legend()

    a = ax[0, 1]                                              # ② 탐지율
    a.bar(["탐지율"], [100 * det], color=G, width=0.5)
    a.text(0, 100 * det + 1, f"{100*det:.1f}%", ha="center", fontweight="bold")
    a.set_ylim(0, 110); a.set_ylabel("%"); a.set_title("② 고장 탐지율 (클린)", fontweight="bold")

    a = ax[0, 2]                                              # ③ 오탐율
    a.bar(["오탐율"], [100 * fpr], color=R, width=0.5)
    a.text(0, 100 * fpr + 0.3, f"{100*fpr:.2f}%", ha="center", fontweight="bold")
    a.set_ylim(0, max(2, 100 * fpr * 1.5)); a.set_ylabel("%"); a.set_title("③ 정상 오탐율 (클린)", fontweight="bold")

    a = ax[1, 0]                                              # ④ 정밀도
    ratios = [0.5, 0.9, 0.95, 0.99]
    precs = [100 * precision_at(fpr, det, r) for r in ratios]
    a.plot([f"{int(r*100)}:{int((1-r)*100)}" for r in ratios], precs, "o-", color=B, lw=2)
    for xi, v in enumerate(precs):
        a.text(xi, v + 1.5, f"{v:.0f}%", ha="center", fontweight="bold")
    a.set_ylim(0, 105); a.set_ylabel("정밀도 %"); a.set_xlabel("정상 : 고장 비율")
    a.set_title("④ 정밀도 ('고장'이라면 맞을 확률)", fontweight="bold")

    a = ax[1, 1]                                              # ⑤ 노이즈 강건성 (입력 노이즈)
    xx = np.arange(len(conds))
    a.plot(xx, [100 * v for v in nz_det], "o-", color=G, lw=2.2, label="고장 탐지율")
    a.plot(xx, [100 * v for v in nz_fpr], "s--", color=R, lw=2.2, label="정상 오탐율")
    a.set_xticks(xx); a.set_xticklabels([c[0] for c in conds], fontsize=8)
    a.set_ylim(0, 105); a.set_ylabel("%"); a.legend(fontsize=9); a.grid(alpha=0.3)
    a.set_title("⑤ 노이즈 강건성 (입력 시점 노이즈)", fontweight="bold")

    ax[1, 2].axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = resolve_path("outputs/figures/aug_model_dashboard.png")
    fig.savefig(out, dpi=120); plt.close(fig)

    print("\n=== 요약 (노이즈 증강 모델) ===")
    print(f"① 분리도: 정상 {ns.mean():.2f} / 고장 {fs_.mean():.2f} (임계값 1.0)")
    print(f"② 탐지율 {100*det:.1f}%  ③ 오탐율 {100*fpr:.2f}%")
    print(f"④ 정밀도 50:50 {precs[0]:.0f}% → 99:1 {precs[-1]:.0f}%")
    print("⑤ 입력 노이즈별 [탐지율/오탐율]:")
    for i, (nm, _) in enumerate(conds):
        print(f"   {nm.replace(chr(10),''):<8} {100*nz_det[i]:>5.0f}% / {100*nz_fpr[i]:>4.0f}%")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
