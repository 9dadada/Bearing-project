"""일반 오토인코더 vs 노이즈 증강 오토인코더 — 노이즈 학습의 효과.

일반:      클린 정상만 학습 → 노이즈에 취약 (노이즈 정상을 오탐, 정상·고장 분별 못함)
노이즈 증강: 클린 + 화이트(+6/+3/0) + 임펄스(중간 랜덤) 학습 → 강건 (오탐↓·고장 탐지 유지)

실행:
    .venv\\Scripts\\python.exe plain_vs_augmented.py
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
        fault += [v["signal"] for v in fa.values() if v["location"] == loc][:1]

    def train_on(sigs):
        specs = to_spectrograms(make_windows_from_signals(sigs, L, ov), cfg)
        sc = fit_spec_scaler(specs); tr = apply_spec_scaler(specs, sc)
        set_seed(cfg["seed"]); m = build_autoencoder(cfg)
        train_ae(m, tr, cfg, device=device, noise_std=0.0)
        return m, sc, fit_threshold_b(recon_errors(m, tr, device), cfg)["threshold"]

    # 일반: 클린 정상만
    clean = {n: np.asarray(s, float) for n, s in normal.items()}
    print("학습: 일반 오토인코더 (클린만) ...")
    plain = train_on(clean)

    # 노이즈 증강: 클린 + 화이트 +6/+3/0 + 임펄스 중간 랜덤
    rng = np.random.default_rng(cfg["seed"])
    aug = {}
    for n, s in normal.items():
        s = np.asarray(s, float); aug[f"{n}_c"] = s
        for db in (6, 3, 0):
            aug[f"{n}_w{db}"] = add_white_noise(s, db, rng)
        for j in range(3):
            aug[f"{n}_i{j}"] = impulse_random(s, rng.uniform(0, 6), rng)
    print("학습: 노이즈 증강 오토인코더 (+6/+3/0) ...")
    augm = train_on(aug)

    models = {"일반 오토인코더": plain, "노이즈 증강 오토인코더": augm}

    conds = [("없음", lambda s, r: s),
             ("화이트\n+10", lambda s, r: add_white_noise(s, 10, r)),
             ("화이트\n+3", lambda s, r: add_white_noise(s, 3, r)),
             ("화이트\n-3", lambda s, r: add_white_noise(s, -3, r)),
             ("화이트\n-5", lambda s, r: add_white_noise(s, -5, r)),
             ("임펄스\n랜덤", lambda s, r: impulse_random(s, 0, r))]

    def err(m, sc, sig):
        w = make_windows(np.asarray(sig, float), L, ov)
        return recon_errors(m, apply_spec_scaler(to_spectrograms(w, cfg), sc), device) if len(w) else np.empty(0)

    fpr = {k: [] for k in models}; det = {k: [] for k in models}
    for _, fn in conds:
        rng = np.random.default_rng(cfg["seed"])
        nn = [fn(np.asarray(s, float), rng) for s in normal.values()]
        nf = [fn(np.asarray(s, float), rng) for s in fault]
        for tag, (m, sc, T) in models.items():
            ne = np.concatenate([err(m, sc, s) for s in nn])
            fe = np.concatenate([err(m, sc, s) for s in nf])
            fpr[tag].append(float((ne > T).mean())); det[tag].append(float((fe > T).mean()))

    x = np.arange(len(conds)); names = [c[0] for c in conds]
    cP, cA = "#e74c3c", "#2980b9"
    cols = {"일반 오토인코더": cP, "노이즈 증강 오토인코더": cA}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.4))
    for k in models:
        a1.plot(x, [100 * v for v in fpr[k]], "o-", color=cols[k], lw=2.4, label=k)
        a2.plot(x, [100 * v for v in det[k]], "o-", color=cols[k], lw=2.4, label=k)
    a1.set_title("① 노이즈 정상 오탐률 (낮을수록 좋음)", fontweight="bold"); a1.set_ylabel("오탐률 %")
    a2.set_title("② 고장 탐지율 (높을수록 좋음)", fontweight="bold"); a2.set_ylabel("탐지율 %")
    a1.annotate("일반: 노이즈 정상을\n전부 오탐", xy=(3, 100), xytext=(2.0, 62),
                color=cP, fontweight="bold", fontsize=10, arrowprops=dict(arrowstyle="->", color=cP))
    a1.annotate("증강: 정상 처리", xy=(4, 3), xytext=(3.2, 28),
                color=cA, fontweight="bold", fontsize=10, arrowprops=dict(arrowstyle="->", color=cA))
    a2.annotate("증강: 고장 탐지 유지", xy=(3, 93), xytext=(1.6, 60),
                color=cA, fontweight="bold", fontsize=10, arrowprops=dict(arrowstyle="->", color=cA))
    for a in (a1, a2):
        a.set_xticks(x); a.set_xticklabels(names, fontsize=8); a.set_ylim(0, 105); a.grid(alpha=0.3); a.legend(fontsize=9)
    fig.suptitle("일반 오토인코더 vs 노이즈 증강 오토인코더 — 노이즈 학습의 효과", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = resolve_path("outputs/figures/plain_vs_augmented.png")
    fig.savefig(out, dpi=120); plt.close(fig)

    print("\n=== [오탐률% / 고장탐지%] ===")
    print(f"{'조건':<8}{'일반':>14}{'노이즈증강':>16}")
    for i, (nm, _) in enumerate(conds):
        a, b = "일반 오토인코더", "노이즈 증강 오토인코더"
        print(f"{nm.replace(chr(10),''):<8}{f'{100*fpr[a][i]:.0f}/{100*det[a][i]:.0f}':>14}{f'{100*fpr[b][i]:.0f}/{100*det[b][i]:.0f}':>16}")
    print("\n해석: 일반은 노이즈서 정상도 고장도 다 '이상'(① 100%·② 100% = 분별 불가).")
    print("      증강은 정상은 정상(① ~0%)·고장은 고장(② 높음) = 분별 가능.")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
