"""이전(노이즈 과다 학습) vs 해결(적정 노이즈 학습) 비교 — 문제와 해결을 한눈에.

이전:  화이트 -5/-3/+3 학습 → 노이즈에 둔감 → 고장 놓침 (문제)
해결:  화이트 +6/+3/0 학습  → 적정 → 오탐↓ 유지하며 고장 탐지 회복
(임펄스는 두 설정 공통: 중간 세기 랜덤)

실행:
    .venv\\Scripts\\python.exe compare_problem_solution.py
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

    def train_cfg(white_dbs):
        rng = np.random.default_rng(cfg["seed"])
        sigs = {}
        for name, sig in normal.items():
            sig = np.asarray(sig, float); sigs[f"{name}_c"] = sig
            for db in white_dbs:
                sigs[f"{name}_w{db}"] = add_white_noise(sig, db, rng)
            for j in range(3):
                sigs[f"{name}_i{j}"] = impulse_random(sig, rng.uniform(0, 6), rng)
        specs = to_spectrograms(make_windows_from_signals(sigs, L, ov), cfg)
        sc = fit_spec_scaler(specs); tr = apply_spec_scaler(specs, sc)
        set_seed(cfg["seed"]); m = build_autoencoder(cfg)
        train_ae(m, tr, cfg, device=device, noise_std=0.0)
        return m, sc, fit_threshold_b(recon_errors(m, tr, device), cfg)["threshold"]

    print("학습: 이전(화이트 -5/-3/+3) ...")
    prev = train_cfg([-5, -3, 3])
    print("학습: 해결(화이트 +6/+3/0) ...")
    fixed = train_cfg([6, 3, 0])
    models = {"이전 (과다: 화이트 -5/-3/+3)": prev, "해결 (적정: 화이트 +6/+3/0)": fixed}

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
    cP, cF = "#e74c3c", "#2980b9"
    cols = {"이전 (과다: 화이트 -5/-3/+3)": cP, "해결 (적정: 화이트 +6/+3/0)": cF}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.4))
    for k in models:
        a1.plot(x, [100 * v for v in fpr[k]], "o-", color=cols[k], lw=2.4, label=k)
        a2.plot(x, [100 * v for v in det[k]], "o-", color=cols[k], lw=2.4, label=k)
    a1.set_title("① 노이즈 정상 오탐률 (낮을수록 좋음)\n→ 둘 다 해결", fontweight="bold"); a1.set_ylabel("오탐률 %")
    a2.set_title("② 고장 탐지율 (높을수록 좋음)\n→ 이전은 붕괴, 해결은 유지", fontweight="bold"); a2.set_ylabel("탐지율 %")
    a2.annotate("이전: 고장 놓침\n(노이즈에 둔감)", xy=(3, 8), xytext=(2.2, 45),
                color=cP, fontweight="bold", fontsize=10,
                arrowprops=dict(arrowstyle="->", color=cP))
    a2.annotate("해결: 탐지 유지", xy=(4, 93), xytext=(3.0, 70),
                color=cF, fontweight="bold", fontsize=10,
                arrowprops=dict(arrowstyle="->", color=cF))
    for a in (a1, a2):
        a.set_xticks(x); a.set_xticklabels(names, fontsize=8); a.set_ylim(0, 105); a.grid(alpha=0.3); a.legend(fontsize=9)
    fig.suptitle("노이즈 증강 학습 — 이전(과다) vs 해결(적정) 모델", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = resolve_path("outputs/figures/problem_vs_solution.png")
    fig.savefig(out, dpi=120); plt.close(fig)

    print("\n=== [오탐률% / 고장탐지%] ===")
    print(f"{'조건':<8}{'이전(과다)':>16}{'해결(적정)':>16}")
    for i, (nm, _) in enumerate(conds):
        a, b = "이전 (과다: 화이트 -5/-3/+3)", "해결 (적정: 화이트 +6/+3/0)"
        print(f"{nm.replace(chr(10),''):<8}{f'{100*fpr[a][i]:.0f}/{100*det[a][i]:.0f}':>16}{f'{100*fpr[b][i]:.0f}/{100*det[b][i]:.0f}':>16}")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
