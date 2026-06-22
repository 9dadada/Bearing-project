"""실험: 일반 AE vs 디노이징 AE — '탐지 성능'으로 비교.

같은 정상 데이터·같은 초기값으로 두 모델을 학습한 뒤, 노이즈 없는 환경 + 노이즈 환경에서
정상/고장 탐지의 AUC·오탐률을 비교한다. (재구성손실이 아니라 '탐지 성능'으로 디노이징의 이득을 본다)
디노이징은 입력 노이즈를 걷어내도록 배우므로 노이즈 환경에서 오탐이 덜 오르고 AUC가 더 유지되어야 한다.

실행:
    .venv\\Scripts\\python.exe -m visualization.ae_vs_dae_detection
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import roc_auc_score

from visualization._common import plt, save_fig


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    from src.autoencoder_detector import build_autoencoder, fit_threshold_b, recon_errors, train_ae
    from src.data_loader import load_fault_signals, load_normal_signals
    from src.evaluator import CONDITIONS, apply_noise
    from src.preprocessing import make_windows, make_windows_from_signals
    from src.spectrogram import apply_spec_scaler, fit_spec_scaler, to_spectrograms
    from src.utils import get_device, load_config, set_seed

    cfg = load_config()
    device = get_device(cfg)
    L, ov = cfg["window"]["length"], cfg["window"]["overlap"]

    normal = load_normal_signals(cfg)
    fault_all = load_fault_signals(cfg)
    fault = []
    for loc in ("IR", "OR", "B"):
        fault += [v["signal"] for v in fault_all.values() if v["location"] == loc][:2]

    tr_specs = to_spectrograms(make_windows_from_signals(normal, L, ov), cfg)
    sc = fit_spec_scaler(tr_specs)
    tr_n = apply_spec_scaler(tr_specs, sc)

    def make(noise_std):
        set_seed(cfg["seed"])
        m = build_autoencoder(cfg)
        train_ae(m, tr_n, cfg, device=device, noise_std=noise_std)
        T = fit_threshold_b(recon_errors(m, tr_n, device), cfg)["threshold"]
        return m, T

    def err(model, sig, kind, val, rng):
        w = make_windows(apply_noise(sig, kind, val, rng), L, ov)
        if len(w) == 0:
            return np.empty(0)
        return recon_errors(model, apply_spec_scaler(to_spectrograms(w, cfg), sc), device)

    print("일반 AE 학습..."); m0 = make(0.0)
    print("디노이징(σ=0.1) 학습..."); m1 = make(0.1)
    print("디노이징(σ=0.5) 학습..."); m5 = make(0.5)
    models = {"일반 AE": m0, "디노이징 σ=0.1": m1, "디노이징 σ=0.5": m5}

    names = [c[0] for c in CONDITIONS]
    aucs = {k: [] for k in models}
    fprs = {k: [] for k in models}
    for name, kind, val in CONDITIONS:
        for k, (m, T) in models.items():
            rng = np.random.default_rng(cfg["seed"])
            ne = np.concatenate([err(m, s, kind, val, rng) for s in normal.values()])
            fe = np.concatenate([err(m, s, kind, val, rng) for s in fault])
            y = np.r_[np.zeros(len(ne)), np.ones(len(fe))]
            aucs[k].append(float(roc_auc_score(y, np.r_[ne, fe])))
            fprs[k].append(float((ne > T).mean()))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(names))
    col = {"일반 AE": "#95a5a6", "디노이징 σ=0.1": "#2980b9", "디노이징 σ=0.5": "#e74c3c"}
    for k in models:
        ax1.plot(x, [100 * a for a in aucs[k]], "o-", color=col[k], lw=2, label=k)
        ax2.plot(x, [100 * f for f in fprs[k]], "o-", color=col[k], lw=2, label=k)
    ax1.set_ylim(40, 105); ax2.set_ylim(0, 105)
    ax1.axhline(50, color="gray", ls=":", lw=1)
    for ax, t, yl in ((ax1, "탐지 AUC (높을수록 좋음)", "AUC %"),
                      (ax2, "정상 오탐률 (낮을수록 좋음)", "오탐률 %")):
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_title(t, fontweight="bold"); ax.set_ylabel(yl); ax.grid(alpha=0.3); ax.legend()
    fig.suptitle("일반 AE vs 디노이징 AE — 탐지 성능 (노이즈 없음 + 노이즈 환경)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = save_fig(fig, "ae_vs_dae_detection.png")

    print("\n=== 탐지 성능 (AUC / 오탐률) ===")
    for i, name in enumerate(names):
        cells = " | ".join(f"{k}: AUC {100*aucs[k][i]:5.1f}% 오탐 {100*fprs[k][i]:5.1f}%" for k in models)
        print(f"{name:<14} | {cells}")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
