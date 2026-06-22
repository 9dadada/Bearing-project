"""실험: 레이어 수(Conv 단 수) 증감에 따른 탐지 성능.

병목(잠재차원)은 128로 고정하고 인코더/디코더 Conv 단 수만 2→3→4→5로 바꿔가며
노이즈 없음 AUC·오탐률, 임펄스 강 AUC(강건성)를 비교한다.

실행:
    .venv\\Scripts\\python.exe -m visualization.ablation_layers
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import roc_auc_score

from visualization._common import plt, save_fig
from visualization.ablation_architecture import ParamAE


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    from src.autoencoder_detector import fit_threshold_b, recon_errors, train_ae
    from src.data_loader import load_fault_signals, load_normal_signals
    from src.evaluator import add_impulse_noise
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
    arr = np.asarray(tr_n)
    in_ch, hw = (arr.shape[1], arr.shape[2:]) if arr.ndim == 4 else (1, arr.shape[1:])

    def specs(sig, impulse, rng):
        s = add_impulse_noise(sig, 10, rng) if impulse else sig
        w = make_windows(s, L, ov)
        return apply_spec_scaler(to_spectrograms(w, cfg), sc) if len(w) else None

    LAYERS = [2, 3, 4, 5]
    auc_clean, fpr_clean, auc_imp = [], [], []
    for ns in LAYERS:
        print(f"학습: Conv {ns}단 (병목 128) ...")
        set_seed(cfg["seed"])
        m = ParamAE(in_ch, hw, ns, 128).to(device)
        train_ae(m, tr_n, cfg, device=device)
        T = fit_threshold_b(recon_errors(m, tr_n, device), cfg)["threshold"]

        def errs(impulse):
            rng = np.random.default_rng(cfg["seed"])
            ne = np.concatenate([recon_errors(m, specs(s, impulse, rng), device) for s in normal.values()])
            fe = np.concatenate([recon_errors(m, specs(s, impulse, rng), device) for s in fault])
            return ne, fe

        ne, fe = errs(False)
        auc_clean.append(roc_auc_score(np.r_[np.zeros(len(ne)), np.ones(len(fe))], np.r_[ne, fe]))
        fpr_clean.append(float((ne > T).mean()))
        nei, fei = errs(True)
        auc_imp.append(roc_auc_score(np.r_[np.zeros(len(nei)), np.ones(len(fei))], np.r_[nei, fei]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    xs = [str(n) for n in LAYERS]

    ax1.plot(xs, [100 * a for a in auc_imp], "o-", color="#2980b9", lw=2.2, label="임펄스 강 AUC")
    ax1.plot(xs, [100 * a for a in auc_clean], "s--", color="#27ae60", lw=2, label="노이즈 없음 AUC")
    for i, a in enumerate(auc_imp):
        ax1.text(i, 100 * a + 1.2, f"{100*a:.1f}%", ha="center", color="#2980b9", fontweight="bold", fontsize=9)
    ax1.set_ylim(40, 105); ax1.axhline(50, color="gray", ls=":", lw=1)
    ax1.set_title("레이어 수 vs 탐지 AUC (높을수록 좋음)", fontweight="bold")
    ax1.set_xlabel("Conv 단 수"); ax1.set_ylabel("AUC %"); ax1.legend(loc="lower right"); ax1.grid(alpha=0.3)

    ax2.plot(xs, [100 * f for f in fpr_clean], "o-", color="#e67e22", lw=2.2)
    for i, f in enumerate(fpr_clean):
        ax2.text(i, 100 * f + 0.04, f"{100*f:.2f}%", ha="center", color="#e67e22", fontweight="bold", fontsize=9)
    ax2.set_ylim(0, max(2.0, max(100 * f for f in fpr_clean) * 1.5))
    ax2.set_title("레이어 수 vs 노이즈 없음 오탐률 (낮을수록 좋음)", fontweight="bold")
    ax2.set_xlabel("Conv 단 수"); ax2.set_ylabel("오탐률 %"); ax2.grid(alpha=0.3)

    fig.suptitle("레이어 수(Conv 단 수) 증감에 따른 탐지 성능 (병목 128 고정)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = save_fig(fig, "ablation_layers.png")

    print("\n=== 레이어 수별 탐지 성능 ===")
    print(f"{'Conv 단':>7} | {'노이즈없음 AUC':>12} | {'노이즈없음 오탐':>13} | {'임펄스강 AUC':>11}")
    for i, ns in enumerate(LAYERS):
        print(f"{ns:>7} | {auc_clean[i]:>12.3f} | {100*fpr_clean[i]:>12.2f}% | {100*auc_imp[i]:>10.1f}%")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
