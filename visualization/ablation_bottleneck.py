"""실험: 병목(잠재차원) 크기별 탐지 성능 — production ConvAutoencoder (32 / 64 / 128).

같은 정상 데이터로 병목만 바꿔 학습한 뒤 탐지 지표를 비교한다.
작을수록 파라미터는 줄지만 압축이 세져 오탐률·정밀도가 나빠지는 트레이드오프를 본다.

실행:
    .venv\\Scripts\\python.exe -m visualization.ablation_bottleneck
"""
from __future__ import annotations

import copy
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
    from src.evaluator import add_impulse_noise
    from src.preprocessing import make_windows, make_windows_from_signals
    from src.reliability import precision_at
    from src.spectrogram import apply_spec_scaler, fit_spec_scaler, to_spectrograms
    from src.utils import get_device, load_config, set_seed

    cfg = load_config()
    device = get_device(cfg)
    L, ov = cfg["window"]["length"], cfg["window"]["overlap"]

    normal = load_normal_signals(cfg)
    norm_by_load = {int(n.split("_")[-1]): s for n, s in normal.items()}
    fault_all = load_fault_signals(cfg)
    fault = []
    for loc in ("IR", "OR", "B"):
        fault += [v["signal"] for v in fault_all.values() if v["location"] == loc][:2]

    tr_specs = to_spectrograms(make_windows_from_signals(normal, L, ov), cfg)
    sc = fit_spec_scaler(tr_specs)
    tr_n = apply_spec_scaler(tr_specs, sc)

    def err(model, sig, impulse, rng):
        s = add_impulse_noise(sig, 10, rng) if impulse else sig
        w = make_windows(s, L, ov)
        return recon_errors(model, apply_spec_scaler(to_spectrograms(w, cfg), sc), device) if len(w) else np.empty(0)

    def evaluate(latent):
        c = copy.deepcopy(cfg); c["ae"]["latent_dim"] = latent
        set_seed(cfg["seed"])
        m = build_autoencoder(c)
        train_ae(m, tr_n, cfg, device=device)
        T = fit_threshold_b(recon_errors(m, tr_n, device), cfg)["threshold"]
        rng = np.random.default_rng(cfg["seed"])
        ne_load = {ld: err(m, s, False, rng) for ld, s in norm_by_load.items()}
        ne = np.concatenate(list(ne_load.values()))
        fe = np.concatenate([err(m, s, False, rng) for s in fault])
        nei = np.concatenate([err(m, s, True, rng) for s in norm_by_load.values()])
        fei = np.concatenate([err(m, s, True, rng) for s in fault])
        fpr = float((ne > T).mean()); det = float((fe > T).mean())
        return {
            "params": sum(p.numel() for p in m.parameters()),
            "auc": roc_auc_score(np.r_[np.zeros(len(ne)), np.ones(len(fe))], np.r_[ne, fe]),
            "det": det, "fpr": fpr, "fpr0": float((ne_load[0] > T).mean()),
            "prec99": precision_at(fpr, det, 0.99),
            "auc_imp": roc_auc_score(np.r_[np.zeros(len(nei)), np.ones(len(fei))], np.r_[nei, fei]),
        }

    LAT = [32, 64, 128]
    res = {}
    for lat in LAT:
        print(f"학습: 병목 {lat} ...")
        res[lat] = evaluate(lat)

    # ===== 출력 =====
    print("\n=== 병목 크기별 탐지 성능 ===")
    print(f"{'병목':>5} {'파라미터':>11} {'AUC':>6} {'탐지율':>7} {'오탐률':>7} {'부하0오탐':>9} {'정밀도99:1':>10} {'임펄스AUC':>9}")
    for lat in LAT:
        r = res[lat]
        print(f"{lat:>5} {r['params']:>11,} {r['auc']:>6.3f} {100*r['det']:>6.1f}% {100*r['fpr']:>6.2f}% "
              f"{100*r['fpr0']:>8.2f}% {100*r['prec99']:>9.1f}% {100*r['auc_imp']:>8.1f}%")

    # ===== 시각화 =====
    xs = [str(l) for l in LAT]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.8))

    a1.plot(xs, [100 * res[l]["fpr"] for l in LAT], "o-", color="#e74c3c", lw=2.2, label="전체 오탐률")
    a1.plot(xs, [100 * res[l]["fpr0"] for l in LAT], "s--", color="#e67e22", lw=2, label="부하0 오탐")
    for i, l in enumerate(LAT):
        a1.text(i, 100 * res[l]["fpr0"] + 0.15, f"{100*res[l]['fpr0']:.1f}%", ha="center", color="#e67e22", fontsize=8, fontweight="bold")
    a1.set_title("오탐률 (낮을수록 좋음)", fontweight="bold"); a1.set_ylabel("오탐률 %")
    a1.set_xlabel("병목 크기"); a1.legend(); a1.grid(alpha=0.3); a1.set_ylim(bottom=0)

    a2.plot(xs, [100 * res[l]["prec99"] for l in LAT], "o-", color="#2980b9", lw=2.2)
    for i, l in enumerate(LAT):
        a2.text(i, 100 * res[l]["prec99"] + 1.5, f"{100*res[l]['prec99']:.0f}%", ha="center", color="#2980b9", fontsize=9, fontweight="bold")
    a2.set_title("정밀도 99:1 (높을수록 좋음)", fontweight="bold"); a2.set_ylabel("정밀도 %")
    a2.set_xlabel("병목 크기"); a2.grid(alpha=0.3); a2.set_ylim(0, 105)

    bars = a3.bar(xs, [res[l]["params"] / 1e6 for l in LAT], color="#27ae60")
    for i, l in enumerate(LAT):
        a3.text(i, res[l]["params"] / 1e6 + 0.02, f"{res[l]['params']/1e6:.2f}M", ha="center", fontsize=9, fontweight="bold")
    a3.set_title("파라미터 수 (작을수록 경량)", fontweight="bold"); a3.set_ylabel("백만 개")
    a3.set_xlabel("병목 크기")

    fig.suptitle("병목(잠재차원) 크기별 탐지 성능 — 32 vs 64 vs 128 (production ConvAutoencoder)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = save_fig(fig, "ablation_bottleneck.png")
    print(f"\n저장: {out}")
    print("해석: 병목↓ → 파라미터↓(경량)이지만 오탐률↑·정밀도↓. 128이 정확도 최선 → production 채택.")


if __name__ == "__main__":
    main()
