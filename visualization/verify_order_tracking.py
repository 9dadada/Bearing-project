"""검증: order tracking(회전수 각도정렬)이 #5(안 본 부하 오탐)를 고치는가?

부하 0,1,2로 학습 → 부하 3(안 본 회전수)으로 시험을, OT 끄고/켜고 비교.
  - OT 끔: 기존 방식(고정 샘플 윈도우) → 부하3 오탐 100%(도메인 시프트)
  - OT 켬: 각 신호를 ref_rpm(1750)으로 각도정렬 후 윈도우 → 회전수 차이 제거 기대

실행:
    .venv\\Scripts\\python.exe -m visualization.verify_order_tracking
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

_TARGET = 12000


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    from src.autoencoder_detector import build_autoencoder, fit_threshold_b, recon_errors, train_ae
    from src.data_loader import load_fault_signals, load_normal_signals
    from src.preprocessing import make_windows_from_signals, order_track
    from src.spectrogram import apply_spec_scaler, fit_spec_scaler, to_spectrograms
    from src.utils import get_device, load_config, set_seed

    cfg = load_config()
    device = get_device(cfg)
    length, overlap = cfg["window"]["length"], cfg["window"]["overlap"]
    rpm_by_load = {int(k): v for k, v in cfg["domain"]["rpm_by_load"].items()}

    normal = load_normal_signals(cfg)                       # normal_0..3
    norm_by_load = {int(name.split("_")[-1]): sig for name, sig in normal.items()}
    fault_all = load_fault_signals(cfg)
    fault = []                                              # (load, signal) 위치별 3개
    for loc in ("IR", "OR", "B"):
        fault += [(v["load"], v["signal"]) for v in fault_all.values() if v["location"] == loc][:3]

    def prep(load_sig_pairs, use_ot, sc=None):
        sigs = {}
        for load, sig in load_sig_pairs:
            s = order_track(sig, _TARGET, rpm_by_load.get(load, cfg["signal"]["ref_rpm"]), cfg) if use_ot else sig
            sigs[f"L{load}_{len(sigs)}"] = s
        specs = to_spectrograms(make_windows_from_signals(sigs, length, overlap), cfg)
        if sc is None:
            sc = fit_spec_scaler(specs)
        return apply_spec_scaler(specs, sc), sc

    def evaluate(use_ot):
        set_seed(cfg["seed"])
        train_pairs = [(0, norm_by_load[0]), (1, norm_by_load[1]), (2, norm_by_load[2])]
        tr, sc = prep(train_pairs, use_ot)
        model = build_autoencoder(cfg)
        train_ae(model, tr, cfg, device=device)
        T = fit_threshold_b(recon_errors(model, tr, device), cfg)["threshold"]
        te, _ = prep([(3, norm_by_load[3])], use_ot, sc)            # 안 본 부하 3
        fpr_unseen = float((recon_errors(model, te, device) > T).mean())
        ferr = []
        for load, sig in fault:
            fs_, _ = prep([(load, sig)], use_ot, sc)
            ferr.append(recon_errors(model, fs_, device))
        det = float((np.concatenate(ferr) > T).mean())
        return fpr_unseen, det

    print("=== order tracking 검증 (부하0,1,2 학습 → 부하3 시험) ===\n")
    print("OT 끔 학습 중...")
    off_fpr, off_det = evaluate(False)
    print("OT 켬 학습 중...")
    on_fpr, on_det = evaluate(True)

    print(f"\n{'':<10}{'안본부하3 오탐':>16}{'고장 탐지율':>14}")
    print("-" * 40)
    print(f"{'OT 끔':<10}{100*off_fpr:>15.1f}%{100*off_det:>13.1f}%")
    print(f"{'OT 켬':<10}{100*on_fpr:>15.1f}%{100*on_det:>13.1f}%")
    print(f"\n→ 부하3 오탐이 {100*off_fpr:.0f}% → {100*on_fpr:.0f}% 로 줄면 order tracking이 효과 있음.")


if __name__ == "__main__":
    main()
