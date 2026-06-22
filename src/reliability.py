"""신뢰도 추가 확인 — 경로 B 오토인코더.

#4 정밀도(Precision): "고장이라 분류한 것 중 진짜 고장일 확률".
    - 정상:고장 비율(base rate)에 좌우되므로 여러 비율로 계산한다.
#5 안 본 부하 일반화(held-out generalization): 부하 0,1,2로 학습 → 부하 3(안 본 조건)
    으로 오탐률 측정. 낮으면 "4개를 외운 게 아니라 정상의 본질을 배웠다"는 증거.

실행:
    .venv\\Scripts\\python.exe -m src.reliability
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils import get_device, load_config, set_seed


def precision_at(fpr: float, det: float, normal_ratio: float) -> float:
    """오탐률·탐지율 + 정상비율 → 정밀도. (정상이 많을수록 정밀도는 떨어짐)"""
    fp = fpr * normal_ratio
    tp = det * (1.0 - normal_ratio)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    from src.autoencoder_detector import (
        build_autoencoder, fit_threshold_b, recon_errors, train_ae,
    )
    from src.data_loader import load_fault_signals, load_normal_signals
    from src.evaluator import _load_ab, _window_scores
    from src.preprocessing import make_windows, make_windows_from_signals
    from src.spectrogram import apply_spec_scaler, fit_spec_scaler, to_spectrograms

    cfg = load_config()
    device = get_device(cfg)
    length, overlap = cfg["window"]["length"], cfg["window"]["overlap"]

    normal = load_normal_signals(cfg)
    fault_all = load_fault_signals(cfg)
    fault = []                                          # 위치별 3개씩
    for loc in ("IR", "OR", "B"):
        fault += [v["signal"] for v in fault_all.values() if v["location"] == loc][:3]

    # ===================== #4 정밀도 =====================
    thr_a, T_b, ae, scaler = _load_ab(cfg, device)

    def b_window_scores(signals):
        out = []
        for s in signals:
            _, sb = _window_scores(s, cfg, thr_a, ae, scaler, T_b, device)
            out.append(sb)
        return np.concatenate(out)

    nb = b_window_scores(list(normal.values()))
    fb = b_window_scores(fault)
    fpr = float((nb > 1).mean())
    det = float((fb > 1).mean())

    print("=" * 60)
    print("#4 정밀도 (B가 '고장'이라 하면 진짜 고장일 확률)")
    print("=" * 60)
    print(f"  B 오탐률 {100*fpr:.2f}% / 탐지율 {100*det:.1f}%")
    for ratio, label in [(0.5, "50 : 50"), (0.90, "정상90 : 고장10"),
                         (0.95, "정상95 : 고장5"), (0.99, "정상99 : 고장1")]:
        p = precision_at(fpr, det, ratio)
        print(f"  정상:고장 = {label:<14} → 정밀도 {100*p:5.1f}%")
    print("  ※ 정상이 많아질수록(현실) 정밀도가 떨어진다 — 오탐이 누적되기 때문")

    # ===================== #5 안 본 부하 일반화 =====================
    print("\n" + "=" * 60)
    print("#5 안 본 부하 일반화 (부하 0,1,2 학습 → 부하 3 시험)")
    print("=" * 60)
    set_seed(cfg["seed"])
    items = sorted(normal.items())                      # normal_0..3 = 부하 0..3
    train_sigs = dict(items[:3])                         # 부하 0,1,2
    test_name, test_sig = items[3]                       # 부하 3 (안 본 조건)

    tr_specs = to_spectrograms(make_windows_from_signals(train_sigs, length, overlap), cfg)
    sc = fit_spec_scaler(tr_specs)
    tr_n = apply_spec_scaler(tr_specs, sc)

    model = build_autoencoder(cfg)
    print(f"  학습 중 (부하 0,1,2, {len(tr_n)} 윈도우)...")
    train_ae(model, tr_n, cfg, device=device)
    thr = fit_threshold_b(recon_errors(model, tr_n, device), cfg)

    # 안 본 정상(부하 3) 오탐률
    te_specs = apply_spec_scaler(to_spectrograms(make_windows(test_sig, length, overlap), cfg), sc)
    te_err = recon_errors(model, te_specs, device)
    fpr_unseen = float((te_err > thr["threshold"]).mean())

    # 고장 탐지율 (같은 모델)
    f_err = []
    for s in fault:
        f_err.append(recon_errors(model, apply_spec_scaler(to_spectrograms(make_windows(s, length, overlap), cfg), sc), device))
    det_unseen = float((np.concatenate(f_err) > thr["threshold"]).mean())

    print(f"  임계값 {thr['threshold']:.4f}")
    print(f"  ▶ 안 본 정상({test_name}, 부하3) 오탐률: {100*fpr_unseen:.2f}%  (낮을수록 = 안 외우고 일반화)")
    print(f"  ▶ 고장 탐지율: {100*det_unseen:.1f}%")
    verdict = "일반화 잘 됨 (외운 게 아님)" if fpr_unseen < 0.10 else "오탐 높음 — 일반화 주의"
    print(f"  판정: {verdict}")


if __name__ == "__main__":
    main()
