"""[신뢰도] ROC 곡선 + AUC — 경로 A·B·융합 (깨끗한 데이터).

모델 신뢰도 = "임계값을 어디 두든 정상/고장을 얼마나 잘 가르나".
  - ROC 곡선: 임계값을 0~100% 쭉 돌리며 (오탐률, 탐지율)을 찍은 선.
  - 곡선이 왼쪽 위(오탐↓·탐지↑)에 붙을수록, AUC(곡선 아래 면적)가 1에 가까울수록 신뢰.
  - 대각선(점선) = 찍기(AUC 0.5).

기존 evaluator.py(노이즈 강건성)와 짝: 이건 '깨끗한 조건의 분별력', 저건 '노이즈 견딤'.

독립 실행 (A·B 결과물 필요):
    .venv\\Scripts\\python.exe -m visualization.reliability_roc
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from visualization._common import plt, save_fig


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    from src.data_loader import load_fault_signals, load_normal_signals
    from src.evaluator import _load_ab, _window_scores
    from src.utils import get_device, load_config

    cfg = load_config()
    device = get_device(cfg)
    thr_a, T_b, ae, scaler = _load_ab(cfg, device)

    normal = list(load_normal_signals(cfg).values())            # 정상 4
    fault_all = load_fault_signals(cfg)
    fault = []                                                  # 위치별 3개씩(매끈한 곡선용)
    for loc in ("IR", "OR", "B"):
        fault += [v["signal"] for v in fault_all.values() if v["location"] == loc][:3]

    # 정상/고장 윈도우별 A·B 점수 수집 (깨끗한 데이터)
    def collect(signals):
        sa_l, sb_l = [], []
        for s in signals:
            a, b = _window_scores(s, cfg, thr_a, ae, scaler, T_b, device)
            sa_l.append(a); sb_l.append(b)
        return np.concatenate(sa_l), np.concatenate(sb_l)

    print("=== ROC·AUC 계산 중 (깨끗한 데이터) ===")
    na, nb = collect(normal)
    fa, fb = collect(fault)
    y = np.r_[np.zeros(len(na)), np.ones(len(fa))]              # 0=정상, 1=고장

    detectors = {
        "경로 A (통계)": (np.r_[na, fa], "#e74c3c"),
        "경로 B (오토인코더)": (np.r_[nb, fb], "#2980b9"),
        "A+B 융합(max)": (np.r_[np.maximum(na, nb), np.maximum(fa, fb)], "#e67e22"),
    }

    fig, ax = plt.subplots(figsize=(7, 6.5))
    for name, (scores, color) in detectors.items():
        fpr, tpr, _ = roc_curve(y, scores)
        auc = roc_auc_score(y, scores)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name}  (AUC={auc:.3f})")
        print(f"  {name}: AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "k:", lw=1, label="찍기 (AUC=0.5)")
    ax.set_xlabel("오탐률 (False Positive Rate)")
    ax.set_ylabel("탐지율 (True Positive Rate)")
    ax.set_title("신뢰도 — ROC 곡선·AUC (깨끗한 데이터)\n왼쪽 위에 붙을수록·AUC 1에 가까울수록 신뢰",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = save_fig(fig, "reliability_roc.png")
    print(f"\n저장: {out}")
    print("해석: 깨끗한 조건에선 셋 다 분별력이 높다(AUC↑). 노이즈 강건성은 evaluator.py 참고.")


if __name__ == "__main__":
    main()
