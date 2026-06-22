"""[강건성] 노이즈 세기별 성능 — 화이트 / 임펄스 (경로 A vs B).

"왜 경로 B를 골랐나"의 근거 그림:
  노이즈가 세질수록 A(통계)는 무너지고 B(오토인코더)는 버틴다.
세로축 = AUC(임계값 무관 분리도, 1=완벽 / 0.5=찍기), 가로축 = 노이즈 세기.
  - 왼쪽: 화이트 노이즈 (깨끗 → +10 → 0 → −10 dB)
  - 오른쪽: 임펄스 노이즈 (깨끗 → 약×3 → 중×5 → 강×10, 무작위)

독립 실행 (A·B 결과물 필요):
    .venv\\Scripts\\python.exe -m visualization.noise_robustness
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
    from src.data_loader import load_fault_signals, load_normal_signals
    from src.evaluator import _load_ab, _window_scores, add_impulse_noise, add_white_noise
    from src.utils import get_device, load_config

    cfg = load_config()
    device = get_device(cfg)
    rng = np.random.default_rng(cfg["seed"])
    thr_a, T_b, ae, scaler = _load_ab(cfg, device)

    normal = list(load_normal_signals(cfg).values())               # 정상 4
    fault_all = load_fault_signals(cfg)
    fault = []                                                     # 위치별 2개씩(속도)
    for loc in ("IR", "OR", "B"):
        fault += [v["signal"] for v in fault_all.values() if v["location"] == loc][:2]

    def auc_ab(transform):
        """노이즈 변환을 적용한 뒤 A·B 각각의 AUC를 구한다."""
        na, nb, fa, fb = [], [], [], []
        for s in normal:
            a, b = _window_scores(transform(s), cfg, thr_a, ae, scaler, T_b, device)
            na.append(a); nb.append(b)
        for s in fault:
            a, b = _window_scores(transform(s), cfg, thr_a, ae, scaler, T_b, device)
            fa.append(a); fb.append(b)
        na, nb = np.concatenate(na), np.concatenate(nb)
        fa, fb = np.concatenate(fa), np.concatenate(fb)
        y = np.r_[np.zeros(len(na)), np.ones(len(fa))]
        auc_a = roc_auc_score(y, np.r_[na, fa])
        auc_b = roc_auc_score(np.r_[np.zeros(len(nb)), np.ones(len(fb))], np.r_[nb, fb])
        return auc_a, auc_b

    # 화이트: 노이즈 없음 + 3단계
    white_x = ["노이즈\n없음", "+10dB", "0dB", "-10dB"]
    white_fns = [lambda s: s,
                 lambda s: add_white_noise(s, 10, rng),
                 lambda s: add_white_noise(s, 0, rng),
                 lambda s: add_white_noise(s, -10, rng)]
    # 임펄스: 노이즈 없음 + 약/중/강
    imp_x = ["노이즈\n없음", "약(×3)", "중(×5)", "강(×10)"]
    imp_fns = [lambda s: s,
               lambda s: add_impulse_noise(s, 3, rng),
               lambda s: add_impulse_noise(s, 5, rng),
               lambda s: add_impulse_noise(s, 10, rng)]

    print("=== 노이즈 세기별 AUC 계산 중 ===")
    white = [auc_ab(fn) for fn in white_fns]
    imp = [auc_ab(fn) for fn in imp_fns]
    for label, xs, vals in (("화이트", white_x, white), ("임펄스", imp_x, imp)):
        for x, (a, b) in zip(xs, vals):
            print(f"  {label} {x:<8} A={a:.3f}  B={b:.3f}")

    # ---- 그림: 좌(화이트) 우(임펄스) ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, xs, vals, title in (
        (ax1, white_x, white, "화이트 노이즈 (점점 강하게)"),
        (ax2, imp_x, imp, "임펄스 노이즈 (점점 강하게, 무작위)"),
    ):
        xi = np.arange(len(xs))
        ax.plot(xi, [v[0] for v in vals], "o-", color="#e74c3c", lw=2, label="경로 A (통계)")
        ax.plot(xi, [v[1] for v in vals], "s-", color="#2980b9", lw=2, label="경로 B (오토인코더)")
        ax.axhline(0.5, color="gray", ls=":", lw=1, label="찍기(0.5)")
        ax.set_xticks(xi); ax.set_xticklabels(xs)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("노이즈 세기 →")
        ax.set_ylim(0.4, 1.02)
        ax.grid(alpha=0.3); ax.legend()
    ax1.set_ylabel("AUC (분리도, 1=완벽 / 0.5=찍기)")
    fig.suptitle("노이즈 강건성 — 경로 B가 A보다 강건 (B 선택 근거)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = save_fig(fig, "noise_robustness_AvsB.png")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
