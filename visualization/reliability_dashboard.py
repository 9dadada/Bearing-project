"""[신뢰도 종합] 경로 B 오토인코더 — 6개 지표 한 장 시각화.

① 분리도        : 정상 vs 고장의 점수 분포가 임계값(1.0)을 사이에 두고 갈리는가
② 탐지율·오탐률  : 고장을 잡는 비율 / 정상을 잘못 알람치는 비율
③ 정밀도         : '고장'이라 하면 진짜 고장일 확률 (정상:고장 비율별)
④ AUC·ROC       : 임계값 무관 분별력
⑤ 노이즈 강건성  : 화이트/임펄스 노이즈 세기별 AUC
⑥ 부하별 오탐률  : 전체 부하(0~3) 학습 후 각 부하의 오탐률 (학습 범위 내 신뢰도)

실행:
    .venv\\Scripts\\python.exe -m visualization.reliability_dashboard
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
    from src.evaluator import _load_ab, _window_scores, add_impulse_noise, add_white_noise
    from src.reliability import precision_at
    from src.utils import get_device, load_config

    cfg = load_config()
    device = get_device(cfg)
    thr_a, T_b, ae, scaler = _load_ab(cfg, device)

    normal = load_normal_signals(cfg)
    fault_all = load_fault_signals(cfg)
    fault = []                                              # 위치별 2개 (속도)
    for loc in ("IR", "OR", "B"):
        fault += [v["signal"] for v in fault_all.values() if v["location"] == loc][:2]

    def b_scores(signals):
        return np.concatenate([_window_scores(s, cfg, thr_a, ae, scaler, T_b, device)[1] for s in signals])

    print("① ~ ④ 깨끗한 데이터 점수 계산...")
    nb = b_scores(list(normal.values()))
    fb = b_scores(fault)
    fpr = float((nb > 1).mean())
    det = float((fb > 1).mean())
    auc_clean = roc_auc_score(np.r_[np.zeros(len(nb)), np.ones(len(fb))], np.r_[nb, fb])

    # ⑤ 노이즈 강건성
    print("⑤ 노이즈 강건성 계산...")
    conds = [("깨끗", None), ("화이트\n+10dB", ("w", 10)), ("화이트\n0dB", ("w", 0)),
             ("화이트\n-10dB", ("w", -10)), ("임펄스\n약", ("i", 3)),
             ("임펄스\n중", ("i", 5)), ("임펄스\n강", ("i", 10))]

    def noisy(sig, spec, rng):
        if spec is None:
            return sig
        kind, val = spec
        return add_white_noise(sig, val, rng) if kind == "w" else add_impulse_noise(sig, val, rng)

    noise_auc = []
    for _, spec in conds:
        rng = np.random.default_rng(cfg["seed"])
        nn = b_scores([noisy(s, spec, rng) for s in normal.values()])
        ff = b_scores([noisy(s, spec, rng) for s in fault])
        noise_auc.append(roc_auc_score(np.r_[np.zeros(len(nn)), np.ones(len(ff))], np.r_[nn, ff]))

    # ⑥ 부하별 오탐률 (production 모델 = 전체 부하 0~3 학습)
    print("⑥ 부하별 오탐률 계산...")
    per_load_fpr = {}
    for name, sig in sorted(normal.items()):
        load = int(name.split("_")[-1])
        sb = _window_scores(sig, cfg, thr_a, ae, scaler, T_b, device)[1]
        per_load_fpr[load] = float((sb > 1).mean())

    # ===================== 시각화 =====================
    fig, ax = plt.subplots(2, 3, figsize=(16, 9.5))
    fig.suptitle("경로 B 오토인코더 — 신뢰도 종합 평가 (6개 지표)", fontsize=15, fontweight="bold")
    B, G, R = "#2980b9", "#27ae60", "#e74c3c"

    # ① 분리도
    a = ax[0, 0]
    bins = np.linspace(0, max(3, float(fb.max())), 40)
    a.hist(np.clip(nb, 0, bins[-1]), bins=bins, color=B, alpha=0.7, label="정상")
    a.hist(np.clip(fb, 0, bins[-1]), bins=bins, color=R, alpha=0.7, label="고장")
    a.axvline(1.0, color="k", ls="--", lw=1.5, label="임계값")
    a.set_title("① 분리도 (정상 vs 고장 점수)", fontweight="bold")
    a.set_xlabel("이상점수 (1=임계값)"); a.set_ylabel("윈도우 수"); a.legend()

    # ② 탐지율·오탐률
    a = ax[0, 1]
    a.bar(["탐지율", "오탐률"], [100 * det, 100 * fpr], color=[G, R])
    for i, v in enumerate([100 * det, 100 * fpr]):
        a.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
    a.set_ylim(0, 110); a.set_ylabel("%")
    a.set_title("② 탐지율 · 오탐률", fontweight="bold")

    # ③ 정밀도
    a = ax[0, 2]
    ratios = [0.5, 0.9, 0.95, 0.99]
    precs = [100 * precision_at(fpr, det, r) for r in ratios]
    a.plot([f"{int(r*100)}:{int((1-r)*100)}" for r in ratios], precs, "o-", color=B, lw=2)
    for x, v in enumerate(precs):
        a.text(x, v + 1.5, f"{v:.0f}%", ha="center", fontweight="bold")
    a.set_ylim(0, 105); a.set_ylabel("정밀도 %"); a.set_xlabel("정상 : 고장 비율")
    a.set_title("③ 정밀도 ('고장'이라면 맞을 확률)", fontweight="bold")

    # ④ ROC·AUC
    a = ax[1, 0]
    fp, tp, _ = roc_curve(np.r_[np.zeros(len(nb)), np.ones(len(fb))], np.r_[nb, fb])
    a.plot(fp, tp, color=B, lw=2.5, label=f"AUC={auc_clean:.3f}")
    a.plot([0, 1], [0, 1], "k:", lw=1, label="찍기(0.5)")
    a.set_title("④ AUC · ROC (분별력)", fontweight="bold")
    a.set_xlabel("오탐률"); a.set_ylabel("탐지율"); a.legend(loc="lower right")

    # ⑤ 노이즈 강건성
    a = ax[1, 1]
    colors = ["#555"] + [B] * 3 + [R] * 3
    a.bar(range(len(conds)), [100 * v for v in noise_auc], color=colors)
    a.set_xticks(range(len(conds)))
    a.set_xticklabels([c[0] for c in conds], fontsize=8)
    a.set_ylim(0, 110); a.set_ylabel("AUC %")
    a.axhline(50, color="k", ls=":", lw=1)
    a.set_title("⑤ 노이즈 강건성 (세기별 AUC)", fontweight="bold")

    # ⑥ 부하별 오탐률 (전체 부하 학습 → 모두 안전)
    a = ax[1, 2]
    loads = sorted(per_load_fpr)
    vals = [100 * per_load_fpr[l] for l in loads]
    a.bar([f"부하 {l}" for l in loads], vals, color=G)
    for i, v in enumerate(vals):
        a.text(i, v + 0.04, f"{v:.2f}%", ha="center", fontweight="bold", fontsize=9)
    a.set_ylim(0, max(2.0, max(vals) * 1.6)); a.set_ylabel("오탐률 %")
    a.set_title("⑥ 부하별 오탐률 (전체 부하 0~3 학습)", fontweight="bold")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = save_fig(fig, "reliability_dashboard.png")

    print("\n=== 요약 ===")
    print(f"① 분리도   : 정상 평균 {nb.mean():.2f} / 고장 평균 {fb.mean():.2f} (임계값 1.0)")
    print(f"② 탐지율 {100*det:.1f}% / 오탐률 {100*fpr:.2f}%")
    print(f"③ 정밀도   : 50:50 {precs[0]:.0f}% → 99:1 {precs[-1]:.0f}%")
    print(f"④ AUC      : {auc_clean:.3f}")
    print(f"⑤ 노이즈   : 화이트 최저 {100*min(noise_auc[1:4]):.0f}% / 임펄스 최저 {100*min(noise_auc[4:]):.0f}%")
    print("⑥ 부하별 오탐 : " + " / ".join(f"부하{l} {100*per_load_fpr[l]:.2f}%" for l in sorted(per_load_fpr)))
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
