"""[신뢰도 종합] 경로 B 오토인코더 — 6개 지표 한 장 시각화.

① 분리도        : 정상 vs 고장의 점수 분포가 임계값(1.0)을 사이에 두고 갈리는가
② 탐지율·오탐률  : 고장을 잡는 비율 / 정상을 잘못 알람치는 비율
③ 정밀도         : '고장'이라 하면 진짜 고장일 확률 (정상:고장 비율별)
④ 노이즈 강건성  : 화이트/임펄스 노이즈 SNR(dB)별 AUC (꺾은선, 같은 dB 기준)
⑤ 부하별 오탐률  : 전체 부하(0~3) 학습 후 각 부하의 오탐률 (학습 범위 내 신뢰도)

실행:
    .venv\\Scripts\\python.exe -m visualization.reliability_dashboard
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
    from src.evaluator import _load_ab, _window_scores, add_white_noise
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

    print("①~③ 깨끗한 데이터 점수 계산...")
    nb = b_scores(list(normal.values()))
    fb = b_scores(fault)
    fpr = float((nb > 1).mean())
    det = float((fb > 1).mean())
    auc_clean = roc_auc_score(np.r_[np.zeros(len(nb)), np.ones(len(fb))], np.r_[nb, fb])

    # ⑤ 노이즈 강건성 — 화이트·임펄스 모두 SNR(dB) 기준 (같은 dB = 같은 노이즈 전력)
    print("④ 노이즈 강건성 계산...")
    conds = [("노이즈\n없음", None),
             ("화이트\n10dB", ("w", 10)), ("화이트\n0dB", ("w", 0)), ("화이트\n-10dB", ("w", -10)),
             ("임펄스\n10dB", ("i", 10)), ("임펄스\n0dB", ("i", 0)), ("임펄스\n-10dB", ("i", -10))]

    def _impulse_at_snr(sig, snr_db, rng, fs=12000, rate_per_sec=10):
        """임펄스(스파이크)를 목표 SNR(dB)에 맞춰 추가 — 화이트와 동일한 dB 잣대로 비교."""
        sig = np.asarray(sig, dtype=np.float64).copy()
        n = len(sig)
        k = max(1, int(rate_per_sec * n / fs))
        train = np.zeros(n)
        train[rng.integers(0, n, k)] = rng.uniform(0.5, 1.5, k) * rng.choice([-1.0, 1.0], k)
        t_rms = np.sqrt(np.mean(train ** 2))
        if t_rms > 0:
            sig_rms = np.sqrt(np.mean(sig ** 2))
            train *= (sig_rms / 10 ** (snr_db / 20.0)) / t_rms      # 임펄스 전력 → 목표 SNR
        return sig + train

    def noisy(sig, spec, rng):
        if spec is None:
            return sig
        kind, val = spec
        return add_white_noise(sig, val, rng) if kind == "w" else _impulse_at_snr(sig, val, rng)

    noise_auc = []
    for _, spec in conds:
        rng = np.random.default_rng(cfg["seed"])
        nn = b_scores([noisy(s, spec, rng) for s in normal.values()])
        ff = b_scores([noisy(s, spec, rng) for s in fault])
        noise_auc.append(roc_auc_score(np.r_[np.zeros(len(nn)), np.ones(len(ff))], np.r_[nn, ff]))

    # ⑥ 부하별 오탐률 (production 모델 = 전체 부하 0~3 학습)
    print("⑤ 부하별 오탐률 계산...")
    per_load_fpr = {}
    for name, sig in sorted(normal.items()):
        load = int(name.split("_")[-1])
        sb = _window_scores(sig, cfg, thr_a, ae, scaler, T_b, device)[1]
        per_load_fpr[load] = float((sb > 1).mean())

    # ===================== 시각화 =====================
    fig, ax = plt.subplots(2, 3, figsize=(16, 9.5))
    fig.suptitle("경로 B 오토인코더 — 신뢰도 종합 평가 (5개 지표)", fontsize=15, fontweight="bold")
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

    # ④ 노이즈 강건성 (꺾은선, SNR dB) — 화이트 vs 임펄스
    a = ax[1, 0]
    xlab = ["노이즈\n없음", "10dB", "0dB", "-10dB"]
    white = [100 * noise_auc[0], 100 * noise_auc[1], 100 * noise_auc[2], 100 * noise_auc[3]]
    imp = [100 * noise_auc[0], 100 * noise_auc[4], 100 * noise_auc[5], 100 * noise_auc[6]]
    a.plot(xlab, white, "o-", color=B, lw=2.2, label="화이트")
    a.plot(xlab, imp, "s-", color=R, lw=2.2, label="임펄스")
    for x, v in enumerate(white):
        a.text(x, v + 1.0, f"{v:.0f}%", ha="center", va="bottom", color=B, fontsize=8, fontweight="bold")
    for x, v in enumerate(imp):
        a.text(x, v - 1.6, f"{v:.0f}%", ha="center", va="top", color=R, fontsize=8, fontweight="bold")
    a.set_ylim(60, 107); a.set_ylabel("AUC %")
    a.set_title("④ 노이즈 강건성 (SNR dB별 AUC)", fontweight="bold")
    a.legend(loc="lower left"); a.grid(alpha=0.3)

    # ⑤ 부하별 오탐률 (전체 부하 0~3 학습 → 모두 안전)
    a = ax[1, 1]
    loads = sorted(per_load_fpr)
    vals = [100 * per_load_fpr[l] for l in loads]
    a.bar([f"부하 {l}" for l in loads], vals, color=G)
    for i, v in enumerate(vals):
        a.text(i, v + 0.3, f"{v:.2f}%", ha="center", fontweight="bold", fontsize=9)
    a.set_ylim(0, 20); a.set_ylabel("오탐률 %")
    a.set_title("⑤ 부하별 오탐률 (전체 부하 0~3 학습)", fontweight="bold")

    ax[1, 2].axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = save_fig(fig, "reliability_dashboard.png")

    print("\n=== 요약 ===")
    print(f"① 분리도   : 정상 평균 {nb.mean():.2f} / 고장 평균 {fb.mean():.2f} (임계값 1.0)")
    print(f"② 탐지율 {100*det:.1f}% / 오탐률 {100*fpr:.2f}%")
    print(f"③ 정밀도   : 50:50 {precs[0]:.0f}% → 99:1 {precs[-1]:.0f}%")
    print(f"④ 노이즈   : 화이트 최저 {100*min(noise_auc[1:4]):.0f}% / 임펄스 최저 {100*min(noise_auc[4:]):.0f}%")
    print("⑤ 부하별 오탐 : " + " / ".join(f"부하{l} {100*per_load_fpr[l]:.2f}%" for l in sorted(per_load_fpr)))
    print(f"(참고) 깨끗한 데이터 AUC = {auc_clean:.3f}")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
