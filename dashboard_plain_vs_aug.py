"""일반 오토인코더 vs 노이즈 증강 오토인코더 — 5개 지표 비교 대시보드.

①분리도 ②탐지율 ③오탐율 ④정밀도 (모두 클린)  +  ⑤⑥ 노이즈 강건성(입력 시점 노이즈) 오탐율·탐지율.
일반: 클린만 학습 / 증강: 클린+화이트(+6/+3/0)+임펄스(중간 랜덤) 학습.

실행:
    .venv\\Scripts\\python.exe dashboard_plain_vs_aug.py
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
from src.reliability import precision_at
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
        fault += [v["signal"] for v in fa.values() if v["location"] == loc][:2]
    fault_nz = [fault[0], fault[2], fault[4]]

    def train_on(sigs):
        specs = to_spectrograms(make_windows_from_signals(sigs, L, ov), cfg)
        sc = fit_spec_scaler(specs); trn = apply_spec_scaler(specs, sc)
        set_seed(cfg["seed"]); m = build_autoencoder(cfg)
        train_ae(m, trn, cfg, device=device, noise_std=0.0)
        return m, sc, fit_threshold_b(recon_errors(m, trn, device), cfg)["threshold"]

    clean = {n: np.asarray(s, float) for n, s in normal.items()}
    print("학습: 일반 ..."); plain = train_on(clean)
    rng = np.random.default_rng(cfg["seed"]); aug = {}
    for n, s in normal.items():
        s = np.asarray(s, float); aug[f"{n}_c"] = s
        for db in (6, 3, 0):
            aug[f"{n}_w{db}"] = add_white_noise(s, db, rng)
        for j in range(3):
            aug[f"{n}_i{j}"] = impulse_random(s, rng.uniform(0, 6), rng)
    print("학습: 노이즈 증강 ..."); augm = train_on(aug)
    MODELS = {"일반 오토인코더": plain, "노이즈 증강 오토인코더": augm}

    def score(model, sc, T, sig):
        w = make_windows(np.asarray(sig, float), L, ov)
        e = recon_errors(model, apply_spec_scaler(to_spectrograms(w, cfg), sc), device) if len(w) else np.empty(0)
        return e / T

    stat = {}
    for tag, (m, sc, T) in MODELS.items():
        ns = np.concatenate([score(m, sc, T, s) for s in normal.values()])
        fs_ = np.concatenate([score(m, sc, T, s) for s in fault])
        stat[tag] = dict(n_mean=ns.mean(), f_mean=fs_.mean(),
                         det=float((fs_ > 1).mean()), fpr=float((ns > 1).mean()))

    conds = [("없음", lambda s, r: s), ("화이트\n+6", lambda s, r: add_white_noise(s, 6, r)),
             ("화이트\n+3", lambda s, r: add_white_noise(s, 3, r)), ("화이트\n0", lambda s, r: add_white_noise(s, 0, r)),
             ("화이트\n-3", lambda s, r: add_white_noise(s, -3, r)), ("화이트\n-5", lambda s, r: add_white_noise(s, -5, r)),
             ("임펄스\n랜덤", lambda s, r: impulse_random(s, 0, r))]
    nz = {tag: dict(det=[], fpr=[]) for tag in MODELS}
    for _, fn in conds:
        for tag, (m, sc, T) in MODELS.items():
            rr = np.random.default_rng(cfg["seed"])
            ne = np.concatenate([score(m, sc, T, fn(np.asarray(s, float), rr)) for s in normal.values()])
            fe = np.concatenate([score(m, sc, T, fn(np.asarray(s, float), rr)) for s in fault_nz])
            nz[tag]["fpr"].append(float((ne > 1).mean())); nz[tag]["det"].append(float((fe > 1).mean()))

    # ===== 시각화 =====
    cP, cA = "#e74c3c", "#2980b9"
    cols = {"일반 오토인코더": cP, "노이즈 증강 오토인코더": cA}
    tags = list(MODELS)
    fig, ax = plt.subplots(2, 3, figsize=(16, 9.5))
    fig.suptitle("일반 오토인코더 vs 노이즈 증강 오토인코더 — 5개 지표 비교", fontsize=15, fontweight="bold")

    a = ax[0, 0]; w = 0.35                                    # ① 분리도(평균 이상점수)
    xc = np.arange(2)
    for i, tag in enumerate(tags):
        a.bar(xc + (i - 0.5) * w, [stat[tag]["n_mean"], stat[tag]["f_mean"]], w, color=cols[tag], label=tag)
    a.axhline(1.0, color="k", ls="--", lw=1.2); a.text(1.5, 1.05, "임계값", fontsize=8)
    a.set_xticks(xc); a.set_xticklabels(["정상", "고장"]); a.set_ylabel("평균 이상점수")
    a.set_title("① 분리도 (평균 점수, 1=임계값)", fontweight="bold"); a.legend(fontsize=8)

    a = ax[0, 1]                                              # ② 탐지율(클린)
    a.bar(xc * 0 + np.arange(2) * 0.5, [100 * stat[t]["det"] for t in tags], 0.4, color=[cP, cA])
    for i, t in enumerate(tags):
        a.text(i * 0.5, 100 * stat[t]["det"] + 1, f"{100*stat[t]['det']:.0f}%", ha="center", fontweight="bold")
    a.set_xticks(np.arange(2) * 0.5); a.set_xticklabels(["일반", "증강"]); a.set_ylim(0, 110)
    a.set_title("② 고장 탐지율 (클린)", fontweight="bold"); a.set_ylabel("%")

    a = ax[0, 2]                                              # ③ 오탐율(클린)
    vals = [100 * stat[t]["fpr"] for t in tags]
    a.bar(np.arange(2) * 0.5, vals, 0.4, color=[cP, cA])
    for i, v in enumerate(vals):
        a.text(i * 0.5, v + 0.02, f"{v:.2f}%", ha="center", fontweight="bold")
    a.set_xticks(np.arange(2) * 0.5); a.set_xticklabels(["일반", "증강"]); a.set_ylim(0, max(1, max(vals) * 1.5))
    a.set_title("③ 정상 오탐율 (클린)", fontweight="bold"); a.set_ylabel("%")

    a = ax[1, 0]                                              # ④ 정밀도
    ratios = [0.5, 0.9, 0.95, 0.99]; xr = [f"{int(r*100)}:{int((1-r)*100)}" for r in ratios]
    for t in tags:
        a.plot(xr, [100 * precision_at(stat[t]["fpr"], stat[t]["det"], r) for r in ratios], "o-", color=cols[t], lw=2, label=t)
    a.set_ylim(0, 105); a.set_xlabel("정상 : 고장 비율"); a.set_ylabel("정밀도 %")
    a.set_title("④ 정밀도 ('고장'이라면 맞을 확률)", fontweight="bold"); a.legend(fontsize=8)

    xx = np.arange(len(conds)); xl = [c[0] for c in conds]
    a = ax[1, 1]                                              # ⑤ 노이즈 오탐율
    for t in tags:
        a.plot(xx, [100 * v for v in nz[t]["fpr"]], "o-", color=cols[t], lw=2.2, label=t)
    a.set_xticks(xx); a.set_xticklabels(xl, fontsize=8); a.set_ylim(0, 105); a.grid(alpha=0.3); a.legend(fontsize=8)
    a.set_title("⑤ 노이즈 강건성 — 오탐율 (입력 노이즈)", fontweight="bold"); a.set_ylabel("오탐율 %")

    a = ax[1, 2]                                              # ⑥ 노이즈 탐지율
    for t in tags:
        a.plot(xx, [100 * v for v in nz[t]["det"]], "o-", color=cols[t], lw=2.2, label=t)
    a.set_xticks(xx); a.set_xticklabels(xl, fontsize=8); a.set_ylim(0, 110); a.grid(alpha=0.3); a.legend(fontsize=8)
    a.set_title("⑥ 노이즈 강건성 — 탐지율 (입력 노이즈)", fontweight="bold"); a.set_ylabel("탐지율 %")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = resolve_path("outputs/figures/dashboard_plain_vs_aug.png")
    fig.savefig(out, dpi=120); plt.close(fig)

    print("\n=== 클린 ===")
    for t in tags:
        print(f"  {t}: 분리 정상{stat[t]['n_mean']:.2f}/고장{stat[t]['f_mean']:.2f}, 탐지 {100*stat[t]['det']:.0f}%, 오탐 {100*stat[t]['fpr']:.2f}%")
    print("=== 입력 노이즈 오탐율 [일반 / 증강] ===")
    for i, (nm, _) in enumerate(conds):
        print(f"  {nm.replace(chr(10),''):<8} {100*nz[tags[0]]['fpr'][i]:>4.0f}% / {100*nz[tags[1]]['fpr'][i]:>3.0f}%")
    print("주의: ⑥에서 일반의 높은 탐지율은 '전부 이상 처리'(⑤ 오탐 100%) 탓 = 분별 아님.")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
