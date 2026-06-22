"""노이즈 학습 범위 sweep — 고장 탐지를 회복하는 화이트 노이즈 범위 찾기.

증강 학습 노이즈를 약/중/강으로 바꿔가며 ① 노이즈 정상 오탐률 ② 고장 탐지율을 비교.
'오탐은 낮추되 고장 탐지는 유지'되는 적정 범위를 찾는다. (임펄스는 모든 설정 공통 中간 세기)

실행:
    .venv\\Scripts\\python.exe sweep_noise_range.py
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

from src.autoencoder_detector import build_autoencoder, fit_threshold_b, load_model, recon_errors, train_ae
from src.data_loader import load_fault_signals, load_normal_signals
from src.evaluator import add_white_noise
from src.preprocessing import make_windows, make_windows_from_signals
from src.spectrogram import apply_spec_scaler, fit_spec_scaler, load_spec_scaler, to_spectrograms
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
    md = cfg["artifacts"]["models"]
    normal = load_normal_signals(cfg)
    fa = load_fault_signals(cfg)
    fault = []
    for loc in ("IR", "OR", "B"):
        fault += [v["signal"] for v in fa.values() if v["location"] == loc][:1]

    def build_aug(white_dbs, rng):
        sigs = {}
        for name, sig in normal.items():
            sig = np.asarray(sig, float)
            sigs[f"{name}_c"] = sig
            for db in white_dbs:
                sigs[f"{name}_w{db}"] = add_white_noise(sig, db, rng)
            for j in range(3):                                  # 임펄스 中간(SNR 0~6 무작위)
                sigs[f"{name}_i{j}"] = impulse_random(sig, rng.uniform(0, 6), rng)
        return sigs

    def train_cfg(white_dbs):
        rng = np.random.default_rng(cfg["seed"])
        specs = to_spectrograms(make_windows_from_signals(build_aug(white_dbs, rng), L, ov), cfg)
        sc = fit_spec_scaler(specs); tr = apply_spec_scaler(specs, sc)
        set_seed(cfg["seed"]); m = build_autoencoder(cfg)
        train_ae(m, tr, cfg, device=device, noise_std=0.0)
        T = fit_threshold_b(recon_errors(m, tr, device), cfg)["threshold"]
        return m, sc, T

    CONFIGS = {"약(+10/+7/+4)": [10, 7, 4], "중(+6/+3/0)": [6, 3, 0], "강(+2/-2/-5)": [2, -2, -5]}
    models = {}
    # production(클린)
    import json
    models["production(클린)"] = (load_model(cfg, f"{md}/autoencoder.pth", device),
                                  load_spec_scaler(f"{md}/spec_scaler.pkl"),
                                  json.loads(resolve_path(cfg["artifacts"]["thresholds"]).read_text("utf-8"))["path_b"]["threshold"])
    for tag, dbs in CONFIGS.items():
        print(f"학습: {tag} ...")
        models[tag] = train_cfg(dbs)

    conds = [("없음", lambda s, r: s),
             ("화이트\n+10", lambda s, r: add_white_noise(s, 10, r)),
             ("화이트\n+3", lambda s, r: add_white_noise(s, 3, r)),
             ("화이트\n-3", lambda s, r: add_white_noise(s, -3, r)),
             ("화이트\n-5", lambda s, r: add_white_noise(s, -5, r)),
             ("임펄스\n랜덤", lambda s, r: impulse_random(s, 0, r))]

    def err(m, sc, sig):
        w = make_windows(np.asarray(sig, float), L, ov)
        return recon_errors(m, apply_spec_scaler(to_spectrograms(w, cfg), sc), device) if len(w) else np.empty(0)

    fpr = {k: [] for k in models}; det = {k: [] for k in models}
    for _, fn in conds:
        rng = np.random.default_rng(cfg["seed"])
        nn = [fn(np.asarray(s, float), rng) for s in normal.values()]
        nf = [fn(np.asarray(s, float), rng) for s in fault]
        for tag, (m, sc, T) in models.items():
            ne = np.concatenate([err(m, sc, s) for s in nn])
            fe = np.concatenate([err(m, sc, s) for s in nf])
            fpr[tag].append(float((ne > T).mean())); det[tag].append(float((fe > T).mean()))

    x = np.arange(len(conds)); names = [c[0] for c in conds]
    cols = {"production(클린)": "#e74c3c", "약(+10/+7/+4)": "#27ae60", "중(+6/+3/0)": "#2980b9", "강(+2/-2/-5)": "#8e44ad"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.2))
    for k in models:
        a1.plot(x, [100 * v for v in fpr[k]], "o-", color=cols[k], lw=2, label=k)
        a2.plot(x, [100 * v for v in det[k]], "o-", color=cols[k], lw=2, label=k)
    a1.set_title("① 노이즈 정상 오탐률 (낮을수록 좋음)", fontweight="bold"); a1.set_ylabel("오탐률 %")
    a2.set_title("② 고장 탐지율 (높을수록 좋음)", fontweight="bold"); a2.set_ylabel("탐지율 %")
    for a in (a1, a2):
        a.set_xticks(x); a.set_xticklabels(names, fontsize=8); a.set_ylim(0, 105); a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.suptitle("화이트 노이즈 학습 범위 sweep — 오탐↓ 유지하며 고장 탐지 회복점 찾기", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = resolve_path("outputs/figures/noise_range_sweep.png")
    fig.savefig(out, dpi=120); plt.close(fig)

    print("\n=== 조건별 [오탐률% / 고장탐지%] ===")
    print(f"{'조건':<8}" + "".join(f"{k.split('(')[0]:>16}" for k in models))
    for i, (nm, _) in enumerate(conds):
        row = "".join(f"{100*fpr[k][i]:>7.0f}/{100*det[k][i]:<8.0f}" for k in models)
        print(f"{nm.replace(chr(10),''):<8}{row}")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
