"""실험: 오토인코더 구조 ablation — 병목 크기·레이어 수에 따른 탐지 성능.

가변 깊이 Conv 오토인코더(출력은 입력 크기로 보간)로 다음을 비교한다:
  · 기준        : Conv 3단, 병목 128
  · 병목 축소   : Conv 3단, 병목 32
  · 레이어 감소 : Conv 2단, 병목 128
  · 레이어 증가 : Conv 4단, 병목 128
지표: 파라미터 수 / 노이즈 없음 AUC·오탐률 / 임펄스 강 AUC(강건성).

실행:
    .venv\\Scripts\\python.exe -m visualization.ablation_architecture
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from visualization._common import plt, save_fig


class ParamAE(nn.Module):
    """가변 깊이 Conv 오토인코더. 출력은 입력 크기로 보간(임의 깊이 허용)."""

    def __init__(self, in_ch, input_hw, n_stages, latent_dim, base=16):
        super().__init__()
        self.input_hw = tuple(int(v) for v in input_hw)
        chs = [base * (2 ** i) for i in range(n_stages)]        # 예: [16,32,64]
        enc, c = [], in_ch
        for o in chs:
            enc += [nn.Conv2d(c, o, 3, stride=2, padding=1), nn.ReLU(inplace=True)]; c = o
        self.enc = nn.Sequential(*enc)
        with torch.no_grad():
            sh = self.enc(torch.zeros(1, in_ch, *self.input_hw)).shape[1:]
        self._sh = tuple(sh); flat = int(np.prod(sh))
        self.efc = nn.Linear(flat, latent_dim)
        self.dfc = nn.Linear(latent_dim, flat)
        dec = []
        for i in range(n_stages):
            ci = chs[n_stages - 1 - i]
            co = chs[n_stages - 2 - i] if i < n_stages - 1 else in_ch
            dec += [nn.ConvTranspose2d(ci, co, 3, stride=2, padding=1, output_padding=1)]
            dec += [nn.ReLU(inplace=True)] if i < n_stages - 1 else []
        self.dec = nn.Sequential(*dec)

    def forward(self, x):
        z = self.efc(self.enc(x).flatten(1))
        d = self.dfc(z).view(-1, *self._sh)
        return F.interpolate(self.dec(d), size=self.input_hw, mode="bilinear", align_corners=False)


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

    def specs(sig, impulse=False, rng=None):
        s = add_impulse_noise(sig, 10, rng) if impulse else sig
        w = make_windows(s, L, ov)
        return apply_spec_scaler(to_spectrograms(w, cfg), sc) if len(w) else None

    def evaluate(n_stages, latent):
        set_seed(cfg["seed"])
        m = ParamAE(in_ch, hw, n_stages, latent).to(device)
        train_ae(m, tr_n, cfg, device=device)
        T = fit_threshold_b(recon_errors(m, tr_n, device), cfg)["threshold"]
        params = sum(p.numel() for p in m.parameters())

        def errs(impulse):
            rng = np.random.default_rng(cfg["seed"])
            ne = np.concatenate([recon_errors(m, specs(s, impulse, rng), device) for s in normal.values()])
            fe = np.concatenate([recon_errors(m, specs(s, impulse, rng), device) for s in fault])
            return ne, fe

        ne, fe = errs(False)
        auc_clean = roc_auc_score(np.r_[np.zeros(len(ne)), np.ones(len(fe))], np.r_[ne, fe])
        fpr_clean = float((ne > T).mean())
        nei, fei = errs(True)
        auc_imp = roc_auc_score(np.r_[np.zeros(len(nei)), np.ones(len(fei))], np.r_[nei, fei])
        return {"params": params, "auc_clean": auc_clean, "fpr_clean": fpr_clean, "auc_imp": auc_imp}

    variants = {
        "기준\n(3단·128)": (3, 128),
        "병목축소\n(3단·32)": (3, 32),
        "레이어감소\n(2단·128)": (2, 128),
        "레이어증가\n(4단·128)": (4, 128),
    }
    res = {}
    for name, (ns, lat) in variants.items():
        print(f"학습: {name.replace(chr(10),' ')} ...")
        res[name] = evaluate(ns, lat)

    names = list(variants)
    x = np.arange(len(names))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.bar(x, [100 * res[n]["auc_imp"] for n in names], color="#2980b9")
    for i, n in enumerate(names):
        ax1.text(i, 100 * res[n]["auc_imp"] + 1, f"{100*res[n]['auc_imp']:.1f}%", ha="center", fontweight="bold", fontsize=9)
    ax1.set_ylim(0, 105); ax1.axhline(50, color="gray", ls=":", lw=1)
    ax1.set_title("임펄스 강 AUC (강건성, 높을수록 좋음)", fontweight="bold"); ax1.set_ylabel("AUC %")

    ax2.bar(x, [100 * res[n]["fpr_clean"] for n in names], color="#e67e22")
    for i, n in enumerate(names):
        ax2.text(i, 100 * res[n]["fpr_clean"] + 0.05, f"{100*res[n]['fpr_clean']:.2f}%", ha="center", fontweight="bold", fontsize=9)
    ax2.set_ylim(0, max(2.0, max(100 * res[n]["fpr_clean"] for n in names) * 1.5))
    ax2.set_title("노이즈 없음 오탐률 (낮을수록 좋음)", fontweight="bold"); ax2.set_ylabel("오탐률 %")
    for ax in (ax1, ax2):
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    fig.suptitle("오토인코더 구조 ablation — 병목 크기·레이어 수에 따른 탐지 성능", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = save_fig(fig, "ablation_architecture.png")

    print("\n=== 구조 ablation (노이즈 없음 AUC는 전부 1.0 예상) ===")
    print(f"{'구조':<16} {'파라미터':>10} {'노이즈없음AUC':>12} {'노이즈없음오탐':>12} {'임펄스강AUC':>11}")
    for n in names:
        r = res[n]
        print(f"{n.replace(chr(10),' '):<16} {r['params']:>10,} {r['auc_clean']:>12.3f} {100*r['fpr_clean']:>11.2f}% {100*r['auc_imp']:>10.1f}%")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
