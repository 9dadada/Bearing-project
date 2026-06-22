"""노이즈 증강 CNN 학습 + 클린 CNN과 노이즈 강건성 비교.

[학습] train 고장 녹음에 노이즈를 섞어(클린 + 화이트 +6/+3/0 + 임펄스) 라벨 그대로 분류 학습.
       (CPU 부담 줄이려 녹음·변형당 윈도우 캡) → 별도 모델 fault_classifier_aug.pth 저장.
[비교] test 녹음에 입력 노이즈를 단계별로 넣고, 클린 CNN vs 증강 CNN 분류 정확도(녹음 다수결) 비교.

실행:
    .venv\\Scripts\\python.exe cnn_aug_compare.py
"""
from __future__ import annotations

import copy
import json
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

import torch
from torch.utils.data import DataLoader

from src.cnn_fault_classifier import (CLASS_TO_IDX, FaultSpectrogramDataset, build_model,
                                      collect_fault_recordings, load_model as load_cnn,
                                      predict_spectrogram, save_model, train_model)
from src.evaluator import add_white_noise
from src.preprocessing import make_windows
from src.schemas import FAULT_CLASSES
from src.spectrogram import to_spectrograms
from src.utils import get_device, load_config, resolve_path, set_seed

FS = 12000
WIN_CAP = 12          # 녹음·변형당 윈도우 최대 수 (CPU 시간 절약)
EPOCHS = 12


def impulse_random(sig, snr_db, rng, fs=FS, rate=10):
    sig = np.asarray(sig, float).copy(); n = len(sig); k = max(1, int(rate * n / fs))
    tr = np.zeros(n); tr[rng.integers(0, n, k)] = rng.uniform(0.5, 1.5, k) * rng.choice([-1.0, 1.0], k)
    t = np.sqrt(np.mean(tr ** 2))
    if t > 0:
        tr *= (np.sqrt(np.mean(sig ** 2)) / 10 ** (snr_db / 20.0)) / t
    return sig + tr


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config(); dev = get_device(cfg)
    Lc, ov = cfg["window"]["length_c"], cfg["window"]["overlap"]
    md = cfg["artifacts"]["models"]
    set_seed(cfg["seed"]); rng = np.random.default_rng(cfg["seed"])

    recs = collect_fault_recordings(cfg, False)
    man = json.load(open("data/splits/c_split_manifest.json", encoding="utf-8"))
    ids = man["split_rec_ids"]
    by_id = {r.rec_id: r for r in recs}
    train_recs = [by_id[i] for i in ids["train"] if i in by_id]
    val_recs = [by_id[i] for i in ids["val"] if i in by_id]
    test_recs = [by_id[i] for i in ids["test"] if i in by_id]

    def cap(w):
        if len(w) > WIN_CAP:
            w = w[rng.choice(len(w), WIN_CAP, replace=False)]
        return w

    # ---- 증강 train (클린 + 노이즈) ----
    noise_fns = [None,
                 lambda s: add_white_noise(s, 6, rng), lambda s: add_white_noise(s, 3, rng),
                 lambda s: add_white_noise(s, 0, rng), lambda s: impulse_random(s, float(rng.uniform(0, 6)), rng)]
    Xw, y = [], []
    for rec in train_recs:
        sig = np.asarray(rec.signal, float); lab = CLASS_TO_IDX[rec.location]
        for fn in noise_fns:
            w = cap(make_windows(fn(sig) if fn else sig, Lc, ov))
            if len(w): Xw.append(w); y += [lab] * len(w)
    Xtr = to_spectrograms(np.concatenate(Xw), cfg); ytr = np.array(y, np.int64)
    # ---- val (클린) ----
    Xv, yv = [], []
    for rec in val_recs:
        w = make_windows(np.asarray(rec.signal, float), Lc, ov)
        if len(w): Xv.append(w); yv += [CLASS_TO_IDX[rec.location]] * len(w)
    Xval = to_spectrograms(np.concatenate(Xv), cfg); yval = np.array(yv, np.int64)
    print(f"증강 train 윈도우 {len(ytr)} / val {len(yval)}")

    cnt = Counter(ytr.tolist()); N = len(ytr); K = len(FAULT_CLASSES)
    cw = {FAULT_CLASSES[i]: N / (K * cnt.get(i, 1)) for i in range(K)}

    bs = cfg["cnn"]["batch_size"]
    loaders = {"train": DataLoader(FaultSpectrogramDataset(Xtr, ytr), batch_size=bs, shuffle=True),
               "val": DataLoader(FaultSpectrogramDataset(Xval, yval), batch_size=bs)}
    cfg2 = copy.deepcopy(cfg); cfg2["cnn"]["epochs"] = EPOCHS
    model = build_model(num_classes=K)
    print("증강 CNN 학습...")
    model, _, best = train_model(model, loaders, cfg2, cw, device=dev)
    save_model(model, f"{md}/fault_classifier_aug.pth")
    print(f"증강 CNN 저장 (val best {best:.3f})")

    # ---- 비교: 클린 CNN vs 증강 CNN, test 입력 노이즈별 ----
    clean = load_cnn(f"{md}/fault_classifier.pth", num_classes=K, device=dev)
    conds = [("없음", lambda s, r: s), ("화이트\n+6", lambda s, r: add_white_noise(s, 6, r)),
             ("화이트\n+3", lambda s, r: add_white_noise(s, 3, r)), ("화이트\n0", lambda s, r: add_white_noise(s, 0, r)),
             ("화이트\n-3", lambda s, r: add_white_noise(s, -3, r)), ("화이트\n-5", lambda s, r: add_white_noise(s, -5, r)),
             ("임펄스\n랜덤", lambda s, r: impulse_random(s, 0, r))]

    def rec_acc(net):
        accs = []
        for _, fn in conds:
            rr = np.random.default_rng(cfg["seed"]); ok = tot = 0
            for rec in test_recs:
                w = make_windows(fn(np.asarray(rec.signal, float), rr), Lc, ov)
                if len(w) == 0: continue
                votes = [predict_spectrogram(net, s).location.value for s in to_spectrograms(w, cfg)]
                top = Counter(votes).most_common(1)[0][0]
                ok += (top == rec.location); tot += 1
            accs.append(ok / tot if tot else 0)
        return accs

    print("비교 평가...")
    a_clean = rec_acc(clean); a_aug = rec_acc(model)

    x = np.arange(len(conds)); names = [c[0] for c in conds]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(x, [100 * v for v in a_clean], "o-", color="#e74c3c", lw=2.4, label="클린 학습 CNN")
    ax.plot(x, [100 * v for v in a_aug], "o-", color="#2980b9", lw=2.4, label="노이즈 증강 CNN")
    for i in range(len(conds)):
        ax.text(i, 100 * a_aug[i] + 1.5, f"{100*a_aug[i]:.0f}%", ha="center", color="#2980b9", fontsize=8, fontweight="bold")
    ax.axhline(33.3, color="gray", ls=":", lw=1); ax.text(0, 35, "찍기 33%", color="gray", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8); ax.set_ylim(0, 105)
    ax.set_ylabel("분류 정확도 %"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("CNN 노이즈 강건성 — 클린 학습 vs 노이즈 증강 (test, 입력 노이즈별)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = resolve_path("outputs/figures/cnn_aug_compare.png")
    fig.savefig(out, dpi=120); plt.close(fig)

    print(f"\n{'조건':<8}{'클린 CNN':>10}{'증강 CNN':>10}")
    for i, (nm, _) in enumerate(conds):
        print(f"{nm.replace(chr(10),''):<8}{100*a_clean[i]:>9.0f}%{100*a_aug[i]:>9.0f}%")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
