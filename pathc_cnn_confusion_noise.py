"""경로 C CNN — 노이즈 입력에서 혼동행렬 비교 (클린 학습 vs 노이즈 증강).

test 고장 녹음에 화이트 0dB 노이즈를 넣은 뒤, 두 모델의 윈도우 단위 혼동행렬·라벨별 정확도를 비교.
  · 클린 학습 CNN (fault_classifier.pth)
  · 노이즈 증강 CNN (fault_classifier_aug.pth)
증강 모델이 노이즈 입력에서 라벨을 더 잘 가르는지 본다.

실행:
    .venv\\Scripts\\python.exe pathc_cnn_confusion_noise.py
"""
from __future__ import annotations

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

from src.cnn_fault_classifier import CLASS_TO_IDX, collect_fault_recordings, load_model as load_cnn, predict_spectrogram
from src.evaluator import add_white_noise
from src.preprocessing import make_windows
from src.schemas import FAULT_CLASSES
from src.spectrogram import to_spectrograms
from src.utils import get_device, load_config, resolve_path

KOR = {"IR": "내륜", "OR": "외륜", "B": "볼"}
SNR = 0  # 화이트 0dB


def confusion(model, test, fn, cfg, dev, Lc, ov):
    M = np.zeros((3, 3), int); ok = tot = 0
    for r in test:
        w = make_windows(fn(np.asarray(r.signal, float)), Lc, ov)
        if len(w) == 0:
            continue
        votes = []
        for s in to_spectrograms(w, cfg):
            p = predict_spectrogram(model, s).location.value
            M[CLASS_TO_IDX[r.location], CLASS_TO_IDX[p]] += 1; votes.append(p)
        ok += (Counter(votes).most_common(1)[0][0] == r.location); tot += 1
    return M, ok / tot if tot else 0


def draw(ax, M, title):
    acc = M.trace() / M.sum()
    im = ax.imshow(M, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_xticklabels([KOR[c] for c in FAULT_CLASSES])
    ax.set_yticks(range(3)); ax.set_yticklabels([KOR[c] for c in FAULT_CLASSES])
    ax.set_xlabel("예측"); ax.set_ylabel("실제")
    for i in range(3):
        for j in range(3):
            pct = 100 * M[i, j] / M[i].sum() if M[i].sum() else 0
            ax.text(j, i, f"{M[i,j]}\n({pct:.0f}%)", ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white" if M[i, j] > M.max() / 2 else "black")
    ax.set_title(f"{title}\n전체 {100*acc:.0f}%", fontweight="bold")
    return acc


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config(); dev = get_device(cfg)
    Lc, ov = cfg["window"]["length_c"], cfg["window"]["overlap"]
    md = cfg["artifacts"]["models"]
    recs = collect_fault_recordings(cfg, False)
    man = json.load(open("data/splits/c_split_manifest.json", encoding="utf-8"))
    keep = set(man["split_rec_ids"]["test"])
    test = [r for r in recs if r.rec_id in keep]

    clean = load_cnn(f"{md}/fault_classifier.pth", num_classes=3, device=dev)
    aug = load_cnn(f"{md}/fault_classifier_aug.pth", num_classes=3, device=dev)
    rng = np.random.default_rng(cfg["seed"])
    noise = lambda s: add_white_noise(s, SNR, rng)

    Mc, rc = confusion(clean, test, noise, cfg, dev, Lc, ov)
    rng = np.random.default_rng(cfg["seed"])
    Ma, ra = confusion(aug, test, lambda s: add_white_noise(s, SNR, rng), cfg, dev, Lc, ov)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.5))
    acc_c = draw(a1, Mc, f"클린 학습 CNN (녹음 다수결 {100*rc:.0f}%)")
    acc_a = draw(a2, Ma, f"노이즈 증강 CNN (녹음 다수결 {100*ra:.0f}%)")
    fig.suptitle(f"노이즈 입력(화이트 {SNR}dB)에서 CNN 혼동행렬 — 클린 vs 노이즈 증강", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = resolve_path("outputs/figures/pathc_cnn_confusion_noise.png")
    fig.savefig(out, dpi=120); plt.close(fig)

    print(f"화이트 {SNR}dB 입력, 윈도우 정확도:  클린 {100*acc_c:.1f}%  /  증강 {100*acc_a:.1f}%")
    print(f"녹음 단위(다수결):                클린 {100*rc:.0f}%  /  증강 {100*ra:.0f}%")
    for name, M in (("클린", Mc), ("증강", Ma)):
        rec = [100 * M[i, i] / M[i].sum() if M[i].sum() else 0 for i in range(3)]
        print(f"  {name} 라벨별: " + " / ".join(f"{KOR[FAULT_CLASSES[i]]} {rec[i]:.0f}%" for i in range(3)))
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
