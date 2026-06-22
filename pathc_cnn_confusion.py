"""경로 C CNN 분류기 — 혼동행렬 + 라벨별 정확도 (클린 test).

test 고장 녹음을 윈도우별로 CNN 예측 → 혼동행렬(IR/OR/B) + 클래스별 재현율 시각화.

실행:
    .venv\\Scripts\\python.exe pathc_cnn_confusion.py
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
from src.preprocessing import make_windows
from src.schemas import FAULT_CLASSES
from src.spectrogram import to_spectrograms
from src.utils import get_device, load_config, resolve_path

KOR = {"IR": "내륜(IR)", "OR": "외륜(OR)", "B": "볼(B)"}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config(); dev = get_device(cfg)
    Lc, ov = cfg["window"]["length_c"], cfg["window"]["overlap"]
    m = load_cnn(cfg["artifacts"]["models"] + "/fault_classifier.pth", num_classes=3, device=dev)
    recs = collect_fault_recordings(cfg, False)
    man = json.load(open("data/splits/c_split_manifest.json", encoding="utf-8"))
    keep = set(man["split_rec_ids"]["test"])
    test = [r for r in recs if r.rec_id in keep]

    M = np.zeros((3, 3), int)          # 행=실제, 열=예측 (윈도우 단위)
    rec_ok = rec_tot = 0
    for r in test:
        w = make_windows(np.asarray(r.signal, float), Lc, ov)
        if len(w) == 0:
            continue
        votes = []
        for s in to_spectrograms(w, cfg):
            pred = predict_spectrogram(m, s).location.value
            M[CLASS_TO_IDX[r.location], CLASS_TO_IDX[pred]] += 1
            votes.append(pred)
        rec_ok += (Counter(votes).most_common(1)[0][0] == r.location); rec_tot += 1

    recall = [M[i, i] / M[i].sum() if M[i].sum() else 0 for i in range(3)]
    overall = M.trace() / M.sum()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))
    # --- 혼동행렬 ---
    im = a1.imshow(M, cmap="Blues")
    a1.set_xticks(range(3)); a1.set_xticklabels([KOR[c] for c in FAULT_CLASSES])
    a1.set_yticks(range(3)); a1.set_yticklabels([KOR[c] for c in FAULT_CLASSES])
    a1.set_xlabel("CNN 예측"); a1.set_ylabel("실제(정답)")
    for i in range(3):
        for j in range(3):
            pct = 100 * M[i, j] / M[i].sum() if M[i].sum() else 0
            col = "white" if M[i, j] > M.max() / 2 else "black"
            a1.text(j, i, f"{M[i, j]}\n({pct:.0f}%)", ha="center", va="center", fontsize=11, fontweight="bold", color=col)
    a1.set_title(f"CNN 혼동행렬 (윈도우 단위, test)\n전체 정확도 {100*overall:.1f}%", fontweight="bold")
    fig.colorbar(im, ax=a1, fraction=0.046, pad=0.04)
    # --- 라벨별 정확도 ---
    colors = ["#27ae60", "#2980b9", "#e67e22"]
    bars = a2.bar([KOR[c] for c in FAULT_CLASSES], [100 * r for r in recall], color=colors)
    for i, r in enumerate(recall):
        a2.text(i, 100 * r + 1.5, f"{100*r:.1f}%", ha="center", fontweight="bold")
    a2.axhline(100 * overall, color="gray", ls="--", lw=1)
    a2.text(2.4, 100 * overall + 1, f"전체 {100*overall:.0f}%", color="gray", fontsize=9, ha="right")
    a2.set_ylim(0, 110); a2.set_ylabel("정확도(재현율) %")
    a2.set_title("라벨별 정확도 (클래스 재현율)", fontweight="bold")
    fig.suptitle("경로 C CNN 분류기 — 혼동행렬 · 라벨별 정확도 (클린 test)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = resolve_path("outputs/figures/pathc_cnn_confusion.png")
    fig.savefig(out, dpi=120); plt.close(fig)

    print(f"윈도우 전체 정확도: {100*overall:.1f}%  /  녹음 단위(다수결): {100*rec_ok/rec_tot:.0f}% ({rec_ok}/{rec_tot})")
    print("라벨별 정확도(재현율):")
    for i, c in enumerate(FAULT_CLASSES):
        print(f"  {KOR[c]}: {100*recall[i]:.1f}%  (윈도우 {M[i].sum()}개)")
    print("혼동행렬(행=실제, 열=예측):")
    print("        " + "  ".join(f"{c:>4}" for c in FAULT_CLASSES))
    for i, c in enumerate(FAULT_CLASSES):
        print(f"  {c:>4}  " + "  ".join(f"{M[i,j]:>4}" for j in range(3)))
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
