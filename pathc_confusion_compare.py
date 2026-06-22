"""경로 C 교차검증 — 부스터 전/후 혼동행렬 비교 (파일 2개).

전(거부권): 물리·CNN 엇갈리면 '미정(검토필요)'으로 보류 → 정확도↓
후(부스터): 엇갈려도 CNN 채택 → 정확도↑
test split 녹음으로 두 방식의 최종 판정을 혼동행렬로 그려 outputs/figures 에 각각 저장.

실행:
    .venv\\Scripts\\python.exe pathc_confusion_compare.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

from src.cnn_fault_classifier import collect_fault_recordings, load_model as load_cnn
from src.physics_diagnosis import diagnose_recording
from src.utils import get_device, load_config, resolve_path

CLS = ["IR", "OR", "B"]
COLS = CLS + ["미정"]


def confmat(pairs):
    M = np.zeros((3, 4), int)
    for truth, pred in pairs:
        c = COLS.index(pred) if pred in COLS else 3
        M[CLS.index(truth), c] += 1
    return M


def draw(M, title, fname):
    acc = np.trace(M[:, :3]) / M.sum()
    fig, ax = plt.subplots(figsize=(6.2, 5))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=M.max() or 1)
    ax.set_xticks(range(4)); ax.set_xticklabels(COLS)
    ax.set_yticks(range(3)); ax.set_yticklabels(CLS)
    ax.set_xlabel("최종 판정"); ax.set_ylabel("실제(정답)")
    for r in range(3):
        for c in range(4):
            if M[r, c]:
                color = "white" if M[r, c] > (M.max() / 2) else "black"
                ax.text(c, r, str(M[r, c]), ha="center", va="center", fontsize=14, fontweight="bold", color=color)
    # 미정 열 강조(빨강 테두리)
    ax.axvline(2.5, color="#e74c3c", lw=1.5, ls="--")
    ax.text(3, -0.65, "검토필요", color="#e74c3c", ha="center", fontsize=9, fontweight="bold")
    ax.set_title(f"{title}\n최종 정확도 {100*acc:.0f}%  (대각선 {int(np.trace(M[:, :3]))}/{M.sum()})",
                 fontweight="bold")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = resolve_path(f"outputs/figures/{fname}")
    fig.savefig(out, dpi=120); plt.close(fig)
    return out, acc


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config(); dev = get_device(cfg)
    m = load_cnn(cfg["artifacts"]["models"] + "/fault_classifier.pth", num_classes=3, device=dev)
    recs = collect_fault_recordings(cfg, False)
    man = json.load(open("data/splits/c_split_manifest.json", encoding="utf-8"))
    keep = set(man["split_rec_ids"]["test"])
    test = [r for r in recs if r.rec_id in keep]

    before, after = [], []
    for r in test:
        res = diagnose_recording(r.signal, 12000, m, cfg)
        p = res.physics.location.value if res.physics.location else None
        c = res.cnn.location.value if res.cnn else None
        # 전(거부권): 일치→그 위치 / 엇갈림→미정 / 한쪽만→그쪽
        if c is None:
            old = p
        elif p is None:
            old = c
        elif p == c:
            old = c
        else:
            old = "미정"
        # 후(부스터): CNN 우선(없으면 물리)
        new = c if c is not None else p
        before.append((r.location, old)); after.append((r.location, new))

    Mb, Ma = confmat(before), confmat(after)
    ob, ab = draw(Mb, "[전] 거부권 방식 (엇갈리면 검토필요)", "pathc_confusion_before.png")
    oa, aa = draw(Ma, "[후] 부스터 방식 (엇갈려도 CNN 채택)", "pathc_confusion_after.png")
    print(f"전(거부권) 최종 정확도: {100*ab:.0f}%  → {ob}")
    print(f"후(부스터) 최종 정확도: {100*aa:.0f}%  → {oa}")
    print(f"개선: {100*ab:.0f}% → {100*aa:.0f}% (미정 {int(Mb[:,3].sum())}건이 모두 CNN으로 정답 처리)")


if __name__ == "__main__":
    main()
