"""경로 C(팀원) 설계·성능 평가 — 물리 vs CNN 교차검증.

test split(공정) + 전체(물리 강건성 보강)로 위치별 정확도·교차검증 분포를 측정·시각화.

실행:
    .venv\\Scripts\\python.exe eval_pathc_design.py
"""
from __future__ import annotations

import os
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

from src.physics_diagnosis import crosscheck_report
from src.utils import get_device, load_config, resolve_path


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config(); dev = get_device(cfg)
    sp = "test" if os.path.exists("data/splits/c_split_manifest.json") else None
    rep = crosscheck_report(cfg, device=dev, split=sp)         # 공정(test)
    rep_all = crosscheck_report(cfg, device=dev, split=None)   # 전체(물리 강건성)

    locs = ["IR", "OR", "B"]
    phys = [100 * rep["per_location"][l]["physics_hit"] / max(1, rep["per_location"][l]["n"]) for l in locs]
    cnn = [100 * rep["per_location"][l]["cnn_hit"] / max(1, rep["per_location"][l]["n"]) for l in locs]
    phys_all = [100 * rep_all["per_location"][l]["physics_hit"] / max(1, rep_all["per_location"][l]["n"]) for l in locs]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.2))
    x = np.arange(3); w = 0.35
    a1.bar(x - w / 2, phys, w, color="#e67e22", label="물리(envelope)")
    a1.bar(x + w / 2, cnn, w, color="#2980b9", label="CNN(MobileNetV2)")
    for i in range(3):
        a1.text(x[i] - w / 2, phys[i] + 2, f"{phys[i]:.0f}%", ha="center", fontsize=9, fontweight="bold")
        a1.text(x[i] + w / 2, cnn[i] + 2, f"{cnn[i]:.0f}%", ha="center", fontsize=9, fontweight="bold")
    a1.set_xticks(x); a1.set_xticklabels([f"{l}\n(n={rep['per_location'][l]['n']})" for l in locs])
    a1.set_ylim(0, 112); a1.set_ylabel("정확도 %"); a1.legend()
    a1.set_title("위치별 정확도 — 물리 vs CNN (test split)", fontweight="bold")

    vc = rep["verdict_count"]
    labels = {"agree": "일치", "disagree": "불일치", "physics_only": "물리단독", "cnn_only": "CNN단독"}
    keys = [k for k in vc if vc[k] > 0]
    a2.bar([labels[k] for k in keys], [vc[k] for k in keys],
           color=["#27ae60" if k == "agree" else "#e74c3c" for k in keys])
    for i, k in enumerate(keys):
        a2.text(i, vc[k] + 0.05, str(vc[k]), ha="center", fontweight="bold")
    a2.set_ylabel("녹음 수")
    a2.set_title(f"교차검증 판정 분포 (일치율 {100*rep['agreement_rate']:.0f}%)", fontweight="bold")
    fig.suptitle("경로 C 평가 — 물리 vs CNN 교차검증 (test split)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = resolve_path("outputs/figures/pathc_design_eval.png")
    fig.savefig(out, dpi=120); plt.close(fig)

    print(f"=== test split (n={rep['n_recordings']}) ===")
    print(f"교차검증 일치율: {100*rep['agreement_rate']:.0f}% (일치 {vc['agree']} / 불일치 {vc['disagree']})")
    for l in locs:
        print(f"  {l}(n={rep['per_location'][l]['n']}): 물리 {phys[locs.index(l)]:.0f}% / CNN {cnn[locs.index(l)]:.0f}%")
    print(f"\n=== 전체(n={rep_all['n_recordings']}) 물리 정확도(강건성 보강) ===")
    for l in locs:
        print(f"  {l}(n={rep_all['per_location'][l]['n']}): 물리 {phys_all[locs.index(l)]:.0f}%")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
