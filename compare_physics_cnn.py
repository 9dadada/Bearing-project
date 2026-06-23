"""물리 진단 vs CNN 진단 — 라벨별 정확도 비교 (녹음 단위).

경로 C의 두 진단(물리 엔벨로프 / CNN)을 같은 test 녹음(48k 포함 16개)에서 위치별로 비교.
물리는 학습이 없어 전체(60녹음) 정확도도 같이 보강 표시.

실행:
    .venv\\Scripts\\python.exe compare_physics_cnn.py
"""
from __future__ import annotations

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

KOR = {"IR": "내륜(IR)", "OR": "외륜(OR)", "B": "볼(B)"}
LOCS = ["IR", "OR", "B"]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config(); dev = get_device(cfg)
    rep = crosscheck_report(cfg, include_48k_downsampled=True, split="test", device=dev)

    def acc(rep, key):
        return [100 * rep["per_location"][l][key] / max(1, rep["per_location"][l]["n"]) for l in LOCS]

    phys = acc(rep, "physics_hit"); cnn = acc(rep, "cnn_hit")
    ns = [rep["per_location"][l]["n"] for l in LOCS]
    ov_p = 100 * sum(rep["per_location"][l]["physics_hit"] for l in LOCS) / rep["n_recordings"]
    ov_c = 100 * sum(rep["per_location"][l]["cnn_hit"] for l in LOCS) / rep["n_recordings"]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(3); w = 0.36
    b1 = ax.bar(x - w / 2, phys, w, color="#e67e22", label=f"물리 진단 (전체 {ov_p:.0f}%)")
    b2 = ax.bar(x + w / 2, cnn, w, color="#2980b9", label=f"CNN 진단 (전체 {ov_c:.0f}%)")
    for i in range(3):
        ax.text(x[i] - w / 2, phys[i] + 1.5, f"{phys[i]:.0f}%", ha="center", fontsize=10, fontweight="bold", color="#b35900")
        ax.text(x[i] + w / 2, cnn[i] + 1.5, f"{cnn[i]:.0f}%", ha="center", fontsize=10, fontweight="bold", color="#1c5a8a")
    ax.axhline(33.3, color="gray", ls=":", lw=1); ax.text(2.4, 35, "찍기 33%", color="gray", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f"{KOR[l]}\n(n={ns[i]})" for i, l in enumerate(LOCS)])
    ax.set_ylim(0, 112); ax.set_ylabel("정확도 % (녹음 단위)")
    ax.legend(loc="lower left")
    ax.set_title("물리 진단 vs CNN 진단 — 라벨별 정확도 (test, 녹음 단위)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = resolve_path("outputs/figures/physics_vs_cnn_label.png")
    fig.savefig(out, dpi=120); plt.close(fig)

    print(f"=== test(n={rep['n_recordings']}) 라벨별 [물리 / CNN] ===")
    for i, l in enumerate(LOCS):
        print(f"  {KOR[l]}(n={ns[i]}): 물리 {phys[i]:.0f}% / CNN {cnn[i]:.0f}%")
    print(f"  전체: 물리 {ov_p:.0f}% / CNN {ov_c:.0f}%")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
