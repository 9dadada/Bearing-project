"""CNN(경로 C) 노이즈 강건성 — 입력에 노이즈를 넣고 분류 정확도 비교.

오토인코더처럼, test 고장 녹음에 화이트/임펄스 노이즈를 단계별로 주입한 뒤
CNN(MobileNetV2)의 고장유형 분류 정확도가 버티는지 측정한다.
  · 윈도우 단위 정확도 (민감)  vs  녹음 단위 다수결 정확도 (강건)
CNN은 클린 스펙트로그램으로만 학습 → 노이즈에 약하면 증강 필요성을 시사.

실행:
    .venv\\Scripts\\python.exe cnn_noise_robustness.py
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

from src.cnn_fault_classifier import collect_fault_recordings, load_model as load_cnn, predict_spectrogram
from src.evaluator import add_white_noise
from src.preprocessing import make_windows
from src.spectrogram import to_spectrograms
from src.utils import get_device, load_config, resolve_path

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
    cfg = load_config(); dev = get_device(cfg)
    Lc, ov = cfg["window"]["length_c"], cfg["window"]["overlap"]
    m = load_cnn(cfg["artifacts"]["models"] + "/fault_classifier.pth", num_classes=3, device=dev)
    recs = collect_fault_recordings(cfg, False)
    man = json.load(open("data/splits/c_split_manifest.json", encoding="utf-8"))
    keep = set(man["split_rec_ids"]["test"])
    test = [r for r in recs if r.rec_id in keep]

    conds = [("없음", lambda s, r: s), ("화이트\n+6", lambda s, r: add_white_noise(s, 6, r)),
             ("화이트\n+3", lambda s, r: add_white_noise(s, 3, r)), ("화이트\n0", lambda s, r: add_white_noise(s, 0, r)),
             ("화이트\n-3", lambda s, r: add_white_noise(s, -3, r)), ("화이트\n-5", lambda s, r: add_white_noise(s, -5, r)),
             ("화이트\n-10", lambda s, r: add_white_noise(s, -10, r)), ("임펄스\n랜덤", lambda s, r: impulse_random(s, 0, r))]

    win_acc, rec_acc = [], []
    for _, fn in conds:
        rng = np.random.default_rng(cfg["seed"])
        wc = wt = rc = rt = 0
        for r in test:
            noisy = fn(np.asarray(r.signal, float), rng)
            windows = make_windows(noisy, Lc, ov)
            if len(windows) == 0:
                continue
            specs = to_spectrograms(windows, cfg)
            votes = [predict_spectrogram(m, s).location.value for s in specs]
            wc += sum(v == r.location for v in votes); wt += len(votes)
            top = Counter(votes).most_common(1)[0][0]
            rc += (top == r.location); rt += 1
        win_acc.append(wc / wt if wt else 0); rec_acc.append(rc / rt if rt else 0)

    x = np.arange(len(conds)); names = [c[0] for c in conds]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(x, [100 * v for v in rec_acc], "o-", color="#2980b9", lw=2.4, label="녹음 단위 (다수결)")
    ax.plot(x, [100 * v for v in win_acc], "s--", color="#e67e22", lw=2, label="윈도우 단위")
    for i in range(len(conds)):
        ax.text(i, 100 * rec_acc[i] + 1.5, f"{100*rec_acc[i]:.0f}%", ha="center", color="#2980b9", fontsize=8, fontweight="bold")
    ax.axhline(33.3, color="gray", ls=":", lw=1); ax.text(0, 35, "찍기(3클래스 33%)", color="gray", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8); ax.set_ylim(0, 105)
    ax.set_ylabel("분류 정확도 %"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("CNN(경로 C) 노이즈 강건성 — 입력 노이즈별 고장유형 분류 정확도 (test)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = resolve_path("outputs/figures/cnn_noise_robustness.png")
    fig.savefig(out, dpi=120); plt.close(fig)

    print(f"=== CNN 노이즈 강건성 (test n={len(test)}) ===")
    print(f"{'조건':<8}{'녹음(다수결)':>12}{'윈도우':>10}")
    for i, (nm, _) in enumerate(conds):
        print(f"{nm.replace(chr(10),''):<8}{100*rec_acc[i]:>11.0f}%{100*win_acc[i]:>9.0f}%")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
