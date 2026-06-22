"""증강 정상 데이터 생성 — 노이즈 강건 학습용 (2단계 증강).

정상 4개(CWRU, 12kHz) 각각에 노이즈를 입혀 '정상' 학습셋을 늘린다(전부 정상, 고장 X).
  · 원본(클린) 1
  · 화이트 -5 / -3 / 3 dB  → 3
  · 임펄스 랜덤 5 (위치·진폭 무작위, SNR -5~3 무작위)
  = 9 / 정상 → 총 36개 → data/normal_aug/ 에 .mat 저장 (키 X_DE_time)

실행:
    .venv\\Scripts\\python.exe make_augmented_normal.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from src.data_loader import load_normal_signals
from src.evaluator import add_white_noise
from src.utils import load_config

OUT = _ROOT / "data" / "normal_aug"
FS = 12000


def impulse_random(sig, snr_db, rng, fs=FS, rate_per_sec=10):
    sig = np.asarray(sig, dtype=float).copy()
    n = len(sig)
    k = max(1, int(rate_per_sec * n / fs))
    train = np.zeros(n)
    train[rng.integers(0, n, k)] = rng.uniform(0.5, 1.5, k) * rng.choice([-1.0, 1.0], k)
    t = np.sqrt(np.mean(train ** 2))
    if t > 0:
        train *= (np.sqrt(np.mean(sig ** 2)) / 10 ** (snr_db / 20.0)) / t
    return sig + train


def save(sig, name):
    sio.savemat(OUT / f"{name}.mat", {"X_DE_time": np.asarray(sig, float).reshape(-1, 1)})


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])

    normal = load_normal_signals(cfg)                       # 정상 4개 (12kHz)
    n = 0
    for name, sig in normal.items():
        sig = np.asarray(sig, float)
        save(sig, f"{name}_clean"); n += 1
        for db in (-5, -3, 3):                              # 화이트 3단계
            save(add_white_noise(sig, db, rng), f"{name}_white_{str(db).replace('-','m')}dB"); n += 1
        for j in range(5):                                  # 임펄스 랜덤 5개
            snr = float(rng.uniform(-5, 3))
            save(impulse_random(sig, snr, rng), f"{name}_impulse_{j}"); n += 1

    print(f"증강 정상 생성: {n}개 (전부 정상) → {OUT}")
    print(f"  구성/정상: 클린 1 + 화이트(-5/-3/3dB) 3 + 임펄스 랜덤 5 = 9")
    print(f"  파일 예: {sorted(p.name for p in OUT.glob('normal_0_*.mat'))}")


if __name__ == "__main__":
    main()
