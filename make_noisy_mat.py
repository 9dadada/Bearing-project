"""데모용 노이즈 .mat 생성 — 정상·고장 신호에 화이트/임펄스 노이즈를 dB별로 주입.

프론트엔드 업로드(/diagnose_mat)로 노이즈 환경 동작을 시연하기 위한 입력 파일.
키는 'X_DE_time'(DE_time 으로 끝남) → 백엔드가 자동 인식. 신호는 12kHz.

실행:
    .venv\\Scripts\\python.exe make_noisy_mat.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from src.data_loader import load_fault_signals, load_normal_signals
from src.evaluator import add_white_noise
from src.utils import load_config

OUT = _ROOT / "data" / "demo_noisy"
FS = 12000


def impulse_at_snr(sig, snr_db, rng, fs=FS, rate_per_sec=10):
    """임펄스(스파이크)를 목표 SNR(dB)에 맞춰 추가 — 화이트와 같은 dB 잣대."""
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
    sio.savemat(OUT / f"{name}.mat", {"X_DE_time": np.asarray(sig, dtype=float).reshape(-1, 1)})


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)

    normal = list(load_normal_signals(cfg).values())[1]          # 정상(부하1, 깨끗)
    fault_all = load_fault_signals(cfg)
    fault = next(v["signal"] for v in fault_all.values() if v["location"] == "IR")  # 고장(내륜)

    bases = {"normal": normal, "fault_IR": fault}
    db_levels = [("p10dB", 10), ("0dB", 0), ("m10dB", -10)]      # +10 / 0 / -10 dB
    count = 0
    for tag, sig in bases.items():
        save(sig, f"{tag}_clean")                                # 노이즈 없음(참고)
        count += 1
        for label, db in db_levels:
            rng = np.random.default_rng(cfg["seed"])
            save(add_white_noise(np.asarray(sig, float), db, rng), f"{tag}_white_{label}")
            rng = np.random.default_rng(cfg["seed"])
            save(impulse_at_snr(sig, db, rng), f"{tag}_impulse_{label}")
            count += 2

    print(f"생성 완료: {count}개 .mat → {OUT}")
    for p in sorted(OUT.glob("*.mat")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
