"""CWRU .mat 로더.

핵심 원칙 — 데이터 누수 차단:
- 정상(normal) 데이터만 학습·임계값 설정에 쓴다.
- 고장(fault) 데이터는 '평가 전용'이다. 학습 코드가 실수로라도 못 쓰게
  load_fault_signals() 라는 별도 함수로만 접근하도록 분리한다.
  (학습 코드는 load_normal_signals() 만 호출한다)
"""
from pathlib import Path

import numpy as np
import scipy.io as sio


def _extract_de_signal(mat_path: Path, key_suffix: str = "DE_time") -> np.ndarray:
    """.mat 파일 하나에서 Drive End 진동 신호(1차원 배열)를 꺼낸다.

    변수명이 파일마다 X097_DE_time, X111_DE_time 처럼 숫자가 달라서,
    '_DE_time' 으로 *끝나는* 키를 찾아서 매칭한다.
    """
    data = sio.loadmat(mat_path)
    de_keys = [k for k in data if k.endswith(key_suffix)]
    if not de_keys:
        raise KeyError(f"{mat_path.name} 에 '{key_suffix}' 로 끝나는 변수가 없음")
    # squeeze: (N,1) 모양을 (N,) 1차원으로 펴 줌
    signal = data[de_keys[0]].squeeze().astype(np.float64)
    return signal


def load_normal_signals(cfg: dict) -> dict[str, np.ndarray]:
    """정상 신호들을 로드한다.

    이게 학습·임계값 설정에 쓰는 '유일한' 데이터다.
    반환: {파일이름: 신호배열}  예) {'normal_0': array([...]), ...}
    """
    folder = Path(cfg["data"]["raw_normal"])
    suffix = cfg["data"]["mat_key_de"]
    signals: dict[str, np.ndarray] = {}
    for mat in sorted(folder.glob("*.mat")):
        signals[mat.stem] = _extract_de_signal(mat, suffix)
    return signals


def parse_fault_label(filename: str) -> str:
    """파일이름에서 고장 종류를 뽑는다.

    IR007_2   -> 'IR' (Inner Race, 내륜)
    OR007_6_2 -> 'OR' (Outer Race, 외륜)
    B007_2    -> 'B'  (Ball, 볼)
    """
    name = filename.upper()
    if name.startswith("IR"):
        return "IR"
    if name.startswith("OR"):
        return "OR"
    if name.startswith("B"):
        return "B"
    return "UNKNOWN"


def load_fault_signals(cfg: dict) -> dict[str, dict]:
    """[평가 전용] 고장 신호들을 로드한다.

    ⚠️ 경고: 학습·임계값 설정에 절대 쓰지 말 것. 평가(EPIC 4)에서만 호출한다.
    반환: {파일이름: {'signal': 신호배열, 'label': 'IR'/'OR'/'B'}}
    """
    folder = Path(cfg["data"]["raw_fault"])
    suffix = cfg["data"]["mat_key_de"]
    signals: dict[str, dict] = {}
    for mat in sorted(folder.glob("*.mat")):
        signals[mat.stem] = {
            "signal": _extract_de_signal(mat, suffix),
            "label": parse_fault_label(mat.stem),
        }
    return signals
