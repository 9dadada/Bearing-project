"""파이프라인 진입점 (T0.1 단계: 아직 뼈대만).

지금 하는 일은 딱 하나 —
"환경과 설정이 제대로 준비됐는지" 확인하는 것이다.
앞으로 EPIC 0~5 를 진행하며 여기에 단계를 하나씩 이어 붙인다.

실행:
    .venv\\Scripts\\python.exe main.py
"""
import sys

# Windows 한글 콘솔(cp949)에서도 한글·특수문자가 깨지지 않도록 출력을 UTF-8 로 고정
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import scipy
import torch

from src.config import load_config, get_device
from src.data.loader import load_normal_signals, load_fault_signals


def main() -> None:
    cfg = load_config()
    device = get_device(cfg)

    print("=== 환경 점검 ===")
    print(f"numpy {np.__version__} / scipy {scipy.__version__} / torch {torch.__version__}")
    print(f"사용 장치(device): {device}")
    print()
    print("=== 설정 확인 ===")
    print(f"샘플링 주파수 : {cfg['signal']['sampling_rate']} Hz")
    print(f"윈도우 길이   : {cfg['window']['length']} 포인트 (겹침 {cfg['window']['overlap']})")
    print(f"정상 데이터   : {cfg['data']['raw_normal']}")
    print(f"고장 데이터   : {cfg['data']['raw_fault']} (평가 전용)")
    print()
    print("=== T0.2 데이터 로드 ===")
    # 정상: 학습에 쓰는 유일한 데이터
    normal = load_normal_signals(cfg)
    print(f"정상 파일 {len(normal)}개 (학습용)")
    for name, sig in normal.items():
        print(f"  - {name}: {len(sig):,} 포인트")

    # 고장: 평가 전용 — 학습 코드는 위 normal 만 쓰고 이건 건드리지 않는다
    fault = load_fault_signals(cfg)
    print(f"고장 파일 {len(fault)}개 (평가 전용 — 학습 금지)")
    # 고장 종류별 개수 세기
    by_label: dict[str, int] = {}
    for info in fault.values():
        by_label[info["label"]] = by_label.get(info["label"], 0) + 1
    for label, count in sorted(by_label.items()):
        print(f"  - {label}: {count}개")
    print()
    print("T0.2 OK — 정상/고장 개수 출력됨. 고장은 별도 함수로만 접근 가능(학습 격리).")


if __name__ == "__main__":
    main()
