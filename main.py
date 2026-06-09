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
    print("T0.1 OK — 빈 파이프라인이 에러 없이 돌아간다.")


if __name__ == "__main__":
    main()
