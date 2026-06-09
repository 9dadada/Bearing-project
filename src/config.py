"""config.yaml 을 읽어서 파이썬 딕셔너리로 돌려주는 모듈.

다른 코드에서는 이렇게 쓴다:
    from src.config import load_config, get_device
    cfg = load_config()
    sr = cfg["signal"]["sampling_rate"]
"""
from pathlib import Path

import yaml

# 이 파일(src/config.py) 기준으로 프로젝트 최상위 폴더를 찾는다.
#   src/config.py  ->  .parent = src  ->  .parent = bearing-project
# 이렇게 하면 어느 위치에서 실행하든 경로가 틀어지지 않는다.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | None = None) -> dict:
    """config.yaml 을 읽어서 dict 로 반환한다.

    path 를 안 주면 기본 위치(configs/config.yaml)를 읽는다.
    """
    if path is None:
        path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_device(cfg: dict) -> str:
    """config 의 device 설정을 실제 사용할 장치 문자열로 바꿔준다.

    'auto' 면 GPU가 있으면 'cuda', 없으면 'cpu' 를 고른다.
    이렇게 해두면 CPU/GPU 어디서 돌려도 코드를 안 고쳐도 된다.
    """
    import torch

    setting = cfg.get("device", "auto")
    if setting == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return setting
