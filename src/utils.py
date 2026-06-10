"""공통 함수 — config 로드, device 선택, 시드 고정 등.

다른 모듈에서 공통으로 쓰는 잡다한 헬퍼를 한 곳에 모은다.

TODO:
- load_config(path) : configs/config.yaml 을 dict 로 로드.
- get_device(cfg)   : 'auto'/'cuda'/'cpu' 해석 (GPU 있으면 cuda).
- set_seed(seed)    : numpy·torch 시드 고정(재현성).
"""
