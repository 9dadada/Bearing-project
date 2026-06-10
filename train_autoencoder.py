"""경로 B — 디노이징 오토인코더 학습 스크립트.

정상(48k) 스펙트로그램만으로 AE 를 학습하고, 모델을 models/autoencoder.pth,
정상 오차 분포 기반 임계값을 models/thresholds.json 에 저장한다.

실행:
    .venv\\Scripts\\python.exe train_autoencoder.py
"""
