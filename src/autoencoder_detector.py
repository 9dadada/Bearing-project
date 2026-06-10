"""경로 B — 디노이징 Conv 오토인코더.

정상 스펙트로그램만 학습(노이즈 증강 후 원본 복원). 재구성 오차가 크면 이상.
이번 단계에서는 48kHz 데이터에서만 학습·추론한다.
임계값은 정상 오차 분포 기반으로 산정해 models/thresholds.json 에 저장,
모델 가중치는 models/autoencoder.pth 로 저장.

TODO: ConvAutoencoder(nn.Module), train(normal), recon_error(x), score(x).
"""
