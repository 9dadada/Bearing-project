"""경로 C — 고장 유형 CNN 분류기 (지도학습).

A·B 융합 판정이 '이상'일 때만 호출되는 게이트형 분류기.
CWRU 고장 데이터(IR/OR/B 등)로 학습하며, 멀티 샘플링 rate(12k·48k)를 다룬다.
출력은 확정 진단이 아니라 '알려진 유형 중 무엇에 가까운지' 참고 추정.
모델 가중치는 models/fault_classifier.pth 로 저장.

TODO: FaultCNN(nn.Module), train(fault_labeled), predict(x) -> (label, prob).
"""
