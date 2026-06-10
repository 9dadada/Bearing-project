"""점수 융합 + 디바운스.

- 경로 A·B 점수를 정규화 후 결합(OR 또는 가중치, config 로 선택).
- 디바운스: 한 윈도우가 아니라 연속 N개 윈도우가 이상일 때만 알람(오탐 억제).
  N(=debounce_n)은 config.yaml 의 fusion 설정값.

TODO: fuse(score_a, score_b, rule), Debouncer(n).
"""
