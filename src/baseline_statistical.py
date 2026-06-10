"""경로 A — 통계 임펄스 탐지기.

윈도우에서 RMS·peak·kurtosis(첨도)·crest factor(크레스트팩터)를 뽑아
정상 분포 기반 임계값(99% 백분위 또는 평균+3σ)을 넘으면 이상으로 본다.
임계값은 정상 데이터로만 산정해 models/thresholds.json 에 저장.

TODO: extract_features(window) -> dict, fit_threshold(normal), score(window).
"""
