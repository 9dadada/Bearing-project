"""오케스트레이터 — 추론 파이프라인 조립.

흐름: 전처리 → 경로 A·B 병렬 → 융합 → (이상이면) 경로 C 호출 → 결과 조립.
게이트 규칙: C 는 A·B 융합 판정이 '이상'일 때만 호출한다(정상엔 호출 안 함).
배포 시 동일 전처리·점수를 보장하기 위해 scaler.pkl 과 thresholds.json 을 로드해 사용.

TODO: load_artifacts(), run(window) -> 최종 판정 dict(schemas 준수).
"""
