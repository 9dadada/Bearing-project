"""CWRU .mat 로더 — 정상/고장 분리, 멀티 샘플링 소스 지원.

핵심 원칙(데이터 누수 차단):
- 정상(normal)만 학습·임계값 설정에 사용.
- 고장(fault)은 평가 전용. 학습 코드가 실수로라도 못 쓰게 별도 함수로 격리.

지원 소스:
- data/raw/normal        : 정상 (학습용, 48k)
- data/raw/fault_48k     : 고장 48kHz (평가 전용)
- data/raw/fault_12k     : 고장 12kHz (평가 전용, 경로 C 멀티 rate 분류용)
- data/raw/fan_end_fault : Fan End 고장 (평가 전용, FE 가속도계)

TODO: load_normal_signals(), load_fault_signals() 작성.
      소스별 sampling_rate / mat_key(DE_time·FE_time) 처리.
"""
