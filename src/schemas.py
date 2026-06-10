"""데이터 타입·최종 출력 JSON 정의 — 팀 공통 계약(contract).

여러 명이 경로 A/B/C·fusion·backend·frontend 를 나눠 만들 때,
'윈도우 1개', '점수', '판정 결과'가 어떤 모양인지 여기서 한 번만 못 박는다.
모든 모듈은 이 정의를 import 해서 같은 모양을 주고받는다.

설계 메모:
- 내부 전달용(Window 등)은 numpy 배열을 그대로 담는다 → 직렬화 안 함, 모듈 간 전달용.
- 최종 출력(InferenceResult)은 스칼라(숫자/불리언/문자열)만 → JSON 으로 프론트에 전송.
- 의존성 0(stdlib dataclass). backend(FastAPI)에서는 이 모양 그대로 pydantic 모델로
  감싸 쓰면 된다.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

import numpy as np


# =============================================================
#  결함 유형 (경로 C 출력 라벨) = 위치 × 손상 크기
#  팀 합의: 위치(IR/OR/B) × 크기(0.007/0.014/0.021) = 9클래스 분류.
#  (CWRU 12k Drive End 기준. 0.028"은 OR에 없어 제외)
# =============================================================
class FaultLocation(str, Enum):
    IR = "IR"   # Inner Race  (내륜)
    OR = "OR"   # Outer Race  (외륜)
    B = "B"     # Ball        (볼)


class DamageSize(str, Enum):
    S007 = "0.007"   # inch
    S014 = "0.014"
    S021 = "0.021"


# 분류기 출력 클래스 9종 = 위치 × 크기 (예: "IR_0.007")
# class_probs 의 키, CNN 출력 순서의 기준으로 쓴다.
FAULT_CLASSES: list[str] = [
    f"{loc.value}_{size.value}"
    for loc in FaultLocation
    for size in DamageSize
]


# =============================================================
#  1. 윈도우 — 전처리 출력 (모듈 내부 전달용, 직렬화 안 함)
#     스펙트로그램은 (N, 1, H, W) 배열 규약. 여기선 1D 신호 윈도우만 담는다.
# =============================================================
@dataclass
class Window:
    signal: np.ndarray          # (L,) 1D 진동 신호
    sampling_rate: int          # 48000(B 경로) / 12000(C 경로)
    source: str                 # 'normal' / 'fault_48k' / 'fault_12k' / 'fan_end_fault'
    index: int                  # 원본 신호 내 윈도우 순번 (0,1,2,...)
    label: str | None = None    # 평가용 정답(있을 때만). 학습 경로에서는 None


# =============================================================
#  2. 경로 A 결과 — 통계 임펄스 탐지기
# =============================================================
@dataclass
class PathAResult:
    score: float                        # 이상 점수(클수록 이상)
    is_anomaly: bool                    # 임계값 초과 여부
    features: dict[str, float]          # {'kurtosis':.., 'crest_factor':.., 'rms':.., 'peak':..}


# =============================================================
#  3. 경로 B 결과 — 디노이징 오토인코더 (48k)
# =============================================================
@dataclass
class PathBResult:
    recon_error: float                  # 재구성 오차(원본)
    score: float                        # 정규화된 이상 점수
    is_anomaly: bool                    # 임계값 초과 여부


# =============================================================
#  4. 융합 결과 — A+B 결합 + 디바운스
# =============================================================
@dataclass
class FusionResult:
    fused_score: float                  # 융합 점수
    is_anomaly: bool                    # 이 윈도우 단독 이상 여부
    alarm: bool                         # 연속 N윈도우 디바운스 통과한 최종 알람


# =============================================================
#  5. 경로 C 결과 — 고장 유형 분류 (12k, 위치 × 크기 9클래스)
#  ⚠️ C 담당자 구현. 계약(필드 모양)만 여기서 공유한다.
# =============================================================
@dataclass
class PathCResult:
    location: FaultLocation             # 추정 결함 위치 (IR/OR/B)
    size: DamageSize                    # 추정 손상 크기 (0.007/0.014/0.021)
    confidence: float                   # 최상위 클래스 확률 (0~1)
    class_probs: dict[str, float]       # 9클래스 확률 {'IR_0.007':.., ..., 'B_0.021':..}


# =============================================================
#  6. 최종 출력 — backend → frontend 로 보내는 JSON
#     스칼라만 담는다(직렬화 가능). 정상이면 C 관련 필드는 None.
# =============================================================
@dataclass
class InferenceResult:
    window_index: int                   # 윈도우 순번
    timestamp: float | None             # 실시간 스트림 시각(초). 오프라인이면 None
    score_a: float                      # 경로 A 점수
    score_b: float                      # 경로 B 점수
    fused_score: float                  # 융합 점수
    is_anomaly: bool                    # 융합 판정(이 윈도우)
    alarm: bool                         # 디바운스 후 최종 알람
    fault_location: str | None = None   # 이상일 때만 C 추정 위치(IR/OR/B). 정상이면 None
    fault_size: str | None = None       # 이상일 때만 C 추정 크기(0.007/..). 정상이면 None
    confidence: float | None = None     # C 추정 신뢰도. 정상이면 None

    def to_dict(self) -> dict:
        """API 응답용 dict 로 변환 (그대로 JSON 직렬화 가능)."""
        return asdict(self)
