# 진동 베어링 이상탐지

CWRU 베어링 진동 신호에서 **정상/이상**을 판단하는 **비지도 이상탐지** 시스템.
정상 데이터만 학습하고, 통계 경로(임펄스)와 디노이징 오토인코더 경로(미묘한 이상)를
병렬로 두어 융합 판정한다. 작업 백로그는 `TASKS.md` 참고.

## 폴더 구조

```
bearing-project/
  configs/config.yaml   # 모든 설정 (윈도우/샘플링/시드 등)
  src/                  # 소스 코드 패키지
    config.py           # config 로드 + 장치(CPU/GPU) 선택
  data/
    raw/normal/         # 정상 .mat (학습용)
    raw/fault_48k/      # 고장 .mat (평가 전용 — 학습 금지)
    processed/          # 전처리 결과 저장
  models/               # 학습된 모델·통계 저장
  main.py               # 파이프라인 진입점
  requirements.txt
```

## 환경 셋업

```powershell
# 가상환경 만들기 (최초 1회)
py -3.12 -m venv .venv

# 라이브러리 설치
.venv\Scripts\python.exe -m pip install -r requirements.txt

# 실행
.venv\Scripts\python.exe main.py
```

## GPU(RTX 5090) 쓰고 싶을 때

기본은 CPU 버전 torch 다. GPU로 바꾸려면 torch 만 다시 깔면 된다
(코드는 device=auto 라 안 고쳐도 됨):

```powershell
.venv\Scripts\python.exe -m pip uninstall -y torch
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```
