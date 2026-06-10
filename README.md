# 진동 베어링 이상탐지

CWRU 베어링 진동 신호에서 **정상/이상**을 판단하는 **비지도 이상탐지** 시스템.
정상 데이터만 학습하고, 통계 경로(임펄스)와 디노이징 오토인코더 경로(미묘한 이상)를
병렬로 두어 융합 판정한다. 이상으로 판정된 신호만 받아 **고장 유형**을 추정하는
지도학습 분류기(경로 C)를 게이트로 연결한다. 작업 백로그는 `TASKS.md` 참고.

## 3경로 구조

- **경로 A (통계)** — RMS·peak·첨도·크레스트팩터로 임펄스성 이상 탐지
- **경로 B (오토인코더)** — 정상 스펙트로그램만 학습, 재구성 오차로 미묘한 이상 탐지
- **경로 C (분류, 지도)** — A·B 융합이 '이상'일 때만 호출, 고장 유형(IR/OR/B) 참고 추정

## 폴더 구조

```
project/
├── data/
│   ├── raw/                  # 원본 CWRU .mat
│   │   ├── normal/           # 정상 (학습용 — 유일한 학습 데이터)
│   │   ├── fault_48k/        # 고장 48kHz (평가 전용)
│   │   ├── fault_12k/        # 고장 12kHz (평가 전용, C 멀티 rate용)
│   │   └── fan_end_fault/    # Fan End 고장 (평가 전용, FE 가속도계)
│   ├── processed/            # 윈도우 슬라이싱 배열
│   ├── features/             # 모델 입력용 스펙트로그램 .npy
│   ├── spectrograms/         # 시각화 이미지(학습 미사용)
│   └── splits/               # train/val/test 분리 정보
├── configs/config.yaml       # 모든 설정(샘플링·윈도우·n_fft·임계값·경로)
├── src/
│   ├── schemas.py            # 데이터 타입·출력 JSON 정의(팀 공통 계약)
│   ├── data_loader.py        # .mat 로드, 정상/고장 분리(멀티 소스)
│   ├── preprocessing.py      # 윈도잉·정규화(정상 통계로만 fit)
│   ├── spectrogram.py        # STFT/CWT 변환(로그 dB)
│   ├── streaming.py          # 실시간 스트리밍 시뮬레이터 + 링버퍼
│   ├── baseline_statistical.py  # 경로 A
│   ├── autoencoder_detector.py  # 경로 B
│   ├── cnn_fault_classifier.py  # 경로 C
│   ├── fusion.py             # A+B 점수 융합 + 디바운스
│   ├── inference.py          # 오케스트레이터(전처리→A·B→융합→게이트 C)
│   ├── evaluator.py          # AUC·pAUC·FPR·confusion matrix
│   └── utils.py              # config 로드·device 선택·시드 고정
├── models/                   # autoencoder.pth / fault_classifier.pth /
│                             # scaler.pkl / thresholds.json
├── outputs/                  # figures / heatmaps / reports
├── backend/api.py            # FastAPI: 추론 결과 JSON 제공
├── frontend/                 # 웹 대시보드(실시간 파형·점수·알람)
├── tests/                    # 단위 테스트(예: 임펄스 주입 검증)
├── run_baseline.py           # 경로 A 실행
├── train_autoencoder.py      # 경로 B 학습
├── train_classifier.py       # 경로 C 학습
└── run_inference.py          # 최종 추론(streaming + inference)
```

> 데이터 누수 차단: 정규화 통계·임계값은 **정상 데이터에서만** 산정해 저장·재사용한다.
> 고장 데이터는 평가(`evaluator.py`)에서만 등장하며 학습 경로에서 접근 불가.

## 환경 셋업

```powershell
# 가상환경 만들기 (최초 1회)
py -3.12 -m venv .venv

# 라이브러리 설치
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 실행

```powershell
.venv\Scripts\python.exe run_baseline.py        # 경로 A 임계값 산정
.venv\Scripts\python.exe train_autoencoder.py   # 경로 B 학습
.venv\Scripts\python.exe train_classifier.py    # 경로 C 학습
.venv\Scripts\python.exe run_inference.py        # 최종 추론(실시간 시뮬)
```

## GPU(RTX 5090) 쓰고 싶을 때

기본은 CPU 버전 torch 다. GPU로 바꾸려면 torch 만 다시 깔면 된다
(코드는 device=auto 라 안 고쳐도 됨):

```powershell
.venv\Scripts\python.exe -m pip uninstall -y torch
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```
