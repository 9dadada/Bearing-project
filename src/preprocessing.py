"""윈도잉·정규화 (전처리).

- window slicing: config 의 window.length / overlap 으로 신호를 (N, L) 윈도우로 자름.
- normalization: 정규화 통계(평균·표준편차)는 '정상 train'에서만 fit 후 저장·재사용.
  고장 데이터에는 절대 fit 하지 않는다(누수 차단). fit 결과는 models/scaler.pkl 로 저장.

TODO: make_windows(signal, length, overlap), fit_scaler(normal), apply_scaler(x).
"""
