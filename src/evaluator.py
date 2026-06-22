"""평가 — 노이즈 강건성 (경로 A·B·융합). "모델을 얼마나 신뢰할 수 있나".

깨끗한 신호 + 화이트/임펄스 노이즈를 단계별로 넣어, 노이즈가 세질수록
탐지 성능이 버티는지(=신뢰성) 측정한다. 모델·임계값은 '깨끗한 정상'으로 정한 것을
그대로 쓰고(재학습 X), 노이즈 낀 데이터로 테스트한다(현실 모사).

지표:
- 탐지율  : 진짜 고장 중 '이상'으로 잡은 비율 (높을수록 좋음)
- 오탐률  : 정상인데 '이상'으로 잘못 친 비율 (낮을수록 좋음)
- AUC     : 임계값 무관 분리도 (1.0=완벽, 0.5=찍기)

노이즈:
- 화이트: SNR +10 / 0 / −10 dB
- 임펄스: 약/중/강 = 신호 peak×(3/5/10), 무작위 위치·무작위 진폭

실행:
    .venv\\Scripts\\python.exe -m src.evaluator
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

from src.utils import load_config, get_device, resolve_path

_FS = 12000


# =============================================================
#  노이즈 주입
# =============================================================
def add_white_noise(sig: np.ndarray, snr_db: float, rng) -> np.ndarray:
    """목표 SNR(dB)에 맞춰 가우시안 노이즈를 더한다."""
    rms = float(np.sqrt(np.mean(sig**2)))
    noise_std = rms / (10 ** (snr_db / 20.0))
    return sig + rng.normal(0.0, noise_std, len(sig))


def add_impulse_noise(sig: np.ndarray, amp_mult: float, rng,
                      fs: int = _FS, rate_per_sec: float = 10.0) -> np.ndarray:
    """무작위 위치·무작위 진폭의 임펄스를 더한다(주기성 없음 = 환경 소음 모사).

    진폭 ≈ 신호 peak × amp_mult × (0.5~1.5 무작위), 부호 무작위.
    """
    out = sig.copy()
    peak = float(np.max(np.abs(sig)))
    n_imp = max(1, int(len(sig) / fs * rate_per_sec))
    idx = rng.integers(0, len(sig), n_imp)                  # 무작위 위치
    amps = peak * amp_mult * rng.uniform(0.5, 1.5, n_imp)   # 무작위 진폭
    signs = rng.choice([-1.0, 1.0], n_imp)
    out[idx] += amps * signs
    return out


# 평가 조건: (이름, 종류, 값)
CONDITIONS = [
    ("깨끗", None, None),
    ("화이트 +10dB", "white", 10),
    ("화이트 0dB", "white", 0),
    ("화이트 -10dB", "white", -10),
    ("임펄스 약(×3)", "impulse", 3),
    ("임펄스 중(×5)", "impulse", 5),
    ("임펄스 강(×10)", "impulse", 10),
]


def apply_noise(sig: np.ndarray, kind, val, rng) -> np.ndarray:
    if kind is None:
        return sig
    if kind == "white":
        return add_white_noise(sig, val, rng)
    return add_impulse_noise(sig, val, rng)


# =============================================================
#  A·B 점수 (윈도우별)
# =============================================================
def _load_ab(cfg, device):
    from src.autoencoder_detector import load_model
    from src.baseline_statistical import load_thresholds
    from src.spectrogram import load_spec_scaler

    thr = load_thresholds(cfg)
    models_dir = cfg["artifacts"]["models"]
    return (
        thr["path_a_statistical"],
        float(thr["path_b"]["threshold"]),
        load_model(cfg, f"{models_dir}/autoencoder.pth", device),
        load_spec_scaler(f"{models_dir}/spec_scaler.pkl"),
    )


def _window_scores(sig, cfg, thr_a, ae, scaler, T_b, device):
    """신호 → (A 점수, B 점수) 윈도우별. 둘 다 1.0 초과면 이상."""
    from src.autoencoder_detector import anomaly_score_b, recon_errors
    from src.baseline_statistical import anomaly_score, extract_features
    from src.preprocessing import make_windows
    from src.spectrogram import apply_spec_scaler, to_spectrograms

    feats = cfg["stats"]["features"]
    w = make_windows(sig, cfg["window"]["length"], cfg["window"]["overlap"])
    if len(w) == 0:
        return np.empty(0), np.empty(0)
    sa = np.array([anomaly_score(extract_features(x, feats), thr_a) for x in w])
    err = recon_errors(ae, apply_spec_scaler(to_spectrograms(w, cfg), scaler), device)
    return sa, np.asarray(anomaly_score_b(err, T_b))


def _rates(normal_scores, fault_scores):
    """점수 배열들 → (오탐률, 탐지율, AUC). 점수>1 이면 이상."""
    fpr = float((normal_scores > 1.0).mean()) if len(normal_scores) else 0.0
    det = float((fault_scores > 1.0).mean()) if len(fault_scores) else 0.0
    y = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(fault_scores))])
    s = np.concatenate([normal_scores, fault_scores])
    auc = float(roc_auc_score(y, s)) if len(set(y.tolist())) == 2 else float("nan")
    return fpr, det, auc


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    from src.data_loader import load_fault_signals, load_normal_signals

    cfg = load_config()
    device = get_device(cfg)
    rng = np.random.default_rng(cfg["seed"])
    thr_a, T_b, ae, scaler = _load_ab(cfg, device)

    normal = list(load_normal_signals(cfg).values())          # 정상 4개
    fault_all = load_fault_signals(cfg)
    fault = []                                                 # 위치별 2개씩(속도)
    for loc in ("IR", "OR", "B"):
        picks = [v["signal"] for v in fault_all.values() if v["location"] == loc][:2]
        fault.extend(picks)

    print("=== 노이즈 강건성 평가 (A·B·융합) ===")
    print(f"정상 {len(normal)}개 / 고장 {len(fault)}개, 장치 {device}\n")
    header = f"{'조건':<14} | {'A(탐지/오탐/AUC)':<22} | {'B(탐지/오탐/AUC)':<22} | {'융합(탐지/오탐/AUC)':<22}"
    print(header)
    print("-" * len(header))

    results = {"A": [], "B": [], "F": []}
    for name, kind, val in CONDITIONS:
        # 점수 모으기 (정상/고장 각각, 노이즈 적용)
        def collect(signals):
            sa_l, sb_l = [], []
            for sig in signals:
                sa, sb = _window_scores(apply_noise(sig, kind, val, rng), cfg, thr_a, ae, scaler, T_b, device)
                sa_l.append(sa); sb_l.append(sb)
            return np.concatenate(sa_l), np.concatenate(sb_l)

        na, nb = collect(normal)
        fa, fb = collect(fault)
        nf, ff = np.maximum(na, nb), np.maximum(fa, fb)        # 융합(OR) = max

        a = _rates(na, fa); b = _rates(nb, fb); f = _rates(nf, ff)
        results["A"].append(a); results["B"].append(b); results["F"].append(f)

        def fmt(r):
            return f"{100*r[1]:5.1f}% /{100*r[0]:5.1f}% /{r[2]:.3f}"
        print(f"{name:<14} | {fmt(a):<22} | {fmt(b):<22} | {fmt(f):<22}")

    _plot(results)
    print("\n해석: 노이즈가 세질수록 탐지율이 버티고 오탐률이 안 치솟으면 → 신뢰할 수 있는 모델.")
    print("      특히 '임펄스'에서 A의 오탐률이 오르는지 보라 (충격을 고장으로 오인).")


def _plot(results):
    names = [c[0] for c in CONDITIONS]
    x = np.arange(len(names))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"A": "#16a085", "B": "#8e44ad", "F": "#e67e22"}
    labels = {"A": "경로 A", "B": "경로 B", "F": "A+B 융합"}
    for k in ("A", "B", "F"):
        det = [r[1] * 100 for r in results[k]]
        fpr = [r[0] * 100 for r in results[k]]
        ax1.plot(x, det, "o-", color=colors[k], label=labels[k])
        ax2.plot(x, fpr, "o-", color=colors[k], label=labels[k])
    for ax, title, ylab in ((ax1, "고장 탐지율 (높을수록 좋음)", "탐지율 %"),
                            (ax2, "정상 오탐률 (낮을수록 좋음)", "오탐률 %")):
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontweight="bold"); ax.set_ylabel(ylab)
        ax.grid(alpha=0.3); ax.legend()
    fig.suptitle("노이즈 강건성 — 모델 신뢰도 평가 (A·B·융합)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = resolve_path("outputs/figures/noise_robustness.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"\n저장: {out.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
