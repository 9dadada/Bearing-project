"""시각화 — 그리는 함수 모음.

여러 곳(run_baseline·evaluator·run_inference)에서 불러 쓰는 matplotlib 함수를 모은다.
나온 이미지는 outputs/figures · outputs/heatmaps · data/spectrograms 에 저장한다.

실행(현재 전처리 결과 보기):
    .venv\\Scripts\\python.exe -m src.visualization
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 화면 없이 파일로 저장
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils import load_config, resolve_path

# 한글 라벨이 깨지지 않게 윈도우 기본 한글 폰트 사용
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def plot_windowing_demo(
    name: str, signal: np.ndarray, save_path: str | Path, title: str | None = None
) -> Path:
    """주어진 신호 하나에 대해 윈도잉 + 정규화를 그린다.

    - 위: 신호 일부 + 윈도우 4개 경계(겹침) 표시
    - 아래: 윈도우 1개 원본 vs 정규화 후
    title 을 안 주면 "{name} windowing + normalization" 으로 자동 생성.
    """
    from src.preprocessing import apply_scaler, load_scaler, make_windows

    cfg = load_config()
    length = cfg["window"]["length"]
    overlap = cfg["window"]["overlap"]
    step = int(round(length * (1.0 - overlap)))

    windows = make_windows(signal, length, overlap)
    scaler = load_scaler(cfg["artifacts"]["scaler"])
    if title is None:
        title = f"{name.replace('_', ' ')} windowing + normalization"

    n_win = windows.shape[0]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))

    # [1] 앞 4개 윈도우로 '겹침' 구조 보기 (깔끔하게 4개만)
    span = step * 3 + length
    axes[0].plot(np.arange(span), signal[:span], color="#333", lw=0.7)
    colors = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6"]
    for i in range(4):
        s = i * step
        axes[0].axvspan(s, s + length, color=colors[i], alpha=0.18)
        axes[0].axvline(s, color=colors[i], ls="--", lw=0.9)
    axes[0].set_xlim(0, span)
    axes[0].set_title(
        f"[1] 윈도잉 (앞 4개) — 길이 {length}, 절반({step})씩 겹침"
        f"   ※ 신호 전체는 이렇게 {n_win}개로 잘림"
    )
    axes[0].set_xlabel("샘플 인덱스")
    axes[0].set_ylabel("진폭")

    # [2] 정규화 효과 — 윈도우 1개 원본 vs 정규화 후
    w = windows[0]
    wn = apply_scaler(w, scaler)
    axes[1].plot(w, color="#e74c3c", lw=0.6, label=f"원본 (std={w.std():.4f})")
    axes[1].plot(wn, color="#2980b9", lw=0.6, alpha=0.8, label=f"정규화 후 (std={wn.std():.4f})")
    axes[1].set_title("[2] 정규화 효과: 윈도우 1개 — 모양은 그대로, 스케일만 표준화")
    axes[1].set_xlabel("샘플 인덱스 (0~1599)")
    axes[1].set_ylabel("진폭")
    axes[1].legend(loc="upper right", fontsize=9)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    out = resolve_path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_signal_overview(save_path: str | Path) -> Path:
    """정상 vs 고장(내륜·외륜·볼) 신호를 '전체 길이'로 비교해 그린다.

    - 가로축: 시간(초) — 모두 48kHz라 실시간 비교 가능
    - 세로축: 진폭 — 네 신호 공통 스케일(고장의 큰 진폭이 한눈에 보이게)
    """
    from src.data_loader import load_fault_signals, load_normal_signals

    cfg = load_config()
    sr = cfg["signal"]["base_sampling_rate"]

    normal = load_normal_signals(cfg)
    fault = load_fault_signals(cfg)

    # 정상 1개 + 고장 위치별(IR/OR/B) 1개씩 고른다 (가능하면 0.007 크기)
    norm_name, norm_sig = next(iter(normal.items()))
    rows = [("정상 (normal)", norm_name, norm_sig, "#2c3e50")]
    picks = {"IR": "내륜 고장", "OR": "외륜 고장", "B": "볼 고장"}
    colors = {"IR": "#e74c3c", "OR": "#e67e22", "B": "#8e44ad"}
    for loc, kor in picks.items():
        cand = [v for v in fault.values() if v["location"] == loc]
        if not cand:
            continue
        cand.sort(key=lambda v: v["size"] or "")  # 작은 손상부터
        v = cand[0]
        rows.append((f"{kor} ({v['label']})", v["source"], v["signal"], colors[loc]))

    # 공통 y-스케일 (네 신호 중 최대 진폭 기준)
    ymax = max(float(np.abs(sig).max()) for _, _, sig, _ in rows) * 1.05

    fig, axes = plt.subplots(len(rows), 1, figsize=(12, 2.2 * len(rows)), sharex=False)
    for ax, (title, _src, sig, color) in zip(axes, rows):
        t = np.arange(len(sig)) / sr
        ax.plot(t, sig, color=color, lw=0.4)
        ax.set_ylim(-ymax, ymax)
        ax.set_title(f"{title}  —  {len(sig):,}점 (약 {len(sig)/sr:.2f}초)", fontsize=10)
        ax.set_ylabel("진폭")
    axes[-1].set_xlabel("시간 (초)")
    fig.suptitle("정상 vs 고장 진동 신호 (전체 길이, 공통 스케일)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    out = resolve_path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    from src.data_loader import load_fault_signals, load_normal_signals

    cfg = load_config()
    normal = load_normal_signals(cfg)
    fault = load_fault_signals(cfg)

    # 정상 4개: normal_0 ~ normal_3 → 각자 파일
    for name, sig in normal.items():
        out = plot_windowing_demo(name, sig, f"outputs/figures/{name}_windowing.png")
        print(f"저장됨: {out.relative_to(_ROOT)}")

    # 고장 위치별 1개씩: IR / OR / B (작은 손상 0.007 우선)
    picks = {"IR": "IR_fault", "OR": "OR_fault", "B": "B_fault"}
    for loc, fname in picks.items():
        cand = [v for v in fault.values() if v["location"] == loc]
        if not cand:
            continue
        cand.sort(key=lambda v: v["size"] or "")
        out = plot_windowing_demo(fname, cand[0]["signal"], f"outputs/figures/{fname}_windowing.png")
        print(f"저장됨: {out.relative_to(_ROOT)}")


if __name__ == "__main__":
    _main()
