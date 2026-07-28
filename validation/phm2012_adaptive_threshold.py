# -*- coding: utf-8 -*-
"""自適應門檻實驗：解決固定 -1.0 全域門檻在 17 顆真實軸承上誤報率 0~67% 嚴重離散的問題。

固定門檻的根本問題：每顆軸承的噪聲底噪不同，同一條 -1.0 分數線在噪聲高的軸承上切得太鬆
（誤報爆高）、在乾淨的軸承上又還好。自適應門檻的想法：不寫死一個全域數字，而是用「每顆
軸承自己健康期的分數分佈」推出這顆專屬的門檻。

關鍵設計（避免偷看答案）：把健康期（壽命前 15%）再切成兩段——
  - fit 段（前 2/3）：只拿來訓練 StandardScaler + OneClassSVM。
  - calib 段（後 1/3）：完全沒參與訓練的健康資料，用它的分數分佈定門檻。
門檻 = mean(calib) - K·std(calib)。K 越大門檻越嚴（越不容易誤報，但也越晚預警）。
這樣門檻會自動貼合每顆軸承的噪聲水平：噪聲大的軸承 std 大、門檻自動往下拉。

對照組：固定 -1.0（沿用 anomaly_detector.py 目前正式服務用的邊際）。

用法（在 ai_service 容器內）：
    python phm2012_adaptive_threshold.py /tmp/Learning_set /tmp/Full_Test_Set
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from anomaly_detector import ANOMALY_SVM_GAMMA
from spectral_features import to_vector
from spectral_features import extract_features as _extract

SAMPLE_RATE_HZ = 25600.0
BAND_HALF_WIDTH_HZ = 15.0
SNAPSHOT_INTERVAL_S = 10.0
CONSECUTIVE_ALARMS = 3
HEALTHY_FRACTION = 0.15  # 壽命前 15% 視為確定健康
FIT_RATIO = 2.0 / 3.0  # 健康期裡前 2/3 拿來 fit，後 1/3 拿來校準門檻
FIXED_MARGIN = -1.0  # 對照組：正式服務目前用的固定門檻
K_VALUES = (2.0, 3.0, 4.0, 5.0)  # 自適應門檻的 std 倍數，掃描比較

CONDITION_FUNDAMENTAL_HZ = {"1": 30.0, "2": 27.5, "3": 25.0}


def fundamental_for(name: str) -> float:
    return CONDITION_FUNDAMENTAL_HZ[name.replace("Bearing", "")[0]]


def _load_one(path: Path) -> np.ndarray:
    with open(path) as fh:
        first = fh.readline()
    delimiter = ";" if ";" in first else ","
    return np.loadtxt(path, delimiter=delimiter, usecols=4)


def load_snapshots(bearing_dir: Path) -> np.ndarray:
    files = sorted(bearing_dir.glob("acc_*.csv"))
    return np.array([_load_one(f) for f in files])


def features_of(snapshots: np.ndarray, fundamental_hz: float) -> np.ndarray:
    return np.array(
        [
            to_vector(_extract(s, sample_rate_hz=SAMPLE_RATE_HZ, fundamental_hz=fundamental_hz, band_half_width_hz=BAND_HALF_WIDTH_HZ))
            for s in snapshots
        ]
    )


def fmt_minutes(n_snapshots: float) -> str:
    m = n_snapshots * SNAPSHOT_INTERVAL_S / 60.0
    return f"{int(m // 60)}h{m % 60:04.1f}m"


def first_persistent_alarm(below: np.ndarray, start: int, n: int) -> int | None:
    for i in range(start, n - CONSECUTIVE_ALARMS + 1):
        if below[i : i + CONSECUTIVE_ALARMS].all():
            return i
    return None


def compute_scores(bearing_dir: Path):
    """回傳 (name, n, scores, n_healthy, calib_scores)。scores 是整段壽命每個快照的分數。"""
    name = bearing_dir.name
    snapshots = load_snapshots(bearing_dir)
    n = len(snapshots)
    n_healthy = int(n * HEALTHY_FRACTION)
    n_fit = max(CONSECUTIVE_ALARMS, int(n_healthy * FIT_RATIO))

    feats = features_of(snapshots, fundamental_for(name))
    scaler = StandardScaler().fit(feats[:n_fit])
    model = OneClassSVM(kernel="rbf", nu=0.05, gamma=ANOMALY_SVM_GAMMA).fit(scaler.transform(feats[:n_fit]))
    scores = model.decision_function(scaler.transform(feats))
    calib_scores = scores[n_fit:n_healthy]  # 健康期後 1/3，沒參與 fit
    return name, n, scores, n_healthy, calib_scores


def evaluate_threshold(name, n, scores, n_healthy, threshold):
    below = scores < threshold
    # 健康期誤報參考區：健康期結束 ~ 壽命 50%（一般仍視為健康）
    fp = below[n_healthy : max(n_healthy + 1, n // 2)].mean()
    idx = first_persistent_alarm(below, n_healthy, n)
    return fp, idx


def run_strategy(bearings, threshold_fn, label):
    """threshold_fn(calib_scores) -> 門檻。回傳每顆的 (name, n, fp, idx)。"""
    rows = []
    for name, n, scores, n_healthy, calib in bearings:
        thr = threshold_fn(calib)
        fp, idx = evaluate_threshold(name, n, scores, n_healthy, thr)
        rows.append((name, n, fp, idx, thr))
    detected = [r for r in rows if r[3] is not None]
    fps = [r[2] for r in rows]
    leads = [(r[1] - r[3]) * SNAPSHOT_INTERVAL_S / 60.0 for r in detected]
    print(f"\n=== {label} ===")
    print(f"  偵測成功率：{len(detected)}/{len(rows)}")
    print(f"  健康期誤報率：中位數 {np.median(fps):.1%}、平均 {np.mean(fps):.1%}、範圍 {min(fps):.1%}~{max(fps):.1%}、標準差 {np.std(fps):.1%}")
    print(f"  超過 10% 誤報的軸承數：{sum(f > 0.10 for f in fps)}/{len(rows)}")
    if leads:
        print(f"  提前預警時間：中位數 {np.median(leads):.0f} 分（範圍 {min(leads):.0f}~{max(leads):.0f}）")
    return rows


def main() -> None:
    roots = [Path(a) for a in sys.argv[1:]]
    dirs = sorted(
        (d for root in roots for d in root.iterdir() if d.is_dir() and d.name.startswith("Bearing")),
        key=lambda p: p.name,
    )
    print(f"載入 {len(dirs)} 顆軸承，計算分數中…")
    bearings = [compute_scores(d) for d in dirs]

    # 對照組：固定門檻
    fixed_rows = run_strategy(bearings, lambda c: FIXED_MARGIN, f"對照組：固定門檻 {FIXED_MARGIN}")

    # 自適應：mean - K·std，掃描 K
    adaptive_results = {}
    for k in K_VALUES:
        adaptive_results[k] = run_strategy(
            bearings, lambda c, k=k: float(np.mean(c) - k * np.std(c)), f"自適應門檻 K={k}（mean - {k:.0f}·std）"
        )

    # 挑一個 K 印每顆軸承的對照明細（用誤報離散度最小、且偵測率仍 100% 的 K）
    best_k = min(
        (k for k in K_VALUES if sum(r[3] is not None for r in adaptive_results[k]) == len(bearings)),
        key=lambda k: np.std([r[2] for r in adaptive_results[k]]),
        default=K_VALUES[1],
    )
    print(f"\n\n每顆軸承對照（固定 -1.0  vs  自適應 K={best_k:.0f}）")
    header = f"{'軸承':<12}{'固定誤報':>9}{'固定預警':>10}{'自適應誤報':>11}{'自適應預警':>11}"
    print(header)
    print("-" * 62)
    for fixed, adap in zip(fixed_rows, adaptive_results[best_k]):
        name, n = fixed[0], fixed[1]
        f_lead = fmt_minutes(n - fixed[3]) if fixed[3] is not None else "未觸發"
        a_lead = fmt_minutes(n - adap[3]) if adap[3] is not None else "未觸發"
        print(f"{name:<12}{fixed[2]:>9.1%}{f_lead:>10}{adap[2]:>11.1%}{a_lead:>11}")


if __name__ == "__main__":
    main()
