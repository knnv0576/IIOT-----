# -*- coding: utf-8 -*-
"""XJTU-SY 跨資料集 CUSUM 驗證：把同一套偵測器（頻譜特徵 + One-Class SVM + CUSUM）
套到 XJTU-SY 的 15 顆真實軸承，回答「PHM2012 的 17/17 是通用還是碰巧」。

方法與 phm2012_cusum_validation.py 完全一致（同一套 SVM 特徵、同樣的 fit/calib 不偷看答案、
同樣的 CUSUM slack/h 掃描），只換資料集相關參數，確保是 apples-to-apples 的跨資料集比較。

XJTU-SY 規格（見 Introduction_to_XJTU-SY_Bearing_Dataset.pdf）：
  25.6kHz、每檔 32768 點(1.28s)、每分鐘一檔（N.csv）、第1欄水平/第2欄垂直振動。
  三工況轉頻：35Hz12kN=35Hz、37.5Hz11kN=37.5Hz、40Hz10kN=40Hz。run-to-failure，無劣化起點標籤。

用法（在 ai_service 容器內、PYTHONPATH=/app、資料唯讀掛載到 /val/data）：
    python xjtu_sy_cusum_validation.py <35Hz12kN路徑> <37.5Hz11kN路徑> <40Hz10kN路徑>
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from anomaly_detector import ANOMALY_SVM_GAMMA
from spectral_features import to_vector
from spectral_features import extract_features as _extract

SAMPLE_RATE_HZ = 25600.0
BAND_HALF_WIDTH_HZ = 15.0
SNAPSHOT_INTERVAL_S = 60.0          # XJTU-SY 每分鐘一檔（PHM2012 是每 10 秒）
READ_SAMPLES = 8192                 # 每檔只讀前 8192 點（0.32s）當一個特徵窗，加速 I/O、FFT 解析度仍夠(3.1Hz)
HEALTHY_FRACTION = 0.15
FIT_RATIO = 2.0 / 3.0
SLACK_K = 0.5
H_VALUES = (3.0, 5.0, 8.0, 12.0)
MIN_SNAPSHOTS = 30                  # 太短的軸承 CUSUM 校準不可靠，跳過

# XJTU-SY 三工況轉頻（Bearing1_x=35Hz、2_x=37.5Hz、3_x=40Hz），對應 folder 內軸承編號前綴
CONDITION_FUNDAMENTAL_HZ = {"1": 35.0, "2": 37.5, "3": 40.0}


def fundamental_for(name: str) -> float:
    return CONDITION_FUNDAMENTAL_HZ[name.replace("Bearing", "")[0]]


def _load_one(path: Path) -> np.ndarray:
    # 第1欄=水平振動；跳過 header；為速度只讀前 READ_SAMPLES 列
    return pd.read_csv(path, usecols=[0], nrows=READ_SAMPLES).iloc[:, 0].to_numpy(dtype=float)


def load_snapshots(bearing_dir: Path) -> np.ndarray:
    files = sorted(bearing_dir.glob("*.csv"), key=lambda p: int(p.stem))  # 檔名數字排序(1,2,...123)不能字典序
    return np.array([_load_one(f) for f in files])


def features_of(snapshots: np.ndarray, fundamental_hz: float) -> np.ndarray:
    return np.array([
        to_vector(_extract(s, sample_rate_hz=SAMPLE_RATE_HZ, fundamental_hz=fundamental_hz, band_half_width_hz=BAND_HALF_WIDTH_HZ))
        for s in snapshots
    ])


def fmt_minutes(n_snapshots: float) -> str:
    m = n_snapshots * SNAPSHOT_INTERVAL_S / 60.0
    return f"{int(m // 60)}h{m % 60:04.1f}m"


def cusum(scores, mu0, slack):
    s = np.zeros(len(scores)); running = 0.0
    for i, sc in enumerate(scores):
        running = max(0.0, running + (mu0 - sc - slack)); s[i] = running
    return s


def compute_bearing(bearing_dir: Path):
    name = bearing_dir.name
    snapshots = load_snapshots(bearing_dir)
    n = len(snapshots)
    if n < MIN_SNAPSHOTS:
        return None
    n_healthy = int(n * HEALTHY_FRACTION)
    n_fit = max(5, min(int(n_healthy * FIT_RATIO), n_healthy - 2))  # 保護：calib 至少留 2 點

    feats = features_of(snapshots, fundamental_for(name))
    scaler = StandardScaler().fit(feats[:n_fit])
    model = OneClassSVM(kernel="rbf", nu=0.05, gamma=ANOMALY_SVM_GAMMA).fit(scaler.transform(feats[:n_fit]))
    scores = model.decision_function(scaler.transform(feats))

    calib = scores[n_fit:n_healthy]
    mu0, sigma0 = float(np.mean(calib)), float(np.std(calib))
    return {"name": name, "n": n, "n_healthy": n_healthy,
            "scores": scores, "sigma0": sigma0, "cusum": cusum(scores, mu0, SLACK_K * sigma0)}


def evaluate_h(b, h_mult):
    n, n_healthy = b["n"], b["n_healthy"]
    h = h_mult * b["sigma0"]
    trig = b["cusum"] > h
    fp = trig[n_healthy: max(n_healthy + 1, n // 2)].mean()
    idx = next((i for i in range(n_healthy, n) if trig[i]), None)
    return fp, idx


def run_h(bearings, h_mult, label):
    rows = [(b["name"], b["n"], *evaluate_h(b, h_mult)) for b in bearings]
    detected = [r for r in rows if r[3] is not None]
    fps = [r[2] for r in rows]
    leads = [(r[1] - r[3]) * SNAPSHOT_INTERVAL_S / 60.0 for r in detected]
    print(f"\n=== {label} ===")
    print(f"  偵測成功率：{len(detected)}/{len(rows)}")
    print(f"  健康期誤報率：中位數 {np.median(fps):.1%}、範圍 {min(fps):.1%}~{max(fps):.1%}、標準差 {np.std(fps):.1%}")
    if leads:
        print(f"  提前預警時間：中位數 {np.median(leads):.0f} 分（範圍 {min(leads):.0f}~{max(leads):.0f}）")
    return rows


def main():
    roots = [Path(a) for a in sys.argv[1:]]
    dirs = sorted((d for root in roots for d in root.iterdir() if d.is_dir() and d.name.startswith("Bearing")),
                  key=lambda p: p.name)
    print(f"載入 {len(dirs)} 顆軸承計算 CUSUM 中…（每檔讀 {READ_SAMPLES} 點、slack={SLACK_K}·sigma0）")
    bearings = [b for b in (compute_bearing(d) for d in dirs) if b is not None]
    print(f"有效軸承 {len(bearings)}/{len(dirs)}（太短的已跳過）")

    all_rows = {h: run_h(bearings, h, f"CUSUM 門檻 h={h:.0f}·sigma0") for h in H_VALUES}

    full = [h for h in H_VALUES if sum(r[3] is not None for r in all_rows[h]) == len(bearings)]
    best_h = min(full, key=lambda h: np.std([r[2] for r in all_rows[h]])) if full else H_VALUES[-1]
    print(f"\n每顆軸承詳細（CUSUM h={best_h:.0f}·sigma0）")
    print(f"{'軸承':<14}{'壽命':>9}{'誤報率':>9}{'提前預警':>11}")
    print("-" * 44)
    for b, r in zip(bearings, all_rows[best_h]):
        lead = fmt_minutes(r[1] - r[3]) if r[3] is not None else "未觸發"
        print(f"{b['name']:<14}{fmt_minutes(r[1]):>9}{r[2]:>9.1%}{lead:>11}")


if __name__ == "__main__":
    main()
