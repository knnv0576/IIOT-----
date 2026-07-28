# -*- coding: utf-8 -*-
"""驗證「密集警報起點 → 損壞」的時間間隔是否可觀察、可預測。

觀念：把之前當成「誤報」的東西翻轉看——警報在壽命後期密集出現，其實是偵測器正確追蹤劣化。
本腳本對幾顆代表性軸承，計算「瞬時警報密度」隨壽命的軌跡，找出密度飆升的起點，
量「密集警報起點 → 損壞」的間隔，看它是否穩定/可預測。

用法（容器內，PYTHONPATH=/app）：python xjtu_sy_alarm_density.py <35Hz資料夾> <37.5Hz資料夾> <40Hz資料夾>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from anomaly_detector import ANOMALY_SVM_GAMMA, ANOMALY_SCORE_MARGIN
from spectral_features import to_vector
from spectral_features import extract_features as _extract

SAMPLE_RATE_HZ, BAND_HALF_WIDTH_HZ, READ_SAMPLES = 25600.0, 15.0, 8192
SNAPSHOT_INTERVAL_S = 60.0
HEALTHY_FRACTION, FIT_RATIO = 0.15, 2.0 / 3.0
DENSITY_WINDOW = 10        # 滾動警報密度的視窗（最近 10 個快照）
DENSITY_ONSET = 0.7        # 密度 >= 70% 視為「密集警報起點」
COND = {"1": 35.0, "2": 37.5, "3": 40.0}
# 挑不同壽命代表：短(1_5=52m)、中(2_5=5h39)、中長(3_3=6h11)、超長(3_1=42h)
TARGETS = {"Bearing1_5", "Bearing2_5", "Bearing3_3", "Bearing3_1"}


def load(bd):
    files = sorted(bd.glob("*.csv"), key=lambda p: int(p.stem))
    return np.array([pd.read_csv(f, usecols=[0], nrows=READ_SAMPLES).iloc[:, 0].to_numpy(float) for f in files])


def fmt(n):
    m = n * SNAPSHOT_INTERVAL_S / 60.0
    return f"{int(m//60)}h{m%60:04.1f}m"


def analyse(bd):
    name = bd.name
    snaps = load(bd)
    n = len(snaps)
    n_h = int(n * HEALTHY_FRACTION)
    n_fit = max(5, min(int(n_h * FIT_RATIO), n_h - 2))
    fhz = COND[name.replace("Bearing", "")[0]]
    feats = np.array([to_vector(_extract(s, sample_rate_hz=SAMPLE_RATE_HZ, fundamental_hz=fhz, band_half_width_hz=BAND_HALF_WIDTH_HZ)) for s in snaps])
    sc = StandardScaler().fit(feats[:n_fit])
    m = OneClassSVM(kernel="rbf", nu=0.05, gamma=ANOMALY_SVM_GAMMA).fit(sc.transform(feats[:n_fit]))
    scores = m.decision_function(sc.transform(feats))
    flags = scores < ANOMALY_SCORE_MARGIN  # 瞬時警報（跟線上偵測器同門檻）

    # 滾動警報密度
    dens = np.array([flags[max(0, i - DENSITY_WINDOW + 1): i + 1].mean() for i in range(n)])
    # 密集警報起點：density 首次持續 >= 門檻（連續 3 個快照都超過，避免單發）
    onset = next((i for i in range(n_h, n - 2) if (dens[i:i+3] >= DENSITY_ONSET).all()), None)

    print(f"\n{name}  壽命={fmt(n)}（{n} 個快照）")
    # 壽命十等分的警報密度，看「爬升」
    seg = "  密度軌跡: " + " ".join(f"{dens[n*d//10:n*(d+1)//10].mean():.0%}" for d in range(10))
    print(seg)
    if onset is None:
        print("  未偵測到密集警報起點")
        return
    lead = n - onset
    print(f"  密集警報起點: 壽命 {onset/n:.0%} 處（開機後 {fmt(onset)}）")
    print(f"  → 損壞的間隔: {fmt(lead)}（{lead} 個快照，占總壽命 {lead/n:.0%}）")


def main():
    roots = [Path(a) for a in sys.argv[1:]]
    dirs = [d for root in roots for d in root.iterdir() if d.is_dir() and d.name in TARGETS]
    print(f"分析 {len(dirs)} 顆代表性軸承的警報密度軌跡…")
    for d in sorted(dirs, key=lambda p: p.name):
        analyse(d)


if __name__ == "__main__":
    main()
