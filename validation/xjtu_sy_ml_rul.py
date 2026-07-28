# -*- coding: utf-8 -*-
"""ML RUL：用監督式模型（RandomForest）學「劣化特徵 → 剩餘壽命」的映射，看能不能打敗
線性趨勢外推那個很爛的成績（15 顆軸承 RUL 相對誤差中位數 164~610%）。

跟線性外推的差別：不再假設「線性」，而是讓模型從一堆真實 run-to-failure 曲線裡「學」出
「這組特徵長這樣 → 大概還剩多久」。關鍵靠「RMS 上升斜率」特徵讓模型能分辨快壞 vs 慢壞。

嚴謹性（不作弊）：留一顆軸承交叉驗證（Leave-One-Bearing-Out）——測試的那顆軸承，模型
訓練時完全沒看過。每顆軸承都輪流當一次測試集。

特徵（全部尺度無關，可跨不同壽命/振幅的軸承）：
  rms_norm      振動 RMS / 健康期基線（劣化程度）
  kurtosis      峰度（軸承故障的衝擊性，會上升）
  crest         波峰因子 = 峰值/RMS
  rms_slope     rms_norm 的局部上升斜率（劣化「速率」——分辨快壞/慢壞的關鍵）
目標：RUL（分鐘）= (總快照數 - i) × 每快照間隔。

用法（容器內）：python xjtu_sy_ml_rul.py <35Hz路徑> <37.5Hz路徑> <40Hz路徑>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as sp_kurtosis
from sklearn.ensemble import RandomForestRegressor

READ_SAMPLES = 8192
SNAPSHOT_INTERVAL_MIN = 1.0
HEALTHY_FRACTION = 0.15
SMOOTH_WIN = 5
EVAL_FRACTIONS = (0.5, 0.7, 0.9)


def bearing_features(bearing_dir: Path):
    """回傳這顆軸承的 (特徵矩陣 X[n,4], RUL 目標 y[n], 快照數 n)。"""
    files = sorted(bearing_dir.glob("*.csv"), key=lambda p: int(p.stem))
    rms, kurt, crest = [], [], []
    for f in files:
        x = pd.read_csv(f, usecols=[0], nrows=READ_SAMPLES).iloc[:, 0].to_numpy(float)
        r = float(np.sqrt(np.mean(x * x)))
        peak = float(np.max(np.abs(x)))
        rms.append(r)
        kurt.append(float(sp_kurtosis(x)))
        crest.append(peak / r if r > 1e-12 else 0.0)
    rms = pd.Series(rms).rolling(SMOOTH_WIN, min_periods=1, center=True).median().to_numpy()
    kurt = np.array(kurt)
    crest = np.array(crest)
    n = len(rms)

    baseline = float(np.mean(rms[: max(1, int(HEALTHY_FRACTION * n))]))
    rms_norm = rms / (baseline if baseline > 1e-12 else 1.0)
    # 局部斜率：rms_norm 對「最近 W 個快照」的線性斜率（劣化速率，尺度無關）
    w = max(5, int(0.1 * n))
    rms_slope = np.zeros(n)
    for i in range(n):
        lo = max(0, i - w + 1)
        seg = rms_norm[lo : i + 1]
        if len(seg) >= 3:
            rms_slope[i] = np.polyfit(np.arange(len(seg)), seg, 1)[0]

    X = np.column_stack([rms_norm, kurt, crest, rms_slope])
    y = (n - np.arange(n)) * SNAPSHOT_INTERVAL_MIN  # RUL（分鐘）
    return X, y, n


def main():
    roots = [Path(a) for a in sys.argv[1:]]
    dirs = sorted((d for root in roots for d in root.iterdir() if d.is_dir() and d.name.startswith("Bearing")),
                  key=lambda p: p.name)
    print(f"載入 {len(dirs)} 顆軸承的特徵…（LOBO 交叉驗證：測試軸承訓練時完全沒看過）")
    data = {d.name: bearing_features(d) for d in dirs}
    names = list(data.keys())

    # 每顆輪流當測試集，其餘訓練
    rel_err = {f: [] for f in EVAL_FRACTIONS}
    per_bearing = {}
    for test_name in names:
        X_tr = np.vstack([data[n][0] for n in names if n != test_name])
        y_tr = np.concatenate([data[n][1] for n in names if n != test_name])
        model = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=0)
        model.fit(X_tr, y_tr)

        X_te, y_te, n_te = data[test_name]
        pred = model.predict(X_te)
        errs = {}
        for f in EVAL_FRACTIONS:
            i = int(f * n_te)
            p, a = float(pred[i]), float(y_te[i])
            e = abs(p - a) / a if a > 0 else 0.0
            errs[f] = (p, a, e)
            rel_err[f].append(e)
        per_bearing[test_name] = (n_te, errs)

    # 每顆詳細
    hdr = f"{'軸承':<13}{'壽命':>7}"
    for f in EVAL_FRACTIONS:
        hdr += f"{'  '+str(int(f*100))+'%:預測/實際':>20}"
    print("\n" + hdr)
    print("-" * len(hdr.encode("ascii", "replace")))
    for name in names:
        n_te, errs = per_bearing[name]
        line = f"{name:<13}{n_te:>7}"
        for f in EVAL_FRACTIONS:
            p, a, e = errs[f]
            line += f"{'  '+str(int(p))+'/'+str(int(a)):>20}"
        print(line)

    print("\n=== 彙總：ML RUL 各壽命階段相對誤差（vs 線性外推基準）===")
    baseline = {0.5: "610%", 0.7: "327%", 0.9: "164%"}
    for f in EVAL_FRACTIONS:
        errs = rel_err[f]
        print(f"  壽命 {int(f*100)}% 處：ML 相對誤差 中位數 {np.median(errs):.0%}、"
              f"範圍 {min(errs):.0%}~{max(errs):.0%}   （線性基準 {baseline[f]}）")


if __name__ == "__main__":
    main()
