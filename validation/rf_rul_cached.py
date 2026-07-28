# -*- coding: utf-8 -*-
"""從快取載入的 RandomForest RUL（跟 lstm_rul.py 用同一份快取、同樣測 IMS 三組，公平對比）。
特徵：[rms_norm, kurt, crest, rms_slope(per-min)]、piecewise 目標。訓練 PHM+XJTU、測 IMS。

用法：python rf_rul_cached.py <cache.pkl>
"""
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

MAX_RUL = 250.0
HEALTHY_FRAC = 0.15
SLOPE_WIN_MIN = 10.0


def bearing_feat(rec):
    rms = pd.Series(rec["rms"]).rolling(5, min_periods=1, center=True).median().to_numpy()
    kurt, crest = np.array(rec["kurt"]), np.array(rec["crest"])
    n = len(rms); interval = rec["interval"]
    base = float(np.mean(rms[: max(1, int(HEALTHY_FRAC * n))]))
    rms_norm = rms / (base if base > 1e-12 else 1.0)
    w = max(5, int(SLOPE_WIN_MIN / interval))
    slope = np.zeros(n)
    for i in range(n):
        seg = rms_norm[max(0, i - w + 1) : i + 1]
        if len(seg) >= 3:
            slope[i] = np.polyfit(np.arange(len(seg)), seg, 1)[0] / interval
    X = np.column_stack([rms_norm, kurt, crest, slope])
    rul = (n - np.arange(n)) * interval
    return X.astype(np.float32), np.minimum(rul, MAX_RUL).astype(np.float32), rul


def main():
    cache = pickle.load(open(sys.argv[1], "rb"))
    train = [k for k in cache if not k.startswith("IMS/")]
    ims = [k for k in cache if k.startswith("IMS/")]
    print(f"RF：訓練 {len(train)} 顆(PHM+XJTU)、測 {len(ims)} 顆(IMS)")

    feats = {k: bearing_feat(cache[k]) for k in cache}
    Xtr = np.vstack([feats[k][0] for k in train])
    ytr = np.concatenate([feats[k][1] for k in train])
    m = RandomForestRegressor(n_estimators=120, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=0)
    m.fit(Xtr, ytr)

    print("\n=== RF 在【完全沒看過的 IMS】測試（可行動窗 RUL 相對誤差）===")
    for k in ims:
        X, _, rul = feats[k]
        pred = m.predict(X)
        mask = rul <= MAX_RUL
        ae = np.abs(pred[mask] - rul[mask]) / np.maximum(rul[mask], 1e-9)
        print(f"  {k}: 可行動窗誤差中位 {np.median(ae):.0%}"
              f"（25~75分位 {np.percentile(ae,25):.0%}~{np.percentile(ae,75):.0%}，壽命 {int(len(rul)*cache[k]['interval']/60)}h）")


if __name__ == "__main__":
    main()
