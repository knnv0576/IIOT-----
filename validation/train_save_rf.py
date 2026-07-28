# -*- coding: utf-8 -*-
"""訓練正式服務要用的 RF RUL 模型並存檔。用全部 32 顆 PHM+XJTU 軸承訓練，
存成 phase3_ai 能載入的 pickle（含模型 + 特徵順序 + 參數，讓正式服務算特徵時對得上）。

用法：python train_save_rf.py <cache.pkl> <out.pkl>
"""
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

MAX_RUL = 250.0
HEALTHY_FRAC = 0.15
SLOPE_WIN_MIN = 10.0
FEATURES = ["rms_norm", "kurt", "crest", "rms_slope_per_min"]


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
    return X.astype(np.float32), np.minimum(rul, MAX_RUL).astype(np.float32)


def main():
    cache = pickle.load(open(sys.argv[1], "rb"))
    train = [k for k in cache if not k.startswith("IMS/")]   # 只用 PHM+XJTU 訓練（IMS 留作獨立驗證）
    X = np.vstack([bearing_feat(cache[k])[0] for k in train])
    y = np.concatenate([bearing_feat(cache[k])[1] for k in train])
    model = RandomForestRegressor(n_estimators=120, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=0)
    model.fit(X, y)
    bundle = {"model": model, "features": FEATURES, "max_rul_min": MAX_RUL,
              "slope_win_min": SLOPE_WIN_MIN, "healthy_frac": HEALTHY_FRAC,
              "trained_on": f"{len(train)} bearings (PHM2012+XJTU)"}
    with open(sys.argv[2], "wb") as f:
        pickle.dump(bundle, f)
    print(f"RF 模型已存：{len(train)} 顆軸承 {len(y)} 樣本 → {sys.argv[2]}")
    print(f"特徵重要度：{dict(zip(FEATURES, np.round(model.feature_importances_, 3)))}")


if __name__ == "__main__":
    main()
