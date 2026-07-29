# -*- coding: utf-8 -*-
"""按「真實剩餘壽命區間」拆解 RUL 準確度，回答：保守估計多久內的預測才可信。
訓練 PHM+XJTU、在完全獨立的 IMS 上評估（跟正式模型同設定）。

用法：python rul_horizon_accuracy.py data/cache/bearing_stats.pkl
"""
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

MAX_RUL = 250.0
HEALTHY_FRAC = 0.15
SLOPE_WIN_MIN = 10.0

# 分析用的真實 RUL 分段（分鐘）
BANDS = [(0, 30), (30, 60), (60, 120), (120, 250)]


def bearing_feat(rec):
    rms = pd.Series(rec["rms"]).rolling(5, min_periods=1, center=True).median().to_numpy()
    kurt, crest = np.array(rec["kurt"]), np.array(rec["crest"])
    n = len(rms)
    interval = rec["interval"]
    base = float(np.mean(rms[: max(1, int(HEALTHY_FRAC * n))]))
    rms_norm = rms / (base if base > 1e-12 else 1.0)
    w = max(5, int(SLOPE_WIN_MIN / interval))
    slope = np.zeros(n)
    for i in range(n):
        seg = rms_norm[max(0, i - w + 1): i + 1]
        if len(seg) >= 3:
            slope[i] = np.polyfit(np.arange(len(seg)), seg, 1)[0] / interval
    X = np.column_stack([rms_norm, kurt, crest, slope])
    rul = (n - np.arange(n)) * interval
    return X.astype(np.float32), np.minimum(rul, MAX_RUL).astype(np.float32), rul


def main():
    cache = pickle.load(open(sys.argv[1], "rb"))
    train = [k for k in cache if not k.startswith("IMS/")]
    ims = [k for k in cache if k.startswith("IMS/")]
    feats = {k: bearing_feat(cache[k]) for k in cache}

    Xtr = np.vstack([feats[k][0] for k in train])
    ytr = np.concatenate([feats[k][1] for k in train])
    m = RandomForestRegressor(n_estimators=120, max_depth=12, min_samples_leaf=5,
                              n_jobs=-1, random_state=0)
    m.fit(Xtr, ytr)
    print(f"訓練 {len(train)} 顆(PHM+XJTU)、測 {len(ims)} 顆(IMS，完全獨立)\n")

    # 匯總 IMS 全部樣本點
    preds, truths = [], []
    for k in ims:
        X, _, rul = feats[k]
        p = m.predict(X)
        mask = rul <= MAX_RUL  # 只看可行動窗（真實 RUL ≤ 250 分，健康平坦段不評）
        preds.append(p[mask])
        truths.append(rul[mask])
    pred = np.concatenate(preds)
    truth = np.concatenate(truths)

    print("真實RUL區間 | 樣本 | 相對誤差中位 | 誤差絕對值中位 | 預測偏差 | ±1.5x命中率 | 方向")
    print("-" * 92)
    for lo, hi in BANDS:
        sel = (truth >= lo) & (truth < hi)
        if sel.sum() == 0:
            continue
        t, p = truth[sel], pred[sel]
        rel = np.abs(p - t) / np.maximum(t, 1e-9)
        abs_err = np.abs(p - t)
        bias = np.median(p - t)  # >0 高估(危險)、<0 低估(保守)
        within = np.mean((p >= t / 1.5) & (p <= t * 1.5))
        direction = "高估(冒險)" if bias > 5 else ("低估(保守)" if bias < -5 else "大致中性")
        print(f"{lo:3d}~{hi:<3d}分  | {sel.sum():5d} | {np.median(rel):10.0%} | "
              f"{np.median(abs_err):9.0f}分 | {bias:+7.0f}分 | {within:9.0%} | {direction}")

    # 「保守可信門檻」：找最大的區間上界，使該界以下所有點的相對誤差中位 ≤ 30%
    print("\n=== 逐步收斂：真實RUL ≤ X 分 時的整體準確度 ===")
    for x in [30, 60, 90, 120, 180, 250]:
        sel = truth <= x
        if sel.sum() == 0:
            continue
        rel = np.abs(pred[sel] - truth[sel]) / np.maximum(truth[sel], 1e-9)
        within = np.mean((pred[sel] >= truth[sel] / 1.5) & (pred[sel] <= truth[sel] * 1.5))
        print(f"  RUL ≤ {x:3d}分：相對誤差中位 {np.median(rel):.0%}、±1.5x命中率 {within:.0%}（{sel.sum()} 點）")


if __name__ == "__main__":
    main()
