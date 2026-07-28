# -*- coding: utf-8 -*-
"""泛化測試：ML RUL 在 PHM2012+XJTU（32 顆）訓練，在【完全沒看過的第三個實驗台 IMS】測 RUL。

這是最有說服力的驗證——如果模型在兩個實驗台訓練、卻能在第三個獨立實驗台（不同軸承型號
Rexnord ZA-2115、不同轉速、20kHz、10 分鐘間隔）也給出合理 RUL，就證明它學到的是「軸承劣化
的通則」而非某個實驗台的特性。IMS 完全不參與訓練。

方法沿用 ml_rul_piecewise.py：特徵=rms_norm/kurtosis/crest/rms_slope(per-min)、piecewise 目標、
RandomForest。差別只在 IMS 當 held-out 測試集。

IMS 資料格式：tab 分隔 ASCII、每檔 20480 點(1秒)、每 10 分鐘一檔、檔名是時間戳(YYYY.MM.DD.HH.MM.SS)。
各組「會壞的那顆軸承」的通道（依標準 IMS 文件）：
  1st_test：8 通道(每軸承 2 ch)，軸承3內圈壞 → col 4
  2nd_test：4 通道(每軸承 1 ch)，軸承1外圈壞 → col 0
  3rd_test：4 通道，軸承3外圈壞 → col 2

用法（容器內）：
  python ims_generalization_test.py --train <PHM/XJTU roots...> --ims <IMS test-set 資料夾...>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as sp_kurtosis
from sklearn.ensemble import RandomForestRegressor

READ_SAMPLES = 8192
HEALTHY_FRACTION = 0.15
SMOOTH_WIN = 5
SLOPE_WINDOW_MIN = 10.0
MAX_RUL_MIN = 250.0
EVAL_FRACTIONS = (0.5, 0.7, 0.9)

IMS_INTERVAL_MIN = 10.0  # IMS 每 10 分鐘一檔


def _file_num(p: Path) -> int:
    digits = "".join(ch for ch in p.stem if ch.isdigit())
    return int(digits) if digits else 0


def _features_from_rms(rms, kurt, crest, interval):
    """把每快照的 RMS/峰度/波峰因子 + 間隔，組成尺度/時間無關的特徵矩陣。"""
    rms = pd.Series(rms).rolling(SMOOTH_WIN, min_periods=1, center=True).median().to_numpy()
    kurt, crest = np.array(kurt), np.array(crest)
    n = len(rms)
    baseline = float(np.mean(rms[: max(1, int(HEALTHY_FRACTION * n))]))
    rms_norm = rms / (baseline if baseline > 1e-12 else 1.0)
    w = max(5, int(SLOPE_WINDOW_MIN / interval))
    slope = np.zeros(n)
    for i in range(n):
        seg = rms_norm[max(0, i - w + 1) : i + 1]
        if len(seg) >= 3:
            slope[i] = np.polyfit(np.arange(len(seg)), seg, 1)[0] / interval
    X = np.column_stack([rms_norm, kurt, crest, slope])
    rul = (n - np.arange(n)) * interval
    return X, np.minimum(rul, MAX_RUL_MIN), rul, n


def _snap_stats(x):
    r = float(np.sqrt(np.mean(x * x)))
    return r, float(sp_kurtosis(x)), (float(np.max(np.abs(x))) / r if r > 1e-12 else 0.0)


def load_csv_bearing(bearing_dir: Path):
    """PHM2012 / XJTU：自動辨識欄位/header/分隔符/間隔。"""
    first = min(bearing_dir.glob("*.csv"), key=_file_num)
    line = open(first).readline()
    if "vibration" in line.lower():
        col, has_header, interval = 0, True, 1.0            # XJTU
    else:
        col, has_header, interval = 4, False, 10.0 / 60.0   # PHM2012
    rms, kurt, crest = [], [], []
    for f in sorted(bearing_dir.glob("*.csv"), key=_file_num):
        delim = ";" if ";" in open(f).readline() else ","
        x = pd.read_csv(f, usecols=[col], nrows=READ_SAMPLES, header=(0 if has_header else None),
                        delimiter=delim).iloc[:, 0].to_numpy(float)
        r, k, c = _snap_stats(x)
        rms.append(r); kurt.append(k); crest.append(c)
    return _features_from_rms(rms, kurt, crest, interval)


def load_ims_bearing(set_dir: Path):
    """IMS：tab 分隔 ASCII、無 header、時間戳檔名。同軸多顆軸承共享一組檔案，
    自動選「失效通道」= 末段 RMS/基線 RMS 比值最大的那欄（就是真正壞掉、RMS 暴衝的軸承）。"""
    files = sorted([f for f in set_dir.rglob("*") if f.is_file()], key=lambda p: p.name)
    ncols = pd.read_csv(files[0], header=None, sep=r"\s+", nrows=5).shape[1]
    cols = [{"rms": [], "kurt": [], "crest": []} for _ in range(ncols)]
    for f in files:
        df = pd.read_csv(f, header=None, sep=r"\s+", nrows=READ_SAMPLES)
        for c in range(ncols):
            r, k, cr = _snap_stats(df.iloc[:, c].to_numpy(float))
            cols[c]["rms"].append(r); cols[c]["kurt"].append(k); cols[c]["crest"].append(cr)
    n = len(files)
    nb = max(1, int(HEALTHY_FRACTION * n))

    def deg_ratio(rms):
        rms = np.array(rms); base = float(np.mean(rms[:nb]))
        return float(np.median(rms[-5:]) / base) if base > 1e-12 else 0.0

    fail = max(range(ncols), key=lambda c: deg_ratio(cols[c]["rms"]))
    print(f"    {set_dir.parent.name}/{set_dir.name}：{ncols} 通道，自動選失效通道 col{fail}"
          f"（末段劣化 {deg_ratio(cols[fail]['rms']):.1f}×）")
    pc = cols[fail]
    return _features_from_rms(pc["rms"], pc["kurt"], pc["crest"], IMS_INTERVAL_MIN)


def evaluate(pred, rul, n, label):
    mask = rul <= MAX_RUL_MIN
    ae = np.abs(pred[mask] - rul[mask]) / np.maximum(rul[mask], 1e-9)
    print(f"\n【{label}】壽命 {int(n*IMS_INTERVAL_MIN/60)}h（{n} 快照）")
    print(f"  可行動窗(距損壞≤{int(MAX_RUL_MIN)}min)RUL 相對誤差：中位數 {np.median(ae):.0%}、"
          f"25~75分位 {np.percentile(ae,25):.0%}~{np.percentile(ae,75):.0%}")
    for f in EVAL_FRACTIONS:
        i = int(f * n)
        print(f"    {int(f*100)}% 處：預測 {pred[i]:.0f} / 實際 {rul[i]:.0f} min")


def main():
    args = sys.argv[1:]
    ti = args.index("--train"); ii = args.index("--ims")
    train_roots = [Path(a) for a in args[ti + 1 : ii]]
    ims_dirs = [Path(a) for a in args[ii + 1 :]]

    tdirs = sorted((d for r in train_roots for d in r.iterdir() if d.is_dir() and d.name.startswith("Bearing")),
                   key=lambda p: (p.parent.name, p.name))
    print(f"訓練集：{len(tdirs)} 顆軸承（PHM2012+XJTU）；載入特徵中…")
    loaded = [load_csv_bearing(d) for d in tdirs]   # 每顆只讀一次
    Xtr = np.vstack([X for X, _, _, _ in loaded])
    ytr = np.concatenate([y for _, y, _, _ in loaded])

    model = RandomForestRegressor(n_estimators=120, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=0)
    model.fit(Xtr, ytr)
    print(f"模型訓練完成（{len(ytr)} 樣本）。開始在【完全沒看過的 IMS】測試：")

    for sd in ims_dirs:
        X, _, rul, n = load_ims_bearing(sd)
        pred = model.predict(X)
        evaluate(pred, rul, n, f"IMS {sd.name}")


if __name__ == "__main__":
    main()
