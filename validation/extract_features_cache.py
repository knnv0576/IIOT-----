# -*- coding: utf-8 -*-
"""把 PHM2012 + XJTU + IMS(含 4th_test) 所有軸承的每快照統計(RMS/峰度/波峰因子)讀一次、
存成 pickle 快取。之後 RF / LSTM 實驗都直接載入快取，不用每次重讀 3 萬多個檔（省 ~35 分鐘/次）。

快取內容：{ name: {"rms":[...], "kurt":[...], "crest":[...], "interval":分鐘, "dataset":標籤} }

用法（容器內，validation 掛 rw）：python extract_features_cache.py <out.pkl> <roots...>
  IMS 資料夾（1st_test/2nd_test/4th_test）會自動辨識並選失效通道。
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as sp_kurtosis

READ_SAMPLES = 8192


def _file_num(p):
    d = "".join(c for c in p.stem if c.isdigit())
    return int(d) if d else 0


def _stats(x):
    r = float(np.sqrt(np.mean(x * x)))
    return r, float(sp_kurtosis(x)), (float(np.max(np.abs(x))) / r if r > 1e-12 else 0.0)


def load_csv_bearing(bd):
    first = min(bd.glob("*.csv"), key=_file_num)
    if "vibration" in open(first).readline().lower():
        col, hh, interval, tag = 0, True, 1.0, "XJTU"
    else:
        col, hh, interval, tag = 4, False, 10.0 / 60.0, "PHM2012"
    rms, kurt, crest = [], [], []
    for f in sorted(bd.glob("*.csv"), key=_file_num):
        delim = ";" if ";" in open(f).readline() else ","
        x = pd.read_csv(f, usecols=[col], nrows=READ_SAMPLES, header=(0 if hh else None),
                        delimiter=delim).iloc[:, 0].to_numpy(float)
        r, k, c = _stats(x)
        rms.append(r); kurt.append(k); crest.append(c)
    return {"rms": rms, "kurt": kurt, "crest": crest, "interval": interval, "dataset": tag}


def load_ims_set(sd):
    files = sorted([f for f in sd.rglob("*") if f.is_file()], key=lambda p: p.name)
    ncols = pd.read_csv(files[0], header=None, sep=r"\s+", nrows=5).shape[1]
    cols = [{"rms": [], "kurt": [], "crest": []} for _ in range(ncols)]
    for f in files:
        df = pd.read_csv(f, header=None, sep=r"\s+", nrows=READ_SAMPLES)
        for c in range(ncols):
            r, k, cr = _stats(df.iloc[:, c].to_numpy(float))
            cols[c]["rms"].append(r); cols[c]["kurt"].append(k); cols[c]["crest"].append(cr)
    n = len(files); nb = max(1, int(0.15 * n))

    def ratio(rms):
        rms = np.array(rms); base = float(np.mean(rms[:nb]))
        return float(np.median(rms[-5:]) / base) if base > 1e-12 else 0.0
    fail = max(range(ncols), key=lambda c: ratio(cols[c]["rms"]))
    pc = cols[fail]
    print(f"  IMS {sd.name}: {ncols}ch, 失效通道 col{fail}（{ratio(pc['rms']):.1f}×）", flush=True)
    return {"rms": pc["rms"], "kurt": pc["kurt"], "crest": pc["crest"], "interval": 10.0, "dataset": "IMS"}


def main():
    out = Path(sys.argv[1])
    roots = [Path(a) for a in sys.argv[2:]]
    cache = {}
    for root in roots:
        name = root.name
        if name in ("1st_test", "2nd_test", "3rd_test", "4th_test"):   # IMS test set
            cache[f"IMS/{name}"] = load_ims_set(root)
        else:  # 含 Bearing* 子夾的 CSV root
            for d in sorted((x for x in root.iterdir() if x.is_dir() and x.name.startswith("Bearing")),
                            key=lambda p: p.name):
                key = f"{d.parent.name}/{d.name}"
                cache[key] = load_csv_bearing(d)
                print(f"  {key}: {len(cache[key]['rms'])} 快照", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump(cache, fh)
    print(f"\n快取完成：{len(cache)} 顆軸承 → {out}")


if __name__ == "__main__":
    main()
