# -*- coding: utf-8 -*-
"""RUL（剩餘壽命）預測「準確度」驗證：用 XJTU-SY 真實軸承的已知壽命，量我們的 RUL 方法
（線性趨勢外推到損壞門檻）預測出來的「還剩多久」跟「實際還剩多久」差多少。

跟前面的「偵測命中率」驗證不同——那個問「有沒有抓到劣化」，這個問「預測的時間數字準不準」。

方法（就是正式服務 RUL 的核心：對健康指標做滑動視窗線性回歸、外推撞門檻）：
  - 健康指標 HI = 每個快照的振動 RMS（水平方向），用滾動中位數平滑去噪。
  - 損壞目標 = 實際最後幾筆快照的 HI（資料驅動、自洽——就是這顆軸承真正壞掉時的 HI 水平）。
  - 在壽命各階段 i：對最近一段 HI 做線性回歸取斜率，外推「還要多久 HI 會到損壞目標」= 預測 RUL。
  - 真實 RUL = (總快照數 - i) × 每快照間隔。
  - 誤差 = |預測 - 真實| / 真實。

用法（容器內）：python xjtu_sy_rul_accuracy.py <35Hz路徑> <37.5Hz路徑> <40Hz路徑>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

READ_SAMPLES = 8192            # 每檔讀前 8192 點算 RMS（夠代表、加速 I/O）
SNAPSHOT_INTERVAL_MIN = 1.0    # XJTU-SY 每分鐘一檔
SMOOTH_WIN = 5                 # HI 滾動中位數平滑視窗
HEALTHY_FRACTION = 0.15
EVAL_FRACTIONS = (0.5, 0.7, 0.9)   # 在壽命這幾個階段評估 RUL 預測
TREND_WIN_FRACTION = 0.15     # 斜率回歸用最近這麼多比例的壽命
MIN_SLOPE = 1e-9


def load_hi(bearing_dir: Path) -> np.ndarray:
    files = sorted(bearing_dir.glob("*.csv"), key=lambda p: int(p.stem))
    hi = []
    for f in files:
        x = pd.read_csv(f, usecols=[0], nrows=READ_SAMPLES).iloc[:, 0].to_numpy(float)
        hi.append(float(np.sqrt(np.mean(x * x))))  # RMS
    return np.array(hi)


def smooth(a: np.ndarray, w: int) -> np.ndarray:
    return pd.Series(a).rolling(w, min_periods=1, center=True).median().to_numpy()


def predict_rul(hi: np.ndarray, i: int, target: float, trend_win: int):
    """在快照 i，用最近 trend_win 個 HI 做線性回歸外推到 target，回傳預測 RUL（快照數）或 None。"""
    lo = max(0, i - trend_win + 1)
    seg = hi[lo : i + 1]
    if len(seg) < 5:
        return None
    t = np.arange(len(seg), dtype=float)
    slope = np.polyfit(t, seg, 1)[0]
    if slope <= MIN_SLOPE or hi[i] >= target:
        return None  # 趨勢還平（無法預測）或已達損壞
    return (target - hi[i]) / slope


def analyse(bearing_dir: Path):
    hi = smooth(load_hi(bearing_dir), SMOOTH_WIN)
    n = len(hi)
    target = float(np.median(hi[-3:]))          # 資料驅動的損壞 HI 水平
    trend_win = max(10, int(TREND_WIN_FRACTION * n))
    rows = {}
    for f in EVAL_FRACTIONS:
        i = int(f * n)
        pred = predict_rul(hi, i, target, trend_win)
        actual = (n - i) * SNAPSHOT_INTERVAL_MIN
        rows[f] = (pred if pred is None else pred * SNAPSHOT_INTERVAL_MIN, actual)
    return bearing_dir.name, n, rows


def main():
    roots = [Path(a) for a in sys.argv[1:]]
    dirs = sorted((d for root in roots for d in root.iterdir() if d.is_dir() and d.name.startswith("Bearing")),
                  key=lambda p: p.name)
    print(f"分析 {len(dirs)} 顆軸承的 RUL 預測準確度…（HI=振動RMS，損壞目標=實際末段 HI）\n")
    hdr = f"{'軸承':<13}{'壽命(min)':>9}"
    for f in EVAL_FRACTIONS:
        hdr += f"{'  '+str(int(f*100))+'%: 預測/實際(min)':>22}"
    print(hdr)
    print("-" * len(hdr.encode("ascii", "replace")))

    results = [analyse(d) for d in dirs]
    rel_err = {f: [] for f in EVAL_FRACTIONS}      # 相對誤差
    predictable = {f: 0 for f in EVAL_FRACTIONS}   # 幾顆在此階段能預測
    for name, n, rows in results:
        line = f"{name:<13}{n:>9}"
        for f in EVAL_FRACTIONS:
            pred, actual = rows[f]
            if pred is None:
                line += f"{'  —/'+str(int(actual)):>22}"
            else:
                line += f"{'  '+str(int(pred))+'/'+str(int(actual)):>22}"
                rel_err[f].append(abs(pred - actual) / actual)
                predictable[f] += 1
        print(line)

    print("\n=== 彙總：各壽命階段的 RUL 預測表現 ===")
    for f in EVAL_FRACTIONS:
        errs = rel_err[f]
        if errs:
            print(f"  壽命 {int(f*100)}% 處：可預測 {predictable[f]}/{len(results)} 顆，"
                  f"相對誤差 中位數 {np.median(errs):.0%}、範圍 {min(errs):.0%}~{max(errs):.0%}")
        else:
            print(f"  壽命 {int(f*100)}% 處：無任一顆可預測（趨勢還太平）")


if __name__ == "__main__":
    main()
