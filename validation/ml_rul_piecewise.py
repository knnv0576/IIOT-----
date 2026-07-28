# -*- coding: utf-8 -*-
"""ML RUL 升級版：piecewise 目標 + PHM2012+XJTU-SY 合訓（32 顆軸承）。

相對第一版（xjtu_sy_ml_rul.py）的兩個改進：
1. **Piecewise-linear RUL 目標**：訓練時把 RUL 上限截斷在 MAX_RUL——遠離損壞的健康期一律當
   「MAX_RUL（還很久）」，只在最後 MAX_RUL 分鐘的劣化段學真正的倒數。這是 PHM 論文治「末端高估」
   的標準招：模型輸出被框在 ≤MAX_RUL，不會再對快壞的軸承亂噴大數字。
2. **PHM2012（17）+ XJTU-SY（15）合訓 = 32 顆**，資料翻倍、泛化更好。兩資料集格式不同（欄位、
   有無 header、快照間隔 10s vs 60s），腳本自動辨識；特徵全部尺度/時間無關，斜率換算成 per-min 才可跨集。

嚴謹性：留一顆軸承交叉驗證（LOBO），測試軸承訓練時完全沒看過（跨資料集也是）。

用法（容器內）：python ml_rul_piecewise.py <root1> <root2> ...   # root = 含 Bearing* 子夾的資料夾
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
SLOPE_WINDOW_MIN = 10.0      # 斜率用最近這麼多「分鐘」的資料（跨資料集用時間而非快照數才一致）
MAX_RUL_MIN = 250.0         # piecewise 截斷：超過這麼多分鐘的 RUL 一律當 MAX_RUL
EVAL_FRACTIONS = (0.5, 0.7, 0.9)


def _file_num(p: Path) -> int:
    """檔名裡的序號：XJTU 是 '1.csv'、PHM2012 是 'acc_00001.csv'，都抽出數字部分。"""
    digits = "".join(ch for ch in p.stem if ch.isdigit())
    return int(digits) if digits else 0


def detect_format(bearing_dir: Path):
    """回傳 (col_index, has_header, delimiter, interval_min)。
    XJTU-SY：有 header('vibration')、2 欄、col0、每分鐘一檔。
    PHM2012：無 header、多欄、col4(水平加速度)、每 10 秒一檔（少數用 ';'）。"""
    first_csv = min(bearing_dir.glob("*.csv"), key=_file_num)
    line = open(first_csv).readline()
    delim = ";" if ";" in line else ","
    if "vibration" in line.lower():          # XJTU header
        return 0, True, delim, 1.0
    return 4, False, delim, 10.0 / 60.0       # PHM2012


def _read_col(path: Path, col: int, has_header: bool) -> np.ndarray:
    """讀單一 CSV 的指定欄；分隔符逐檔偵測（PHM2012 有些檔用 ';'）。"""
    line = open(path).readline()
    delim = ";" if ";" in line else ","
    return pd.read_csv(path, usecols=[col], nrows=READ_SAMPLES, header=(0 if has_header else None),
                       delimiter=delim).iloc[:, 0].to_numpy(float)


def bearing_features(bearing_dir: Path):
    col, has_header, _, interval = detect_format(bearing_dir)
    files = sorted(bearing_dir.glob("*.csv"), key=_file_num)
    rms, kurt, crest = [], [], []
    for f in files:
        x = _read_col(f, col, has_header)
        r = float(np.sqrt(np.mean(x * x)))
        rms.append(r)
        kurt.append(float(sp_kurtosis(x)))
        crest.append(float(np.max(np.abs(x))) / r if r > 1e-12 else 0.0)
    rms = pd.Series(rms).rolling(SMOOTH_WIN, min_periods=1, center=True).median().to_numpy()
    kurt, crest = np.array(kurt), np.array(crest)
    n = len(rms)

    baseline = float(np.mean(rms[: max(1, int(HEALTHY_FRACTION * n))]))
    rms_norm = rms / (baseline if baseline > 1e-12 else 1.0)
    w = max(5, int(SLOPE_WINDOW_MIN / interval))       # 斜率視窗（快照數，換算自時間）
    rms_slope = np.zeros(n)
    for i in range(n):
        seg = rms_norm[max(0, i - w + 1) : i + 1]
        if len(seg) >= 3:
            rms_slope[i] = np.polyfit(np.arange(len(seg)), seg, 1)[0] / interval  # per-min，跨集一致

    X = np.column_stack([rms_norm, kurt, crest, rms_slope])
    rul = (n - np.arange(n)) * interval                # 真實 RUL（分鐘）
    y = np.minimum(rul, MAX_RUL_MIN)                    # piecewise 截斷目標
    return X, y, rul, n, interval


def main():
    roots = [Path(a) for a in sys.argv[1:]]
    dirs = sorted((d for root in roots for d in root.iterdir() if d.is_dir() and d.name.startswith("Bearing")),
                  key=lambda p: (str(p.parent.name), p.name))
    print(f"載入 {len(dirs)} 顆軸承特徵（PHM2012+XJTU 合訓，piecewise 截斷 {MAX_RUL_MIN:.0f}min）…")
    data = {}
    for d in dirs:
        key = f"{d.parent.name}/{d.name}"
        data[key] = bearing_features(d)
    names = list(data.keys())
    print(f"共 {len(names)} 顆，開始 LOBO 交叉驗證\n")

    # 三個指標：(a) 可行動窗(真實RUL≤MAX)內誤差 (b) 90% 處誤差 (c) 50/70% 參考
    action_err, frac_err = [], {f: [] for f in EVAL_FRACTIONS}
    for test in names:
        X_tr = np.vstack([data[n][0] for n in names if n != test])
        y_tr = np.concatenate([data[n][1] for n in names if n != test])
        m = RandomForestRegressor(n_estimators=120, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=0)
        m.fit(X_tr, y_tr)
        X_te, y_te, rul_te, n_te, _ = data[test]
        pred = m.predict(X_te)
        # (a) 可行動窗：真實 RUL ≤ MAX 的快照
        mask = rul_te <= MAX_RUL_MIN
        if mask.sum() > 0:
            action_err.extend((np.abs(pred[mask] - rul_te[mask]) / np.maximum(rul_te[mask], 1e-9)).tolist())
        # (b/c) 各壽命階段
        for f in EVAL_FRACTIONS:
            i = int(f * n_te)
            a = float(rul_te[i])
            frac_err[f].append((abs(float(pred[i]) - a) / a, float(pred[i]), a))

    print("=== ML RUL（piecewise + 合訓）結果 ===\n")
    print("【核心指標】可行動窗內（距損壞 ≤250 分鐘的所有時刻）的 RUL 相對誤差：")
    ae = np.array(action_err)
    print(f"  中位數 {np.median(ae):.0%}、平均 {np.mean(ae):.0%}、25~75 分位 {np.percentile(ae,25):.0%}~{np.percentile(ae,75):.0%}"
          f"（樣本 {len(ae)} 個時刻）\n")
    print("【各壽命階段】相對誤差中位數（對照：第一版ML / 線性）：")
    ref = {0.5: "87% / 610%", 0.7: "78% / 327%", 0.9: "235% / 164%"}
    for f in EVAL_FRACTIONS:
        errs = [e for e, _, _ in frac_err[f]]
        print(f"  {int(f*100)}% 處：中位數 {np.median(errs):.0%}   （對照 {ref[f]}）")


if __name__ == "__main__":
    main()
