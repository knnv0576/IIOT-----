# -*- coding: utf-8 -*-
"""CUSUM 變化點偵測驗證：解決固定/自適應絕對門檻在 17 顆真實軸承上誤報率離散度
（標準差 22~25%）收斂不了的問題。

前兩次實驗的教訓：
  1. 固定門檻 -1.0：健康期誤報中位數 19.5%，範圍 0~63%（標準差 22.0%）。
  2. 自適應門檻 mean-K·std：離散度完全沒收斂（K=3 時仍是 24.1%），因為快速損壞的
     軸承在「健康期」校準窗口內就已經開始劣化，校準基準本身被污染。
根因都指向同一件事：One-Class SVM 的 decision_function 分數在不同軸承間「絕對值的
意義不一致」——同一個 -1.0 或 mean-K·std 對雜訊底噪不同的軸承，代表完全不同的相對地位。
任何基於絕對分數切一刀的門檻，都補不了這個洞。

CUSUM 換一個問題來問：不看「這一刻的分數夠不夠低」，而看「分數有沒有開始持續、
系統性地往下掉」。做法（單邊 CUSUM，只抓劣化方向）：
    S[0] = 0
    S[t] = max(0, S[t-1] + (mu0 - score[t] - slack))
其中 mu0 是健康期分數的均值，slack（漂移容忍量）通常取健康期標準差的某個倍數，
用來吸收正常的隨機波動，只累積「持續系統性劣化」的部分。S[t] 持續走高 → 觸發警報
（S[t] > h，h 是決策門檻，同樣用健康期統計量校準）。

這跟 RUL 層對電流均值做斜率偵測是同一個哲學：不看瞬時值，看趨勢/累積偏移。

關鍵仍然遵守「不偷看答案」：mu0、std、slack、h 全部只用「校準段」（健康期後段、
沒參與模型訓練）算出來，不用到後面壽命資料。

用法（在 ai_service 容器內）：
    python phm2012_cusum_validation.py /tmp/Learning_set /tmp/Full_Test_Set
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from anomaly_detector import ANOMALY_SVM_GAMMA
from spectral_features import to_vector
from spectral_features import extract_features as _extract

SAMPLE_RATE_HZ = 25600.0
BAND_HALF_WIDTH_HZ = 15.0
SNAPSHOT_INTERVAL_S = 10.0
HEALTHY_FRACTION = 0.15  # 壽命前 15% 視為（大致）健康，同前次實驗
FIT_RATIO = 2.0 / 3.0  # 健康期裡前 2/3 拿來 fit SVM，後 1/3 拿來校準 CUSUM 統計量
SLACK_K = 0.5  # slack = SLACK_K * calib_std，容忍多少正常波動不累積
H_VALUES = (3.0, 5.0, 8.0, 12.0)  # 決策門檻 h = H_MULT * calib_std，掃描比較

CONDITION_FUNDAMENTAL_HZ = {"1": 30.0, "2": 27.5, "3": 25.0}


def fundamental_for(name: str) -> float:
    return CONDITION_FUNDAMENTAL_HZ[name.replace("Bearing", "")[0]]


def _load_one(path: Path) -> np.ndarray:
    with open(path) as fh:
        first = fh.readline()
    delimiter = ";" if ";" in first else ","
    return np.loadtxt(path, delimiter=delimiter, usecols=4)


def load_snapshots(bearing_dir: Path) -> np.ndarray:
    files = sorted(bearing_dir.glob("acc_*.csv"))
    return np.array([_load_one(f) for f in files])


def features_of(snapshots: np.ndarray, fundamental_hz: float) -> np.ndarray:
    return np.array(
        [
            to_vector(_extract(s, sample_rate_hz=SAMPLE_RATE_HZ, fundamental_hz=fundamental_hz, band_half_width_hz=BAND_HALF_WIDTH_HZ))
            for s in snapshots
        ]
    )


def fmt_minutes(n_snapshots: float) -> str:
    m = n_snapshots * SNAPSHOT_INTERVAL_S / 60.0
    return f"{int(m // 60)}h{m % 60:04.1f}m"


def cusum(scores: np.ndarray, mu0: float, slack: float) -> np.ndarray:
    """單邊 CUSUM，只累積「分數比 mu0 低超過 slack」的部分（劣化方向）。"""
    s = np.zeros(len(scores))
    running = 0.0
    for i, sc in enumerate(scores):
        running = max(0.0, running + (mu0 - sc - slack))
        s[i] = running
    return s


def compute_bearing(bearing_dir: Path) -> dict:
    """回傳一顆軸承的完整分析素材：score 序列 + CUSUM 校準統計量。"""
    name = bearing_dir.name
    snapshots = load_snapshots(bearing_dir)
    n = len(snapshots)
    n_healthy = int(n * HEALTHY_FRACTION)
    n_fit = max(10, int(n_healthy * FIT_RATIO))

    feats = features_of(snapshots, fundamental_for(name))
    scaler = StandardScaler().fit(feats[:n_fit])
    model = OneClassSVM(kernel="rbf", nu=0.05, gamma=ANOMALY_SVM_GAMMA).fit(scaler.transform(feats[:n_fit]))
    scores = model.decision_function(scaler.transform(feats))

    calib = scores[n_fit:n_healthy]  # 沒參與 SVM 訓練的健康資料，用來校準 CUSUM
    mu0, sigma0 = float(np.mean(calib)), float(np.std(calib))
    slack = SLACK_K * sigma0

    s = cusum(scores, mu0, slack)
    return {
        "name": name,
        "n": n,
        "n_healthy": n_healthy,
        "scores": scores,
        "mu0": mu0,
        "sigma0": sigma0,
        "slack": slack,
        "cusum": s,
    }


def evaluate_h(b: dict, h_mult: float):
    """給定 h = h_mult * sigma0，回傳 (健康期誤報率, 首次觸發位置或 None)。"""
    n, n_healthy = b["n"], b["n_healthy"]
    h = h_mult * b["sigma0"]
    triggered = b["cusum"] > h

    # 健康期誤報參考區：健康期結束 ~ 壽命 50%
    ref_lo, ref_hi = n_healthy, max(n_healthy + 1, n // 2)
    fp = triggered[ref_lo:ref_hi].mean()

    idx = None
    for i in range(n_healthy, n):
        if triggered[i]:
            idx = i
            break
    return fp, idx, h


def run_h(bearings: list[dict], h_mult: float, label: str):
    rows = []
    for b in bearings:
        fp, idx, h = evaluate_h(b, h_mult)
        rows.append((b["name"], b["n"], fp, idx, h))
    detected = [r for r in rows if r[3] is not None]
    fps = [r[2] for r in rows]
    leads = [(r[1] - r[3]) * SNAPSHOT_INTERVAL_S / 60.0 for r in detected]
    print(f"\n=== {label} ===")
    print(f"  偵測成功率：{len(detected)}/{len(rows)}")
    print(
        f"  健康期誤報率：中位數 {np.median(fps):.1%}、平均 {np.mean(fps):.1%}、"
        f"範圍 {min(fps):.1%}~{max(fps):.1%}、標準差 {np.std(fps):.1%}"
    )
    print(f"  超過 10% 誤報的軸承數：{sum(f > 0.10 for f in fps)}/{len(rows)}")
    if leads:
        print(f"  提前預警時間：中位數 {np.median(leads):.0f} 分（範圍 {min(leads):.0f}~{max(leads):.0f}）")
    return rows


def main() -> None:
    roots = [Path(a) for a in sys.argv[1:]]
    dirs = sorted(
        (d for root in roots for d in root.iterdir() if d.is_dir() and d.name.startswith("Bearing")),
        key=lambda p: p.name,
    )
    print(f"載入 {len(dirs)} 顆軸承，計算 CUSUM 中…（slack = {SLACK_K}·sigma0）")
    bearings = [compute_bearing(d) for d in dirs]

    all_rows = {}
    for h_mult in H_VALUES:
        all_rows[h_mult] = run_h(bearings, h_mult, f"CUSUM 門檻 h = {h_mult:.0f}·sigma0")

    # 挑「偵測率仍 100%、且誤報離散度(std)最小」的 h，印每顆詳細對照
    full_detect_hs = [h for h in H_VALUES if sum(r[3] is not None for r in all_rows[h]) == len(bearings)]
    best_h = min(full_detect_hs, key=lambda h: np.std([r[2] for r in all_rows[h]])) if full_detect_hs else H_VALUES[-1]

    print(f"\n\n每顆軸承詳細（CUSUM h={best_h:.0f}·sigma0）")
    header = f"{'軸承':<12}{'sigma0':>8}{'slack':>8}{'誤報率':>9}{'提前預警':>11}"
    print(header)
    print("-" * 50)
    for b, row in zip(bearings, all_rows[best_h]):
        name, n, fp, idx, h = row
        lead = fmt_minutes(n - idx) if idx is not None else "未觸發"
        print(f"{name:<12}{b['sigma0']:>8.3f}{b['slack']:>8.3f}{fp:>9.1%}{lead:>11}")


if __name__ == "__main__":
    main()
