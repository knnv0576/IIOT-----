# -*- coding: utf-8 -*-
"""用 IEEE PHM 2012（FEMTO）真實軸承 run-to-failure 資料，重放驗證 Phase3 的頻譜異常偵測管線。

為什麼需要這個驗證：專案的模擬器不會真的劣化，先前只能用「人工疊加諧波」驗證偵測邏輯
會不會觸發；這裡改用真實軸承從健康一路運轉到損壞的振動記錄，回答人工注入回答不了的
問題——真實劣化的早期頻譜異變，能比實際損壞提早多久被抓到（提前預警時間）。

資料集：25.6kHz 取樣，每 10 秒記錄一個 0.1 秒快照（2560 筆），實驗在振動超過 20g 時
停機（視為損壞）。三種工況轉速不同、轉頻（基頻）也不同，本腳本依軸承編號自動套用：
  Bearing1_x：1800rpm → 30.0Hz、Bearing2_x：1650rpm → 27.5Hz、Bearing3_x：1500rpm → 25.0Hz

管線與正式服務（anomaly_detector.py）完全一致：同一套 extract_features 六維特徵
（只把取樣率/基頻參數換成本資料集的規格）、同樣的 StandardScaler +
OneClassSVM(nu=0.05, gamma=0.02)、同樣的 -1.0 分數邊際。差異只有訓練集來源：
正式服務用「最近 300 秒」，這裡用「壽命最前段（確定健康）的快照」。

用法（在 ai_service 容器內，PYTHONPATH 指向 phase3_ai）：
    python phm2012_replay_validation.py <bearing_dir>              # 單顆，印完整走勢
    python phm2012_replay_validation.py --summary <root_dir> ...   # 掃描底下所有 Bearing*，印彙總表
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from anomaly_detector import ANOMALY_SCORE_MARGIN, ANOMALY_SVM_GAMMA
from spectral_features import to_vector
from spectral_features import extract_features as _extract

SAMPLE_RATE_HZ = 25600.0
BAND_HALF_WIDTH_HZ = 15.0  # 快照只有 0.1 秒，頻率解析度 10Hz，頻帶要比電流的 ±3Hz 寬
SNAPSHOT_INTERVAL_S = 10.0  # 快照間隔
CONSECUTIVE_ALARMS = 3  # 連續 3 個快照（30 秒）都超過邊際才算「持續異常」，抑制單發毛刺
TRAIN_FRACTION = 0.15  # 用壽命最前段當「確定健康」的訓練集

# 依軸承編號（工況）對應的轉頻。名稱形如 Bearing1_1 / Bearing2_3 / Bearing3_2。
CONDITION_FUNDAMENTAL_HZ = {"1": 30.0, "2": 27.5, "3": 25.0}


def fundamental_for(bearing_name: str) -> float:
    condition = bearing_name.replace("Bearing", "")[0]
    return CONDITION_FUNDAMENTAL_HZ[condition]


def _load_one(path: Path) -> np.ndarray:
    # PHM2012 資料集內分隔符不一致：多數軸承用逗號，少數（如 Bearing1_4）用分號，
    # 看第一行有沒有分號決定。第 5 欄（index 4）是水平方向加速度，劣化特徵最明顯的通道。
    with open(path) as fh:
        first = fh.readline()
    delimiter = ";" if ";" in first else ","
    return np.loadtxt(path, delimiter=delimiter, usecols=4)


def load_snapshots(bearing_dir: Path) -> np.ndarray:
    files = sorted(bearing_dir.glob("acc_*.csv"))
    if not files:
        raise SystemExit(f"{bearing_dir} 裡沒有 acc_*.csv")
    return np.array([_load_one(f) for f in files])


def features_of(snapshots: np.ndarray, fundamental_hz: float) -> np.ndarray:
    return np.array(
        [
            to_vector(
                _extract(
                    s,
                    sample_rate_hz=SAMPLE_RATE_HZ,
                    fundamental_hz=fundamental_hz,
                    band_half_width_hz=BAND_HALF_WIDTH_HZ,
                )
            )
            for s in snapshots
        ]
    )


def fmt_minutes(n_snapshots: float) -> str:
    minutes = n_snapshots * SNAPSHOT_INTERVAL_S / 60.0
    return f"{int(minutes // 60)}h{minutes % 60:04.1f}m"


def evaluate(bearing_dir: Path) -> dict:
    """跑完一顆軸承，回傳關鍵指標（供彙總表與單顆詳細輸出共用）。"""
    name = bearing_dir.name
    fundamental_hz = fundamental_for(name)
    snapshots = load_snapshots(bearing_dir)
    n = len(snapshots)
    n_train = int(n * TRAIN_FRACTION)

    feats = features_of(snapshots, fundamental_hz)
    scaler = StandardScaler().fit(feats[:n_train])
    model = OneClassSVM(kernel="rbf", nu=0.05, gamma=ANOMALY_SVM_GAMMA).fit(scaler.transform(feats[:n_train]))
    scores = model.decision_function(scaler.transform(feats))
    below = scores < ANOMALY_SCORE_MARGIN

    detection_idx = None
    for i in range(n_train, n - CONSECUTIVE_ALARMS + 1):
        if below[i : i + CONSECUTIVE_ALARMS].all():
            detection_idx = i
            break

    # 健康期誤報參考區：訓練期結束 ~ 壽命 50%（此段一般仍視為健康）
    healthy_fp = below[n_train : max(n_train + 1, n // 2)].mean()

    return {
        "name": name,
        "fundamental_hz": fundamental_hz,
        "n": n,
        "n_train": n_train,
        "scores": scores,
        "below": below,
        "detection_idx": detection_idx,
        "healthy_fp": healthy_fp,
    }


def print_detail(r: dict) -> None:
    n, scores, below = r["n"], r["scores"], r["below"]
    print(
        f"{r['name']}（工況轉頻 {r['fundamental_hz']}Hz）: {n} 個快照（壽命 {fmt_minutes(n)}），"
        f"前 {r['n_train']} 個（{fmt_minutes(r['n_train'])}）當健康訓練集"
    )
    print("\n壽命分段走勢（每段：分數中位數 / 超過邊際比例）")
    for d in range(10):
        lo, hi = n * d // 10, n * (d + 1) // 10
        seg_scores, seg_below = scores[lo:hi], below[lo:hi]
        bar = "#" * int(seg_below.mean() * 20)
        print(f"  {d*10:3d}~{(d+1)*10:3d}%  median={np.median(seg_scores):8.3f}  異常率={seg_below.mean():6.1%}  {bar}")

    print(f"\n訓練期後~壽命 50% 的異常率（越低代表健康期越乾淨）: {r['healthy_fp']:.1%}")
    if r["detection_idx"] is None:
        print("整段壽命都沒有觸發持續異常警報（偵測失敗）")
        return
    idx = r["detection_idx"]
    lead = n - idx
    print(f"首次持續異常警報：第 {idx} 個快照（壽命 {idx / n:.1%} 處，開機後 {fmt_minutes(idx)}）")
    print(f"提前預警時間：損壞前 {fmt_minutes(lead)}（{lead} 個快照）")
    print(f"警報當下分數 {scores[idx]:.3f}，損壞前最後一段中位數 {np.median(scores[-30:]):.3f}")


def print_summary(roots: list[Path]) -> None:
    bearing_dirs = sorted(
        (d for root in roots for d in root.iterdir() if d.is_dir() and d.name.startswith("Bearing")),
        key=lambda p: p.name,
    )
    print(f"掃描到 {len(bearing_dirs)} 顆軸承\n")
    header = f"{'軸承':<12}{'壽命':>9}{'健康期誤報':>10}{'警報時機':>10}{'提前預警':>11}{'末段分數':>10}"
    print(header)
    print("-" * len(header.encode("ascii", "replace")))

    rows = []
    for d in bearing_dirs:
        r = evaluate(d)
        rows.append(r)
        n = r["n"]
        if r["detection_idx"] is None:
            timing, lead = "未觸發", "—"
        else:
            timing = f"{r['detection_idx'] / n:.0%}處"
            lead = fmt_minutes(n - r["detection_idx"])
        end_med = np.median(r["scores"][-30:])
        print(
            f"{r['name']:<12}{fmt_minutes(n):>9}{r['healthy_fp']:>10.1%}"
            f"{timing:>10}{lead:>11}{end_med:>10.2f}"
        )

    # 統整
    detected = [r for r in rows if r["detection_idx"] is not None]
    print(f"\n偵測成功率：{len(detected)}/{len(rows)} 顆軸承在壽命結束前觸發持續警報")
    if detected:
        leads_min = [(r["n"] - r["detection_idx"]) * SNAPSHOT_INTERVAL_S / 60.0 for r in detected]
        timings = [r["detection_idx"] / r["n"] for r in detected]
        fps = [r["healthy_fp"] for r in rows]
        print(f"提前預警時間：中位數 {np.median(leads_min):.0f} 分鐘（範圍 {min(leads_min):.0f}~{max(leads_min):.0f}）")
        print(f"警報時機：中位數落在壽命 {np.median(timings):.0%} 處")
        print(f"健康期誤報率：中位數 {np.median(fps):.1%}（範圍 {min(fps):.1%}~{max(fps):.1%}）")


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--summary":
        print_summary([Path(a) for a in args[1:]])
    else:
        print_detail(evaluate(Path(args[0])))


if __name__ == "__main__":
    main()
