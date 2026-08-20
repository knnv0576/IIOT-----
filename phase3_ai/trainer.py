"""獨立的模型訓練模組：與推論服務分離。

為什麼要分離：
- 訓練是「偶爾做一次、吃 CPU、需要一段乾淨的健康資料」；推論是「一直跑、要即時」。
  綁在一起會讓每次重啟都被迫重訓（等 300 秒 + 吃 CPU），期間完全無法偵測。
- 更關鍵的是**分數尺度**：SVM 重訓後 decision_function 的離散度實測會變 1.5~3 倍，
  而 CUSUM 基線是「機台 × 該次模型」的屬性 → 模型一換，基線全部失效、健康通道大量誤報。
  模型固定下來之後，這個問題從根本消失。

三種使用方式：
1. **自動**（預設）：ai_service 發現某通道沒有模型，且資料量足夠，就自動訓練並存檔。
   之後不論重啟幾次都直接載入，不會再訓練。這也是「接上真實感測器後，資料一進來就自動
   訓練」的路徑——訓練讀的是 sensor_readings，模擬器和 MQTT 真實資料寫進來的是同一張表。
2. **手動全量**：docker compose run --rm ai_service python trainer.py --all
3. **強制重訓某台/某通道**：python trainer.py --device SCT-05 --force

重要前提：訓練資料必須是「機台健康時」的資料。這支程式無法自己判斷健康與否
（那是雞生蛋問題），所以自動訓練只在「該通道還沒有模型」時發生，等同假設
「第一次看到這台機台時它是健康的」。真實產線上線時，應該在確認機台狀態正常後
用 --all --force 做一次正式的 commissioning 訓練。
"""

import argparse
import os
import sys

import psycopg2

import model_store
from anomaly_detector import train_model
from spectral_features import FUNDAMENTAL_HZ

CURRENT_FUNDAMENTAL_HZ = 60.0   # 電流：60Hz 交流電
VIBRATION_FUNDAMENTAL_HZ = 30.0  # 振動：對應 edge_pipeline/formulas.py 的 _VIBRATION_FREQ_HZ

DEVICES = [f"SCT-{i:02d}" for i in range(1, 11)]
PHASES = ("A", "B", "C")
AXES = ("X", "Y", "Z")


def channels_of(device_id: str) -> list[tuple[str, float]]:
    """回傳這台機台要訓練的通道 [(sensor_id, fundamental_hz), ...]（電流三相 + 振動三軸）。"""
    n = int(device_id.split("-")[1])
    return (
        [(f"E{n:02d}-{p}", CURRENT_FUNDAMENTAL_HZ) for p in PHASES]
        + [(f"V{n:02d}-{ax}", VIBRATION_FUNDAMENTAL_HZ) for ax in AXES]
    )


def ensure_model(conn, device_id: str, sensor_id: str, fundamental_hz: float,
                 force: bool = False, quiet: bool = False):
    """取得這個通道可用的模型：優先載入既有的，沒有（或 force）才訓練並存檔。

    回傳 (bundle, fingerprint, source)，資料不足而無法訓練時回傳 None。
    source 是 'loaded' 或 'trained'，讓呼叫端知道有沒有真的重訓過。
    """
    if not force:
        got = model_store.load(device_id, sensor_id)
        if got is not None:
            bundle, meta = got
            return bundle, meta["fingerprint"], "loaded"

    bundle = train_model(conn, device_id, sensor_id, fundamental_hz=fundamental_hz)
    if bundle is None:
        return None  # 歷史資料還不夠（量或覆蓋度不足），呼叫端之後再試
    fingerprint = model_store.save(device_id, sensor_id, bundle)
    if not quiet:
        kind = "電流" if sensor_id.startswith("E") else "振動"
        print(
            f"[trainer] {device_id}/{sensor_id}（{kind} {fundamental_hz:.0f}Hz）訓練完成並存檔"
            f"（{bundle['n_train_windows']} 個訓練窗口，指紋 {fingerprint}）"
        )
    return bundle, fingerprint, "trained"


def train_all(conn, devices=None, force: bool = False) -> dict:
    """對指定機台（預設全部）的每個通道確保有模型。回傳統計。"""
    stats = {"trained": 0, "loaded": 0, "insufficient": 0}
    for device_id in devices or DEVICES:
        for sensor_id, fundamental_hz in channels_of(device_id):
            result = ensure_model(conn, device_id, sensor_id, fundamental_hz, force=force)
            if result is None:
                stats["insufficient"] += 1
                print(f"[trainer] {device_id}/{sensor_id} 資料不足，略過（稍後可再跑一次）")
            else:
                stats[result[2] if result[2] == "loaded" else "trained"] += 1
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase3 AI 頻譜異常模型訓練（與推論服務分離）")
    ap.add_argument("--all", action="store_true", help="對全部 10 台機台訓練")
    ap.add_argument("--device", action="append", help="只訓練指定機台（可重複，例：--device SCT-05）")
    ap.add_argument("--force", action="store_true", help="即使已有模型也重新訓練並覆蓋")
    ap.add_argument("--list", action="store_true", help="列出目前已存的模型")
    args = ap.parse_args()

    if args.list:
        metas = model_store.list_meta()
        if not metas:
            print("目前沒有任何已存模型（第一次跑 ai_service 時會自動訓練）")
            return 0
        print(f"已存模型 {len(metas)} 個：")
        for m in metas:
            print(f"  {m['device_id']}/{m['sensor_id']}  指紋={m['fingerprint']}  "
                  f"訓練於={m['trained_at']}  窗口數={m.get('n_train_windows')}")
        return 0

    if not args.all and not args.device:
        ap.error("請指定 --all 或 --device（或用 --list 查看現況）")

    conn = psycopg2.connect(os.environ["TIMESCALEDB_DSN"])
    conn.autocommit = True
    try:
        stats = train_all(conn, devices=args.device, force=args.force)
    finally:
        conn.close()
    print(f"[trainer] 完成：新訓練 {stats['trained']}、沿用既有 {stats['loaded']}、"
          f"資料不足 {stats['insufficient']}")
    return 1 if stats["insufficient"] and not stats["trained"] else 0


if __name__ == "__main__":
    sys.exit(main())
