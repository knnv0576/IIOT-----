#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""匯出「健康基線」的原始波形快照，供日後重訓／重算特徵用。

## 為什麼需要這支工具

sensor_readings 的保留政策是 **20 分鐘**（2kHz × 70 感測器 ≈ 1 GB/分鐘，一天就 1.4 TB，
不可能長期保存）。也就是說：

    **你必須在「機台當下確認健康」時主動存下來，事後無法回溯。**

三種保存層級，這支工具負責最後一種：

| 層級 | 大小 | 能做什麼 | 工具 |
|---|---|---|---|
| 不保存 | 0 | 健康時直接重訓 | `trainer.py --all --force` |
| 特徵矩陣 | ~數百 KB | 完全重現當初的訓練 | `trainer.py --save-features DIR` |
| **原始波形** | ~1 GB/5分鐘 | **日後改了特徵定義也能重算** | **本工具** |

只有原始波形能應付「特徵定義改變」（`SCHEMA_VERSION` 或 `FEATURE_NAMES` 變更）的情況——
那時存好的特徵集會作廢，只能從波形重算。

## 用法

    # 匯出最近 5 分鐘（訓練視窗長度）的全機隊資料
    python tools/export_baseline_snapshot.py --minutes 5 --note "上線 commissioning，全機台目視正常"

    # 只匯出特定機台
    python tools/export_baseline_snapshot.py --minutes 5 --device SCT-05 --device SCT-07

    # 指定時間範圍（UTC）
    python tools/export_baseline_snapshot.py --from "2026-08-25 12:00:00" --to "2026-08-25 12:05:00"

輸出到 validation/data/baseline_snapshots/<時間戳>/，該目錄已在 .gitignore 內（檔案太大不進 repo）。

本工具在**主機端**執行，透過 `docker compose exec timescaledb psql` 取資料，
所以主機不需要安裝 psycopg2。
"""

import argparse
import gzip
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "validation" / "data" / "baseline_snapshots"
# 訓練視窗是 300 秒（anomaly_detector.TRAIN_WINDOW_SECONDS），預設就抓這麼長
DEFAULT_MINUTES = 5.0


def psql(sql: str, *, stream: bool = False):
    """透過 docker compose exec 跑 psql。stream=True 時回傳 Popen 供逐行讀取。"""
    cmd = ["docker", "compose", "exec", "-T", "timescaledb",
           "psql", "-U", "iiot", "-d", "iiot", "--no-align", "--tuples-only", "-c", sql]
    if stream:
        return subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE)
    r = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"psql 失敗：{r.stderr.strip()}")
    return r.stdout.strip()


def resolve_range(args) -> tuple[str, str]:
    """決定匯出的時間範圍，並確認資料真的還在（保留政策可能已經刪掉了）。"""
    if args.from_ts and args.to_ts:
        lo, hi = args.from_ts, args.to_ts
    else:
        row = psql("SELECT max(ts), max(ts) - (%f * interval '1 minute') FROM sensor_readings"
                   % args.minutes)
        if not row or "|" not in row:
            raise RuntimeError("sensor_readings 沒有資料")
        hi, lo = [x.strip() for x in row.split("|")]
    avail = psql("SELECT min(ts), max(ts) FROM sensor_readings")
    print(f"  DB 內可用範圍：{avail}")
    print(f"  預計匯出範圍：{lo}  ~  {hi}")
    return lo, hi


def main() -> int:
    ap = argparse.ArgumentParser(
        description="匯出健康基線的原始波形快照（供日後重訓／重算特徵）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=DEFAULT_MINUTES,
                    help=f"從最新資料往回抓幾分鐘（預設 {DEFAULT_MINUTES}，即訓練視窗長度）")
    ap.add_argument("--from", dest="from_ts", help="起始時間（UTC，例：'2026-08-25 12:00:00'）")
    ap.add_argument("--to", dest="to_ts", help="結束時間（UTC）")
    ap.add_argument("--device", action="append", help="只匯出指定機台（可重複）")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="輸出根目錄")
    ap.add_argument("--note", default="", help="註記這份快照的情境（強烈建議填，例：保養後、目視正常）")
    args = ap.parse_args()

    if bool(args.from_ts) != bool(args.to_ts):
        ap.error("--from 與 --to 必須成對使用")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    out_dir = Path(args.out) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[snapshot] 輸出目錄：{out_dir}")

    lo, hi = resolve_range(args)
    dev_filter = ""
    if args.device:
        ids = ",".join(f"'{d}'" for d in args.device)
        dev_filter = f" AND device_id IN ({ids})"

    # 用 COPY ... TO STDOUT 串流出來直接 gzip，不在記憶體裡堆整份資料
    sql = (f"COPY (SELECT ts, device_id, sensor_id, value FROM sensor_readings "
           f"WHERE ts >= '{lo}' AND ts <= '{hi}'{dev_filter} ORDER BY ts) "
           f"TO STDOUT WITH CSV HEADER")
    csv_path = out_dir / "sensor_readings.csv.gz"
    print("[snapshot] 匯出中（原始波形量大，可能要幾分鐘）...")

    rows = 0
    proc = psql(sql, stream=True)
    with gzip.open(csv_path, "wb") as gz:
        for line in proc.stdout:
            gz.write(line)
            rows += 1
    proc.wait()
    if proc.returncode != 0:
        print("[snapshot] 匯出失敗", file=sys.stderr)
        return 1
    rows = max(0, rows - 1)  # 扣掉 CSV 標頭

    size_mb = csv_path.stat().st_size / 1024 / 1024
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "range_from": lo, "range_to": hi,
        "devices": args.device or "ALL",
        "rows": rows,
        "file_size_mb": round(size_mb, 1),
        "note": args.note,
        # 記下取樣率，日後重算特徵時必須用同一個值，否則 FFT 的頻率軸會錯
        "sample_rate_hz": 2000,
        "purpose": "健康基線原始波形快照，供日後重訓或特徵定義變更後重算特徵",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[snapshot] 完成：{rows:,} 筆、{size_mb:.1f} MB → {csv_path}")
    if not args.note:
        print("[snapshot] 提醒：沒有填 --note。日後回頭看時會不知道這份快照當時的機台狀態，"
              "建議編輯 manifest.json 補上。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
