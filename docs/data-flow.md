# 資料庫寫入與刪除模式

> 本文件整理 IIoT 系統目前資料庫（TimescaleDB）的寫入與刪除全貌，作為調參與除錯時的對照。
> 數值來源：`edge_pipeline/constants.py`、`edge_pipeline/main.py`、`edge_pipeline/db_writer.py`、
> `timescaledb/init.sql`（2026-07 現況）。改動這些檔案時請同步更新本文件。

---

## 一、寫入模式

### 1. 主資料流：原始感測資料（佔 99.9% 的量）

四層批次，層層把「跨程序 / 跨磁碟」的動作頻率往下壓：

```
10 台機台各自的模擬程序（2kHz tick）
   │  每 tick 產生 6 筆（電流 3 相 E-A/B/C + 振動 3 軸 V-X/Y/Z）
   ▼
① 本地雙緩衝  FLUSH_TICKS=500（250ms）              ← constants.py
   │  湊滿 500 tick 或 250ms → 整批丟進「該機台專屬 Queue」
   ▼
② 每台一條 multiprocessing.Queue（互不干擾，Lock-Free）
   │
   ▼
③ 4 條平行 DB writer thread（NUM_DB_WRITERS=4，10 台分 4 份）  ← main.py
   │  各自累積 DB_BATCH_SIZE=5000 筆 或 DB_BATCH_SECONDS=0.5 秒 → 觸發一次寫入
   ▼
④ 寫入 TimescaleDB（db_writer.py）
      TRUNCATE 暫存表 → COPY 進暫存表 → INSERT…SELECT 轉型進 sensor_readings
```

**關鍵設定與數字：**

| 項目 | 值 | 說明 |
|---|---|---|
| 總寫入速率 | ~120,010 筆/秒 | 10 台 × 2kHz × 6 通道 + 溫度 10 筆/秒（1Hz） |
| 取樣率 | 2kHz | 原為 4kHz，因搬到 SATA SSD（實測上限 ~19.5萬筆/秒）而降速 |
| 寫入路徑 | COPY→暫存表→INSERT SELECT | 實測 ~120k 筆/秒；逐筆 INSERT / execute_values 只有 ~55k |
| 平行連線 | 4 條（`NUM_DB_WRITERS`，可用環境變數覆寫） | 單連線約 12~14 萬筆/秒天花板，需平行才追得上全速 |
| `synchronous_commit` | **OFF** | 換取速度，接受當機時遺失最後幾筆未落地資料 |
| 暫存表 | session TEMP `_sensor_readings_staging`，每批 TRUNCATE 重用 | 不重複建表 |
| 斷線緩衝上限 | `MAX_BUFFER_SIZE = 50000`（= DB_BATCH_SIZE × 10） | DB 連不上時本地最多堆這麼多，滿了丟棄新資料保護記憶體 |
| 重連冷卻 | `RECONNECT_COOLDOWN = 2.0` 秒 | 斷線時避免瘋狂重試 |

**批次大小的意義**（常見誤解澄清）：
- 放大批次幫的是**磁碟負載（IOPS / fsync 次數）與寫入效率**，讓慢磁碟撐得住。
- 放大批次**不會**減少磁碟空間佔用——存多少由保留政策決定，資料總量不變。
- 目前 5000 筆已調得不錯，再放大邊際效益小，且副作用是延遲上升 + 當機遺失更多緩衝資料。

### 2. 衍生資料流：背景自動寫入（量很小）

| 目標表 | 寫入者 | 頻率 | 觸發方式 |
|---|---|---|---|
| `current_1s_agg` | TimescaleDB 背景 | 每 20 秒 | Continuous Aggregate 政策，從 sensor_readings 算 1 秒均值 |
| `vibration_1s_agg` | TimescaleDB 背景 | 每 20 秒 | 同上，取三軸最大值 |
| `sensor_daily_summary` | TimescaleDB 背景 | 每 1 天 | 每日摘要（長期保存） |
| `rul_predictions` | ai_service（phase3_ai） | 每 15 秒 | RUL 線性迴歸結果 |
| `anomaly_scores` | ai_service（phase3_ai） | 每 30 秒 | One-Class SVM 頻譜異常評分 |
| `system_health` | edge_pipeline | 定期 | CPU / RAM / TPS / 寫入延遲 |

---

## 二、刪除模式

原始資料**完全不用逐筆 DELETE**——全靠「整塊 chunk 丟棄」，幾乎零成本。共三個機制：

| 機制 | 對象 | 觸發 | 動作 |
|---|---|---|---|
| **① 時間保留政策** `policy_retention` | sensor_readings | 每 5 分鐘 | 丟棄超過 **20 分鐘** 的已關閉 chunk（整塊 drop_chunks） |
| **② 容量保險絲** `purge_sensor_readings_when_oversize` | sensor_readings | 每 5 分鐘 | 若 hypertable > **40GB** → 丟棄所有已封存 chunk 重新累計（第二道防線） |
| **③ 暫存表清空** | `_sensor_readings_staging` | 每次批次寫入前 | `TRUNCATE`（不是刪資料，是清空重用） |

**設計重點：**
- **chunk 間隔 10 分鐘**：正在寫入的「當前 chunk」永遠不會被上述政策動到（它的時間範圍上界在未來），
  所以刪除**完全不中斷寫入**。
- 保留 20 分鐘 = 隨時只有約 2~3 個 chunk 存在（1 個當前 + 1~2 個待刪），磁碟穩定在 ~15~25GB。
- **壓縮政策已移除**：資料活不到壓縮（15 分鐘）就被刪了；且慢磁碟上壓縮的讀取 I/O 尖峰會跟串流寫入
  搶頻寬，造成寫入延遲螺旋。若日後改為「長保留」需求，再考慮重開壓縮（並最好搬回 NVMe）。

**不會被刪的資料**（獨立儲存，不受 20 分鐘保留影響）：
- 三個 Continuous Aggregate（`current_1s_agg`、`vibration_1s_agg`、`sensor_daily_summary`）——
  這是原始資料被丟棄後，趨勢仍能長期保存的形式。
- `rul_predictions`、`anomaly_scores`、`system_health`、`sensor_anomalies`——都沒設保留政策，量小，預設永久保留。

---

## 三、一句話總結

- **寫入**：高頻原始資料經「本地緩衝 500 筆 → 專屬佇列 → 4 條平行 writer 各批 5000 筆 → COPY 大量寫入」，
  `synchronous_commit=OFF` 求快；衍生的彙總 / AI 結果由背景低頻寫入。
- **刪除**：原始資料靠「20 分鐘時間保留 + 40GB 容量保險絲」整塊丟棄 chunk，永不逐筆刪、永不中斷寫入；
  有價值的彙總與 AI 結果獨立永久保存。

---

## 四、注意事項（運維踩雷）

- **磁碟實體位置**：Docker 資料碟在 E: 槽（SATA SSD），經 junction 生效。vhdx 只增不減，已用
  `~/.wslconfig` 的 `sparseVhd=true` 自動回收。詳見專案記憶。
- **關機順序**：關 Docker/WSL 前務必 `docker compose stop -t 60`（預設 10 秒太短，timescaledb 來不及
  checkpoint 會被 SIGKILL）。不乾淨關機會觸發崩潰恢復，**且磁碟滿時的崩潰會讓 Postgres WAL 遺失、
  把保留政策（20分）與保險絲（40GB）回滾成舊值**。重開後應查 `timescaledb_information.jobs` 確認
  retention/purge config，不對就重套。
- **這些政策是 live catalog 狀態，不是 init.sql**：`init.sql` 只在資料卷「首次初始化」時執行一次；
  之後的調整都要靠 live SQL（`add_retention_policy` / `alter_job` 等），並確保乾淨關機才會持久化。
