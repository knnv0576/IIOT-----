# 模組功能與使用場景指南

本文件說明 IIoT 預防性維護系統各模組的功能、實際能做什麼、怎麼操作，以及對應的使用場景。
適合日後接手、教育訓練或對外報告時作為導覽。

## 一句話總覽

模擬 **10 台機台（SCT-01~10）** 的高頻電流/振動感測 → 邊緣即時處理 →
時序資料庫 → **AI 異常偵測 + 剩餘壽命預測（RUL）** → Grafana 戰情室，
構成完整的四階段預測性維護鏈。

## 整體資料流

```
edge_pipeline（10 進程模擬 + degrade 注入）
   │ 4 條平行寫入（斷線自癒）
   ▼
timescaledb（原始 → 1 秒彙總 → 20 分清除 → 150GB 保險絲）
   │            ▲
   ▼            │ 讀彙總算特徵、寫回分數/RUL
phase3_ai（SVM 異常 + CUSUM 劣化 + 三種 RUL）
   │
   ▼
grafana（戰情室：即時圖 + 警報 + 三種 RUL）

validation（離線）：真實軸承資料 → 訓練 RF → 驗準度/泛化 → 產出 rul_rf.pkl
```

五個 Docker 服務：`edge_pipeline`、`mosquitto`、`timescaledb`、`ai_service`、`grafana`。

---

## 1. `edge_pipeline/` — 邊緣模擬器 + 資料匯流（Phase 1/2/4）

整套系統的「工廠現場」。單一 `main.py` 同時扛三件事：模擬 10 台機台、把資料匯入資料庫、
監控自己的健康。

### 場景 A：動態注入異常（最常用的操作介面）

程式起來後會開一個 **HTTP 控制 API（port 8080）**，不用重啟就能對任一台機台注入故障：

| 操作 | 指令 | 場景 |
|---|---|---|
| 瞬間故障 | `POST /danger/SCT-05` | 模擬突發撞擊/斷裂，振幅瞬間拉高 → 驗證即時警報 |
| **漸進劣化** | `POST /degrade/SCT-05` | 模擬軸承慢慢磨損，諧波邊帶 + 振幅隨時間成長 → 驗證 AI 提前預警 / RUL 倒數 |
| 恢復正常 | `POST /normal/SCT-05` | 解除上述兩種，回健康 |
| 查狀態 | `GET /status` | 一次看 10 台誰在 danger / 誰在 degrading |

前景執行時也能用 stdin：`SCT-05 DANGER` / `SCT-05 NORMAL` / `STATUS` / `QUIT`。
這是做每一次 AI 端到端驗證的入口。

範例：
```bash
curl -X POST http://localhost:8080/degrade/SCT-05   # 開始漸進劣化
curl http://localhost:8080/status                    # 查狀態
curl -X POST http://localhost:8080/normal/SCT-05    # 解除
```

### 場景 B：硬體隔離的多機台壓力測試

10 台各是**獨立 OS 進程**（不是執行緒，繞開 GIL）。每台掛了會單獨警示、不互相拖累
（`process_monitor_loop`，死掉不自動重啟、只警示，讓人工決定）。
適合驗證「單機故障不影響其他 9 台」這類可靠度場景。

### 場景 C：高吞吐 + 斷線自癒的資料匯流

- **4 條平行寫入連線**（`NUM_DB_WRITERS`，可環境變數覆寫），把 10 台輪流分攤，實測約 25.5 萬筆/秒。
- **背壓保護**：資料庫斷線時本地 buffer 到上限（`MAX_BUFFER_SIZE`）就停止抽 Queue，
  靠各機台 Queue 的 maxsize 頂住，不會把記憶體吃爆。
- **斷線自動補寫**：寫入失敗保留未寫資料、遵守冷卻時間重連，資料庫恢復後自動接續補寫。
- 同時吃 **MQTT 進線**（`mqtt_ingest`），可用 `mqtt_test_publisher.py` 灌外部感測器資料進同一條管線。

### 場景 D：自我驗收

`self_test.py` 一鍵驗證 4 項：資料格式、每秒總量、時序連續性（相鄰筆嚴格 0.5ms）、危險模式注入生效。
用於交付前驗收或回歸測試。

### 關鍵檔案

| 檔案 | 職責 |
|---|---|
| `main.py` | 進入點：組裝進程、寫入 worker、HTTP 控制、系統健康監控 |
| `machine_process.py` | 單台機台主迴圈，`perf_counter` 精準計時避免累積誤差 |
| `formulas.py` | 向量化物理公式：電流 `I_base+A·sin(2π·60·t)+雜訊`、三軸振動、溫度；含漸進劣化注入（諧波邊帶 + 振幅成長） |
| `machine_config.py` | 每台機台參數化基準值與門檻（取代寫死 Excel） |
| `double_buffer.py` | 本地雙緩衝，滿 500 tick 或 250ms 出貨 |
| `db_writer.py` | COPY 批次寫入 staging，高吞吐 |
| `anomaly_extractor.py` | 抽頻譜/時域特徵 |
| `records.py` | 把一批 NumPy 矩陣攤平成逐筆 tuple |
| `self_test.py` | 4 項驗收 |

---

## 2. `phase3_ai/` — AI 推論服務（整套的大腦，Phase 3）

單一 `main.py` 主迴圈，每 15 秒跑 RUL、每 30 秒跑異常評分，三種能力疊在一起同時運作。

### 場景 A：早期異常偵測（振幅還正常就抓到）

對**每台 6 通道**（電流三相 60Hz + 振動三軸 30Hz）各訓練一個 **One-Class SVM**，
學「正常長怎樣」，再對最新波形算頻譜 + 時域特徵去評分。
可抓到均值還看不出來、但頻譜結構已開始跑掉的早期劣化——這是門檻規則抓不到的。

參數上的教訓（見 `memory` / 調參結論）：nu=0.05、gamma=0.02（固定值，非 "scale"）、
分數邊際 -1.0，把誤報率從 43.7% 降到可用。

### 場景 B：漸進劣化的提前預警

每通道一個線上 **CUSUM** 追蹤器，抓「異常分數持續系統性下滑」而非「低於某條線」。
跨資料集驗證 **32/32 真實軸承全命中**（PHM2012 17 + XJTU-SY 15）。
場景：在故障發生前數十分鐘就收到「這台開始劣化了」，安排停機保養。

### 場景 C：剩餘壽命 RUL —— 三種視角

| 種類 | 追蹤量 | 適用 | 函式 |
|---|---|---|---|
| 電流均值 RUL | 電流均值撞上下限 [5.5, 6.5]% | 負載漂移類故障 | `predict_rul` |
| 振動頻譜 RUL | thd_ratio 撞 0.05 | 振幅還沒動、頻譜先變 | `predict_spectral_rul` |
| **振動 ML RUL** | RF 模型輸出分鐘數 | 一般軸承磨損（正式服務採用） | `ml_vibration_rul` |

ML RUL 用 `[rms_norm, kurt, crest, slope]` 四特徵餵 **32 顆真實軸承訓練的 RandomForest**
（`models/rul_rf.pkl`），直接吐剩餘分鐘，自動分 STABLE / WARNING（<200 分）/ CRITICAL（<60 分）。
- `rms_norm` = 現值 rms / 健康基線（該通道最早 20 筆 rms 的中位數），是最重要特徵（重要度約 57%）。
- 端到端驗證：健康 → RUL ~220 分 STABLE；劣化推進 rms_norm 1→6 → RUL 一路降到 ~90 分、轉 WARNING。

> 註：模型學的是真實軸承劣化。跑在模擬器上時，RUL 數字反映「這波動有多像真實軸承劣化」，
> 接真實感測器才最準。為了讓模擬器的劣化能觸發振幅型 ML RUL，`formulas.py` 的漸進劣化
> 除了加諧波，也讓整體振幅（RMS）隨劣化成長（`DEGRADE_VIBRATION_AMPLITUDE_GROWTH`），
> 模擬真實軸承磨損末期 RMS 暴增的特性。

### 場景 D：高可用韌性

- 模型載入失敗只停用 ML RUL，不讓整個服務掛掉。
- 資料庫錯誤本輪中止、自動重連（遵守冷卻時間）。
- 每通道模型訓練成功就快取，只重試還沒訓練成功的通道。

### 關鍵檔案

| 檔案 | 職責 |
|---|---|
| `main.py` | **推論主迴圈**（不再負責訓練）+ 三種 RUL + 異常評分排程 |
| `trainer.py` | **獨立訓練模組**：載入優先、缺少才訓練；可單獨用 CLI 執行 |
| `model_store.py` | **模型儲存層**：存/載/指紋/中繼資料，訓練與推論之間唯一的介面 |
| `spectral_features.py` | FFT 頻譜特徵（thd_ratio）+ 時域（RMS、kurtosis、crest）；SAMPLE_RATE_HZ=2000 |
| `anomaly_detector.py` | One-Class SVM 訓練演算法 + CUSUM 追蹤器 |
| `models/rul_rf.pkl` | image 內建的 RandomForest RUL 模型（由 validation/train_save_rf.py 產出，唯讀） |

### 訓練與推論分離（2026-08）

原本 SVM 在 ai_service 啟動時「順便」訓練，模型只活在記憶體。這造成兩個問題：

1. 每次重啟都要等 300 秒累積資料 + 重訓，期間無法偵測。
2. **重訓後 decision_function 的分數尺度會變**（實測離散度差 1.5~3 倍），而 CUSUM 基線是
   「機台 × 該次模型」的屬性 → 模型一換基線就失效 → slack 與門檻一起縮水 → 健康通道大量誤報。

現在改成：

```
sensor_readings（模擬器 + MQTT 真實資料寫進同一張表）
      │
      ▼
trainer.py ──訓練──> model_store（Docker volume，跨容器重建保留）
      │                    │
      │                    ▼
      └──載入優先────> main.py 推論（重啟不重訓）
                           │
                           └─ CUSUM 基線以「模型指紋」為鍵存取，指紋不符就重新暖機
```

**使用方式**

| 情境 | 指令 |
|---|---|
| 自動（預設） | ai_service 發現通道沒有模型且資料足夠 → 自動訓練並存檔 |
| 查看已存模型 | `docker compose exec ai_service python trainer.py --list` |
| 全機隊重訓 | `docker compose run --rm ai_service python trainer.py --all --force` |
| 單台重訓 | `docker compose run --rm ai_service python trainer.py --device SCT-05 --force` |

**接真實感測器時**：訓練讀的是 `sensor_readings`，MQTT 進線寫的也是同一張表，所以真實資料
一進來就會被自動用於訓練，不需要改程式。但**必須在機台確認健康時**做一次
`--all --force` 的正式 commissioning 訓練——程式無法自行判斷資料是否代表健康狀態。

### 重訓用的歷史資料要放哪裡

**關鍵限制**：`sensor_readings` 的保留政策是 **20 分鐘**（2kHz × 70 感測器 ≈ 1 GB/分鐘，
一天就 1.4 TB）。所以**必須在「機台當下確認健康」時主動存下來，事後無法回溯**。

三個保存層級，按需要選：

| 層級 | 大小 | 能做什麼 | 指令 |
|---|---|---|---|
| **不保存** | 0 | 健康時直接重訓（工業實務常用） | `trainer.py --all --force` |
| **特徵矩陣** | 全機隊約 **420 KB** | 完全重現當初的訓練（指紋一致），可進 git 版控 | `trainer.py --all --force --save-features DIR` |
| **原始波形** | 全機隊 5 分鐘約 **420 MB**（壓縮後） | 日後改了特徵定義也能重算 | `python tools/export_baseline_snapshot.py --minutes 5 --note "..."` |

**特徵集重現訓練**（實測：指紋完全一致，且不碰 DB、快 75 倍）：
```bash
# 存
docker compose exec ai_service python trainer.py --all --force --save-features /app/model_store/baseline_20260825
# 事後重現
docker compose exec ai_service python trainer.py --all --force --from-features /app/model_store/baseline_20260825
```
特徵集帶 `manifest.json` 記錄 `schema_version` 與 `feature_names`；若日後特徵定義變更，
載入時會被擋下（不會靜默訓練出錯的模型），此時只能靠原始波形快照重算。

**原始波形快照**輸出到 `validation/data/baseline_snapshots/<時間戳>/`（已在 `.gitignore` 內），
含 `manifest.json` 記錄時間範圍、筆數、取樣率與情境註記。工具在**主機端**執行
（透過 `docker compose exec psql`），主機不需安裝 psycopg2。

**建議做法**：平常什麼都不存，需要重訓時在健康狀態下 `--force`；每次正式 commissioning
額外存一份特徵集（進 git）+ 一份原始快照（離線保存）。

---

## 3. `timescaledb/init.sql` — 時序資料庫（系統的記憶）

不只是存資料，內建自動維運能力：

- **Hypertable + 1 秒連續彙總**：Grafana 查的是彙總表（`current_1s_agg` / `vibration_1s_agg`），
  不是逐筆原始，才能秒級刷新。
- **20 分鐘保留政策**：原始資料自動 `drop_chunks`，不無限膨脹。
- **150GB 容量保險絲**（自訂 job）：即使塞爆也會自動清最舊 chunk，防磁碟撐爆崩潰（踩過 SIGBUS 教訓）。
- 三張輸出表：
  - `sensor_readings`（逐筆原始）
  - `anomaly_scores`（AI 特徵/分數/CUSUM；含 rms/kurt/crest）
  - `rul_predictions`（三種 RUL，靠 `metric_kind` 區分：`current_mean` / `thd_ratio` / `ml_rf`）

場景：長期無人值守運行，不用手動清資料。

---

## 4. `grafana/` — 戰情室（給人看的介面）

單一 dashboard「IIoT 全系統視覺化戰情室」，分區呈現，頂部下拉可切機台（`$device`）：

- **即時圖表**：溫度、電流三相、三軸合成振動 vs ISO 門檻（綠=優良、黃=可接受、紅=危險）。
- **AI 警報**：大色塊即時狀態。
- **系統健康**：CPU% / RAM / TPS / 寫入延遲（證明高負載下平穩）。
- **三種 RUL 區**：
  - 電流均值 RUL 表
  - 振動頻譜 RUL 表 + thd_ratio 趨勢圖 + CUSUM 累積圖
  - 振動 ML RUL 表 + RUL 倒數趨勢圖 + rms_norm 趨勢圖

場景：現場大螢幕，一眼看出哪台紅了、還剩多久。

儀表板定義：`grafana/provisioning/dashboards/iiot-overview.json`（provisioning 自動載入）。

---

## 5. `validation/` — 離線驗證與訓練（讓 demo 有真實資料背書）

這區不進正式服務，是科學驗證與模型生產用。用真實 run-to-failure 資料集：
PHM2012（FEMTO，17 軸承）、XJTU-SY（15 軸承）、IMS（NASA）。

| 檔案 | 用途 |
|---|---|
| `train_save_rf.py` | 訓練並輸出 `rul_rf.pkl`（正式服務用的模型就是這來的） |
| `extract_features_cache.py` | 抽特徵快取（→ data/cache/bearing_stats.pkl，35 軸承） |
| `xjtu_sy_rul_accuracy.py` / `phm2012_replay_validation.py` | 驗 RUL 準度 |
| `xjtu_sy_cusum_validation.py` / `phm2012_cusum_validation.py` | 驗 CUSUM 偵測命中率 |
| `ims_generalization_test.py` | 用完全沒訓練過的 IMS 驗泛化（確認學到通則、非過擬合） |
| `lstm_rul.py` | LSTM 對照組（結論：只比 RF 好約 5 分，瓶頸是資料量而非模型） |
| `ml_rul_piecewise.py` | 分段式 RUL 目標 + 多軸承合併訓練 |
| `Dockerfile.torch` | 在 ai_service 映像上加 PyTorch CPU，供 LSTM 實驗 |

### 能力誠實評估（驗證結論）

| 能力 | 狀態 |
|---|---|
| 異常偵測 | 32/32 軸承 100% 命中，可靠 |
| RUL（線性外推） | 誤差 164~610%，不堪用 |
| **RUL（ML RandomForest）** | 中位數約 62% 可動作窗口（約 1.6 倍誤差），跨資料集（IMS）泛化通過 |
| LSTM | 只比 RF 好約 5 分，瓶頸是資料量（32 軌跡）不是模型 |

---

## 重要約束與運維提醒

- **取樣率 2kHz（12 萬筆/秒）**：瓶頸是 CPU 不是磁碟。4kHz + 完整 AI（60 通道）+ 10 進程會壓垮
  8 核 CPU，故定在 2kHz。定義在 `edge_pipeline/constants.py` 與 `phase3_ai/spectral_features.py`
  兩處（刻意重複，兩服務不共用模組）。
- **關機順序**：關 Docker/WSL 前務必 `docker compose stop --timeout 90`，讓 timescaledb 完成
  checkpoint，否則 SIGKILL 會遺失 WAL、把保留政策/保險絲設定回滾。乾淨關機標誌：log 出現
  「database system is shut down」且 exit code 0。
- **磁碟空間回收**：Docker 資料在 `F:\DockerData\wsl\disk\docker_data.vhdx`。vhdx 只增不減，
  需 `fstrim`（釋放空塊）+ diskpart `compact vdisk`（縮小實體檔）回收。`.wslconfig` 已開
  `sparseVhd=true` 讓 WSL 自動歸還。
- **門檻一致性**：`phase3_ai/main.py` 的 `CURRENT_UPPER/LOWER` 與振動基頻，須與
  `edge_pipeline/machine_config.py` / `formulas.py` 保持一致（兩服務分屬不同 Docker context）。
