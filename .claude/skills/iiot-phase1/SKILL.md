---
name: iiot-phase1
description: IIoT 專案 Phase 1（10 台機台 4kHz 高頻電流/振動模擬、machine_config.py 參數化基準值疊加、多進程硬體隔離、本地雙緩衝、跨進程專屬 Queue、危險模式動態切換）的規格與驗收指引。當在本專案中撰寫、檢查、除錯 Phase 1 邊緣模擬器（SCT-01~10 資料產生）程式碼時使用。
---

# IIoT Phase 1：硬體隔離級高頻資料產生 (Edge Emulation)

## 適用範圍

本 Skill 只涵蓋 **Phase 1（Edge Emulation）**：10 台機台的資料產生、多進程硬體隔離、本地雙緩衝、跨進程專屬 Queue、危險模式切換。
Phase 2（TimescaleDB 寫入）、Phase 3（AI 推論）、Phase 4（Grafana 視覺化）不適用本規格，請勿混用；但 Phase1 產出的 Queue 型態與批次大小會直接影響 Phase2 的消費邏輯，修改時需注意介面一致。

## 架構總覽（重要：已從執行緒改為多進程）

- 10 台機台 SCT-01 ~ SCT-10，各自由**獨立的 `multiprocessing.Process`** 執行（**不是** `threading.Thread`），徹底繞開 Python GIL，可吃滿多核心 CPU，達成硬體級隔離。**每台機台只用 1 個進程**，不要為了 7 顆感測器再拆出額外的子進程/子執行緒。
- 每個進程內部維護一個**本地雙緩衝區（Double Buffer）**，用 **NumPy 陣列**作為容器（不是 Python list）。4kHz tick 迴圈產生的每筆數值先寫入這個本地陣列，不直接跨進程傳輸。
- **出貨（Flush）觸發條件（符合其一即觸發）**：
  1. 本地緩衝累積滿 **1,000 個 tick**（在標準 4kHz 下換算剛好是 250ms，兩個門檻設計上會同時命中）。
  2. 距離上次出貨已經過 **250 毫秒**，即使 tick 數不足 1,000 也要強制出貨（例如系統延遲導致採樣變慢時，避免尾端資料滯留在本地緩衝區出不去）。
- 出貨時，把整批本地緩衝（NumPy 陣列）一次性放入**該機台專屬的 `multiprocessing.Queue`**。10 台機台各自一條 Queue，彼此獨立、互不干擾。
- **徹底移除「全域共享 list + 全域 `threading.Lock()`」設計**：不再有跨機台共用的緩衝區或鎖。與外部的同步只發生在「自己的 Queue.put()」這個動作上，且頻率從舊設計「每筆都要搶鎖」（每秒 8 萬次）驟降到「每批次才 put 一次」（10 台機台 × 每秒 4 次出貨 = 每秒約 40 次），本質上是 Lock-Free 設計。
- 溫度是 1Hz：做法是每 4,000 個 tick（即每秒，相當於 4 個出貨週期）才額外多產生 1 筆溫度值，寫入本地緩衝區的溫度欄位；溫度值不需要每個批次都出貨，可以累積到下一批一起送出，或用獨立的小批次送出，實作時擇一並保持一致即可。

## 機台與感測器總覽

每台機台實際上有 **7 顆物理感測器**，取樣頻率不同：

| 感測器 | sensor_id 範例（以 SCT-01 為例） | 頻率 |
|---|---|---|
| 溫度 x1 | `T1` | 每秒 1 筆（1Hz） |
| 電流（3 相）x3 | `E01-A`、`E01-B`、`E01-C` | 每秒 4,000 筆（4kHz） |
| 三軸振動 x3（1 顆感測器的 3 軸） | `V01-X`、`V01-Y`、`V01-Z` | 每秒 4,000 筆（4kHz） |

- `sensor_id` 依機台編號遞增，例如 SCT-02 對應 `T2`、`E02-A/B/C`、`V02-X/Y/Z`。
- **每個電流相（A/B/C）與每個振動軸（X/Y/Z）都各自是獨立的一筆記錄**（`device_id`+`sensor_id`+`timestamp`+`value`），不要把三個值合併成一筆複合記錄；在本地 NumPy 緩衝陣列裡可以用矩陣欄位表示（例如 shape `(1000, 6)` 對應 6 個 4kHz 通道），出貨後由 Phase2 負責攤平/寫入。

機台依大小分級，分級會影響振動 ISO 門檻與危險模式數值（見下方「危險模式動態切換規則」）：

| 機台 | 分級 | Class |
|---|---|---|
| SCT-01 ~ SCT-06 | 小型 | Class I |
| SCT-07 ~ SCT-08 | 中型 | Class II |
| SCT-09 ~ SCT-10 | 大型 | Class III |

## 4kHz 無延遲採樣規則

- 電流與振動的採樣率為 **4,000 筆/秒**（間隔 0.25ms）。**禁止**單純用 `time.sleep(1/4000)` 逐次呼叫，因為呼叫本身的開銷會造成誤差累積、長時間運行後產生明顯漂移（drift）。
- 正確做法：以進程啟動時的基準時間為 0，第 n 個 tick 的目標時間 = `start + n * interval`，每次用目標時間減去當前時間來決定要 sleep 多久，藉此讓誤差不會累積。計時基準一律用 `time.perf_counter()`（單調時鐘），不要用 `time.time()` 做間隔計算。
- 示意邏輯（僅供理解計時策略，非完整實作；每 1,000 tick 或每 250ms 觸發一次出貨，`n % 4000 == 0` 用來每秒額外觸發一次溫度讀值）：

```python
interval = 1 / 4000
start = time.perf_counter()
last_flush = start
buf = LocalDoubleBuffer()  # 內部用 NumPy 陣列，容量 1000
n = 0
while running:
    target = start + n * interval
    now = time.perf_counter()
    if target > now:
        time.sleep(target - now)
    buf.append_current_and_vibration(n)   # 寫入本地 NumPy 陣列，非跨進程操作
    if n % 4000 == 0:
        buf.append_temperature(n)          # 1Hz
    now = time.perf_counter()
    if buf.count >= 1000 or (now - last_flush) >= 0.25:
        queue.put(buf.flush())              # 整批出貨，僅此處才跨進程通訊
        last_flush = now
    n += 1
```

- **驗收硬規則**：同一感測器（同一個 `sensor_id`）前後兩筆資料的 timestamp 差，電流/振動必須嚴格等於 0.25ms，溫度必須嚴格等於 1 秒，不可有遺漏、重複或時間倒退——這條規則不因為批次出貨而改變，批次只影響「何時送出」，不影響「產生時的時間戳精度」。

## NumPy 向量化規則

- **全面禁止**用原生 Python `for` 迴圈逐筆呼叫 `numpy.random.normal()` 等函式產生單一數值——這樣不但沒有效能優勢，反而比原生 `random` 模組更慢（因為每次呼叫 NumPy 都有 overhead）。
- 正確做法：每次出貨批次（最多 1,000 個 tick）**先一次性用 NumPy 向量化算出整批數值**，例如：
  - 先產生整批的時間點陣列 `t = start + np.arange(n0, n0+1000) * interval`。
  - 電流：`I = I_base + A * np.sin(2*np.pi*60*t) + np.random.normal(0, sigma, size=1000)`，三相各算一次（3 次向量化呼叫，而非 3000 次逐筆呼叫）。
  - 振動：X/Y/Z 三軸同理，各自一次向量化呼叫算出 1000 筆。
- 逐 tick 的**節奏控制（sleep 校時）仍然需要逐筆進行**（因為要維持精確的 0.25ms 間隔），但「數值運算」跟「節奏控制」是兩件事：可以先在批次開始前一次性算好整批 1,000 筆數值存進本地陣列，迴圈裡只負責照節奏把預先算好的值填入緩衝區、控制何時 sleep，不要在迴圈內重複呼叫 NumPy 隨機函式。

## 參數設定規則（重要：不在執行期讀取 Excel）

- 所有機台/物理參數寫死在 `edge_pipeline/machine_config.py`（`MACHINES`、`TEMPERATURE`、`CURRENT`、`VIBRATION_ISO`），程式執行期**不讀取** `config/machine_config.xlsx`。Excel 檔案只保留在 `config/` 目錄下作為需求對照參考，不是執行時的資料來源，**不要**在程式碼裡加入 `pandas.read_excel`/`openpyxl` 之類的讀取邏輯。
- 要調整任何基準值、上下限、機台分級、ISO 門檻，直接改 `machine_config.py` 裡對應的常數即可；改完不需要重新產生或讀取任何檔案，重啟進程就會套用新值。
- `machine_config.py` 的資料結構（`load_config()` 回傳的 dict）維持跟 Excel 內容一致，方便日後對照：
  - `MACHINES`：10 台機台的 `device_id` + `size_class`（`I`=小型、`II`=中型、`III`=大型）。
  - `TEMPERATURE` / `CURRENT`：全域（10 台共用）的 `upper`/`setpoint`/`lower`，電流是「百分比數字」（6.0 代表 6.00%），不是安培。
  - `VIBRATION_ISO`：依 Class 分的優良/可接受/警示/危險四段門檻（mm/s），供危險模式與未來 Phase3/4 的 ISO 比對使用。
- 保留 `danger_threshold(config, size_class)` 與 `good_zone_upper(config, size_class)` 兩個輔助函式，語意不變（分別回傳該 Class 的危險門檻、優良區間上限）。

## 物理量模擬公式（不可變更的規格）

1. **溫度**：以 `machine_config.py` 的 `TEMPERATURE["setpoint"]`（26°C）為中心，疊加高斯常態分佈雜訊 `numpy.random.normal(loc=26, scale=σ)`，使數值自然漂移在 **25.5 ~ 26.5°C** 之間（可調整 σ 使漂移範圍符合此區間，並做邊界裁切）。
2. **電流**：A/B/C 三相各自獨立產生，都以 `machine_config.py` 的 `CURRENT`（設定值 6.00%，允許範圍 5.50%~6.50%）為基準，疊加 60Hz 交流電正弦波，再加微幅雜訊：
   $$I_{phase} = I_{base}(\%) + A \cdot \sin(2\pi \cdot 60 \cdot t) + \text{noise}, \quad phase \in \{A, B, C\}$$
   三相可各自使用獨立的雜訊種子/相位差，避免三相數值完全相同。
3. **三軸振動**：單顆三軸感測器，X/Y/Z 三軸各自有獨立的基礎振幅與相位差，疊加微幅雜訊，模擬空間方向上的差異（例如三軸各自用不同相位的正弦/餘弦基底 + 高斯雜訊）。正常模式下數值應落在該機台 Class 對應的「優良/可接受」區間內（見 `machine_config.py` 的 `VIBRATION_ISO`）。

以上三條公式的實作**都必須套用「NumPy 向量化規則」**，一次算整批而非逐筆算。

## 資料格式規格

每筆記錄必須包含：

| 欄位 | 說明 |
|---|---|
| `device_id` | 機台編號，`SCT-01` ~ `SCT-10` |
| `sensor_id` | 感測器編號，例如 `T3`、`E03-A`、`E03-B`、`E03-C`、`V03-X`、`V03-Y`、`V03-Z` |
| `timestamp` | 精確到毫秒，西元年/月/日/時/分/毫秒，例如 `2026-07-01 19:23:45.125` |
| `value` | 感測數值 |

注意：Python 的 `datetime.now()` 預設精度到微秒，**格式化輸出時務必截斷/四捨五入到毫秒**，不要直接留微秒尾數。

單筆記錄範例（dict 形式，僅供理解欄位；批次出貨時實際是 NumPy 矩陣，不會逐筆用 dict 傳遞）：

```python
{"device_id": "SCT-03", "sensor_id": "V03-X", "timestamp": "2026-07-01 19:23:45.125", "value": 0.42}
{"device_id": "SCT-03", "sensor_id": "E03-A", "timestamp": "2026-07-01 19:23:45.125", "value": 6.02}
```

## 跨進程 Queue 與背壓（Backpressure）規則

- 每台機台的出貨批次進入**自己專屬的 `multiprocessing.Queue`**（進程安全佇列），不是 `asyncio.Queue`——`asyncio.Queue` 只能在單一進程的事件迴圈內使用，無法跨進程傳遞資料。文件中提到的「非同步 Queue」在 Phase1（跨進程 IPC）語境下應理解為 `multiprocessing.Queue`；Phase2 消費端要用非同步方式去讀取這些進程安全佇列時，可以搭配 executor/thread 包一層 async 介面，但底層資料通道本身仍是 `multiprocessing.Queue`。
- Queue 上限（`maxsize`，由 Phase2 設定，目前為 500 個批次）是防止後端斷線/處理跟不上時記憶體爆掉的保護機制，取代舊設計裡「全域共享 list 上限 50 萬筆」的角色。
- 當 Queue 已滿，`queue.put(batch)` 預設會**阻塞**，這就是「暫停接收新資料」的保護行為，Phase1 端不需要另外實作容量檢查。若不希望完全阻塞導致節奏被打亂，可改用 `queue.put(batch, timeout=...)` 搭配例外處理並記錄警告，但**不可**因為滿了就靜默丟資料。
- 因為機台彼此是獨立進程、獨立 Queue，一台機台的 Queue 阻塞**不會**影響其他機台的採樣節奏，這是相對於舊版「全域共享 list + 單一鎖」設計的主要優勢，撰寫程式碼時應確認真的沒有引入任何跨進程共用的鎖或變數，破壞這個隔離性。

## 危險模式動態切換規則（重要：跨進程共享機制已改變）

- 提供一個可從外部即時切換的全域狀態旗標：`NORMAL` / `DANGER`（D 模式），依機台（`device_id`）分別控制。
- **因為機台改用獨立進程，記憶體不再像執行緒一樣共享**，不能用模組層級的 Python 全域變數當開關（子進程看不到父進程之後才修改的值）。必須用官方提供的跨進程 IPC 原語，例如：
  - 每個機台進程各自一個 `multiprocessing.Value('i', 0)` 或 `multiprocessing.Event()`，建立進程時當作參數傳入子進程；主控端修改這個共享物件的值即可讓子進程在下一個 tick 讀到新狀態。
  - 或者用一條額外的「控制 Queue」，主控端推送 `("SET_MODE", device_id, "DANGER")` 這類指令，各機台進程在自己的 4kHz 迴圈中非阻塞地輪詢（例如每個 tick 用 `queue.get_nowait()` 檢查一次）。
- **Windows 平台注意**：`multiprocessing` 在 Windows 上預設啟動方式是 **spawn**（不是 Unix 的 `fork`），子進程不會繼承父進程當下的記憶體狀態。所有傳給子進程的物件（目標函式、參數）都必須是可 pickle 的，共享狀態一律透過 `Value`/`Event`/`Queue`/`Manager` 傳遞，絕對不能假設模組層級全域變數會在進程之間同步。
- 切換至 `DANGER` 模式時，振動數值需刻意產生超過該機台 **Class 對應的危險門檻**（來自 `machine_config.py` 的 `VIBRATION_ISO`），而非統一用單一數字：

  | 機台 | Class | 危險(D)門檻 |
  |---|---|---|
  | SCT-01 ~ SCT-06（小型） | I | > 4.5 mm/s |
  | SCT-07 ~ SCT-08（中型） | II | > 7.10 mm/s |
  | SCT-09 ~ SCT-10（大型） | III | > 11.20 mm/s |

  切換邏輯需依 `device_id` 查出對應 Class 後再決定要超過哪個數字，不可對所有機台都套用 4.5。
- 切換動作本身**不可**造成任何進程卡死或明顯延遲；只需要在下一個 tick 生效即可，不需要中斷正在進行中的 sleep；其餘機台的採樣時序不應受切換動作影響。

## 自我測試 / 驗收清單

實作完成後，逐項驗證：

1. **格式符合度**：每台機台先限制只輸出 10 筆，檢查 `device_id`、`sensor_id`（含 T/E-A/B/C/V-X/Y/Z 是否對應正確機台編號）、毫秒級 `timestamp` 是否完整且格式正確。
2. **數據總量檢查**：統計 1 秒內 10 台機台的總筆數，4kHz 通道（電流 3 相 + 振動 3 軸）應精確等於
   $$10 \text{ 台} \times 4{,}000 \text{ 筆/秒} \times 6 \text{（電流A/B/C + 振動X/Y/Z）} = 240{,}000 \text{ 筆/秒}$$
   另加溫度 `10 台 × 1 筆/秒 = 10 筆/秒`（1Hz，不併入 4kHz 通道的總量檢查）。
3. **時序連續性測試**：對同一 `sensor_id`，逐筆計算前後 timestamp 差值；電流/振動需嚴格等於 0.25ms，溫度需嚴格等於 1 秒，且無時間倒退、遺漏或重複。
4. **動態注入測試**：切換至 `DANGER` 模式後，驗證**該機台 Class 對應門檻**的振動數值立即超標（小型 >4.5、中型 >7.10、大型 >11.20），同時確認其他機台的進程沒有卡死、其他機台仍正常運作、其他機台的時序連續性未被中斷。
5. **雙緩衝出貨測試**：驗證兩種出貨觸發條件都各自生效——(a) 正常 4kHz 下應該每 250ms 剛好累積滿 1,000 筆並出貨；(b) 故意模擬降速（例如人為插入延遲）讓 tick 數不足 1,000，驗證 250ms 時間門檻仍會強制出貨，不會讓資料卡在本地緩衝區出不去。
6. **多進程隔離測試**：驗證 10 個機台確實是獨立作業系統進程（而非執行緒，可用 `psutil`/工作管理員檢查 PID），且任一機台的 Queue 阻塞或切換危險模式，不會拖慢其他機台的 tick 節奏。

## 常見陷阱

- Windows 上 `time.sleep()` 的實際精度可能粗於 1ms，高頻計時迴圈要用忙等（busy-wait）搭配短 sleep 做補償，單純 sleep 可能導致實際間隔大於 0.25ms。
- `datetime.now()` 預設輸出到微秒，忘記格式化到毫秒是常見的資料格式不符合規格的原因。
- 誤把危險模式門檻寫死成單一數字（例如全部機台都用 >4.5），沒有依機台 Class 查表，會導致中型/大型機台的故障測試不符合 ISO 規格表。
- 電流基準值是百分比（%），若程式碼誤用安培級的數字（例如把 6.00 當成 6 安培）會讓正弦波振幅公式失真。
- **用模組層級全域變數傳遞危險模式旗標**：在 threading 版本裡這樣做沒問題，但改成 `multiprocessing` 之後，子進程（尤其 Windows 的 spawn 模式）根本看不到父進程之後修改的全域變數，必須改用 `Value`/`Event`/`Queue`/`Manager`。
- **對進程目標函式使用 lambda 或不可 pickle 的物件**：Windows spawn 模式要求傳給 `Process(target=..., args=...)` 的東西都能被 pickle，否則啟動子進程時會直接報錯。
- **逐 tick 呼叫 NumPy 隨機函式**：只在批次層級（每 1,000 筆）呼叫一次向量化函式，而不是在 4kHz 迴圈內每個 tick 都呼叫一次 `numpy.random.normal()`，否則效能反而比原生 Python `random` 更差。
- **出貨批次大小/時間門檻不要隨意更動**：1,000 筆／250ms 是已經抓好的平衡點（序列化成本 vs. IPC 次數），批次太小會讓 `Queue.put()` 呼叫過於頻繁（IPC overhead 高），批次太大則單次序列化/傳輸耗時變長、危險模式切換的生效延遲也會變大。
