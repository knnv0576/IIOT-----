CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS sensor_readings (
    ts          TIMESTAMPTZ       NOT NULL,
    device_id   TEXT              NOT NULL,
    sensor_id   TEXT              NOT NULL,
    value       DOUBLE PRECISION  NOT NULL
);

-- chunk 間隔 10 分鐘：預設 7 天對高頻寫入來說太大。原本設 1 小時，但實測全速下
-- 單一未壓縮 chunk 會膨脹到上百 GB（壓縮政策只處理「已關閉」的 chunk，正在寫入的
-- 當前 chunk 完全不受控），直接把磁碟撐爆；縮到 10 分鐘讓當前 chunk 的峰值可控
--（4kHz、24 萬筆/秒下約 22GB），也讓容量保險絲能更快介入。
SELECT create_hypertable(
    'sensor_readings', 'ts',
    chunk_time_interval => INTERVAL '10 minutes',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_sensor_readings_device_sensor_ts
    ON sensor_readings (device_id, sensor_id, ts DESC);

-- 原始資料保留 20 分鐘、每 5 分鐘檢查一次（drop_chunks 整塊丟棄，幾乎零成本）。
-- 為什麼這麼短：資料庫實體磁碟現在放 F: 槽（Crucial T500 NVMe，~930GB），4kHz 全速寫入
-- 約 130GB/小時，20 分鐘保留讓原始資料的磁碟佔用穩定在 ~45GB。下游都不需要更長的原始資料：
-- AI 頻譜異常偵測只要最近 400 秒、Grafana 電流/振動面板查的是 1 秒彙總（見下方 CAgg）、
-- 長期趨勢由彙總視圖保存（獨立儲存，不受本保留政策影響）。若想留更長歷史，F: 空間充裕可拉長。
-- 註：原本配有 TimescaleDB 原生壓縮（實測 17 倍），但資料只活 20 分鐘根本輪不到壓縮（15 分才壓、
-- 20 分就丟），且壓縮 job 的 I/O 尖峰會跟串流寫入互搶，故整組移除。若日後拉長保留期，可恢復
-- 壓縮設定（git 歷史有完整範例）。
SELECT add_retention_policy('sensor_readings',
    drop_after => INTERVAL '20 minutes',
    schedule_interval => INTERVAL '5 minutes');

-- 每日摘要（Downsampling）：正常資料的長期保存形式，TimescaleDB 原生 Continuous Aggregate，
-- 不受 30 天保留政策限制（這是獨立的物化視圖，raw chunk 被丟掉後摘要仍然保留）。
CREATE MATERIALIZED VIEW IF NOT EXISTS sensor_daily_summary
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', ts) AS day,
    device_id,
    sensor_id,
    avg(value)    AS avg_value,
    min(value)    AS min_value,
    max(value)    AS max_value,
    stddev(value) AS stddev_value,
    count(*)      AS sample_count
FROM sensor_readings
GROUP BY day, device_id, sensor_id
WITH NO DATA;

SELECT add_continuous_aggregate_policy('sensor_daily_summary',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day');

-- 異常資料明細：跟每日摘要分開存放，判斷邏輯在 Python（edge_pipeline/anomaly_extractor.py），
-- 因為門檻值要直接沿用 machine_config.py，避免跟 SQL 兩邊各存一份、之後改一邊忘了改另一邊。
-- 這張表不設保留政策：異常資料量遠小於原始資料，且是 AI 訓練/事後調查最有價值的部分，預設永久保留。
CREATE TABLE IF NOT EXISTS sensor_anomalies (
    ts         TIMESTAMPTZ       NOT NULL,
    device_id  TEXT              NOT NULL,
    sensor_id  TEXT              NOT NULL,
    value      DOUBLE PRECISION  NOT NULL,
    reason     TEXT              NOT NULL
);

SELECT create_hypertable('sensor_anomalies', 'ts', chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_sensor_anomalies_device_sensor_ts
    ON sensor_anomalies (device_id, sensor_id, ts DESC);

-- Phase4 用：機台振動 ISO 門檻參考表，給 Grafana 畫優良/可接受/警示/危險門檻橫線。
-- 純粹是視覺化參考資料（機台分級幾乎不會變動），數值需與 machine_config.py 的 VIBRATION_ISO 保持一致。
CREATE TABLE IF NOT EXISTS machine_vibration_thresholds (
    device_id        TEXT PRIMARY KEY,
    size_class       TEXT             NOT NULL,
    good_upper       DOUBLE PRECISION NOT NULL,
    acceptable_upper DOUBLE PRECISION NOT NULL,
    danger_lower     DOUBLE PRECISION NOT NULL  -- 警示上限 = 危險下限
);

INSERT INTO machine_vibration_thresholds (device_id, size_class, good_upper, acceptable_upper, danger_lower) VALUES
    ('SCT-01', 'I',   0.71, 1.80, 4.50),
    ('SCT-02', 'I',   0.71, 1.80, 4.50),
    ('SCT-03', 'I',   0.71, 1.80, 4.50),
    ('SCT-04', 'I',   0.71, 1.80, 4.50),
    ('SCT-05', 'I',   0.71, 1.80, 4.50),
    ('SCT-06', 'I',   0.71, 1.80, 4.50),
    ('SCT-07', 'II',  1.12, 2.80, 7.10),
    ('SCT-08', 'II',  1.12, 2.80, 7.10),
    ('SCT-09', 'III', 1.80, 4.50, 11.20),
    ('SCT-10', 'III', 1.80, 4.50, 11.20)
ON CONFLICT (device_id) DO NOTHING;

-- Phase4 用：系統自身健康度（CPU/RAM/TPS/寫入延遲），main.py 的 system_health_loop 定期寫入。
CREATE TABLE IF NOT EXISTS system_health (
    ts               TIMESTAMPTZ       NOT NULL,
    cpu_percent      DOUBLE PRECISION,
    memory_mb        DOUBLE PRECISION,
    tps              DOUBLE PRECISION,
    write_latency_ms DOUBLE PRECISION
);

SELECT create_hypertable('system_health', 'ts', chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);

-- Phase4 儀表板用：電流/振動是 4kHz 原始資料，Grafana 面板如果直接查 sensor_readings 原始表，
-- 5 分鐘視窗就要掃幾百萬筆，實測要 5~12 秒、超過 Grafana 預設的查詢逾時，面板會一直轉圈圈。
-- 改成 Continuous Aggregate 讓 TimescaleDB 背景先算好 1 秒一筆的彙總，面板改查這裡，
-- 實測從 5~12 秒降到 0.05 秒。TimescaleDB 預設支援 real-time aggregate（最新幾秒的資料
-- 會即時併入查詢結果），所以不會因為背景彙總的排程間隔而看起來延遲。
CREATE MATERIALIZED VIEW IF NOT EXISTS current_1s_agg
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 second', ts) AS bucket,
    device_id,
    sensor_id,
    avg(value) AS avg_value
FROM sensor_readings
WHERE sensor_id LIKE 'E%'
GROUP BY bucket, device_id, sensor_id
WITH NO DATA;

-- start_offset 原本設 10 分鐘、schedule_interval 設 10 秒，實測在高頻寫入下每次刷新都要重掃
-- 10 分鐘視窗，工作卡住跑了 2.5 分鐘以上都完成不了（爭 I/O）；縮小成 2 分鐘視窗、拉長刷新間隔到
-- 20 秒之後，單次刷新只要 ~1.6 秒。
SELECT add_continuous_aggregate_policy('current_1s_agg',
    start_offset => INTERVAL '2 minutes',
    end_offset => INTERVAL '5 seconds',
    schedule_interval => INTERVAL '20 seconds');

CREATE MATERIALIZED VIEW IF NOT EXISTS vibration_1s_agg
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 second', ts) AS bucket,
    device_id,
    max(value) FILTER (WHERE sensor_id LIKE '%-X') AS x,
    max(value) FILTER (WHERE sensor_id LIKE '%-Y') AS y,
    max(value) FILTER (WHERE sensor_id LIKE '%-Z') AS z
FROM sensor_readings
WHERE sensor_id LIKE 'V%'
GROUP BY bucket, device_id
WITH NO DATA;

SELECT add_continuous_aggregate_policy('vibration_1s_agg',
    start_offset => INTERVAL '2 minutes',
    end_offset => INTERVAL '5 seconds',
    schedule_interval => INTERVAL '20 seconds');

-- Phase3 用：AI 推論服務（phase3_ai/main.py）定期寫入的剩餘壽命（RUL）預測結果。
-- 對每個電流通道用滑動視窗做線性回歸，抓「均值+飄移速率」，外推還要多久會撞到電流上下限。
CREATE TABLE IF NOT EXISTS rul_predictions (
    ts                  TIMESTAMPTZ       NOT NULL,
    device_id           TEXT              NOT NULL,
    sensor_id           TEXT              NOT NULL,
    current_mean        DOUBLE PRECISION,  -- 被追蹤量的目前值（見 metric_kind：電流均值 或 thd_ratio）
    drift_rate_per_min  DOUBLE PRECISION,  -- 每分鐘飄移速率
    r_squared           DOUBLE PRECISION,  -- 趨勢可信度，越接近1代表越像真的在飄移而不是雜訊
    target_threshold    DOUBLE PRECISION,  -- 預測會撞到的門檻（電流上下限 或 thd 劣化門檻）
    eta_seconds         DOUBLE PRECISION,  -- 預估還要多久會撞到門檻；NULL 代表目前判定為穩定、無法/不需要預測
    status              TEXT              NOT NULL,  -- STABLE / WARNING / CRITICAL
    metric_kind         TEXT              -- 'current_mean'(電流均值 RUL) 或 'thd_ratio'(振動頻譜 RUL)
);

-- 舊資料卷升級用：metric_kind 是後來才加的。
ALTER TABLE rul_predictions ADD COLUMN IF NOT EXISTS metric_kind TEXT;

SELECT create_hypertable('rul_predictions', 'ts', chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_rul_predictions_device_sensor_ts
    ON rul_predictions (device_id, sensor_id, ts DESC);

-- Phase3 用：One-Class SVM 頻譜異常評分（phase3_ai/anomaly_detector.py）。
-- 每個通道各自訓練一個模型（用正常運作時的原始波形抽時域+頻域特徵當訓練集），
-- 之後定期對最新窗口評分，score 越負代表跟正常樣子差越多，用來抓早期、均值還看不出來的異變。
-- 涵蓋電流(E,60Hz) + 振動(V,30Hz) 共 6 通道/台，device_id+sensor_id 自然區分。
CREATE TABLE IF NOT EXISTS anomaly_scores (
    ts          TIMESTAMPTZ       NOT NULL,
    device_id   TEXT              NOT NULL,
    sensor_id   TEXT              NOT NULL,
    score       DOUBLE PRECISION,
    is_anomaly  BOOLEAN           NOT NULL,  -- 瞬時判定：score < ANOMALY_SCORE_MARGIN(-1.0)
    mean_value  DOUBLE PRECISION,
    std_value   DOUBLE PRECISION,
    thd_ratio   DOUBLE PRECISION,  -- 諧波能量/基頻能量比值，方便事後debug是不是頻譜異常觸發的
    cusum       DOUBLE PRECISION,  -- 線上 CUSUM 累積值；校準期(暖機)為 NULL
    cusum_alarm BOOLEAN,           -- CUSUM 是否越過 h：抓「分數持續系統性下滑」的漸進劣化
    rms         DOUBLE PRECISION,  -- 時域均方根（給 ML RUL 模型當特徵）
    kurt        DOUBLE PRECISION,  -- 峰度
    crest       DOUBLE PRECISION   -- 波峰因子 = 峰值/rms
);

-- 舊資料卷升級用：欄位是後來才加的，用 ADD COLUMN IF NOT EXISTS 讓既有 DB 也補上。
ALTER TABLE anomaly_scores ADD COLUMN IF NOT EXISTS cusum DOUBLE PRECISION;
ALTER TABLE anomaly_scores ADD COLUMN IF NOT EXISTS cusum_alarm BOOLEAN;
ALTER TABLE anomaly_scores ADD COLUMN IF NOT EXISTS rms DOUBLE PRECISION;
ALTER TABLE anomaly_scores ADD COLUMN IF NOT EXISTS kurt DOUBLE PRECISION;
ALTER TABLE anomaly_scores ADD COLUMN IF NOT EXISTS crest DOUBLE PRECISION;

SELECT create_hypertable('anomaly_scores', 'ts', chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_anomaly_scores_device_sensor_ts
    ON anomaly_scores (device_id, sensor_id, ts DESC);

-- CUSUM 基線持久化：每通道一列，存 Phase3 AI 現場暖機得到的健康基線 mu0/sigma0。服務重啟時載回，
-- 免除「每次重啟都要重新暖機」的偵測盲窗，也讓一次乾淨的長校準結果能被反覆沿用。
-- 非 hypertable（資料量極小，最多 60 列）。ai_service 啟動時也會 CREATE IF NOT EXISTS，對既有 DB 安全。
CREATE TABLE IF NOT EXISTS cusum_baselines (
    device_id      TEXT              NOT NULL,
    sensor_id      TEXT              NOT NULL,
    mu0            DOUBLE PRECISION  NOT NULL,  -- 健康分數基準均值
    sigma0         DOUBLE PRECISION  NOT NULL,  -- 健康分數基準散布（robust 估計）
    low_confidence BOOLEAN           NOT NULL DEFAULT FALSE,  -- 校準可信度旗標（低信心者不存，此欄留給未來）
    rejected_frac  DOUBLE PRECISION,            -- 暖機被 robust 修剪剔除的比例
    n_warmup       INTEGER,                     -- 校準用的暖機樣本數
    updated_at     TIMESTAMPTZ       NOT NULL DEFAULT now(),
    PRIMARY KEY (device_id, sensor_id)
);

-- 容量保險絲：TimescaleDB 的保留政策只支援「按時間」，但以每秒十幾萬筆的持續寫入，
-- 30 天時間保留根本來不及擋（實測 4kHz 全速跑 9 小時就累積 81GB、曾把系統碟撐爆）。
-- 這個自訂排程工作每 5 分鐘檢查 sensor_readings 的實際大小，超過 limit_gb
-- 就丟棄所有已封存的 chunk 重新累計——drop_chunks(older_than => now()) 不會動到
-- 「正在寫入的當前 chunk」（其時間範圍上界在未來），所以寫入完全不中斷；
-- 連續彙總（每日摘要、1 秒彙總）是獨立儲存，已彙總的歷史趨勢不會跟著消失。
CREATE OR REPLACE PROCEDURE purge_sensor_readings_when_oversize(job_id INT, config JSONB)
LANGUAGE plpgsql AS $BODY$
DECLARE
    size_gb  DOUBLE PRECISION;
    limit_gb DOUBLE PRECISION := COALESCE((config->>'limit_gb')::DOUBLE PRECISION, 200);
BEGIN
    SELECT hypertable_size('sensor_readings') / 1024.0^3 INTO size_gb;
    IF size_gb > limit_gb THEN
        PERFORM drop_chunks('sensor_readings', older_than => now());
        RAISE WARNING 'sensor_readings 已達 % GB，超過上限 % GB：丟棄所有已封存 chunk，重新累計',
            round(size_gb::numeric, 1), limit_gb;
    END IF;
END
$BODY$;

-- 上限 150GB：這是「時間保留政策失效時」的第二道防線（正常情況下 20 分鐘保留政策會把
-- 大小壓在 ~45GB，摸不到這條線）。資料庫實體位置在 F: 槽（Crucial T500 NVMe，~930GB），
-- 4kHz 下 20 分保留約 45GB，留大量餘裕故上限設 150GB。用 alter_job(<job_id>, config => ...) 可調。
SELECT add_job('purge_sensor_readings_when_oversize', INTERVAL '5 minutes',
               config => '{"limit_gb": 150}');
