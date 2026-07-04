CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS sensor_readings (
    ts          TIMESTAMPTZ       NOT NULL,
    device_id   TEXT              NOT NULL,
    sensor_id   TEXT              NOT NULL,
    value       DOUBLE PRECISION  NOT NULL
);

-- chunk 間隔改成 1 小時：預設 7 天對 240,000 筆/秒的寫入速度來說太大，
-- 全速跑不到 1 天單一 chunk 就會膨脹到 TB 等級，拖慢查詢與壓縮。
SELECT create_hypertable(
    'sensor_readings', 'ts',
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_sensor_readings_device_sensor_ts
    ON sensor_readings (device_id, sensor_id, ts DESC);

-- 原生壓縮：實測壓縮比約 17 倍（153 bytes/筆 -> 約 9 bytes/筆）。
-- segmentby 用 device_id+sensor_id，因為查詢/壓縮都是以「同一感測器的時序」為單位最有效率。
ALTER TABLE sensor_readings SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'device_id, sensor_id',
    timescaledb.compress_orderby = 'ts DESC'
);

-- 超過 1 小時沒有新資料寫入的 chunk（也就是不再是「當前」那個 chunk）就自動壓縮。
SELECT add_compression_policy('sensor_readings', INTERVAL '1 hour');

-- 原始高頻資料只保留 30 天，超過就整個 chunk 直接丟棄（不是逐筆 DELETE，幾乎零成本）。
SELECT add_retention_policy('sensor_readings', INTERVAL '30 days');

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
