"""每日異常擷取：在保留政策把舊的原始 chunk 丟掉之前，
把超過「可接受」門檻（警示 C + 危險 D 都算異常）的原始資料另外存進 sensor_anomalies，
其餘正常資料則交給 TimescaleDB 原生的 Continuous Aggregate（sensor_daily_summary）做每日平均壓縮。

前提：原始資料保留期要超過 1 天，這支程式處理「昨天」時資料才還在。本機開發環境只保留
20 分鐘（見 timescaledb/init.sql），所以在本機跑會擷取到 0 筆，這是預期中的結果。
另外這支程式目前沒有被任何排程呼叫，部署時要自己用 cron 之類的工具每天跑一次。

異常判斷門檻直接沿用 machine_config.py，不在 SQL 端另外存一份門檻常數，
避免以後改了 Excel/machine_config.py 的門檻卻忘記同步改 SQL。

用法：
    python anomaly_extractor.py              # 處理「昨天」整天（UTC）
    python anomaly_extractor.py 2026-07-01   # 處理指定日期（UTC 當天 00:00~24:00）
"""

import os
import sys
from datetime import datetime, timedelta, timezone

import psycopg2
from psycopg2.extras import execute_values

import machine_config

TIMESCALEDB_DSN = os.environ["TIMESCALEDB_DSN"]


def _day_range(day: datetime) -> tuple[datetime, datetime]:
    start = day.replace(hour=0, minute=0, second=0, microsecond=0)
    return start, start + timedelta(days=1)


def _insert_anomalies(conn, rows: list[tuple]) -> None:
    if not rows:
        return
    with conn.cursor() as cur:
        execute_values(
            cur,
            "INSERT INTO sensor_anomalies (ts, device_id, sensor_id, value, reason) VALUES %s",
            rows,
        )


def extract_temperature_and_current_anomalies(conn, config: dict, start: datetime, end: datetime) -> int:
    """溫度/電流各自的正常範圍是全域統一的，超出上下限就算異常，直接逐筆比對即可。"""
    temp = config["temperature"]
    current = config["current"]
    total = 0

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT ts, device_id, sensor_id, value FROM sensor_readings
            WHERE sensor_id LIKE 'T%%' AND ts >= %s AND ts < %s
              AND (value < %s OR value > %s)
            """,
            (start, end, temp["lower"], temp["upper"]),
        )
        rows = [(ts, dev, sid, val, "TEMP_OUT_OF_RANGE") for ts, dev, sid, val in cur.fetchall()]
        _insert_anomalies(conn, rows)
        total += len(rows)

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT ts, device_id, sensor_id, value FROM sensor_readings
            WHERE sensor_id LIKE 'E%%' AND ts >= %s AND ts < %s
              AND (value < %s OR value > %s)
            """,
            (start, end, current["lower"], current["upper"]),
        )
        rows = [(ts, dev, sid, val, "CURRENT_OUT_OF_RANGE") for ts, dev, sid, val in cur.fetchall()]
        _insert_anomalies(conn, rows)
        total += len(rows)

    return total


def extract_vibration_anomalies(conn, config: dict, start: datetime, end: datetime) -> int:
    """振動要三軸合成（sqrt(X^2+Y^2+Z^2)）才有意義，門檻依機台 Class 查表。

    magnitude 計算和門檻比對都推進 SQL 做（用 VALUES 把 machine_config.py 的門檻值動態帶進查詢），
    只把真正超標的那一小部分結果拉回 Python 展開成 X/Y/Z 三筆——
    早期版本是把當天「全部」振動分組（不管正不正常）整批 fetchall 到 Python 再逐筆比對，
    千萬筆規模下會把記憶體吃到數 GB、跑好幾十秒，這裡改成讓資料庫先篩選再回傳。
    """
    thresholds = [
        (
            m["device_id"],
            machine_config.acceptable_zone_upper(config, m["size_class"]),
            machine_config.danger_threshold(config, m["size_class"]),
        )
        for m in config["machines"]
    ]
    values_sql = ", ".join(["(%s, %s, %s)"] * len(thresholds))
    params = [v for row in thresholds for v in row] + [start, end]

    query = f"""
        WITH device_thresholds (device_id, acceptable_upper, danger_limit) AS (
            VALUES {values_sql}
        ),
        grouped AS (
            SELECT ts, device_id,
                   max(value) FILTER (WHERE sensor_id LIKE '%%-X') AS x,
                   max(value) FILTER (WHERE sensor_id LIKE '%%-Y') AS y,
                   max(value) FILTER (WHERE sensor_id LIKE '%%-Z') AS z
            FROM sensor_readings
            WHERE sensor_id LIKE 'V%%' AND ts >= %s AND ts < %s
            GROUP BY ts, device_id
        )
        SELECT g.ts, g.device_id, g.x, g.y, g.z,
               CASE WHEN sqrt(g.x*g.x + g.y*g.y + g.z*g.z) > dt.danger_limit
                    THEN 'VIBRATION_DANGER' ELSE 'VIBRATION_WARNING' END AS reason
        FROM grouped g
        JOIN device_thresholds dt ON dt.device_id = g.device_id
        WHERE g.x IS NOT NULL AND g.y IS NOT NULL AND g.z IS NOT NULL
          AND sqrt(g.x*g.x + g.y*g.y + g.z*g.z) > dt.acceptable_upper
    """

    with conn.cursor() as cur:
        cur.execute(query, params)
        anomalous_groups = cur.fetchall()

    rows_to_insert: list[tuple] = []
    for ts, device_id, x, y, z, reason in anomalous_groups:
        n = int(device_id.split("-")[1])
        rows_to_insert.append((ts, device_id, f"V{n:02d}-X", x, reason))
        rows_to_insert.append((ts, device_id, f"V{n:02d}-Y", y, reason))
        rows_to_insert.append((ts, device_id, f"V{n:02d}-Z", z, reason))

    _insert_anomalies(conn, rows_to_insert)
    return len(rows_to_insert)


def run(day: datetime) -> None:
    config = machine_config.load_config()
    start, end = _day_range(day)
    conn = psycopg2.connect(TIMESCALEDB_DSN)
    conn.autocommit = True
    try:
        n_temp_current = extract_temperature_and_current_anomalies(conn, config, start, end)
        n_vibration = extract_vibration_anomalies(conn, config, start, end)
        print(
            f"[anomaly_extractor] {start.date()} 共擷取 {n_temp_current + n_vibration} 筆異常"
            f"（溫度/電流 {n_temp_current} 筆，振動 {n_vibration} 筆）"
        )
    finally:
        conn.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_day = datetime.strptime(sys.argv[1], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        target_day = datetime.now(timezone.utc) - timedelta(days=1)
    run(target_day)
