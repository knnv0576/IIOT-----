"""Phase2：把攤平後的 (epoch_timestamp, device_id, sensor_id, value) 批次寫入 TimescaleDB。

用「COPY 進暫存表 -> 單一 INSERT SELECT 轉型進 hypertable」，而不是逐筆 INSERT/execute_values，
實測在本機環境下前者約 120,000 筆/秒，後者只有約 55,000 筆/秒（詳見開發過程的量測結果）。
"""

import io

import psycopg2

_STAGING_TABLE = "_sensor_readings_staging"


def connect(dsn: str):
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    with conn.cursor() as cur:
        # 高頻寫入場景可以接受「當機時遺失最後幾筆尚未 flush 到磁碟的資料」，換取寫入速度。
        cur.execute("SET synchronous_commit = OFF;")
        cur.execute(
            f"CREATE TEMP TABLE IF NOT EXISTS {_STAGING_TABLE} "
            "(epoch_ts DOUBLE PRECISION, device_id TEXT, sensor_id TEXT, value DOUBLE PRECISION);"
        )
    return conn


def insert_rows(conn, rows: list[tuple]) -> None:
    """rows: (epoch_timestamp, device_id, sensor_id, value) 的 tuple list。"""
    if not rows:
        return

    buf = io.StringIO()
    for ts, device_id, sensor_id, value in rows:
        buf.write(f"{ts}\t{device_id}\t{sensor_id}\t{value}\n")
    buf.seek(0)

    with conn.cursor() as cur:
        cur.execute(f"TRUNCATE {_STAGING_TABLE};")
        cur.copy_expert(f"COPY {_STAGING_TABLE} (epoch_ts, device_id, sensor_id, value) FROM STDIN", buf)
        cur.execute(
            "INSERT INTO sensor_readings (ts, device_id, sensor_id, value) "
            f"SELECT to_timestamp(epoch_ts), device_id, sensor_id, value FROM {_STAGING_TABLE};"
        )
