"""Phase2：把攤平後的 (epoch_timestamp, device_id, sensor_id, value) 批次寫入 TimescaleDB。

用「COPY 進暫存表 -> 單一 INSERT SELECT 轉型進 hypertable」，而不是逐筆 INSERT/execute_values，
實測在本機環境下前者約 120,000 筆/秒，後者只有約 55,000 筆/秒（詳見開發過程的量測結果）。
"""

import io

import psycopg2

_STAGING_TABLE = "_sensor_readings_staging"


def connect(dsn: str):
    """建立一條寫入專用連線。連不上時拋出 psycopg2.Error，由呼叫端負責重試。"""
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


def insert_rows_isolating_bad(conn, rows: list[tuple]) -> list[tuple]:
    """跟 insert_rows() 一樣整批寫入，但遇到「資料本身有問題」（psycopg2.DataError：
    COPY 格式被字串裡的 tab/換行/反斜線打亂、NUL 字元、時間戳超出範圍…）時，
    用二分法把壞掉的那幾筆找出來剔除，其餘正常資料照樣寫進去。回傳被剔除的 rows。

    為什麼需要：呼叫端寫入失敗時會保留 buffer 重試。如果失敗原因是資料本身有問題，
    重試一萬次都一樣會失敗——這條寫入 thread 會永久卡住，負責的機台全部被背壓擋停。
    MQTT 進線雖然已經先做驗證，這裡是最後一道防線。

    連線類錯誤（OperationalError 等）不在這裡處理，照常往上拋，讓呼叫端保留資料、重連再試。
    autocommit 模式下每個 statement 各自成敗，失敗的那一輪不會有部分寫入殘留。
    """
    try:
        insert_rows(conn, rows)
        return []
    except psycopg2.DataError:
        if len(rows) == 1:
            return list(rows)
    mid = len(rows) // 2
    return insert_rows_isolating_bad(conn, rows[:mid]) + insert_rows_isolating_bad(conn, rows[mid:])
