"""把 batch（NumPy 矩陣形式）攤平成 device_id+sensor_id+timestamp+value 的個別記錄。"""

from datetime import datetime


def format_timestamp(epoch_seconds: float) -> str:
    dt = datetime.fromtimestamp(epoch_seconds)
    return dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{dt.microsecond // 1000:03d}"


def _machine_number(device_id: str) -> int:
    return int(device_id.split("-")[1])


def explode_batch(batch: dict) -> list[dict]:
    device_id = batch["device_id"]
    n = _machine_number(device_id)
    timestamps = batch["timestamps"]
    records = []

    for phase, values in batch["current"].items():
        sensor_id = f"E{n:02d}-{phase}"
        for ts, value in zip(timestamps, values):
            records.append(
                {"device_id": device_id, "sensor_id": sensor_id, "timestamp": format_timestamp(ts), "value": float(value)}
            )

    for axis, values in batch["vibration"].items():
        sensor_id = f"V{n:02d}-{axis}"
        for ts, value in zip(timestamps, values):
            records.append(
                {"device_id": device_id, "sensor_id": sensor_id, "timestamp": format_timestamp(ts), "value": float(value)}
            )

    if "temperature" in batch:
        sensor_id = f"T{n}"
        temp = batch["temperature"]
        for ts, value in zip(temp["timestamps"], temp["values"]):
            records.append(
                {"device_id": device_id, "sensor_id": sensor_id, "timestamp": format_timestamp(ts), "value": float(value)}
            )

    return records


def explode_batch_raw(batch: dict) -> list[tuple]:
    """跟 explode_batch() 一樣攤平，但 timestamp 保留原始 epoch 浮點數、不做字串格式化，
    給 Phase2 高吞吐寫入 TimescaleDB 用（格式化字串留給展示/自我測試用途）。
    回傳 (epoch_timestamp, device_id, sensor_id, value) tuple 的 list。
    """
    device_id = batch["device_id"]
    n = _machine_number(device_id)
    timestamps = batch["timestamps"]
    rows = []

    for phase, values in batch["current"].items():
        sensor_id = f"E{n:02d}-{phase}"
        rows.extend((float(ts), device_id, sensor_id, float(value)) for ts, value in zip(timestamps, values))

    for axis, values in batch["vibration"].items():
        sensor_id = f"V{n:02d}-{axis}"
        rows.extend((float(ts), device_id, sensor_id, float(value)) for ts, value in zip(timestamps, values))

    if "temperature" in batch:
        sensor_id = f"T{n}"
        temp = batch["temperature"]
        rows.extend(
            (float(ts), device_id, sensor_id, float(value)) for ts, value in zip(temp["timestamps"], temp["values"])
        )

    return rows
