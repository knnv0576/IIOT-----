"""外部 IoT 資料進線（MQTT）：把真實裝置透過 MQTT 送來的資料，轉成跟 Phase1 模擬器一致的
(epoch_timestamp, device_id, sensor_id, value) tuple，餵進共用的 queue.Queue()，
讓 Phase2 的 db_writer_loop 可以用同一套邏輯把模擬器資料和真實 MQTT 資料一起寫進 TimescaleDB。

主題（topic）格式："<topic_prefix>/<device_id>/<sensor_id>"，例如 "iiot/SCT-05/E05-A"。
Payload 是 JSON：{"timestamp": "2026-07-01 19:23:45.125", "value": 6.02}；
timestamp 可以是毫秒精度字串、epoch 數字，或直接省略（省略時用訊息到達時間補上）。
"""

import json
import queue
import time
from datetime import datetime

import paho.mqtt.client as mqtt


def _parse_topic(topic: str, topic_prefix: str):
    prefix, _, rest = topic.partition("/")
    device_id, _, sensor_id = rest.partition("/")
    if prefix != topic_prefix or not device_id or not sensor_id:
        raise ValueError(f"topic 格式不符（應為 {topic_prefix}/<device_id>/<sensor_id>）: {topic!r}")
    return device_id, sensor_id


def _parse_timestamp(raw) -> float:
    if raw is None:
        return time.time()
    if isinstance(raw, (int, float)):
        return float(raw)
    return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S.%f").timestamp()


def run_mqtt_ingest(
    broker_host: str,
    broker_port: int,
    topic_prefix: str,
    output_queue: "queue.Queue",
    stop_event,
):
    def on_connect(client, userdata, flags, reason_code, properties=None):
        client.subscribe(f"{topic_prefix}/#")
        print(f"[mqtt_ingest] 已連線到 {broker_host}:{broker_port}，訂閱 {topic_prefix}/#")

    def on_disconnect(client, userdata, flags, reason_code, properties=None):
        print(f"[mqtt_ingest] 與 broker 斷線（reason_code={reason_code}），將自動重連")

    def on_message(client, userdata, msg):
        try:
            device_id, sensor_id = _parse_topic(msg.topic, topic_prefix)
            payload = json.loads(msg.payload.decode("utf-8"))
            timestamp = _parse_timestamp(payload.get("timestamp"))
            row = (timestamp, device_id, sensor_id, float(payload["value"]))
            output_queue.put(row)
        except Exception as exc:
            print(f"[mqtt_ingest] 忽略無法解析的訊息 topic={msg.topic!r} payload={msg.payload!r}: {exc}")

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    client.connect(broker_host, broker_port)
    client.loop_start()

    try:
        while not stop_event.is_set():
            time.sleep(0.2)
    finally:
        client.loop_stop()
        client.disconnect()
