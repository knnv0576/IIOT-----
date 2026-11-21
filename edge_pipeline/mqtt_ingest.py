"""外部 IoT 資料進線（MQTT）：把真實裝置透過 MQTT 送來的資料，轉成跟 Phase1 模擬器一致的
(epoch_timestamp, device_id, sensor_id, value) tuple，餵進共用的 queue.Queue()，
讓 Phase2 的 db_writer_loop 可以用同一套邏輯把模擬器資料和真實 MQTT 資料一起寫進 TimescaleDB。

主題（topic）格式："<topic_prefix>/<device_id>/<sensor_id>"，例如 "iiot/SCT-05/E05-A"。
Payload 是 JSON：{"timestamp": "2026-07-01 19:23:45.125", "value": 6.02}；timestamp 可以是：
- ISO 8601 字串，建議帶時區，例如 "2026-07-01T19:23:45.125+08:00" 或 "...Z"（UTC）。
- 不帶時區的字串，例如 "2026-07-01 19:23:45.125"：當成 MQTT_NAIVE_TIMESTAMP_TZ 時區的時間
  （預設 Asia/Taipei）。舊版會當成容器時區（UTC），台灣裝置送來的時間就差 8 小時。
- epoch 秒數（數字，本身就是絕對時間，不受時區影響）。
- 直接省略：用訊息到達時間補上。

外部進來的資料一律視為不可信，進 Queue 之前先驗證，不合格的直接丟棄並印 log：
- device_id / sensor_id 只允許英數字與 `_ . : -`，長度 1~64。tab、換行、反斜線這類字元
  會打亂 db_writer 用的 COPY 文字格式，讓整批寫入失敗。
- value 必須是有限的數字。NaN/Infinity 會讓 1 秒彙總的 avg() 也變成 NaN，污染儀表板。
- timestamp 不能是負的，也不能比現在晚超過 MAX_FUTURE_SECONDS。最常見的錯誤是把
  「epoch 毫秒」當成秒送進來，會變成好幾萬年後的時間。上限故意設得很緊：原始資料的保留
  政策是「比現在舊多少就刪」，未來時間的資料要等時間真的走到才會被刪，等於可以繞過保留政策
  把磁碟撐爆。
"""

import json
import math
import os
import queue
import re
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import paho.mqtt.client as mqtt

_ID_PATTERN = re.compile(r"[A-Za-z0-9_.:\-]{1,64}")  # device_id / sensor_id 允許的字元與長度
# timestamp 最多允許比現在晚多少秒（容忍裝置時鐘誤差；有對時的裝置誤差通常在 1 秒內）
MAX_FUTURE_SECONDS = float(os.environ.get("MQTT_MAX_FUTURE_SECONDS", "300"))
# 不帶時區的字串時間戳要當成哪個時區。模組載入時就解析，時區名稱打錯會在啟動時直接報錯，
# 而不是等到每一筆訊息都解析失敗才發現。
NAIVE_TIMESTAMP_TZ = ZoneInfo(os.environ.get("MQTT_NAIVE_TIMESTAMP_TZ", "Asia/Taipei"))


def _parse_topic(topic: str, topic_prefix: str):
    prefix, _, rest = topic.partition("/")
    device_id, _, sensor_id = rest.partition("/")
    if prefix != topic_prefix or not device_id or not sensor_id:
        raise ValueError(f"topic 格式不符（應為 {topic_prefix}/<device_id>/<sensor_id>）: {topic!r}")
    if not _ID_PATTERN.fullmatch(device_id) or not _ID_PATTERN.fullmatch(sensor_id):
        raise ValueError(
            f"device_id/sensor_id 含不允許的字元或過長（只允許英數字與 _.:-，1~64 字）: {topic!r}"
        )
    return device_id, sensor_id


def _parse_timestamp(raw) -> float:
    if raw is None:
        return time.time()
    if isinstance(raw, bool):  # bool 是 int 的子類別，要先排除，否則 true 會被當成 epoch 1
        raise ValueError(f"timestamp 不能是布林值: {raw!r}")
    if isinstance(raw, (int, float)):
        ts = float(raw)
    elif isinstance(raw, str):
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=NAIVE_TIMESTAMP_TZ)
        ts = dt.timestamp()
    else:
        raise ValueError(f"timestamp 型別不支援: {raw!r}")
    if not math.isfinite(ts) or ts < 0:
        raise ValueError(f"timestamp 不合法: {raw!r}")
    if ts > time.time() + MAX_FUTURE_SECONDS:
        raise ValueError(f"timestamp 遠在未來（是不是把 epoch 毫秒當成秒送進來了？）: {raw!r}")
    return ts


def _parse_value(raw) -> float:
    if isinstance(raw, bool):
        raise ValueError(f"value 不能是布林值: {raw!r}")
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"value 必須是有限數字: {raw!r}")
    return value


class _ThrottledLog:
    """限制 log 頻率：同一類訊息每 interval 秒最多印一次，期間被略過的次數併在下一次一起報告。
    外部裝置一旦送錯格式或灌爆訊息，通常是每秒成千上萬筆，逐筆印 log 會洗掉其他所有訊息。"""

    def __init__(self, interval: float = 5.0):
        self.interval = interval
        self._last = 0.0
        self._suppressed = 0

    def __call__(self, message: str) -> None:
        now = time.time()
        if now - self._last < self.interval:
            self._suppressed += 1
            return
        extra = f"（另有 {self._suppressed} 筆同類訊息已略過）" if self._suppressed else ""
        print(message + extra)
        self._last = now
        self._suppressed = 0


def run_mqtt_ingest(
    broker_host: str,
    broker_port: int,
    topic_prefix: str,
    output_queue: "queue.Queue",
    stop_event,
    username: "str | None" = None,
    password: "str | None" = None,
):
    """username/password：broker 帳密。mosquitto 已設定禁止匿名連線，正式環境一定要提供。"""
    log_invalid = _ThrottledLog()
    log_queue_full = _ThrottledLog()

    def on_connect(client, userdata, flags, reason_code, properties=None):
        if reason_code.is_failure:
            print(f"[mqtt_ingest] broker 拒絕連線（reason_code={reason_code}），稍後自動重試")
            return
        client.subscribe(f"{topic_prefix}/#")
        print(f"[mqtt_ingest] 已連線到 {broker_host}:{broker_port}，訂閱 {topic_prefix}/#")

    def on_connect_fail(client, userdata):
        print(f"[mqtt_ingest] 連不上 broker {broker_host}:{broker_port}，稍後自動重試")

    def on_disconnect(client, userdata, flags, reason_code, properties=None):
        print(f"[mqtt_ingest] 與 broker 斷線（reason_code={reason_code}），將自動重連")

    def on_message(client, userdata, msg):
        try:
            device_id, sensor_id = _parse_topic(msg.topic, topic_prefix)
            payload = json.loads(msg.payload.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("payload 必須是 JSON 物件")
            timestamp = _parse_timestamp(payload.get("timestamp"))
            row = (timestamp, device_id, sensor_id, _parse_value(payload["value"]))
        except Exception as exc:
            log_invalid(
                f"[mqtt_ingest] 忽略無法解析的訊息 topic={msg.topic!r} payload={msg.payload[:200]!r}: {exc}"
            )
            return
        try:
            # put_nowait：佇列滿代表寫入端跟不上，寧可丟這筆也不能卡住 MQTT 的網路 thread
            output_queue.put_nowait(row)
        except queue.Full:
            log_queue_full(
                f"[mqtt_ingest] 進線佇列已滿（上限 {output_queue.maxsize} 筆），丟棄新訊息；"
                f"可能是裝置送太快或資料庫寫入跟不上"
            )

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    if username:
        client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.on_connect_fail = on_connect_fail
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    client.reconnect_delay_set(min_delay=1, max_delay=30)
    # 用 connect_async 而不是 connect()：connect() 是同步連線，broker 還沒起來（compose 的
    # depends_on 不等 mosquitto 就緒）就直接拋例外、整條 MQTT 進線 thread 永久死掉。
    # connect_async 只記下連線參數，由 loop_start() 的背景 thread 負責連線，
    # 第一次連線失敗也會依 reconnect_delay_set 的間隔一直重試。
    client.connect_async(broker_host, broker_port)
    client.loop_start()

    try:
        while not stop_event.is_set():
            time.sleep(0.2)
    finally:
        client.loop_stop()
        client.disconnect()
