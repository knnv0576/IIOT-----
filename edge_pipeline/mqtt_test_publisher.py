"""測試用：模擬一台外部 IoT 裝置，透過 MQTT 發布幾筆感測資料，驗證 mqtt_ingest.py 的進線是否work。

用法：docker compose run --rm --no-deps edge_pipeline python mqtt_test_publisher.py
"""

import json
import os
import time

import paho.mqtt.client as mqtt

BROKER_HOST = os.environ.get("MQTT_BROKER_HOST", "mosquitto")
BROKER_PORT = int(os.environ.get("MQTT_BROKER_PORT", "1883"))
TOPIC_PREFIX = os.environ.get("MQTT_TOPIC_PREFIX", "iiot")


def main():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.connect(BROKER_HOST, BROKER_PORT)
    client.loop_start()

    samples = [
        ("SCT-05", "E05-A", 6.03),
        ("SCT-05", "V05-X", 0.42),
        ("SCT-09", "T9", 26.1),
    ]
    for device_id, sensor_id, value in samples:
        topic = f"{TOPIC_PREFIX}/{device_id}/{sensor_id}"
        payload = json.dumps({"value": value})  # 不帶 timestamp，讓 mqtt_ingest 用到達時間補上
        client.publish(topic, payload, qos=1)
        print(f"發布 topic={topic} payload={payload}")
        time.sleep(0.2)

    time.sleep(1)
    client.loop_stop()
    client.disconnect()


if __name__ == "__main__":
    main()
