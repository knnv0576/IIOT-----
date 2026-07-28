"""Phase1 自我測試：對應 SKILL.md「自我測試 / 驗收清單」的每一項。"""

import multiprocessing as mp
import os
import time

import numpy as np

import machine_config
from constants import FLUSH_SECONDS, FLUSH_TICKS, SAMPLE_INTERVAL, SAMPLE_RATE_HZ
from formulas import vector_magnitude
from machine_process import run_machine
from records import explode_batch


def _start_machine(device_id, size_class, config, stop_event, queue_size=20, **kwargs):
    queue = mp.Queue(maxsize=queue_size)
    danger_flag = mp.Value("b", 0)
    process = mp.Process(
        target=run_machine,
        args=(device_id, size_class, config, queue, danger_flag, stop_event),
        kwargs=kwargs,
        daemon=True,
    )
    process.start()
    return process, queue, danger_flag


def test_format(config):
    machine = config["machines"][2]  # SCT-03
    stop_event = mp.Event()
    process, queue, _ = _start_machine(
        machine["device_id"], machine["size_class"], config, stop_event, max_batches=1
    )
    batch = queue.get(timeout=5)
    stop_event.set()
    process.join(timeout=2)

    records = explode_batch(batch)[:10]
    ok = all(
        r["device_id"] == machine["device_id"]
        and r["sensor_id"]
        and len(r["timestamp"]) == 23  # "YYYY-MM-DD HH:MM:SS.mmm"
        for r in records
    )
    print(f"[1. 格式符合度] {'PASS' if ok else 'FAIL'}，前 10 筆範例：")
    for r in records:
        print("   ", r)
    return ok


def test_total_count(config, duration=3.0):
    stop_event = mp.Event()
    processes = []
    queues = {}
    for machine in config["machines"]:
        process, queue, _ = _start_machine(
            machine["device_id"], machine["size_class"], config, stop_event, queue_size=200
        )
        processes.append(process)
        queues[machine["device_id"]] = queue

    counts = {device_id: 0 for device_id in queues}
    deadline = time.time() + duration
    while time.time() < deadline:
        for device_id, queue in queues.items():
            while True:
                try:
                    batch = queue.get_nowait()
                except Exception:
                    break
                counts[device_id] += len(batch["timestamps"]) * 6
                if "temperature" in batch:
                    counts[device_id] += len(batch["temperature"]["timestamps"])
        time.sleep(0.02)

    stop_event.set()
    for process in processes:
        process.join(timeout=2)

    total = sum(counts.values())
    expected = 10 * SAMPLE_RATE_HZ * 6 * duration
    ratio = total / expected
    ok = 0.9 <= ratio <= 1.1
    print(
        f"[2. 數據總量檢查] {'PASS' if ok else 'FAIL'}："
        f"{duration:.0f} 秒內共 {total} 筆（預期約 {int(expected)} 筆，比例 {ratio:.2%}）"
    )
    return ok


def test_continuity(config, duration=2.0):
    machine = config["machines"][4]  # SCT-05
    stop_event = mp.Event()
    process, queue, _ = _start_machine(
        machine["device_id"], machine["size_class"], config, stop_event, queue_size=50
    )

    all_timestamps = []
    temp_timestamps = []
    deadline = time.time() + duration
    while time.time() < deadline:
        try:
            batch = queue.get(timeout=0.5)
        except Exception:
            continue
        all_timestamps.append(batch["timestamps"])
        if "temperature" in batch:
            temp_timestamps.extend(batch["temperature"]["timestamps"])

    stop_event.set()
    process.join(timeout=2)

    ts = np.concatenate(all_timestamps)
    diffs = np.diff(ts)
    current_ok = bool(np.all(np.abs(diffs - SAMPLE_INTERVAL) < 1e-6)) and bool(np.all(diffs > 0))

    temp_ok = True
    if len(temp_timestamps) >= 2:
        temp_diffs = np.diff(np.array(temp_timestamps))
        temp_ok = bool(np.all(np.abs(temp_diffs - 1.0) < 1e-3))

    ok = current_ok and temp_ok
    print(
        f"[3. 時序連續性測試] {'PASS' if ok else 'FAIL'}："
        f"{SAMPLE_RATE_HZ}Hz 通道間隔誤差最大值 {np.max(np.abs(diffs - SAMPLE_INTERVAL)):.2e}s，"
        f"溫度樣本數 {len(temp_timestamps)}（間隔應為 1s）"
    )
    return ok


def test_danger_injection(config):
    representatives = ["SCT-01", "SCT-07", "SCT-09"]  # Class I / II / III
    all_ok = True
    for device_id in representatives:
        machine = next(m for m in config["machines"] if m["device_id"] == device_id)
        size_class = machine["size_class"]
        threshold = machine_config.danger_threshold(config, size_class)

        stop_event = mp.Event()
        process, queue, danger_flag = _start_machine(device_id, size_class, config, stop_event, queue_size=20)

        normal_batch = queue.get(timeout=5)
        normal_mag = float(vector_magnitude(normal_batch["vibration"]).mean())

        danger_flag.value = 1
        danger_mag = None
        for _ in range(6):
            batch = queue.get(timeout=5)
            mag = float(vector_magnitude(batch["vibration"]).mean())
            if mag > threshold:
                danger_mag = mag
                break

        stop_event.set()
        process.join(timeout=2)

        ok = normal_mag <= threshold and danger_mag is not None
        all_ok = all_ok and ok
        print(
            f"[4. 動態注入測試] {device_id} (Class {size_class}, 門檻 {threshold}) "
            f"{'PASS' if ok else 'FAIL'}：正常均值 {normal_mag:.3f}，危險均值 {danger_mag}"
        )
    return all_ok


def test_flush_trigger(config):
    machine = config["machines"][0]

    # (a) 正常速度：應該剛好滿 1000 tick 出貨
    stop_event_a = mp.Event()
    process_a, queue_a, _ = _start_machine(machine["device_id"], machine["size_class"], config, stop_event_a)
    batch_a = queue_a.get(timeout=5)
    stop_event_a.set()
    process_a.join(timeout=2)
    normal_ok = len(batch_a["timestamps"]) == FLUSH_TICKS

    # (b) 人為降速：250ms 時間門檻應強制出貨（tick 數遠低於 1000）
    stop_event_b = mp.Event()
    process_b, queue_b, _ = _start_machine(
        machine["device_id"],
        machine["size_class"],
        config,
        stop_event_b,
        debug_extra_delay_seconds=0.001,
        debug_extra_delay_every_ticks=1,
    )
    t0 = time.perf_counter()
    batch_b = queue_b.get(timeout=5)
    elapsed = time.perf_counter() - t0
    stop_event_b.set()
    process_b.join(timeout=2)
    timeout_ok = len(batch_b["timestamps"]) < FLUSH_TICKS and elapsed < 1.0

    ok = normal_ok and timeout_ok
    print(
        f"[5. 雙緩衝出貨測試] {'PASS' if ok else 'FAIL'}："
        f"正常速度 {len(batch_a['timestamps'])} tick（應={FLUSH_TICKS}），"
        f"降速後 {len(batch_b['timestamps'])} tick / {elapsed:.3f}s（應 <{FLUSH_TICKS} tick 且提早出貨）"
    )
    return ok


def test_process_isolation(config):
    stop_event = mp.Event()
    processes = []
    for machine in config["machines"][:3]:
        process, _, _ = _start_machine(machine["device_id"], machine["size_class"], config, stop_event)
        processes.append(process)
    time.sleep(0.3)
    pids = {p.pid for p in processes}
    ok = len(pids) == len(processes) and os.getpid() not in pids
    stop_event.set()
    for process in processes:
        process.join(timeout=2)
    print(f"[6. 多進程隔離測試] {'PASS' if ok else 'FAIL'}：主進程 PID={os.getpid()}，子進程 PID={sorted(pids)}")
    return ok


def main():
    config = machine_config.load_config()
    results = {
        "格式符合度": test_format(config),
        "數據總量檢查": test_total_count(config),
        "時序連續性測試": test_continuity(config),
        "動態注入測試": test_danger_injection(config),
        "雙緩衝出貨測試": test_flush_trigger(config),
        "多進程隔離測試": test_process_isolation(config),
    }

    print("\n===== 測試總結 =====")
    all_pass = True
    for name, ok in results.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        all_pass = all_pass and ok

    raise SystemExit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
