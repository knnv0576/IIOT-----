"""單一機台的 multiprocessing.Process 主迴圈：4kHz tick、雙緩衝、危險模式讀取。"""

import time

import numpy as np

import machine_config
from constants import FLUSH_SECONDS, FLUSH_TICKS, SAMPLE_INTERVAL, TEMP_TICK_INTERVAL
from double_buffer import LocalDoubleBuffer
from formulas import generate_current_batch, generate_temperature_batch, generate_vibration_batch


def run_machine(
    device_id: str,
    size_class: str,
    config: dict,
    queue,
    danger_flag,
    stop_event,
    max_batches: int | None = None,
    debug_extra_delay_seconds: float = 0.0,
    debug_extra_delay_every_ticks: int = 0,
):
    """Process target。debug_extra_delay_* 僅供自我測試模擬降速使用，正式運作時保持預設值 0。"""

    rng = np.random.default_rng()
    buffer = LocalDoubleBuffer(device_id)

    temp_cfg = config["temperature"]
    current_cfg = config["current"]
    temp_sigma = (temp_cfg["upper"] - temp_cfg["lower"]) / 6

    good_upper = machine_config.good_zone_upper(config, size_class)
    danger_limit = machine_config.danger_threshold(config, size_class)
    normal_target_magnitude = good_upper * 0.6
    danger_target_magnitude = danger_limit * 1.3

    wall_start = time.time()
    perf_start = time.perf_counter()
    tick = 0
    batches_sent = 0

    while not stop_event.is_set():
        n0 = tick
        t_array = (n0 + np.arange(FLUSH_TICKS)) * SAMPLE_INTERVAL
        timestamps = wall_start + t_array

        is_danger = bool(danger_flag.value)
        target_magnitude = danger_target_magnitude if is_danger else normal_target_magnitude

        current_batch = generate_current_batch(
            t_array, current_cfg["setpoint"], current_cfg["upper"], current_cfg["lower"], rng
        )
        vibration_batch = generate_vibration_batch(t_array, target_magnitude, rng)

        buffer.load_batch(timestamps, current_batch, vibration_batch)
        last_flush_perf = time.perf_counter()

        while not buffer.full and not stop_event.is_set():
            target_perf = perf_start + tick * SAMPLE_INTERVAL
            now = time.perf_counter()
            if target_perf > now:
                time.sleep(target_perf - now)

            if debug_extra_delay_every_ticks and tick % debug_extra_delay_every_ticks == 0:
                time.sleep(debug_extra_delay_seconds)

            buffer.advance()
            tick += 1

            if tick % TEMP_TICK_INTERVAL == 0:
                temp_value = generate_temperature_batch(
                    np.array([tick * SAMPLE_INTERVAL]),
                    temp_cfg["setpoint"],
                    temp_cfg["lower"],
                    temp_cfg["upper"],
                    temp_sigma,
                    rng,
                )[0]
                buffer.add_temperature(wall_start + tick * SAMPLE_INTERVAL, temp_value)

            now = time.perf_counter()
            if buffer.full or (now - last_flush_perf) >= FLUSH_SECONDS:
                queue.put(buffer.flush())
                batches_sent += 1
                last_flush_perf = now
                break

        if max_batches is not None and batches_sent >= max_batches:
            break
