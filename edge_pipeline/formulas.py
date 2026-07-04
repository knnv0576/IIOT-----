"""物理量模擬公式，全部向量化（一次算一批 NumPy 陣列，不逐筆呼叫）。"""

import math

import numpy as np

from constants import AC_FREQ_HZ, CURRENT_PHASES, VIBRATION_AXES

_PHASE_OFFSETS = {"A": 0.0, "B": 2 * math.pi / 3, "C": 4 * math.pi / 3}
_AXIS_OFFSETS = {"X": 0.0, "Y": 2 * math.pi / 3, "Z": 4 * math.pi / 3}
_VIBRATION_FREQ_HZ = 30.0  # 機械振動基頻（僅供模擬用，非規格值）


def generate_temperature_batch(t, setpoint, lower, upper, sigma, rng):
    """回傳長度 len(t) 的溫度陣列，漂移裁切在 [lower, upper]。"""
    values = rng.normal(setpoint, sigma, size=t.shape)
    return np.clip(values, lower, upper)


def generate_current_batch(t, setpoint_pct, upper_pct, lower_pct, rng):
    """回傳 {"A": arr, "B": arr, "C": arr}，每相各自獨立相位與雜訊。"""
    amplitude = (upper_pct - lower_pct) / 2 * 0.8  # 留一點餘裕給雜訊，避免常態超出上下限太多
    noise_sigma = (upper_pct - lower_pct) / 2 * 0.1
    result = {}
    for phase in CURRENT_PHASES:
        offset = _PHASE_OFFSETS[phase]
        noise = rng.normal(0, noise_sigma, size=t.shape)
        result[phase] = setpoint_pct + amplitude * np.sin(2 * np.pi * AC_FREQ_HZ * t + offset) + noise
    return result


def generate_vibration_batch(t, target_magnitude, rng):
    """回傳 {"X": arr, "Y": arr, "Z": arr}，三軸向量合成後平均落在 target_magnitude 附近。

    target_magnitude：期望的 sqrt(X^2+Y^2+Z^2) 平均值（正常模式傳優良區間內的值，
    危險模式傳超過危險門檻的值）。
    """
    per_axis_base = target_magnitude / math.sqrt(3)
    mod_amplitude = per_axis_base * 0.15
    noise_sigma = per_axis_base * 0.05
    result = {}
    for axis in VIBRATION_AXES:
        offset = _AXIS_OFFSETS[axis]
        noise = rng.normal(0, noise_sigma, size=t.shape)
        values = per_axis_base + mod_amplitude * np.sin(2 * np.pi * _VIBRATION_FREQ_HZ * t + offset) + noise
        result[axis] = np.clip(values, 0, None)
    return result


def vector_magnitude(vibration_batch):
    """給定 {"X":arr,"Y":arr,"Z":arr}，回傳逐筆的 sqrt(X^2+Y^2+Z^2)。"""
    return np.sqrt(
        vibration_batch["X"] ** 2 + vibration_batch["Y"] ** 2 + vibration_batch["Z"] ** 2
    )
