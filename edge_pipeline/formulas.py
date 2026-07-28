"""物理量模擬公式，全部向量化（一次算一批 NumPy 陣列，不逐筆呼叫）。"""

import math

import numpy as np

from constants import (
    AC_FREQ_HZ,
    CURRENT_PHASES,
    DEGRADE_HARMONIC_MULT,
    DEGRADE_MAX_HARMONIC_RATIO,
    DEGRADE_VIBRATION_AMPLITUDE_GROWTH,
    VIBRATION_AXES,
)

_PHASE_OFFSETS = {"A": 0.0, "B": 2 * math.pi / 3, "C": 4 * math.pi / 3}
_AXIS_OFFSETS = {"X": 0.0, "Y": 2 * math.pi / 3, "Z": 4 * math.pi / 3}
_VIBRATION_FREQ_HZ = 30.0  # 機械振動基頻（僅供模擬用，非規格值）


def generate_temperature_batch(t, setpoint, lower, upper, sigma, rng):
    """回傳長度 len(t) 的溫度陣列，漂移裁切在 [lower, upper]。"""
    values = rng.normal(setpoint, sigma, size=t.shape)
    return np.clip(values, lower, upper)


def generate_current_batch(t, setpoint_pct, upper_pct, lower_pct, rng, degradation_level=0.0):
    """回傳 {"A": arr, "B": arr, "C": arr}，每相各自獨立相位與雜訊。

    degradation_level（0~1）：漸進劣化程度，>0 時在 3 次諧波（180Hz）上疊加一個成長中的
    成分，模擬早期故障的頻譜邊帶。整體振幅幾乎不變（諧波能量小），刻意讓「只看均值/振幅」
    看不出來、只有頻譜分析抓得到——這正是頻譜 SVM + CUSUM 要驗證的早期偵測能力。
    """
    amplitude = (upper_pct - lower_pct) / 2 * 0.8  # 留一點餘裕給雜訊，避免常態超出上下限太多
    noise_sigma = (upper_pct - lower_pct) / 2 * 0.1
    result = {}
    for phase in CURRENT_PHASES:
        offset = _PHASE_OFFSETS[phase]
        noise = rng.normal(0, noise_sigma, size=t.shape)
        values = setpoint_pct + amplitude * np.sin(2 * np.pi * AC_FREQ_HZ * t + offset) + noise
        if degradation_level > 0.0:
            harmonic_amp = amplitude * DEGRADE_MAX_HARMONIC_RATIO * degradation_level
            values = values + harmonic_amp * np.sin(2 * np.pi * AC_FREQ_HZ * DEGRADE_HARMONIC_MULT * t + offset)
        result[phase] = values
    return result


def generate_vibration_batch(t, target_magnitude, rng, degradation_level=0.0):
    """回傳 {"X": arr, "Y": arr, "Z": arr}，三軸向量合成後平均落在 target_magnitude 附近。

    target_magnitude：期望的 sqrt(X^2+Y^2+Z^2) 平均值（正常模式傳優良區間內的值，
    危險模式傳超過危險門檻的值）。
    degradation_level（0~1）：漸進劣化，>0 時在 3 次諧波（90Hz）上疊加成長中的成分，
    以 30Hz 調變振幅為基準，模擬軸承早期故障的頻譜邊帶（見 generate_current_batch 說明）。
    """
    per_axis_base = target_magnitude / math.sqrt(3)
    mod_amplitude = per_axis_base * 0.15
    noise_sigma = per_axis_base * 0.05
    result = {}
    for axis in VIBRATION_AXES:
        offset = _AXIS_OFFSETS[axis]
        noise = rng.normal(0, noise_sigma, size=t.shape)
        values = per_axis_base + mod_amplitude * np.sin(2 * np.pi * _VIBRATION_FREQ_HZ * t + offset) + noise
        if degradation_level > 0.0:
            harmonic_amp = mod_amplitude * DEGRADE_MAX_HARMONIC_RATIO * degradation_level
            values = values + harmonic_amp * np.sin(
                2 * np.pi * _VIBRATION_FREQ_HZ * DEGRADE_HARMONIC_MULT * t + offset
            )
            # 整體振幅隨劣化成長（模擬真實軸承 RMS 暴增），讓振幅型的 ML RUL 也抓得到
            values = values * (1.0 + degradation_level * DEGRADE_VIBRATION_AMPLITUDE_GROWTH)
        result[axis] = np.clip(values, 0, None)
    return result


def vector_magnitude(vibration_batch):
    """給定 {"X":arr,"Y":arr,"Z":arr}，回傳逐筆的 sqrt(X^2+Y^2+Z^2)。"""
    return np.sqrt(
        vibration_batch["X"] ** 2 + vibration_batch["Y"] ** 2 + vibration_batch["Z"] ** 2
    )
