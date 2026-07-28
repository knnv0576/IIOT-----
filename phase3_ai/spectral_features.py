"""從一段原始電流波形窗口，抽取時域統計特徵 + 頻域（FFT）特徵。

給 One-Class SVM 用來判斷「這個窗口的統計/頻譜特徵，跟正常運作時的樣子像不像」——
真正的馬達/軸承早期故障，常常會先在電流頻譜裡冒出額外的諧波成分（邊帶頻率），
但均值/振幅本身還看不太出來，這就是為什麼只看均值+趨勢（RUL）會太晚才發現。
"""

import numpy as np
from scipy.stats import kurtosis, skew

# 需與 edge_pipeline/constants.py 的 SAMPLE_RATE_HZ 一致（兩個服務是分開的 Docker context
# 沒有共用模組，跟 main.py 的電流門檻一樣是刻意接受的重複）。
SAMPLE_RATE_HZ = 2000.0
FUNDAMENTAL_HZ = 60.0
HARMONIC_MULTIPLES = (2, 3, 4)  # 抓 2/3/4 次諧波（120/180/240Hz）
BAND_HALF_WIDTH_HZ = 3.0  # 頻帶容許誤差範圍

FEATURE_NAMES = ["mean", "std", "skewness", "kurtosis", "thd_ratio", "noise_floor_ratio"]


def extract_features(
    values: np.ndarray,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
    fundamental_hz: float = FUNDAMENTAL_HZ,
    band_half_width_hz: float = BAND_HALF_WIDTH_HZ,
) -> dict:
    """values：一個窗口的原始感測數值，依時間排序、等間隔取樣。

    預設參數對應正式服務的電流通道（2kHz、60Hz 基頻）；可傳入其他取樣率/基頻，
    讓同一套特徵邏輯用在不同來源的波形上（例如公開軸承資料集的重放驗證）。
    """
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    std = float(values.std())
    skewness = float(skew(values)) if std > 1e-12 else 0.0
    kurt = float(kurtosis(values)) if std > 1e-12 else 0.0
    # 時域健康指標：ML RUL 模型（RF）吃這幾個。rms=均方根、crest=峰值/rms（衝擊性）。
    rms = float(np.sqrt(np.mean(values**2)))
    crest = float(np.max(np.abs(values)) / rms) if rms > 1e-12 else 0.0

    n = len(values)
    centered = values - mean  # 先去掉直流成分，FFT 只看交流的頻譜結構
    magnitudes = np.abs(np.fft.rfft(centered))
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)

    def band_energy(center_hz: float) -> float:
        mask = (freqs >= center_hz - band_half_width_hz) & (freqs <= center_hz + band_half_width_hz)
        return float(np.sum(magnitudes[mask] ** 2))

    fundamental_energy = band_energy(fundamental_hz)
    harmonic_energy = sum(band_energy(fundamental_hz * k) for k in HARMONIC_MULTIPLES)
    total_energy = float(np.sum(magnitudes**2))

    # THD-like 比值：諧波能量相對基頻能量的比例，健康馬達應該很接近 0。
    thd_ratio = harmonic_energy / fundamental_energy if fundamental_energy > 1e-9 else 0.0
    # 扣掉基頻跟已知諧波之後，剩下的頻譜能量占比——用來抓「不在預期頻率上的」異常成分。
    noise_floor_ratio = (
        (total_energy - fundamental_energy - harmonic_energy) / total_energy if total_energy > 1e-9 else 0.0
    )

    return {
        "mean": mean,
        "std": std,
        "skewness": skewness,
        "kurtosis": kurt,
        "thd_ratio": thd_ratio,
        "noise_floor_ratio": noise_floor_ratio,
        "rms": rms,
        "crest": crest,
    }


def to_vector(features: dict) -> list:
    return [features[name] for name in FEATURE_NAMES]
