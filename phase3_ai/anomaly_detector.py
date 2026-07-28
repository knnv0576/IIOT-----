"""One-Class SVM 頻譜異常偵測：每個通道各自訓練一個模型（只用正常運作時的資料），
之後定期對最新窗口算特徵、餵進模型評分。

跟 rul_predictor.py 的差異：RUL 抓的是「均值本身的線性趨勢」，這裡抓的是「這個窗口的
統計/頻譜特徵組合，跟訓練時看過的正常樣子差多少」——對早期、均值還看不出明顯偏移、
但頻譜結構已經開始跑掉的異變更敏感。
"""

import os

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from spectral_features import FUNDAMENTAL_HZ, SAMPLE_RATE_HZ, extract_features, to_vector

WINDOW_SAMPLES = 4096  # 一個特徵窗口的取樣數，2kHz 下約 2.05 秒，取 2 的冪方便 FFT
TRAIN_WINDOW_SECONDS = 300  # 訓練用的歷史資料長度
MIN_TRAIN_WINDOWS = 130  # 訓練樣本數太少模型不可靠，寧可不訓練也不要訓練出爛模型（300s@2kHz 約 146 窗）
# 覆蓋度門檻：要求最近 TRAIN_WINDOW_SECONDS 秒的取樣覆蓋率。原本 0.95，但 4kHz 下 60 通道的
# 重量級訓練查詢（每通道讀 120 萬列）會跟串流寫入搶 I/O、把寫入率拖到 ~40%，覆蓋度掉到 56%
# 就永遠達不到 95%、訓練卡死在 24/60（死結）。那是 4kHz 特有問題；降回 2kHz 後 CPU 有餘裕、
# 寫入穩定，覆蓋度自然夠，故預設回到 0.95（拒絕真正有 gap 的資料，訓練品質較好）。可用環境變數覆寫。
COVERAGE_MIN_RATIO = float(os.environ.get("COVERAGE_MIN_RATIO", "0.95"))

# 實測發現：訓練窗口只有 ~56 個（60 秒）時，即使是完全正常、held-out 的資料，
# 誤報率高達 43.7%（gamma="scale" 在小樣本下把邊界fit得太緊，過擬合到訓練噪音上，
# 尤其是 thd_ratio 這種本身取樣變異就很大 CV約31% 的頻譜特徵）。
# 拉長訓練視窗到 300 秒（~292 個窗口）後，用完全沒參與訓練的 held-out 資料驗證：
# gamma="scale" 誤報率降到 8~20%，改成固定 gamma=0.02（更平滑的邊界）後穩定落在
# 3~6%，非常接近 nu=0.05 的設計目標。三個不同機台/通道交叉驗證過，不是單一巧合。
ANOMALY_SVM_GAMMA = 0.02

# 判定門檻不用 SVM 預設的 decision_function < 0，而是要求分數低於這個邊際才算異常。
# 依據（2kHz 實測）：正常資料的分數幾乎全部落在 -0.7 以上（500+ 筆線上評分最差 -0.72、
# 99% 在 -0.37 以上），而人工注入諧波異常（即使弱到只有基頻 2.5% 的振幅）分數中位數
# 就低於 -3——中間有超過 2 個單位的分離帶。門檻取 -1.0 可把正常資料殘餘誤報
#（predict() 邊界下約 5~8%）壓到 ~0%，同時對真異常的偵測率完全不受影響。
ANOMALY_SCORE_MARGIN = -1.0


def fetch_raw_series(conn, device_id: str, sensor_id: str, seconds: float) -> list[float]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT value FROM sensor_readings
            WHERE device_id = %s AND sensor_id = %s
              AND ts > now() - (%s * interval '1 second')
            ORDER BY ts
            """,
            (device_id, sensor_id, seconds),
        )
        return [row[0] for row in cur.fetchall()]


def _chunk_windows(series: list[float], window_samples: int) -> list[list[float]]:
    n_windows = len(series) // window_samples
    return [series[i * window_samples : (i + 1) * window_samples] for i in range(n_windows)]


def train_model(conn, device_id: str, sensor_id: str, fundamental_hz: float = FUNDAMENTAL_HZ) -> dict | None:
    """用最近 TRAIN_WINDOW_SECONDS 秒的原始資料訓練一個通道專屬的 One-Class SVM。
    資料不足時回傳 None（呼叫端應該之後再重試，而不是硬訓練一個不可靠的模型）。

    fundamental_hz：這個通道的基頻，電流用 60Hz、振動用 30Hz（模擬器 _VIBRATION_FREQ_HZ）。
    模型 bundle 會把 fundamental_hz 一起存起來，評分時沿用同一個值，確保特徵一致。
    """
    series = fetch_raw_series(conn, device_id, sensor_id, TRAIN_WINDOW_SECONDS)
    # 覆蓋度檢查：實測發現服務啟動初期，資料剛累積到窗口數下限就搶先訓練的話，
    # 訓練集會涵蓋暖機期/寫入中斷造成的不連續段——chunk 切窗時把「有時間跳躍的樣本」
    # 串成一個窗口，FFT 特徵整組被污染，之後對正常資料的誤報率會飆到 40% 以上。
    # 要求最近 TRAIN_WINDOW_SECONDS 秒內取樣覆蓋率 >= 95% 才訓練，寧可多等幾輪。
    if len(series) < int(TRAIN_WINDOW_SECONDS * SAMPLE_RATE_HZ * COVERAGE_MIN_RATIO):
        return None
    windows = _chunk_windows(series, WINDOW_SAMPLES)
    if len(windows) < MIN_TRAIN_WINDOWS:
        return None

    feature_vectors = np.array(
        [to_vector(extract_features(np.array(w), fundamental_hz=fundamental_hz)) for w in windows]
    )

    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_vectors)

    model = OneClassSVM(kernel="rbf", nu=0.05, gamma=ANOMALY_SVM_GAMMA)
    model.fit(scaled)

    return {"model": model, "scaler": scaler, "n_train_windows": len(windows), "fundamental_hz": fundamental_hz}


def score_latest_window(conn, device_id: str, sensor_id: str, model_bundle: dict):
    """對最新一個窗口的原始資料評分。回傳 (score, is_anomaly, features)，資料不足回傳 None。
    score 是 One-Class SVM 的 decision_function 值，越負代表跟訓練時看過的正常樣子差越多。
    """
    window_seconds = WINDOW_SAMPLES / SAMPLE_RATE_HZ + 1.0  # 多抓一點餘裕，避免剛好卡在批次邊界湊不滿
    series = fetch_raw_series(conn, device_id, sensor_id, window_seconds)
    if len(series) < WINDOW_SAMPLES:
        return None

    window = np.array(series[-WINDOW_SAMPLES:])  # 只取最新的 WINDOW_SAMPLES 筆，確保長度一致
    features = extract_features(window, fundamental_hz=model_bundle.get("fundamental_hz", FUNDAMENTAL_HZ))
    vector = np.array([to_vector(features)])
    scaled = model_bundle["scaler"].transform(vector)

    score = float(model_bundle["model"].decision_function(scaled)[0])
    is_anomaly = score < ANOMALY_SCORE_MARGIN
    return score, is_anomaly, features


# ── 線上 CUSUM 變化點偵測 ─────────────────────────────────────────────
# 動機：固定門檻 -1.0 在 17 顆真實軸承（PHM2012）上誤報離散度收斂不了（0~63%），因為
# One-Class SVM 分數的絕對值在不同通道間意義不一致。CUSUM 改累積「分數持續系統性下滑」的
# 部分，對每個通道自己的噪聲水平自適應。跟 RUL 層看均值斜率同哲學：看趨勢/累積偏移，不看瞬時值。
# 驗證狀態：CUSUM 已在「模擬器漸進劣化」的 SCT-05 端到端實驗中實測有效（見 main.py 說明）；
# 17 顆 PHM2012 軸承的離線 CUSUM 對照腳本備妥於 validation/phm2012_cusum_validation.py，但尚未實跑。
CUSUM_WARMUP_SAMPLES = 20  # 模型訓練後，先收集這麼多筆「健康分數」估基準（每輪一筆）才啟動 CUSUM
CUSUM_SLACK_K = 0.5        # slack = CUSUM_SLACK_K * sigma0，吸收正常隨機波動，只累積系統性劣化
CUSUM_H_MULT = 8.0         # 決策門檻 h = CUSUM_H_MULT * sigma0（實驗中 3~12 都能 17/17 偵測，取偏保守值少誤報）
CUSUM_SIGMA_FLOOR = 0.05   # sigma0 下限，避免某通道健康分數過度集中導致 h 太小、CUSUM 過度敏感


class CusumTracker:
    """對單一通道的異常分數時序做單邊 CUSUM，偵測「分數持續往下掉」的漸進劣化。

    校準期：模型訓練後先收集 CUSUM_WARMUP_SAMPLES 筆分數（此時資料假設仍健康）估基準
    mu0/sigma0，再開始累積——這對應離線實驗裡「用沒參與訓練的 held-out 健康資料校準」。
    """

    def __init__(self) -> None:
        self._warmup: list[float] = []
        self.mu0: float | None = None
        self.sigma0: float | None = None
        self.slack: float | None = None
        self.h: float | None = None
        self.value: float = 0.0

    @property
    def calibrated(self) -> bool:
        return self.mu0 is not None

    def update(self, score: float):
        """吃一筆新分數，回傳 (cusum 值, 是否持續劣化)。校準期間回傳 (None, False)。"""
        if not self.calibrated:
            self._warmup.append(score)
            if len(self._warmup) >= CUSUM_WARMUP_SAMPLES:
                self.mu0 = float(np.mean(self._warmup))
                self.sigma0 = max(float(np.std(self._warmup)), CUSUM_SIGMA_FLOOR)
                self.slack = CUSUM_SLACK_K * self.sigma0
                self.h = CUSUM_H_MULT * self.sigma0
            return None, False

        # score 越低越異常；(mu0 - score) 在劣化時為正 → 累積，slack 把正常波動漏掉。
        self.value = max(0.0, self.value + (self.mu0 - score - self.slack))
        return self.value, self.value > self.h
