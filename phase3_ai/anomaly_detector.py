"""One-Class SVM 頻譜異常偵測：每個通道各自訓練一個模型（只用正常運作時的資料），
之後定期對最新窗口算特徵、餵進模型評分。

跟 main.py 的電流 RUL 的差異：RUL 抓的是「均值本身的線性趨勢」，這裡抓的是「這個窗口的
統計/頻譜特徵組合，跟訓練時看過的正常樣子差多少」——對早期、均值還看不出明顯偏移、
但頻譜結構已經開始跑掉的異變更敏感。

模型的存檔、載入與指紋（fingerprint）在 model_store.py，這裡只負責演算法本身。
"""

import os

import numpy as np
from scipy.stats import norm as _normal
from scipy.stats import t as _student_t
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


def extract_training_features(conn, device_id: str, sensor_id: str,
                              fundamental_hz: float = FUNDAMENTAL_HZ):
    """從 DB 取最近 TRAIN_WINDOW_SECONDS 秒的原始波形，算出訓練用的特徵矩陣。
    回傳 shape=(n_windows, n_features) 的 ndarray；資料量或覆蓋度不足時回傳 None。

    抽出來當獨立函式的原因：讓「特徵」可以被存檔／版控／事後重現訓練。原始波形在
    2kHz × 70 感測器下是 ~1GB/5分鐘、保留政策只留 20 分鐘，不可能長期保存；
    但特徵矩陣整個機隊只有幾百 KB，存得起。見 trainer.py 的 --save-features。
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

    return np.array(
        [to_vector(extract_features(np.array(w), fundamental_hz=fundamental_hz)) for w in windows]
    )


def fit_model_from_features(feature_vectors, fundamental_hz: float = FUNDAMENTAL_HZ) -> dict:
    """由特徵矩陣擬合 One-Class SVM，回傳模型 bundle。

    這一段是純計算、不碰 DB——所以「從存檔的特徵重訓」跟「從即時資料重訓」走的是
    完全相同的程式碼路徑，結果（含模型指紋）可完全重現。
    """
    feature_vectors = np.asarray(feature_vectors, dtype=float)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_vectors)

    model = OneClassSVM(kernel="rbf", nu=0.05, gamma=ANOMALY_SVM_GAMMA)
    model.fit(scaled)

    return {"model": model, "scaler": scaler,
            "n_train_windows": len(feature_vectors), "fundamental_hz": fundamental_hz}


def train_model(conn, device_id: str, sensor_id: str, fundamental_hz: float = FUNDAMENTAL_HZ) -> dict | None:
    """用最近 TRAIN_WINDOW_SECONDS 秒的原始資料訓練一個通道專屬的 One-Class SVM。
    資料不足時回傳 None（呼叫端應該之後再重試，而不是硬訓練一個不可靠的模型）。

    fundamental_hz：這個通道的基頻，電流用 60Hz、振動用 30Hz（模擬器 _VIBRATION_FREQ_HZ）。
    模型 bundle 會把 fundamental_hz 一起存起來，評分時沿用同一個值，確保特徵一致。
    """
    feature_vectors = extract_training_features(conn, device_id, sensor_id, fundamental_hz)
    if feature_vectors is None:
        return None
    return fit_model_from_features(feature_vectors, fundamental_hz)


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


# ── 線上 CUSUM 變化點偵測（Self-Starting，Hawkins 1987）─────────────────────
# 動機：固定門檻 -1.0 在 17 顆真實軸承（PHM2012）上誤報離散度收斂不了（0~63%），因為
# One-Class SVM 分數的絕對值在不同通道間意義不一致。CUSUM 改累積「分數持續系統性下滑」的
# 部分，對每個通道自己的噪聲水平自適應。跟 RUL 層看均值斜率同哲學：看趨勢/累積偏移，不看瞬時值。
#
# ── 為什麼改成 self-starting（2026-08 實測教訓）────────────────────────────
# 舊版是「固定暖機 20 筆估出 mu0/sigma0 後凍結」的傳統做法，實測會失控：
#   增量 = mu0 - score - slack，其中 slack = 0.5*sigma0
#   只要 mu0 估高了超過 slack，這個增量長期為正 → CUSUM 單調累積 → 必定誤報，且不會自我修復
#   （CUSUM 跨過門檻後不歸零）。實測告警機台數隨時間 1→2→3→4 持續上升。
# 根因是樣本數：20 筆估均值的 95% 誤差界是 2σ/√20 ≈ 0.45σ，而容錯只有 slack = 0.5σ——
# **估計誤差幾乎用光整個容錯**。控制圖文獻對此有明確結論：估計參數的管制圖需要數百個
# Phase I 樣本才穩定（m=50 時 80% 使用者拿到的 ARL 落在 87.6~290.4，名目 200）。
#
# Self-starting 的做法（Hawkins & Olwell）不預先估參數，而是把每個新觀測用「它之前所有
# 觀測」的 running mean/SD 標準化：
#     T_n = (x_n - x̄_{n-1}) / s_{n-1} * sqrt((n-1)/n)   ~ t(n-2)
#     U_n = Φ⁻¹( F_{t,n-2}(T_n) )                        ~ N(0,1)  【在製程穩定時嚴格成立】
# 關鍵性質：U_n 是嚴格的 iid 標準常態，**與未知的 μ、σ 無關**。所以：
#   - 沒有參數估計誤差，門檻 h 從一開始就有它設計上的 ARL
#   - 不需要暖機盲窗（第 3 筆就能開始判斷，實務上從第 5 筆起）
# 代價：早期樣本少時檢定力較弱；且若劣化在最初幾筆就已存在，會被當成「正常基準」吸收
#（self-starting 的已知限制，無法避免）。
#
# **注意：這裡不會追蹤長期漂移。** running mean/SD 用 Welford 對「服務啟動以來的全部歷史」
# 累計，第 n 筆對基準的影響只有 1/n。剛啟動時基準還會跟著資料移動；跑了幾百筆
#（每筆 30 秒 → 幾小時）之後，基準幾乎就固定在這段期間的平均值。
# 所以如果真實機台的分數因為季節溫度、負載或正常磨耗而緩慢漂移，漂移量大過 slack
#（0.5σ）就會慢慢累積成告警——這時告警會是「真的偏離了啟動時的狀態」，但不一定代表故障。
# 目前的處理方式是：重啟 ai_service（或重新 commissioning 訓練）就會重新建立基準。
# 若要自動適應漂移，可以把 Welford 換成指數遺忘或固定長度滑動視窗，但那樣 U_n 就只是
# 近似 N(0,1)、h 的 ARL 也要重新驗證（validation/ 的誤報重放），不能直接沿用 h=7。
CUSUM_SLACK_K = 0.5  # 標準化後的 slack（單位是 σ）。CUSUM 只累積超過這個幅度的系統性下滑。

# 決策門檻，單位是「標準化後的 σ」——因為 U_n 已經是 N(0,1)，h 是純數字，不再乘 sigma0。
# 取 7 的依據（ISA-18.2）：操作員感受到的是整個機隊，機隊告警率 = 單通道率 × 60 通道。
# k=0.5 時各 h 的理論 ARL（Siegmund 近似）：5σ≈938、6σ≈2573、7σ≈7020、8σ≈19112 樣本。
# 每筆間隔 30s、60 通道 → 機隊告警率 = 60 / (ARL × 30s)。ISA-18.2 要求穩態 < 6 則/小時：
#   h=5 → 8.0 則/小時（超標）、h=6 → 2.8、h=7 → 1.0（約 6 倍餘裕）。
# **與舊版最大的差別**：舊版這個 ARL 是「理論值」，實際因估計誤差而崩到 448；
# self-starting 讓理論值就是實際值，所以 h=7 是真的 h=7。
CUSUM_H = float(os.environ.get("CUSUM_H", "7.0"))

# 開始判斷前至少要看過幾筆。理論上第 3 筆（df=1）起 U_n 就已是嚴格 N(0,1)、誤報率受控，
# 但那時檢定力極低；取 5 是「盡快上線」與「基準不要只靠 2 筆」的折衷（約 2.5 分鐘，
# 相較舊版 20 筆暖機的 10 分鐘大幅縮短）。
CUSUM_SELFSTART_MIN_N = int(os.environ.get("CUSUM_SELFSTART_MIN_N", "5"))

# 機率截斷：F_t(T) 逼近 0 或 1 時 Φ⁻¹ 會發散。截到 1e-6 → |U| ≤ 4.75，
# 順帶得到一個好性質：**任何單一離群分數對 CUSUM 的貢獻上限是 4.75-0.5=4.25 < h=7**，
# 所以孤立的瞬態不可能單獨觸發告警，至少要連續兩筆。這是刻意保留的抗突波設計。
CUSUM_P_CLIP = 1e-6

# 相容用：main.py 存 cusum_baselines 時記錄的 n_warmup 欄位。self-starting 沒有暖機期，
# 這裡就是「開始判斷所需的最少觀測數」。
CUSUM_WARMUP_SAMPLES = CUSUM_SELFSTART_MIN_N


class CusumTracker:
    """單一通道的 self-starting 單邊 CUSUM，偵測「異常分數持續往下掉」的漸進劣化。

    對外介面與舊版相同（update / calibrated / mu0 / sigma0 / value / h ...），
    所以 main.py 不需要改動。內部改用 running 統計量 + t→常態轉換，不再有「暖機 20 筆就凍結」
    的基線；基準會持續納入新觀測，但影響力隨樣本數遞減（見上方「不會追蹤長期漂移」的說明）。
    """

    def __init__(self) -> None:
        # Welford 線上統計：只需 O(1) 記憶體，且數值穩定（不用累加平方和）
        self._n: int = 0        # 已納入基準的觀測數
        self._mean: float = 0.0
        self._m2: float = 0.0   # 離差平方和，變異數 = _m2 / (n-1)

        self.value: float = 0.0
        self.h: float = CUSUM_H
        self.slack: float = CUSUM_SLACK_K

        # 告警後凍結基準更新：否則持續的劣化會被 running 統計吸收成「新的正常」，
        # 導致 CUSUM 自己把警報消掉（self-starting 的標準對策）。回到 value=0 才解凍。
        self._frozen: bool = False

        # ── 以下欄位純為相容 main.py 既有的記錄/存檔邏輯 ──
        self.low_confidence: bool = False
        self.rejected_frac: float = 0.0
        self.source: str = "self-starting"
        self.calib_generation: int = 0
        self.reval_failed: bool = False

    # ── 相容屬性：main.py 會讀這兩個值存進 cusum_baselines 當診斷用 ──
    @property
    def mu0(self) -> "float | None":
        return self._mean if self._n >= 1 else None

    @property
    def sigma0(self) -> "float | None":
        if self._n < 2:
            return None
        return float(np.sqrt(self._m2 / (self._n - 1)))

    @property
    def calibrated(self) -> bool:
        """已看過足夠觀測、可以開始判斷。self-starting 下這只需要幾筆，不是 10 分鐘暖機。"""
        return self._n >= CUSUM_SELFSTART_MIN_N

    def _push(self, x: float) -> None:
        """Welford 線上更新 running mean / 離差平方和。"""
        self._n += 1
        delta = x - self._mean
        self._mean += delta / self._n
        self._m2 += delta * (x - self._mean)

    def load_baseline(self, mu0: float, sigma0: float, low_confidence: bool) -> None:
        """用先前存下的基線當「起始基準」，讓 running 統計不必從零開始。

        self-starting 本來就不需要這個；提供它只是為了在「確定同一個模型、且確信那段資料
        代表健康」時，能少花幾分鐘就達到較好的檢定力。等效樣本數刻意只給
        CUSUM_SELFSTART_MIN_N（而非原本的暖機筆數），讓後續真實觀測能很快把基準拉回正確位置——
        載入的值只是起點，不是凍結的真理。這也是它跟舊版最本質的差別。
        """
        n_eff = max(2, CUSUM_SELFSTART_MIN_N)
        self._n = n_eff
        self._mean = float(mu0)
        s = max(float(sigma0), 1e-9)
        self._m2 = (s ** 2) * (n_eff - 1)
        self.value = 0.0
        self._frozen = False
        self.low_confidence = bool(low_confidence)
        self.source = "seeded"
        self.calib_generation += 1

    def update(self, score: float):
        """吃一筆新分數，回傳 (cusum 值, 是否持續劣化)；觀測還不夠時回傳 (None, False)。"""
        x = float(score)
        n_prev = self._n

        # 需要至少 2 筆先驗觀測才能算 s_{n-1}（自由度 n-2 >= 1）
        if n_prev < 2:
            self._push(x)
            return None, False

        mean_prev = self._mean
        var_prev = self._m2 / (n_prev - 1)
        if var_prev <= 0.0:
            # 退化情形：先前觀測完全相同（例如分數被量化、感測器卡住）。
            # 無法標準化，先只納入基準，不做判斷。
            self._push(x)
            return None, False
        sd_prev = float(np.sqrt(var_prev))

        # Hawkins self-starting 標準化：T ~ t(n_prev - 1)
        n_now = n_prev + 1
        t_stat = (x - mean_prev) / sd_prev * float(np.sqrt(n_prev / n_now))
        df = n_prev - 1
        p = float(_student_t.cdf(t_stat, df))
        p = min(max(p, CUSUM_P_CLIP), 1.0 - CUSUM_P_CLIP)
        u = float(_normal.ppf(p))  # 製程穩定時 u ~ N(0,1)，且與未知的 μ、σ 無關

        # 告警期間凍結基準（不 _push），避免劣化被當成新的正常而把警報自己消掉
        if not self._frozen:
            self._push(x)

        if not self.calibrated:
            return None, False

        # 單邊 CUSUM：分數越低越異常 → 累積 (-u)，slack 濾掉正常波動
        self.value = max(0.0, self.value + (-u - self.slack))
        alarming = self.value > self.h

        if alarming:
            self._frozen = True
        elif self.value == 0.0 and self._frozen:
            self._frozen = False  # 完全回復才解凍，基準重新開始納入新觀測

        return self.value, alarming
