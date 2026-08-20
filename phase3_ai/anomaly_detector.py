"""One-Class SVM 頻譜異常偵測：每個通道各自訓練一個模型（只用正常運作時的資料），
之後定期對最新窗口算特徵、餵進模型評分。

跟 rul_predictor.py 的差異：RUL 抓的是「均值本身的線性趨勢」，這裡抓的是「這個窗口的
統計/頻譜特徵組合，跟訓練時看過的正常樣子差多少」——對早期、均值還看不出明顯偏移、
但頻譜結構已經開始跑掉的異變更敏感。
"""

import hashlib
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


def model_fingerprint(model_bundle: dict) -> str:
    """算出這個訓練好的模型的指紋，用來當 CUSUM 基線快取鍵的一部分。

    為什麼需要（實測踩到的坑）：健康分數的分布是「機台 × **該次訓練出來的模型**」的屬性，
    不是機台本身的屬性。原本 cusum_baselines 只用 (device_id, sensor_id) 當鍵，等於快取鍵不完整；
    而 SVM 每次服務啟動都會重訓，取回的 mu0/sigma0 其實屬於另一個模型——實測 sigma0 被低估
    1.5~3 倍，導致 slack 與門檻 h 一起縮水，健康通道的隨機遊走輕易跨過 h 而大量誤報。
    把指紋納入鍵之後，模型不同就不會取到舊基線（自然退回現場暖機，正確行為）；
    等日後模型本身也持久化了，指紋會相符，基線快取就會自動開始生效，不需要再改這裡。
    """
    model, scaler = model_bundle["model"], model_bundle["scaler"]
    h = hashlib.sha256()
    for arr in (model.dual_coef_, model.support_vectors_, np.atleast_1d(model.intercept_),
                scaler.mean_, scaler.scale_):
        h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    return h.hexdigest()[:16]


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

# 決策門檻 h 改為「隨校準後時間幾何衰減、但有地板」而非固定值。
# 動機：固定 h=8*sigma0 偏保守，緩慢劣化要拖到接近末期 value 才跨得過去（onset 太晚才亮）；
# 而且暖機期若碰到吵的一段、sigma0 被高估，h 在絕對值上過大會「永不觸發」（V05-X 就是這樣，
# 才臨時加了 thd-floor 補丁）。改成 h 從 8*sigma0 起、每過一個衰減週期 *0.8、滑到地板 5*sigma0 就停住。
# 地板取 7*sigma0 的依據（2026-08 依 ISA-18.2 修正，原本訂 5σ 是錯的）：
# 偵測力上 3~12σ 都能 17/17 命中，但「誤報率」對 h 極敏感——用 Siegmund 近似（k=0.5），單通道假警
# 平均間隔 ARL 每降 1σ 約縮 3 倍。原本只看「單通道」ARL 覺得 5σ 還算安靜，但**操作員感受到的是整個
# 機隊**：60 通道（6/台×10 台）同時跑，機隊告警率 = 單通道率 × 60。用 validation/cusum_false_alarm_replay.py
# 實測換算：5σ→約 21 則/小時、8σ→約 3 則/小時。ISA-18.2 要求穩態下每位操作員 < 6 則/小時
#（>30 則/小時視為告警洪水），故 5σ 嚴重超標、7σ 才落在合格帶內。
# 註：起始 8σ 經一個衰減週期 ×0.8=6.4 已低於地板，故實務上是「8σ 跑一個週期後固定在 7σ」——
# 仍保有約 7 倍的 ARL 靈敏度改善（防末期遲鈍的原始動機），只是不再滑到誤報懸崖。
CUSUM_H_MULT = 8.0         # 校準後的「起始」門檻倍率：h_start = CUSUM_H_MULT * sigma0
CUSUM_H_FLOOR_MULT = float(os.environ.get("CUSUM_H_FLOOR_MULT", "7.0"))  # 衰減下限 h_floor = 此值 * sigma0
CUSUM_DECAY_FACTOR = 0.8   # 每過一個衰減週期，h *= 此值（幾何衰減）
# 每這麼多筆「校準後樣本」衰減一次。CUSUM 每輪一筆、每輪 ANOMALY_CHECK_INTERVAL≈30s，
# 故 70 筆≈35 分；8σ→6.4→5.12→(clamp)5σ 共約 1.75 小時滑到地板。用樣本數而非牆鐘，重放測試才好重現。
CUSUM_DECAY_PERIOD_SAMPLES = int(os.environ.get("CUSUM_DECAY_PERIOD_SAMPLES", "70"))
CUSUM_SIGMA_FLOOR = 0.05   # sigma0 下限，避免某通道健康分數過度集中導致 h 太小、CUSUM 過度敏感

# 方案②+① robust 校準參數（見 CusumTracker._calibrate）。動機：20 筆暖機用 mean/std 估基線時，
# 一顆瞬態離群（負載尖峰、單筆壞值、旁邊機台干擾）就能把 sigma0 灌歪，而 ARL 對 h/σ 指數敏感 →
# 假警放大約 40 倍（實測，V05-X 永不觸發就是這樣）。改成「先 median/MAD 擋離群、再在乾淨點上用
# mean/std 取回效率」，並用「mean↔median、std↔MAD 分歧」當校準可信度旗標。
CUSUM_TRIM_K = 4.0            # 修剪門檻：|x-median| > K*robust_scale 才剔除。取寬鬆值，保護健康分數
                             # 不對稱的向下深谷、只砍極端瞬態（K 太小會誤砍健康下探 → 反而更會誤報）
CUSUM_MIN_KEEP = 10          # 修剪後至少留這麼多點才用 mean/std，否則退回 median/MAD
CUSUM_MIN_KEEP_FRAC = 0.5    # 或至少保留這個比例（兩條件取較大者）
CUSUM_GATE_SHIFT = 0.5       # 交叉檢查：|mean-median| 超過此倍 robust_scale → 低信心
CUSUM_GATE_RATIO = 1.5       # 交叉檢查：std/robust_scale 超過此值 → 低信心（重尾/污染）
CUSUM_GATE_REJECT_FRAC = 0.3 # 交叉檢查：修剪剔除比例超過此值 → 低信心
CUSUM_LOWCONF_FLOOR_BUMP = 1.0  # 低信心時把衰減地板抬高這麼多 σ（更保守、少誤報）

# 基線持久化的載入再驗證：mu0/sigma0 存 DB、重啟載回可免掉每次重啟重新暖機的盲窗。但 SVM 模型
# 每次啟動都重訓，分數尺度可能微漂移，故載入後先用前幾筆新分數確認與載入基線一致（期間仍照常偵測
# ＝零盲窗），一致就採用、不一致就丟棄基線改重新暖機（自我修復）。
# 再驗證/重新對準的樣本數，取與暖機相同（20）。**不能取小**：重新對準要估的是 mu0，而中位數的
# 標準誤約 1.25σ/√n——n=10 時約 0.4σ，跟 CUSUM 的 slack(0.5σ) 同一量級，估計誤差本身就會被
# CUSUM 當成劣化累積成假警（實測 n=10 時健康通道誤報 8%）。n=20 把標準誤壓到約 0.28σ。
# 注意：拉長不會產生偵測盲窗——再驗證期間照常偵測，只是期間累積的 CUSUM 量會在對準後歸零。
CUSUM_REVALIDATE_SAMPLES = 20
CUSUM_REVALIDATE_TOL = 3.0     # 新分數中位數與載入 mu0 相差超過此倍 sigma0 → 判定漂移、改重新暖機


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
        self.h: float | None = None          # 目前生效的門檻（會隨時間從 h_start 衰減到 h_floor）
        self.h_start: float | None = None     # 校準後起始門檻 = CUSUM_H_MULT * sigma0
        self.h_floor: float | None = None     # 衰減地板 = CUSUM_H_FLOOR_MULT * sigma0
        self._n: int = 0                      # 校準後已吃進的樣本數，用來算衰減步數
        self.value: float = 0.0
        self.low_confidence: bool = False     # 校準可信度旗標（方案①交叉檢查閘）
        self.rejected_frac: float = 0.0       # 暖機被 robust 修剪剔除的比例
        self.source: str = "warmup"           # 校準來源：'warmup'（現場暖機）或 'loaded'（DB 載回）
        self.calib_generation: int = 0        # 每完成一次校準 +1，供呼叫端偵測「剛校準」事件、只記一次
        self._provisional: bool = False       # 載入基線後的再驗證期（期間仍偵測、零盲窗）
        self._reval: list[float] = []         # 再驗證期收集的新分數
        self.reval_failed: bool = False       # 一次性旗標：載入基線與新模型分數不一致、已改重新暖機

    @property
    def calibrated(self) -> bool:
        return self.mu0 is not None

    def _apply_baseline(self, floor_mult: float) -> None:
        """由已定好的 mu0/sigma0 推導 slack 與（會衰減的）門檻，並重置 CUSUM 累積與衰減時鐘。"""
        self.slack = CUSUM_SLACK_K * self.sigma0
        self.h_start = CUSUM_H_MULT * self.sigma0
        self.h_floor = min(self.h_start, floor_mult * self.sigma0)  # 地板不超過起始值
        self.h = self.h_start
        self._n = 0
        self.value = 0.0
        self.calib_generation += 1

    def load_baseline(self, mu0: float, sigma0: float, low_confidence: bool) -> None:
        """從持久化的 DB 基線直接校準（跳過現場暖機），進入再驗證期。"""
        self.mu0 = float(mu0)
        self.sigma0 = max(float(sigma0), CUSUM_SIGMA_FLOOR)
        self.low_confidence = bool(low_confidence)
        floor_mult = CUSUM_H_FLOOR_MULT + (CUSUM_LOWCONF_FLOOR_BUMP if self.low_confidence else 0.0)
        self._apply_baseline(floor_mult)
        self.source = "loaded"
        self._provisional = True
        self._reval = []

    def _calibrate(self) -> None:
        """從暖機 buffer 估健康基線 mu0/sigma0。
        方案②（robust 修剪→古典估計）：先用 median/MAD 擋掉暖機期的瞬態離群，再在存活的乾淨點上
        用 mean/std 取回效率。K 取寬鬆值，刻意保護健康分數本身不對稱的向下深谷、只剔除極端瞬態。
        方案①（交叉檢查閘）：比對 mean↔median、std↔MAD、剔除比例，分歧大就把這條校準標為低信心，
        並自動把衰減地板抬高一階（更保守），交由呼叫端記錄告警。
        """
        w = np.asarray(self._warmup, dtype=float)
        med = float(np.median(w))
        mad_sigma = 1.4826 * float(np.median(np.abs(w - med)))  # MAD 換算成常態等價 σ
        scale = max(mad_sigma, CUSUM_SIGMA_FLOOR)

        keep = w[np.abs(w - med) <= CUSUM_TRIM_K * scale]
        min_keep = max(CUSUM_MIN_KEEP, int(CUSUM_MIN_KEEP_FRAC * len(w)))
        if len(keep) >= min_keep:
            self.mu0 = float(np.mean(keep))                       # 乾淨點：mean（效率）
            self.sigma0 = max(float(np.std(keep)), CUSUM_SIGMA_FLOOR)
        else:
            # 剔太兇＝污染過重，別硬算古典估計，退回全體的 robust 值
            self.mu0 = med
            self.sigma0 = scale
        self.rejected_frac = 1.0 - len(keep) / len(w)

        # 交叉檢查閘：兩組估計分歧 → 暖機資料不乾淨 → 低信心
        mean_all, std_all = float(np.mean(w)), float(np.std(w))
        self.low_confidence = (
            abs(mean_all - med) > CUSUM_GATE_SHIFT * scale
            or std_all > CUSUM_GATE_RATIO * scale
            or self.rejected_frac > CUSUM_GATE_REJECT_FRAC
        )
        floor_mult = CUSUM_H_FLOOR_MULT + (CUSUM_LOWCONF_FLOOR_BUMP if self.low_confidence else 0.0)
        self._apply_baseline(floor_mult)
        self.source = "warmup"
        self._provisional = False  # 現場暖機的校準不需再驗證

    def update(self, score: float):
        """吃一筆新分數，回傳 (cusum 值, 是否持續劣化)。校準期間回傳 (None, False)。"""
        if not self.calibrated:
            self._warmup.append(score)
            if len(self._warmup) >= CUSUM_WARMUP_SAMPLES:
                self._calibrate()
            return None, False

        # 載入持久化基線後的短暫再驗證：用前幾筆新分數確認與載入基線一致。期間仍照常往下偵測
        # （零盲窗）；不一致代表模型重訓後分數漂移，丟棄基線改重新暖機（自我修復）。
        if self._provisional:
            self._reval.append(score)
            if len(self._reval) >= CUSUM_REVALIDATE_SAMPLES:
                # 用與 _calibrate 相同的 robust 修剪後再取平均：先 median/MAD 擋掉再驗證期間
                # 的瞬態，再在乾淨點上取 mean（效率較高），避免瞬態把重新對準的中心帶偏。
                _w = np.asarray(self._reval, dtype=float)
                _m = float(np.median(_w))
                _s = max(1.4826 * float(np.median(np.abs(_w - _m))), CUSUM_SIGMA_FLOOR)
                _keep = _w[np.abs(_w - _m) <= CUSUM_TRIM_K * _s]
                med = float(np.mean(_keep)) if len(_keep) >= max(3, len(_w) // 2) else _m
                drift = abs(med - self.mu0)
                if drift <= CUSUM_REVALIDATE_TOL * self.sigma0:
                    # **保留持久化的 sigma0，但用新樣本重新對準 mu0。**
                    # 原因（實測踩到）：SVM 每次重訓，分數的「中心」會漂移（實測 0.6~0.9σ），
                    # 但「離散度」相當穩定。而 CUSUM 的 slack 只有 0.5σ——它天生就是設計來偵測
                    # 這種量級的持續偏移，所以只要沿用舊 mu0，健康通道就會被自己的基線誤差
                    # 一路累積成假警（實測 34/180 通道誤報）。再驗證容忍 3σ 是「模型是不是同一
                    # 個世界」的粗檢查，遠比 CUSUM 的靈敏度鬆，不能當成 mu0 夠準的保證。
                    # 只重估 mu0 所需樣本遠少於估好 sigma0，故持久化的價值（省下長暖機）仍保留。
                    self.mu0 = med
                    self.value = 0.0  # 丟棄再驗證期間用舊 mu0 累積的量
                    self._provisional = False
                else:
                    self._warmup = list(self._reval)  # 不一致，退回重新暖機
                    self.mu0 = None
                    self._provisional = False
                    self.reval_failed = True
                    return None, False

        # 門檻隨校準後樣本數幾何衰減到地板：h = max(h_floor, h_start * 0.8^steps)。
        # 讓緩慢劣化的 onset 不會拖到末期才亮，又不低於 h_floor 而讓健康雜訊誤報。
        self._n += 1
        steps = self._n // CUSUM_DECAY_PERIOD_SAMPLES
        self.h = max(self.h_floor, self.h_start * (CUSUM_DECAY_FACTOR ** steps))

        # score 越低越異常；(mu0 - score) 在劣化時為正 → 累積，slack 把正常波動漏掉。
        self.value = max(0.0, self.value + (self.mu0 - score - self.slack))
        return self.value, self.value > self.h
