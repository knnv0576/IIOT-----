"""Phase3 AI 推論服務：多層預測疊在一起。

1. RUL（剩餘壽命）預測，兩條路徑：
   (a) 電流 RUL：對電流均值做滑動視窗線性回歸，外推還要多久撞到電流上下限
       （metric_kind='current_mean'）。
   (b) 振動 RUL：RandomForest 模型（models/rul_rf.pkl，32 顆真實軸承訓練）從時域特徵
       [rms_norm, kurt, crest, rms 斜率] 直接預測剩餘分鐘，附 P10/P90 信賴區間
       （metric_kind='ml_rf'）。早期版本用 thd_ratio 線性外推（含「CUSUM onset 後才 fit」
       的兩段式改良），已被 ML RUL 取代並移除；DB 裡 metric_kind='thd_ratio' 的舊資料
       是那個時期留下的。
2. 頻譜異常評分（One-Class SVM）：對每個通道訓練一個「正常樣子長怎樣」的模型，
   定期對最新一段原始波形算時域+頻域特徵去評分，能抓到均值還看不出來、但頻譜結構
   已經開始跑掉的早期異變（實作見 anomaly_detector.py / spectral_features.py）。
   **電流(60Hz) + 振動(30Hz) 三相三軸共 6 通道/台都做**——真實軸承/機械故障其實在
   振動頻譜裡更早顯現，這也是公開 PHM 資料集清一色是振動的原因。
3. 線上 CUSUM 漸進劣化偵測：對每個通道的異常分數時序做單邊累積，抓「分數持續系統性
   下滑」而非「低於某條線」。跨資料集驗證過（PHM2012 17/17 + XJTU-SY 15/15 = 32/32 全命中）。
4. 振動時域跳線（災難兜底）：三軸 RMS 合成幅值超過 ISO 危險線就強制 CRITICAL 並閂鎖，
   獨立執行緒、獨立連線，不受 SVM/ML 路徑影響（metric_kind='vib_trip'）。

跟 Phase1 的「危險模式」（瞬間切換的 0/1 開關）不同，這裡抓的是漸進式趨勢；
如果資料本身很穩定，各層結果都應該顯示正常，這是正確行為，不是没抓到東西。

CURRENT_UPPER/CURRENT_LOWER 這兩個門檻值必須跟 edge_pipeline/machine_config.py 的
CURRENT 設定保持一致（兩個服務目前是分開的 Docker context，沒有共用模組，
所以這裡是刻意接受的重複，不是遺漏)。
"""

import os
import pickle
import threading
import time

import numpy as np
import psycopg2

from anomaly_detector import (
    CUSUM_WARMUP_SAMPLES,
    CusumTracker,
    WINDOW_SAMPLES,
    score_latest_window,
)
from trainer import ensure_model

TIMESCALEDB_DSN = os.environ["TIMESCALEDB_DSN"]

# 訓練好的 ML RUL 模型（RandomForest，用 32 顆真實軸承 PHM2012+XJTU 訓練，跨 IMS 泛化驗證過）。
# 取代振動 RUL 原本的線性外推——線性外推對「平坦→暴衝」的非線性劣化誤差 3~7 倍，ML 降到可行動窗 ~1.6 倍。
# 註：模型學的是真實軸承劣化；跑在模擬器上時，RUL 數字反映「這波動有多像真實軸承劣化」，接真實感測器才最準。
RUL_MODEL_PATH = os.environ.get("RUL_MODEL_PATH", "/app/models/rul_rf.pkl")
try:
    with open(RUL_MODEL_PATH, "rb") as _f:
        _RUL_BUNDLE = pickle.load(_f)
    _RUL_MODEL = _RUL_BUNDLE["model"]
    ML_MAX_RUL_MIN = float(_RUL_BUNDLE["max_rul_min"])
    ML_SLOPE_WIN_MIN = float(_RUL_BUNDLE["slope_win_min"])
except Exception as _exc:  # 模型載入失敗不讓整個服務掛掉，只停用 ML RUL
    _RUL_MODEL, ML_MAX_RUL_MIN, ML_SLOPE_WIN_MIN = None, 250.0, 10.0
    print(f"[ai_service] 警告：ML RUL 模型載入失敗（{_exc.__class__.__name__}: {_exc}），振動 RUL 停用")

ML_RUL_MIN_POINTS = 10       # 至少要有這麼多 rms 歷史點才預測
ML_BASELINE_POINTS = 20      # 用最早這麼多筆 rms 當「健康基線」
ML_RUL_WARNING_MIN = 200.0   # RUL < 這麼多分鐘 → WARNING；< ML_RUL_CRITICAL_MIN → CRITICAL
ML_RUL_CRITICAL_MIN = 60.0
# 判定 status 時要用哪個 RUL 值（ISO 13381-1 精神：在不確定性下做決策）。
# 'mean'（預設）＝用點估計，維持既有告警行為。
# 'p10'＝用悲觀界（有 10% 機率比這更早壞），可抵銷「模型在最後 60 分鐘高估 RUL」的已知失效模式
#（IMS 0~30 分帶平均高估 47 分）——但**切換時必須同時重新校準下面兩個門檻**：實測健康態的
# 點估計約 223 分、P10 約 190 分，直接沿用 200/60 會讓健康通道全部站在 WARNING（常駐告警，
# 違反 ISA-18.2）。若要改用 p10，建議 ML_RUL_WARNING_MIN≈170、ML_RUL_CRITICAL_MIN≈50。
# 不論用哪個判定，P10/P90 都會照常存進 DB 與儀表板供人判讀（ISO 13381-1 的信賴區間要求）。
ML_RUL_STATUS_QUANTILE = os.environ.get("ML_RUL_STATUS_QUANTILE", "mean")

PREDICTION_INTERVAL = float(os.environ.get("RUL_PREDICTION_INTERVAL", "15"))  # 每隔幾秒跑一輪 RUL 預測
WINDOW_SECONDS = float(os.environ.get("RUL_WINDOW_SECONDS", "120"))  # RUL 回歸視窗長度（秒）
MIN_R_SQUARED = float(os.environ.get("RUL_MIN_R_SQUARED", "0.3"))  # 趨勢顯著性門檻，太低代表只是雜訊
MIN_POINTS = 10  # 視窗內至少要有這麼多筆彙總點才做回歸，太少統計上不可靠

ANOMALY_CHECK_INTERVAL = float(os.environ.get("ANOMALY_CHECK_INTERVAL", "30"))  # 頻譜異常評分的週期

CURRENT_UPPER = 6.5
CURRENT_LOWER = 5.5

# 時域絕對門檻的災難兜底（改動2）：三軸 RMS 合成幅值超過 ISO 危險線就強制 CRITICAL，不看 SVM/CUSUM。
# 這是「附加」在既有頻譜偵測之上的最後一道保險，專門接住學習型模型失效（模型沒訓練成功、基線被污染）
# 時的嚴重超標。門檻讀 machine_vibration_thresholds.danger_lower（定義在三軸合成幅值上，故用三軸 RMS
# 平方和開根還原到同一量綱；用 RMS 而非峰值＝較平滑、與 1 秒彙總的顯示口徑一致）。
#
# 跑在「獨立執行緒 + 獨立連線」上，不跟昂貴的 FFT/SVM 路徑綁在一起：保險絲只需要 RMS（很便宜），
# 不該被 ML 路徑拖慢或餓死。這是安全兜底該有的性質。
VIB_TRIP_CHECK_INTERVAL = float(os.environ.get("VIB_TRIP_CHECK_INTERVAL", "10"))  # 跳線自己的檢查週期（秒）
VIB_TRIP_WINDOW_SECONDS = float(os.environ.get("VIB_TRIP_WINDOW_SECONDS", "2.0"))  # 每次算 RMS 的取樣窗長
VIB_TRIP_DEBOUNCE_CYCLES = int(os.environ.get("VIB_TRIP_DEBOUNCE_CYCLES", "3"))  # 連續超標幾輪才判（3×10s=30s）
# 重置遲滯（ISA-18.2 防抖動）：跳閘後要掉到「門檻 × 此比例」以下才算條件解除，
# 否則在門檻邊緣徘徊會 trip/clear 反覆跳。
VIB_TRIP_RESET_RATIO = float(os.environ.get("VIB_TRIP_RESET_RATIO", "0.9"))

RECONNECT_COOLDOWN = 2.0

DEVICES = [f"SCT-{i:02d}" for i in range(1, 11)]
PHASES = ("A", "B", "C")  # 電流三相
AXES = ("X", "Y", "Z")  # 振動三軸

# 頻譜異常偵測要用的基頻，決定 FFT 抓哪個頻帶當基波。電流是 60Hz 交流電；
# 振動是 30Hz（對應 edge_pipeline/formulas.py 的 _VIBRATION_FREQ_HZ，兩個服務分屬不同
# Docker context、沒共用模組，跟 CURRENT_UPPER/LOWER 一樣是刻意接受的重複）。
CURRENT_FUNDAMENTAL_HZ = 60.0
VIBRATION_FUNDAMENTAL_HZ = 30.0


def anomaly_channels(n: int) -> list[tuple[str, float]]:
    """回傳這台機台要做頻譜異常偵測的通道 [(sensor_id, fundamental_hz), ...]。
    電流三相 + 振動三軸都納入——早期軸承/機械故障其實在振動頻譜裡更早顯現。
    """
    current = [(f"E{n:02d}-{p}", CURRENT_FUNDAMENTAL_HZ) for p in PHASES]
    vibration = [(f"V{n:02d}-{ax}", VIBRATION_FUNDAMENTAL_HZ) for ax in AXES]
    return current + vibration


def fetch_window(conn, device_id: str, sensor_id: str, window_seconds: float) -> list[tuple[float, float]]:
    """從 current_1s_agg 撈最近 window_seconds 秒的 (時間, 均值) 序列。"""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT extract(epoch FROM bucket) AS t, avg_value
            FROM current_1s_agg
            WHERE device_id = %s AND sensor_id = %s
              AND bucket > now() - (%s * interval '1 second')
            ORDER BY bucket
            """,
            (device_id, sensor_id, window_seconds),
        )
        return cur.fetchall()


def fit_trend(rows: list[tuple[float, float]]):
    """對視窗內的資料做線性回歸，回傳 (每秒飄移速率, R平方, 最近均值)；資料不足回傳 None。"""
    if len(rows) < MIN_POINTS:
        return None

    t = np.array([r[0] for r in rows], dtype=float)
    v = np.array([r[1] for r in rows], dtype=float)
    t0 = t - t[0]

    slope, intercept = np.polyfit(t0, v, 1)
    predicted = slope * t0 + intercept
    ss_res = float(np.sum((v - predicted) ** 2))
    ss_tot = float(np.sum((v - v.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    recent_mean = float(v[-10:].mean())  # 用視窗尾端這幾筆當「目前值」，比整個視窗的均值更即時
    return float(slope), r_squared, recent_mean


def predict_rul(slope: float, r_squared: float, current_mean: float):
    """回傳 (目標門檻, 預估還要幾秒撞到, 狀態)。趨勢不顯著或幾乎沒有飄移就判定 STABLE。"""
    if r_squared < MIN_R_SQUARED or abs(slope) < 1e-6:
        return None, None, "STABLE"

    if slope > 0:
        target = CURRENT_UPPER
        if current_mean >= target:
            return target, 0.0, "CRITICAL"
        eta_seconds = (target - current_mean) / slope
    else:
        target = CURRENT_LOWER
        if current_mean <= target:
            return target, 0.0, "CRITICAL"
        eta_seconds = (current_mean - target) / (-slope)

    if eta_seconds < 300:
        status = "CRITICAL"
    elif eta_seconds < 1800:
        status = "WARNING"
    else:
        status = "STABLE"
    return target, eta_seconds, status


def insert_prediction(
    conn, device_id, sensor_id, value, slope, r_squared, target, eta_seconds, status,
    metric_kind: str = "current_mean", eta_p10_seconds=None, eta_p90_seconds=None,
) -> None:
    """metric_kind：哪一種預測——'current_mean'（電流均值 RUL）、'ml_rf'（振動 ML RUL，
    value 存 rms_norm）、'vib_trip'（時域跳線，value 存三軸合成 RMS）。value 欄位存的就是該量的目前值。
    eta_p10/p90：ISO 13381-1 要求的信賴區間（目前只有 ML RUL 有，其餘傳 None）。"""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO rul_predictions
                (ts, device_id, sensor_id, current_mean, drift_rate_per_min, r_squared,
                 target_threshold, eta_seconds, status, metric_kind,
                 eta_p10_seconds, eta_p90_seconds)
            VALUES (now(), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (device_id, sensor_id, value, slope * 60, r_squared, target, eta_seconds, status,
             metric_kind, eta_p10_seconds, eta_p90_seconds),
        )


def ensure_cusum_baseline_table(conn) -> None:
    """建立 CUSUM 基線持久化表（若不存在）。存 healthy 校準結果，讓服務重啟能載回、免除每次
    重啟都要重新暖機（盲窗）。用 CREATE IF NOT EXISTS，對既有 DB 也安全（init.sql 也有一份供全新安裝）。"""
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS cusum_baselines (
                device_id TEXT NOT NULL,
                sensor_id TEXT NOT NULL,
                mu0 DOUBLE PRECISION NOT NULL,
                sigma0 DOUBLE PRECISION NOT NULL,
                low_confidence BOOLEAN NOT NULL DEFAULT FALSE,
                rejected_frac DOUBLE PRECISION,
                n_warmup INTEGER,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (device_id, sensor_id)
            )
            """
        )


def load_cusum_baseline(conn, device_id, sensor_id, fingerprint):
    """回傳 (mu0, sigma0, low_confidence)；沒有「同一個模型」的基線時回傳 None。

    **指紋是鍵的一部分,不是附註**：健康分數的分布屬於「機台 × 該次訓練出的模型」。
    早期版本只用 (device_id, sensor_id) 當鍵，重訓後取到別的模型的基線，實測 sigma0 被低估
    1.5~3 倍 → slack 與門檻 h 一起縮水 → 健康通道的隨機遊走輕易跨過 h，誤報率一度達 19%。
    指紋不符就回 None、退回現場暖機——這是正確行為。
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT mu0, sigma0, low_confidence FROM cusum_baselines "
            "WHERE device_id = %s AND sensor_id = %s AND model_fingerprint = %s",
            (device_id, sensor_id, fingerprint),
        )
        return cur.fetchone()


def save_cusum_baseline(conn, device_id, sensor_id, tracker, fingerprint) -> None:
    """把現場暖機得到的健康基線連同模型指紋存起來，下次同一個模型啟動就能免暖機。"""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO cusum_baselines
                (device_id, sensor_id, mu0, sigma0, low_confidence, rejected_frac, n_warmup,
                 model_fingerprint, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, now())
            ON CONFLICT (device_id, sensor_id) DO UPDATE SET
                mu0 = EXCLUDED.mu0, sigma0 = EXCLUDED.sigma0,
                low_confidence = EXCLUDED.low_confidence, rejected_frac = EXCLUDED.rejected_frac,
                n_warmup = EXCLUDED.n_warmup, model_fingerprint = EXCLUDED.model_fingerprint,
                updated_at = now()
            """,
            (device_id, sensor_id, tracker.mu0, tracker.sigma0, tracker.low_confidence,
             tracker.rejected_frac, CUSUM_WARMUP_SAMPLES, fingerprint),
        )


def insert_anomaly_score(conn, device_id, sensor_id, score, is_anomaly, features, cusum, cusum_alarm) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO anomaly_scores
                (ts, device_id, sensor_id, score, is_anomaly, mean_value, std_value, thd_ratio,
                 cusum, cusum_alarm, rms, kurt, crest)
            VALUES (now(), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                device_id, sensor_id, score, is_anomaly,
                features["mean"], features["std"], features["thd_ratio"], cusum, cusum_alarm,
                features["rms"], features["kurtosis"], features["crest"],
            ),
        )


# {(device_id, sensor_id): model_bundle}；訓練一次成功就快取起來，之後只重試還沒訓練成功的通道。
_anomaly_models: dict[tuple[str, str], dict] = {}
# {(device_id, sensor_id): CusumTracker}；每個通道一個線上 CUSUM 追蹤器，跟模型同時建立。
_cusum_trackers: dict[tuple[str, str], CusumTracker] = {}
# {(device_id, sensor_id): 模型指紋}；CUSUM 基線的存取都要帶這個，確保基線與模型是同一組。
_model_fingerprints: dict[tuple[str, str], str] = {}
# {(device_id, sensor_id): 已記錄的 calib_generation}；每完成一次校準才記/存一次（含重新校準）。
_cusum_calib_gen: dict[tuple[str, str], int] = {}
# {(device_id, sensor_id): 健康基線 rms}；ML RUL 的 rms_norm=現值/基線，用最早幾筆健康 rms 算一次。
_rms_baseline: dict[tuple[str, str], float] = {}
# {device_id: danger_lower}；振動危險門檻快取（靜態設定，讀一次）。None=該台無門檻，跳線 fail-safe 跳過。
_vib_danger_threshold: dict[str, "float | None"] = {}
# {device_id: 連續超標輪數}；時域跳線去抖動計數。
_vib_trip_streak: dict[str, int] = {}


def fetch_rms_history(conn, device_id, sensor_id, window_seconds):
    """撈最近 window_seconds 秒的 (epoch, rms, kurt, crest)，給 ML RUL 算特徵。"""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT extract(epoch FROM ts), rms, kurt, crest FROM anomaly_scores
            WHERE device_id=%s AND sensor_id=%s AND rms IS NOT NULL
              AND ts > now() - (%s * interval '1 second') ORDER BY ts
            """,
            (device_id, sensor_id, window_seconds),
        )
        return cur.fetchall()


def get_rms_baseline(conn, device_id, sensor_id):
    """健康基線 rms = 該通道最早 ML_BASELINE_POINTS 筆的中位數（假設服務啟動初期是健康的）。
    算一次快取起來；anomaly_scores 無保留政策，最早的健康資料一直都在。"""
    key = (device_id, sensor_id)
    if key in _rms_baseline:
        return _rms_baseline[key]
    with conn.cursor() as cur:
        cur.execute(
            "SELECT rms FROM anomaly_scores WHERE device_id=%s AND sensor_id=%s AND rms IS NOT NULL "
            "ORDER BY ts ASC LIMIT %s",
            (device_id, sensor_id, ML_BASELINE_POINTS),
        )
        vals = [r[0] for r in cur.fetchall()]
    if len(vals) < ML_BASELINE_POINTS:
        return None  # 健康資料還不夠，等下一輪
    base = float(np.median(vals))
    _rms_baseline[key] = base
    return base


def ml_vibration_rul(conn, device_id, sensor_id):
    """用 RF 模型從時域特徵預測 RUL（分鐘）。
    回傳 (rul_min, p10_min, p90_min, rms_norm, slope_per_min)，資料不足回傳 None。

    p10/p90 是 ISO 13381-1 要求的信賴區間：RandomForest 的每一棵樹各自是一個獨立預測，
    取這個分布的分位數就得到「免費」的預測區間，不需要重訓或額外模型。
    P10=悲觀界（有 10% 機率比這更早壞）、P90=樂觀界。
    """
    if _RUL_MODEL is None:
        return None
    baseline = get_rms_baseline(conn, device_id, sensor_id)
    if baseline is None or baseline <= 1e-12:
        return None
    rows = fetch_rms_history(conn, device_id, sensor_id, ML_SLOPE_WIN_MIN * 60.0)
    if len(rows) < ML_RUL_MIN_POINTS:
        return None
    t = np.array([r[0] for r in rows], dtype=float)  # epoch 是 Decimal，要轉 float
    rms = np.array([r[1] for r in rows], dtype=float)
    rms_norm_series = rms / baseline
    rms_norm = float(np.median(rms_norm_series[-5:]))  # 平滑當前值
    # 斜率：rms_norm 對時間（每分鐘）
    tm = (t - t[0]) / 60.0
    slope = float(np.polyfit(tm, rms_norm_series, 1)[0]) if len(rows) >= 3 else 0.0
    kurt, crest = float(rows[-1][2]), float(rows[-1][3])
    features = [[rms_norm, kurt, crest, slope]]
    rul_min = float(_RUL_MODEL.predict(features)[0])
    try:
        tree_preds = np.array([est.predict(features)[0] for est in _RUL_MODEL.estimators_])
        p10, p90 = (float(v) for v in np.percentile(tree_preds, [10, 90]))
    except AttributeError:  # 非樹系模型（例如日後換成 LSTM）就沒有區間，不讓服務掛掉
        p10 = p90 = None
    return rul_min, p10, p90, rms_norm, slope


def get_vib_danger_threshold(conn, device_id):
    """讀該台振動危險門檻（三軸合成幅值上的 danger_lower），快取。無門檻回 None（跳線會 fail-safe 跳過）。"""
    if device_id in _vib_danger_threshold:
        return _vib_danger_threshold[device_id]
    with conn.cursor() as cur:
        cur.execute("SELECT danger_lower FROM machine_vibration_thresholds WHERE device_id = %s", (device_id,))
        row = cur.fetchone()
    val = float(row[0]) if row and row[0] is not None else None
    _vib_danger_threshold[device_id] = val
    return val


def ensure_vib_trip_latch_table(conn) -> None:
    """建立跳閘閂鎖表（若不存在）。閂鎖＝跳閘後即使振動回到正常也維持 CRITICAL，直到人工確認，
    避免災難事件的紀錄自己消失（ISA-18.2 標準告警語意）。"""
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS vib_trip_latches (
                device_id       TEXT PRIMARY KEY,
                tripped_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
                danger_line     DOUBLE PRECISION NOT NULL,
                peak_mag_rms    DOUBLE PRECISION NOT NULL,
                last_mag_rms    DOUBLE PRECISION,
                acknowledged    BOOLEAN NOT NULL DEFAULT FALSE,
                acknowledged_at TIMESTAMPTZ
            )
            """
        )


def fetch_vib_rms_all(conn) -> dict:
    """一次查回全機隊各振動通道最近 VIB_TRIP_WINDOW_SECONDS 秒的 RMS。
    定義與 spectral_features.extract_features 的 rms 一致（含直流），確保與既有門檻口徑相同。
    回傳 {device_id: {軸: rms}}。"""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT device_id, sensor_id, sqrt(avg(value * value)) AS rms
            FROM sensor_readings
            WHERE sensor_id LIKE 'V%%' AND ts > now() - (%s * interval '1 second')
            GROUP BY device_id, sensor_id
            """,
            (VIB_TRIP_WINDOW_SECONDS,),
        )
        out: dict = {}
        for device_id, sensor_id, rms in cur.fetchall():
            if rms is None:
                continue
            out.setdefault(device_id, {})[sensor_id.split("-")[-1]] = float(rms)
        return out


def load_vib_latches(conn) -> dict:
    """回傳 {device_id: acknowledged}，代表目前處於閂鎖中的機台。"""
    with conn.cursor() as cur:
        cur.execute("SELECT device_id, acknowledged FROM vib_trip_latches")
        return {row[0]: bool(row[1]) for row in cur.fetchall()}


def set_vib_latch(conn, device_id, danger, mag_rms) -> None:
    """新跳閘：建立閂鎖。已存在則只更新峰值/最近值（ON CONFLICT 保住原始 tripped_at 與確認狀態）。"""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO vib_trip_latches (device_id, danger_line, peak_mag_rms, last_mag_rms)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (device_id) DO UPDATE SET
                peak_mag_rms = GREATEST(vib_trip_latches.peak_mag_rms, EXCLUDED.peak_mag_rms),
                last_mag_rms = EXCLUDED.last_mag_rms
            """,
            (device_id, danger, mag_rms, mag_rms),
        )


def clear_vib_latch(conn, device_id) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM vib_trip_latches WHERE device_id = %s", (device_id,))


def run_vibration_trip_cycle(conn) -> None:
    """時域絕對門檻的災難兜底一輪：三軸 RMS 合成幅值 ≥ ISO 危險線就強制 CRITICAL（不看 SVM/CUSUM）。

    護欄：
    - fail-safe：三軸沒湊齊或該台無門檻就跳過（寧可不跳，也不因缺資料誤跳/崩潰）。
    - 去抖動：需連續 VIB_TRIP_DEBOUNCE_CYCLES 輪超標才判，防單發瞬態。
    - 遲滯：跳閘後要掉到 danger×VIB_TRIP_RESET_RATIO 以下才算條件解除，防門檻邊緣抖動。
    - 閂鎖：跳閘後維持 CRITICAL 直到「人工確認 且 條件已解除」才復歸（ISA-18.2 語意）。
    出口：rul_predictions(metric_kind='vib_trip', status='CRITICAL', eta=0)。
    """
    rms_by_device = fetch_vib_rms_all(conn)
    latches = load_vib_latches(conn)

    for device_id, axes in rms_by_device.items():
        if len(axes) < 3:
            continue  # fail-safe：三軸沒湊齊，無法算合成幅值
        danger = get_vib_danger_threshold(conn, device_id)
        if danger is None or danger <= 0:
            continue  # fail-safe：沒有門檻就不跳線
        mag_rms = float(np.sqrt(sum(v * v for v in axes.values())))
        latched = device_id in latches
        over = mag_rms >= danger
        cleared = mag_rms < danger * VIB_TRIP_RESET_RATIO  # 遲滯：介於兩者之間視為「仍在告警帶」

        if over:
            _vib_trip_streak[device_id] = _vib_trip_streak.get(device_id, 0) + 1
        elif cleared:
            _vib_trip_streak[device_id] = 0
        # else：落在遲滯帶內（danger×ratio ~ danger），維持現狀不累加也不重置

        n = int(device_id.split("-")[1])
        if _vib_trip_streak.get(device_id, 0) >= VIB_TRIP_DEBOUNCE_CYCLES and not latched:
            set_vib_latch(conn, device_id, danger, mag_rms)
            latched = True
            latches[device_id] = False
            print(
                f"[ai_service] 時域超標強制 CRITICAL（已閂鎖）: {device_id} 三軸合成 RMS={mag_rms:.3f} "
                f"≥ 危險線={danger:.3f}（連續 {VIB_TRIP_DEBOUNCE_CYCLES} 輪超標，不看 SVM）。"
                f"確認方式：UPDATE vib_trip_latches SET acknowledged=true, acknowledged_at=now() "
                f"WHERE device_id='{device_id}';"
            )

        if latched:
            # 復歸條件（ISA-18.2）：必須「已人工確認」且「條件已解除」，兩者缺一不可。
            if latches[device_id] and cleared:
                clear_vib_latch(conn, device_id)
                _vib_trip_streak[device_id] = 0
                print(f"[ai_service] 跳閘已確認且振動回到門檻下，閂鎖解除: {device_id}")
                continue
            set_vib_latch(conn, device_id, danger, mag_rms)  # 更新峰值/最近值
            insert_prediction(
                conn, device_id, f"V{n:02d}", mag_rms, 0.0, None, danger, 0.0, "CRITICAL",
                metric_kind="vib_trip",
            )


def vibration_trip_loop() -> None:
    """獨立執行緒：跳線跑自己的節奏、用自己的連線，不被昂貴的 FFT/SVM 路徑拖慢或餓死。"""
    conn = None
    while True:
        try:
            if conn is None:
                conn = connect()
                ensure_vib_trip_latch_table(conn)
            run_vibration_trip_cycle(conn)
        except Exception as exc:
            # 刻意捕捉所有例外，不只 psycopg2.Error：這是安全兜底，執行緒「靜默死掉」是最糟的
            # 失效模式（之後完全沒保護、卻沒人知道）。任何錯誤都記錄下來、重連後繼續跑。
            print(
                f"[ai_service] 振動跳線本輪失敗（{exc.__class__.__name__}: {exc}），"
                f"重連後繼續（保險絲不會因單次錯誤停止運作）"
            )
            try:
                if conn is not None:
                    conn.close()
            except Exception:
                pass
            conn = None
        time.sleep(VIB_TRIP_CHECK_INTERVAL)


def run_anomaly_detection_cycle(conn) -> None:
    for device_id in DEVICES:
        n = int(device_id.split("-")[1])
        for sensor_id, fundamental_hz in anomaly_channels(n):
            key = (device_id, sensor_id)

            if key not in _anomaly_models:
                # 載入優先、缺少才自動訓練（見 trainer.py）。模型持久化之後，重啟不會重訓，
                # 分數尺度固定 → CUSUM 基線才可以安全沿用。
                got = ensure_model(conn, device_id, sensor_id, fundamental_hz, quiet=True)
                if got is None:
                    continue  # 歷史資料還不夠，下一輪再試
                bundle, fingerprint, source = got

                tracker = CusumTracker()
                # **基線必須與模型指紋綁定**：健康分數的分布是「機台 × 該次模型」的屬性。
                # 實測若只用機台當鍵，重訓後取到的舊基線 sigma0 會被低估 1.5~3 倍，
                # 導致 slack 與門檻一起縮水、健康通道大量誤報。指紋不符就退回現場暖機。
                # 跨 session 沿用基線預設關閉（CUSUM_USE_PERSISTED_BASELINE=1 才啟用）。
                # **實測證據**：即使模型指紋完全相同，上個 session 存的基線對這個 session 的資料
                # 仍系統性偏差——mu0 高出 0.16~0.39、sigma0 低估 1.5~2.7 倍。因為分數分布不只取決於
                # 模型，也取決於當下的資料（模擬器重啟、真實機台的溫度/負載/磨耗都會讓分數漂移）。
                # 後果是 (mu0 - score - slack) 每一筆都為正 → CUSUM 單調累積 → **所有通道遲早都會
                # 誤報**，實測告警機台數隨時間 1→2→3→4 持續上升。凍結的基線本質上就是錯的。
                # 現在一律現場暖機（20 筆≈10 分鐘），這是唯一能保證基線對應當下資料的做法。
                if os.environ.get("CUSUM_USE_PERSISTED_BASELINE", "0") == "1":
                    baseline = load_cusum_baseline(conn, device_id, sensor_id, fingerprint)
                    if baseline is not None:
                        tracker.load_baseline(baseline[0], baseline[1], baseline[2])

                # 三個 dict 一起登記（放在最後）：中途若拋例外就當這個通道還沒準備好，
                # 下一輪整段重來。若先登記 _anomaly_models 再失敗，下一輪會跳過這段初始化，
                # 卻讀不到 tracker 而 KeyError——實測踩過。
                _anomaly_models[key] = bundle
                _model_fingerprints[key] = fingerprint
                _cusum_trackers[key] = tracker

                kind = "電流" if sensor_id.startswith("E") else "振動"
                action = "載入既有模型" if source == "loaded" else "訓練完成並存檔"
                print(
                    f"[ai_service] {device_id}/{sensor_id}（{kind} {fundamental_hz:.0f}Hz）{action}"
                    f"（{bundle['n_train_windows']} 個訓練窗口，指紋 {fingerprint}）"
                )

            result = score_latest_window(conn, device_id, sensor_id, _anomaly_models[key])
            if result is None:
                continue
            score, is_anomaly, features = result
            tracker = _cusum_trackers[key]
            cusum_value, is_degrading = tracker.update(score)
            insert_anomaly_score(conn, device_id, sensor_id, score, is_anomaly, features, cusum_value, is_degrading)

            # 載入基線與新模型分數不一致（模型重訓漂移）→ tracker 已自動改重新暖機，這裡記一次。
            if tracker.reval_failed:
                tracker.reval_failed = False
                _cusum_calib_gen.pop(key, None)  # 讓接下來重新暖機的校準能再記/存一次
                print(
                    f"[ai_service] CUSUM 載入基線與新模型分數不一致（重訓漂移），改重新暖機: "
                    f"{device_id}/{sensor_id}"
                )

            # 每完成一次校準（含重新校準）記/存一次：載入的印提示、現場暖機的存進 DB 供下次免暖機。
            if tracker.calibrated and _cusum_calib_gen.get(key) != tracker.calib_generation:
                _cusum_calib_gen[key] = tracker.calib_generation
                if tracker.source == "seeded":  # CusumTracker.load_baseline() 載入時設的值
                    print(
                        f"[ai_service] CUSUM 基線已載回（跳過暖機、再驗證中）: {device_id}/{sensor_id} "
                        f"mu0={tracker.mu0:.3f} sigma0={tracker.sigma0:.3f}"
                    )
                elif tracker.low_confidence:
                    print(
                        f"[ai_service] CUSUM 校準低信心: {device_id}/{sensor_id}"
                        f"（暖機剔除 {tracker.rejected_frac:.0%}、地板已自動抬高保守處理，"
                        f"建議檢查該通道暖機期是否非穩態；不存檔）"
                    )
                else:
                    # 乾淨校準才存，並綁定當前模型指紋
                    save_cusum_baseline(conn, device_id, sensor_id, tracker, _model_fingerprints[key])
                    print(
                        f"[ai_service] CUSUM 基線已存檔（CUSUM_USE_PERSISTED_BASELINE=1 時下次啟動會載入）: "
                        f"{device_id}/{sensor_id} "
                        f"mu0={tracker.mu0:.3f} sigma0={tracker.sigma0:.3f}"
                    )

            if is_anomaly:
                print(
                    f"[ai_service] 頻譜異常(瞬時): {device_id}/{sensor_id} score={score:.4f} "
                    f"thd_ratio={features['thd_ratio']:.5f}"
                )
            if is_degrading:
                print(
                    f"[ai_service] 持續劣化(CUSUM): {device_id}/{sensor_id} "
                    f"cusum={cusum_value:.3f} > h={_cusum_trackers[key].h:.3f}"
                )


def connect() -> "psycopg2.extensions.connection":
    conn = psycopg2.connect(TIMESCALEDB_DSN)
    conn.autocommit = True
    return conn


def main() -> None:
    ml = "已載入" if _RUL_MODEL is not None else "未載入(停用)"
    print(
        f"[ai_service] RUL 預測服務啟動：視窗={WINDOW_SECONDS:.0f}s、間隔={PREDICTION_INTERVAL:.0f}s、"
        f"電流均值 RUL 門檻=[{CURRENT_LOWER},{CURRENT_UPPER}]%；"
        f"振動 RUL 改用 ML 模型（RF，{ml}；32 顆真實軸承訓練、跨 IMS 泛化驗證，"
        f"附 P10~P90 信賴區間、status 以 {ML_RUL_STATUS_QUANTILE} 判定）"
    )
    print(
        f"[ai_service] 頻譜異常評分啟動：每 {ANOMALY_CHECK_INTERVAL:.0f}s 一輪、每個特徵窗口 {WINDOW_SAMPLES} 筆，"
        f"涵蓋電流({CURRENT_FUNDAMENTAL_HZ:.0f}Hz) + 振動({VIBRATION_FUNDAMENTAL_HZ:.0f}Hz) 共 6 通道/台，"
        f"含線上 self-starting CUSUM 漸進劣化偵測（前 {CUSUM_WARMUP_SAMPLES} 筆只累積基準、不判斷）"
    )

    print(
        f"[ai_service] 振動時域跳線啟動（災難兜底）：獨立執行緒每 {VIB_TRIP_CHECK_INTERVAL:.0f}s 檢查一次、"
        f"連續 {VIB_TRIP_DEBOUNCE_CYCLES} 輪超標才判 CRITICAL"
        f"（≈{VIB_TRIP_CHECK_INTERVAL * VIB_TRIP_DEBOUNCE_CYCLES:.0f}s），"
        f"遲滯 {VIB_TRIP_RESET_RATIO:.0%}、跳閘閂鎖至人工確認"
    )

    conn = connect()
    ensure_cusum_baseline_table(conn)  # CUSUM 基線持久化表（重啟免暖機）
    ensure_vib_trip_latch_table(conn)  # 跳閘閂鎖表
    threading.Thread(target=vibration_trip_loop, daemon=True).start()
    connection_ok = True
    last_reconnect_attempt = 0.0
    last_anomaly_check = 0.0

    while True:
        if not connection_ok:
            now = time.time()
            if now - last_reconnect_attempt < RECONNECT_COOLDOWN:
                time.sleep(PREDICTION_INTERVAL)
                continue
            last_reconnect_attempt = now
            try:
                conn = connect()
                connection_ok = True
                print("[ai_service] 重新連線成功")
            except psycopg2.Error as exc:
                print(f"[ai_service] 重新連線失敗（{exc.__class__.__name__}），{RECONNECT_COOLDOWN:.0f} 秒後再試")
                time.sleep(PREDICTION_INTERVAL)
                continue

        warnings = []
        try:
            for device_id in DEVICES:
                n = int(device_id.split("-")[1])
                for phase in PHASES:
                    sensor_id = f"E{n:02d}-{phase}"
                    rows = fetch_window(conn, device_id, sensor_id, WINDOW_SECONDS)
                    fit = fit_trend(rows)
                    if fit is None:
                        continue
                    slope, r_squared, current_mean = fit
                    target, eta_seconds, status = predict_rul(slope, r_squared, current_mean)
                    insert_prediction(
                        conn, device_id, sensor_id, current_mean, slope, r_squared, target, eta_seconds, status
                    )
                    if status in ("WARNING", "CRITICAL"):
                        warnings.append((device_id, sensor_id, status, eta_seconds))

                # 振動 RUL：用訓練好的 RF 模型從時域特徵 [rms_norm, kurt, crest, rms_slope] 直接預測
                # RUL（分鐘），取代原本的線性/兩段式外推。模型會自己處理「健康→遠、劣化→近」。
                for axis in AXES:
                    sensor_id = f"V{n:02d}-{axis}"
                    result = ml_vibration_rul(conn, device_id, sensor_id)
                    if result is None:
                        continue  # 模型未載入或資料不足（含健康基線還沒建好），下一輪再說
                    rul_min, p10_min, p90_min, rms_norm, slope = result
                    # 判定基準：預設用點估計（ML_RUL_STATUS_QUANTILE="mean"）；設成 "p10" 才改用
                    # 悲觀界來抵銷模型末期高估 RUL 的已知失效模式（切換時要一併調門檻，見常數說明）。
                    basis = p10_min if (ML_RUL_STATUS_QUANTILE == "p10" and p10_min is not None) else rul_min
                    if basis < ML_RUL_CRITICAL_MIN:
                        status = "CRITICAL"
                    elif basis < ML_RUL_WARNING_MIN:
                        status = "WARNING"
                    else:
                        status = "STABLE"
                    insert_prediction(
                        conn, device_id, sensor_id, rms_norm, slope, None, ML_MAX_RUL_MIN,
                        rul_min * 60.0, status, metric_kind="ml_rf",
                        eta_p10_seconds=None if p10_min is None else p10_min * 60.0,
                        eta_p90_seconds=None if p90_min is None else p90_min * 60.0,
                    )
                    if status in ("WARNING", "CRITICAL"):
                        warnings.append((device_id, sensor_id, status, rul_min * 60.0))
        except psycopg2.Error as exc:
            print(f"[ai_service] 資料庫錯誤（{exc.__class__.__name__}），本輪中止，嘗試重新連線")
            connection_ok = False
            try:
                conn.close()
            except Exception:
                pass
            time.sleep(PREDICTION_INTERVAL)
            continue

        if warnings:
            for device_id, sensor_id, status, eta_seconds in warnings:
                eta_txt = f"預估 {eta_seconds:.0f} 秒後觸及門檻" if eta_seconds is not None else "劣化中（趨勢待明朗）"
                print(f"[ai_service] {status}: {device_id}/{sensor_id} {eta_txt}")
        else:
            print(f"[ai_service] 本輪完成，全部通道穩定（STABLE）")

        now = time.time()
        if connection_ok and now - last_anomaly_check >= ANOMALY_CHECK_INTERVAL:
            try:
                run_anomaly_detection_cycle(conn)
            except psycopg2.Error as exc:
                print(f"[ai_service] 頻譜異常評分資料庫錯誤（{exc.__class__.__name__}），本輪跳過")
            last_anomaly_check = now

        time.sleep(PREDICTION_INTERVAL)


if __name__ == "__main__":
    main()
