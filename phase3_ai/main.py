"""Phase3 AI 推論服務：多層預測疊在一起。

1. RUL（剩餘壽命）趨勢外推：觀察線性飄移速率，外推還要多久撞到門檻。兩種輸入：
   (a) 電流均值 → 撞電流上下限；(b) 振動 thd_ratio（諧波比值）→ 撞頻譜劣化門檻。
   為什麼振動用 thd_ratio 而非振幅：SCT-05 實驗證實漸進劣化時振動「振幅」幾乎不動
   （std +4%）、但 thd_ratio 大幅上升（×20），所以追蹤頻譜訊號才能提前預測、追蹤振幅會太晚。
2. 頻譜異常評分（One-Class SVM）：對每個通道訓練一個「正常樣子長怎樣」的模型，
   定期對最新一段原始波形算時域+頻域特徵去評分，能抓到均值還看不出來、但頻譜結構
   已經開始跑掉的早期異變（實作見 anomaly_detector.py / spectral_features.py）。
   **電流(60Hz) + 振動(30Hz) 三相三軸共 6 通道/台都做**——真實軸承/機械故障其實在
   振動頻譜裡更早顯現，這也是公開 PHM 資料集清一色是振動的原因。
3. 線上 CUSUM 漸進劣化偵測：對每個通道的異常分數時序做單邊累積，抓「分數持續系統性
   下滑」而非「低於某條線」。跨資料集驗證過（PHM2012 17/17 + XJTU-SY 15/15 = 32/32 全命中）。

振動 RUL 是「兩段式」：CUSUM 警報當階段1「偵測 onset」（劣化確實開始了），onset 後才進
階段2「預測」——只用 onset 之後的 thd 資料 fit 斜率外推撞門檻。這樣斜率不會被健康期平線
污染，修掉單段式固定視窗早期 ETA 高估的毛病。onset 前一律 STABLE、不亂預測。

跟 Phase1 的「危險模式」（瞬間切換的 0/1 開關）不同，這裡抓的是漸進式趨勢；
如果資料本身很穩定，各層結果都應該顯示正常，這是正確行為，不是没抓到東西。

CURRENT_UPPER/CURRENT_LOWER 這兩個門檻值必須跟 edge_pipeline/machine_config.py 的
CURRENT 設定保持一致（兩個服務目前是分開的 Docker context，沒有共用模組，
所以這裡是刻意接受的重複，不是遺漏)。
"""

import os
import pickle
import time

import numpy as np
import psycopg2

from anomaly_detector import CusumTracker, WINDOW_SAMPLES, score_latest_window, train_model

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

PREDICTION_INTERVAL = float(os.environ.get("RUL_PREDICTION_INTERVAL", "15"))  # 每隔幾秒跑一輪 RUL 預測
WINDOW_SECONDS = float(os.environ.get("RUL_WINDOW_SECONDS", "120"))  # RUL 回歸視窗長度（秒）
MIN_R_SQUARED = float(os.environ.get("RUL_MIN_R_SQUARED", "0.3"))  # 趨勢顯著性門檻，太低代表只是雜訊
MIN_POINTS = 10  # 視窗內至少要有這麼多筆彙總點才做回歸，太少統計上不可靠

ANOMALY_CHECK_INTERVAL = float(os.environ.get("ANOMALY_CHECK_INTERVAL", "30"))  # 頻譜異常評分的週期

CURRENT_UPPER = 6.5
CURRENT_LOWER = 5.5

# 頻譜訊號版振動 RUL：不追蹤振動「振幅」（SCT-05 實驗證實振幅在頻譜劣化時幾乎不動），
# 改追蹤 thd_ratio（諧波/基頻能量比）的上升趨勢——它在劣化早期就穩定爬升。做法跟電流 RUL
# 同架構（滑動視窗線性回歸 + 外推撞門檻），只是輸入換成 anomaly_scores 表累積的 thd_ratio 時序。
SPECTRAL_RUL_WINDOW_SECONDS = float(os.environ.get("SPECTRAL_RUL_WINDOW_SECONDS", "900"))  # thd 回歸視窗（anomaly_scores 每 30s 一筆，900s≈30 點）
SPECTRAL_MIN_POINTS = 10  # thd 序列至少這麼多點才回歸
SPECTRAL_MIN_R_SQUARED = float(os.environ.get("SPECTRAL_MIN_R_SQUARED", "0.3"))  # 趨勢顯著性門檻
VIBRATION_THD_DANGER = float(os.environ.get("VIBRATION_THD_DANGER", "0.05"))  # thd 劣化門檻：健康約0.005，SCT-05全劣化達0.10，取0.05當危險線

# 兩段式 RUL：CUSUM 警報當「偵測 onset」，onset 後只用劣化段資料 fit 斜率。
# 動機（SCT-05 實驗教訓）：單段式固定 900s 視窗會把「健康期平線」混進 fit，斜率被拉平、
# 早期 ETA 高估（曾一開始估 5.4 小時才慢慢收斂）。錨到 onset、只 fit 劣化段 → 斜率乾淨、ETA 早就準。
SPECTRAL_RUL_MAX_WINDOW = float(os.environ.get("SPECTRAL_RUL_MAX_WINDOW", "1800"))  # onset 後最多回看這麼久的 thd（避免視窗無限長、斜率變太平均）
# onset 安全網：光靠 CUSUM(score) 當閘門太脆弱——實測 V05-X 的 CUSUM 沒觸發，害 RUL 卡在 STABLE
# 即使 thd 已飆到 0.099。所以再加一條：thd 本身升破這個 floor（約健康基線 0.005 的 3 倍）也算 onset。
THD_ONSET_FLOOR = float(os.environ.get("THD_ONSET_FLOOR", "0.015"))

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


def fetch_thd_series(conn, device_id: str, sensor_id: str, window_seconds: float) -> list[tuple[float, float]]:
    """從 anomaly_scores 撈最近 window_seconds 秒的 (時間, thd_ratio) 序列，給頻譜 RUL 做趨勢外推。"""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT extract(epoch FROM ts) AS t, thd_ratio
            FROM anomaly_scores
            WHERE device_id = %s AND sensor_id = %s AND thd_ratio IS NOT NULL
              AND ts > now() - (%s * interval '1 second')
            ORDER BY ts
            """,
            (device_id, sensor_id, window_seconds),
        )
        return cur.fetchall()


def predict_spectral_rul(slope: float, r_squared: float, recent_thd: float):
    """回傳 (目標門檻, 預估還要幾秒撞到, 狀態)。thd_ratio 只會因劣化而「上升」，
    所以只看正斜率；趨勢不顯著或沒在上升就判 STABLE。"""
    if r_squared < SPECTRAL_MIN_R_SQUARED or slope <= 1e-9:
        return None, None, "STABLE"
    target = VIBRATION_THD_DANGER
    if recent_thd >= target:
        return target, 0.0, "CRITICAL"
    eta_seconds = (target - recent_thd) / slope
    if eta_seconds < 300:
        status = "CRITICAL"
    elif eta_seconds < 1800:
        status = "WARNING"
    else:
        status = "STABLE"
    return target, eta_seconds, status


def insert_prediction(
    conn, device_id, sensor_id, value, slope, r_squared, target, eta_seconds, status,
    metric_kind: str = "current_mean",
) -> None:
    """metric_kind：被追蹤的量——電流 RUL 是 'current_mean'（電流均值），
    振動頻譜 RUL 是 'thd_ratio'（諧波比值）。value 欄位存的就是該量的目前值。"""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO rul_predictions
                (ts, device_id, sensor_id, current_mean, drift_rate_per_min, r_squared,
                 target_threshold, eta_seconds, status, metric_kind)
            VALUES (now(), %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (device_id, sensor_id, value, slope * 60, r_squared, target, eta_seconds, status, metric_kind),
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
# {(device_id, sensor_id): onset_epoch}；CUSUM 首次警報的時刻，兩段式 RUL 用它當劣化起點錨。
_rul_onset: dict[tuple[str, str], float] = {}
# {(device_id, sensor_id): 健康基線 rms}；ML RUL 的 rms_norm=現值/基線，用最早幾筆健康 rms 算一次。
_rms_baseline: dict[tuple[str, str], float] = {}


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
    """用 RF 模型從時域特徵預測 RUL（分鐘）。回傳 (rul_min, rms_norm, slope_per_min)，資料不足回傳 None。"""
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
    rul_min = float(_RUL_MODEL.predict([[rms_norm, kurt, crest, slope]])[0])
    return rul_min, rms_norm, slope


def run_anomaly_detection_cycle(conn) -> None:
    for device_id in DEVICES:
        n = int(device_id.split("-")[1])
        for sensor_id, fundamental_hz in anomaly_channels(n):
            key = (device_id, sensor_id)

            if key not in _anomaly_models:
                bundle = train_model(conn, device_id, sensor_id, fundamental_hz=fundamental_hz)
                if bundle is None:
                    continue  # 歷史資料還不夠，下一輪再試
                _anomaly_models[key] = bundle
                _cusum_trackers[key] = CusumTracker()
                kind = "電流" if sensor_id.startswith("E") else "振動"
                print(
                    f"[ai_service] {device_id}/{sensor_id}（{kind} {fundamental_hz:.0f}Hz）頻譜異常模型訓練完成"
                    f"（{bundle['n_train_windows']} 個訓練窗口）"
                )

            result = score_latest_window(conn, device_id, sensor_id, _anomaly_models[key])
            if result is None:
                continue
            score, is_anomaly, features = result
            cusum_value, is_degrading = _cusum_trackers[key].update(score)
            insert_anomaly_score(conn, device_id, sensor_id, score, is_anomaly, features, cusum_value, is_degrading)

            # 兩段式 RUL 的階段1：CUSUM 首次警報 = 劣化 onset，記下時刻當趨勢 fit 的起點錨。
            if is_degrading and key not in _rul_onset:
                _rul_onset[key] = time.time()
                print(f"[ai_service] 劣化 onset 標記: {device_id}/{sensor_id}（RUL 進入階段2，開始倒數）")

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
        f"振動 RUL 改用 ML 模型（RF，{ml}；32 顆真實軸承訓練、跨 IMS 泛化驗證）"
    )
    print(
        f"[ai_service] 頻譜異常評分啟動：每 {ANOMALY_CHECK_INTERVAL:.0f}s 一輪、每個特徵窗口 {WINDOW_SAMPLES} 筆，"
        f"涵蓋電流({CURRENT_FUNDAMENTAL_HZ:.0f}Hz) + 振動({VIBRATION_FUNDAMENTAL_HZ:.0f}Hz) 共 6 通道/台，含線上 CUSUM 漸進劣化偵測"
    )

    conn = connect()
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
                    rul_min, rms_norm, slope = result
                    if rul_min < ML_RUL_CRITICAL_MIN:
                        status = "CRITICAL"
                    elif rul_min < ML_RUL_WARNING_MIN:
                        status = "WARNING"
                    else:
                        status = "STABLE"
                    insert_prediction(
                        conn, device_id, sensor_id, rms_norm, slope, None, ML_MAX_RUL_MIN,
                        rul_min * 60.0, status, metric_kind="ml_rf",
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
