"""Phase1 共用常數，對應 SKILL.md 規格。"""

# 取樣率 2kHz（12 萬筆/秒）。原始規格是 4kHz，磁碟搬到 F: NVMe 後一度恢復 4kHz，但實測發現
# 「4kHz + 完整 AI（60 通道 FFT/SVM 評分）+ 10 個機台進程」會把 8 核 CPU（Ryzen 5700X3D）壓垮，
# 機台進程被 CPU 飢餓餓死（10 台死剩 4 台）、吞吐量崩到 ~40%。所以瓶頸從「磁碟」變成「CPU」，
# 只能降回 2kHz 讓 edge_pipeline 的 CPU 負載減半、8 核撐得住。頻譜 Nyquist 從 2kHz 降到 1kHz，
# 60Hz 基頻與 120/180/240Hz 諧波偵測不受影響。若日後換更多核心的 CPU 可再評估 4kHz。
SAMPLE_RATE_HZ = 2000
SAMPLE_INTERVAL = 1 / SAMPLE_RATE_HZ  # 0.5ms

TEMP_TICK_INTERVAL = 2000  # 每 2000 個 2kHz tick 才多產生 1 筆溫度（1Hz）

FLUSH_TICKS = 500  # 本地雙緩衝出貨門檻：滿 500 tick（2kHz 下剛好 250ms，與時間門檻同時命中）
FLUSH_SECONDS = 0.25  # 或經過 250ms（兩者擇一先到）

AC_FREQ_HZ = 60  # 電流疊加的交流電頻率

CURRENT_PHASES = ("A", "B", "C")
VIBRATION_AXES = ("X", "Y", "Z")

QUEUE_MAXSIZE = 500  # 每台機台專屬 Queue 的批次上限

# 漸進劣化注入（預設不啟用，透過 HTTP 控制 API 對指定機台觸發，供 Phase3 AI 端到端驗證用）：
# 觸發後在基頻上疊加一個「隨時間線性成長」的諧波，模擬真實軸承/機械早期故障在頻譜裡逐漸
# 冒出的邊帶。振幅從 0 線性爬到「基波振幅 × DEGRADE_MAX_HARMONIC_RATIO」，費時
# DEGRADATION_RAMP_SECONDS。用來驗證頻譜 One-Class SVM + CUSUM 能在整體振幅還沒明顯變化前
# 就提前預警——這是危險模式（瞬間拉高振幅）驗不出來的漸進趨勢能力。
DEGRADATION_RAMP_SECONDS = 600.0  # 諧波從 0 爬到滿的時間（10 分鐘）
DEGRADE_MAX_HARMONIC_RATIO = 0.30  # 滿劣化時，注入諧波振幅相對基波振幅的比例
DEGRADE_HARMONIC_MULT = 3  # 疊加幾次諧波（電流 60×3=180Hz、振動 30×3=90Hz）
# 振動劣化時整體振幅(RMS)的成長倍率：真實軸承磨損 RMS 會暴增（末期約 6~10 倍），
# 但原本的模擬器劣化只加諧波、RMS 幾乎不變，導致「振幅型」的 ML RUL 模型抓不到。
# 加這個讓振動 RMS 隨劣化成長（滿劣化時 ×(1+此值)），同時觸發頻譜(諧波)與 ML RUL(振幅)兩條路徑。
DEGRADE_VIBRATION_AMPLITUDE_GROWTH = 5.0
