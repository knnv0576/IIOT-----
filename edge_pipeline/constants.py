"""Phase1 共用常數，對應 SKILL.md 規格。"""

SAMPLE_RATE_HZ = 4000
SAMPLE_INTERVAL = 1 / SAMPLE_RATE_HZ  # 0.25ms

TEMP_TICK_INTERVAL = 4000  # 每 4000 個 4kHz tick 才多產生 1 筆溫度（1Hz）

FLUSH_TICKS = 1000  # 本地雙緩衝出貨門檻：滿 1000 tick
FLUSH_SECONDS = 0.25  # 或經過 250ms（兩者擇一先到）

AC_FREQ_HZ = 60  # 電流疊加的交流電頻率

CURRENT_PHASES = ("A", "B", "C")
VIBRATION_AXES = ("X", "Y", "Z")

QUEUE_MAXSIZE = 500  # 每台機台專屬 Queue 的批次上限
