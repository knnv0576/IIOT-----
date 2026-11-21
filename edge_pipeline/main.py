"""Phase 1+2+4 進入點。

這支程式做三件事：
1. 依 machine_config.py 的設定產生 10 台機台的模擬進程（Phase1）。
2. 把機台模擬器與 MQTT 進線的資料，統一批次寫入 TimescaleDB（Phase2）。
3. 定期把自己的 CPU/記憶體/吞吐量寫進資料庫，給 Grafana 儀表板監控用（Phase4）。

危險模式切換提供兩種介面：
- 標準輸入的命令列介面（適合 `docker compose run` 前景互動執行）。
- HTTP API（適合 `docker compose up -d` 背景執行時，用 curl/瀏覽器操作）。
兩者共用同一份 danger_flags，切換效果一樣，可以同時開著互不干擾。
"""

import hmac
import json
import multiprocessing as mp
import os
import queue
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import psutil
import psycopg2

import db_writer
import machine_config
from constants import QUEUE_MAXSIZE
from machine_process import run_machine
from mqtt_ingest import run_mqtt_ingest
from records import explode_batch_raw

# --- MQTT 進線設定 ---
# 對應 docker-compose.yml 裡 edge_pipeline 服務的環境變數；本機測試沒設的話用預設值。
MQTT_BROKER_HOST = os.environ.get("MQTT_BROKER_HOST", "mosquitto")  # MQTT broker 的主機名稱
MQTT_BROKER_PORT = int(os.environ.get("MQTT_BROKER_PORT", "1883"))  # MQTT broker 的連接埠
MQTT_TOPIC_PREFIX = os.environ.get("MQTT_TOPIC_PREFIX", "iiot")  # 訂閱主題的前綴，實際訂閱 "<prefix>/#"
MQTT_USERNAME = os.environ.get("MQTT_USERNAME") or None  # broker 帳密（mosquitto 已禁止匿名連線）
MQTT_PASSWORD = os.environ.get("MQTT_PASSWORD") or None
# MQTT 進線佇列上限：外部裝置灌爆訊息時，寧可丟掉新訊息（mqtt_ingest 會印 log），
# 也不要讓沒有上限的 queue 把記憶體吃光、拖垮整個 edge_pipeline。
MQTT_QUEUE_MAXSIZE = int(os.environ.get("MQTT_QUEUE_MAXSIZE", "100000"))

# --- 危險模式控制 HTTP API 設定 ---
HTTP_CONTROL_PORT = int(os.environ.get("HTTP_CONTROL_PORT", "8080"))
# POST 類操作（切換危險模式、劣化注入、/quit）必須帶 "Authorization: Bearer <token>"。
# 沒設定 token 時 POST 一律拒絕（fail-closed），不會變成「任何人都能操作」。
# GET /status 只回傳狀態、不改任何東西，維持免驗證，方便監控。
CONTROL_API_TOKEN = os.environ.get("CONTROL_API_TOKEN", "")

# --- TimescaleDB 連線與批次寫入設定 ---
TIMESCALEDB_DSN = os.environ["TIMESCALEDB_DSN"]  # postgresql://user:password@host:port/dbname，必須由外部提供

DB_BATCH_SIZE = 5000  # 本地緩衝累積滿這麼多筆，就觸發一次批次寫入
DB_BATCH_SECONDS = 0.5  # 就算筆數不到 DB_BATCH_SIZE，只要經過這麼久也強制寫入一次，避免資料卡在記憶體太久

# 斷線期間（資料庫連不上）本地緩衝最多允許累積這麼多筆（DB_BATCH_SIZE 的 10 倍）。
# 一旦超過，db_writer_loop 就會暫停從機台 Queue 撈新資料，讓機台自己的 Queue
# （原本就有 QUEUE_MAXSIZE 上限）去頂住背壓，而不是在這裡的 Python list 裡無限累積、把記憶體吃光。
MAX_BUFFER_SIZE = DB_BATCH_SIZE * 10

# 寫入失敗、嘗試重新連線又失敗之後，至少要間隔這麼久（秒）才會再試一次，
# 避免資料庫真的斷線很久時，程式瘋狂嘗試連線、把錯誤訊息洗版。
RECONNECT_COOLDOWN = 2.0

# 單一條資料庫連線實測寫入速度約 120,000~140,000 筆/秒，追不上模擬器全速的 240,000 筆/秒。
# 所以改用多條連線平行分攤 10 台機台的 Queue；實測 4 條並行可達 ~255,000 筆/秒，留有安全餘裕。
# 可以用環境變數 NUM_DB_WRITERS 覆寫這個數字（例如測試單一連線的效能時設成 1）。
NUM_DB_WRITERS = int(os.environ.get("NUM_DB_WRITERS", "4"))

# --- Phase4 系統健康監控共用的狀態 ---
# 所有 db_writer_loop（可能有好幾條平行 thread）每次成功寫入後，都會呼叫 _record_write_stats()
# 把這次寫了幾筆、花了多久時間記錄到這個共用字典裡；system_health_loop 每隔幾秒讀一次、歸零，
# 換算成 TPS（每秒寫入筆數）之後存進 system_health 表，給 Grafana 的系統監控面板使用。
# 用 threading.Lock() 保護是因為會有多條 db_writer thread 同時寫入這個字典。
_health_lock = threading.Lock()
_health_stats = {
    "rows_since_sample": 0,  # 累積筆數，system_health_loop 讀取後會歸零
    "last_latency_ms": 0.0,  # 最近一次批次寫入（INSERT）實際花費的時間，毫秒
}


def _record_write_stats(n_rows: int, latency_ms: float) -> None:
    """給 db_writer_loop 呼叫：回報這次批次寫入的筆數與耗時，累加進全域統計字典。"""
    with _health_lock:
        _health_stats["rows_since_sample"] += n_rows
        _health_stats["last_latency_ms"] = latency_ms


def partition_queues(queues: dict, n: int) -> list[dict]:
    """把 10 台機台的 Queue 依序輪流分配成 n 份（round-robin），讓每個 db_writer 各自負責一部分機台。

    參數：
        queues: {device_id: multiprocessing.Queue} 的完整對照表（10 台機台全部）。
        n: 要分成幾份，對應要開幾條平行寫入 thread（NUM_DB_WRITERS）。
    回傳：
        長度為 n 的 list，每個元素是一個 {device_id: Queue} 的子集合。
    """
    partitions: list[dict] = [dict() for _ in range(n)]
    for i, (device_id, machine_queue) in enumerate(queues.items()):
        partitions[i % n][device_id] = machine_queue
    return partitions


def build_workers(config: dict, stop_event):
    """依設定檔產生 10 台機台各自的 multiprocessing.Process、專屬 Queue、危險模式旗標。

    每台機台都是一個獨立的作業系統進程（不是執行緒），彼此不共用記憶體，
    這是 Phase1 規格要求的「硬體隔離」設計，繞開 Python GIL 的限制。

    參數：
        config: machine_config.load_config() 回傳的設定字典（機台清單、溫度/電流/振動門檻）。
        stop_event: multiprocessing.Event()，所有機台進程共用同一個，設定後所有進程都會收到停止訊號。
    回傳：
        (processes, queues, danger_flags, degrade_flags) 四個 list/dict：
        - processes: 10 個 multiprocessing.Process 物件（尚未 start()）
        - queues: {device_id: 該機台專屬的 multiprocessing.Queue}，機台把資料批次塞進這裡
        - danger_flags: {device_id: multiprocessing.Value}，跨進程共享的危險模式開關（0=正常,1=危險）
        - degrade_flags: {device_id: multiprocessing.Value}，漸進劣化起始時間（-1=未劣化）
    """
    queues = {}
    danger_flags = {}
    degrade_flags = {}
    processes = []
    for machine in config["machines"]:
        device_id = machine["device_id"]
        size_class = machine["size_class"]  # 機台大小分級（I/II/III），決定振動 ISO 門檻
        machine_queue = mp.Queue(maxsize=QUEUE_MAXSIZE)  # 這台機台專屬的輸出佇列，容量見 constants.py
        danger_flag = mp.Value("b", 0)  # 'b' = signed char，跨進程共享的布林值（0/1）
        degrade_flag = mp.Value("d", -1.0)  # 'd' = double，-1=未劣化，否則=劣化起始的 wall time
        process = mp.Process(
            target=run_machine,  # 實際在子進程裡跑的函式，見 machine_process.py
            args=(device_id, size_class, config, machine_queue, danger_flag, stop_event, degrade_flag),
            name=device_id,  # 讓 process_monitor_loop 印警示訊息時能直接顯示機台編號
            daemon=True,  # 主進程結束時，這些子進程也會被強制結束，不會變成孤兒進程
        )
        queues[device_id] = machine_queue
        danger_flags[device_id] = danger_flag
        degrade_flags[device_id] = degrade_flag
        processes.append(process)
    return processes, queues, danger_flags, degrade_flags


def db_writer_loop(
    writer_id: int,
    queues: dict,
    mqtt_queue: "queue.Queue | None",
    stop_event,
    report_interval: float = 5.0,
):
    """Phase2 寫入 worker：負責 `queues` 這一部分機台（外加其中一個 worker 順便扛 MQTT 進線），
    把資料攤平成統一的 (timestamp, device_id, sensor_id, value) tuple，累積到門檻後批次寫入 TimescaleDB。
    多個 worker 各自用獨立連線平行運作，才追得上模擬器全速的吞吐量（見 NUM_DB_WRITERS 的說明）。

    斷線復原：寫入失敗不會讓這條 thread 整個死掉——會保留還沒寫進去的資料、
    嘗試重新連線，並在 RECONNECT_COOLDOWN 冷卻時間內避免瘋狂重試；
    資料庫恢復正常後，會自動接著把累積下來的資料補寫進去。

    參數：
        writer_id: 這條 worker 的編號（0 ~ NUM_DB_WRITERS-1），只用來在 log 訊息裡標示是哪一條。
        queues: 這條 worker 負責的機台子集合，{device_id: Queue}（來自 partition_queues()）。
        mqtt_queue: 只有其中一條 worker（writer_id == 0）會拿到這個非 None 的值，
            負責順便把 MQTT 進線的資料也一起寫進資料庫；其餘 worker 傳 None。
        stop_event: 全域停止旗標，設定後這條迴圈會結束並做善後（flush 剩餘資料、關閉連線）。
        report_interval: 每隔幾秒印一次「過去 N 秒寫入了幾筆」的統計訊息。
    """
    # 連線一律走 ensure_connection()，不在這裡直接 connect()：docker compose 的 depends_on
    # 只保證 timescaledb 容器「已啟動」、不保證資料庫「已可連線」（第一次 up 時 init.sql 要跑好幾秒）。
    # 舊版在這裡直接 connect()，一失敗整條 thread 就死掉，主進程卻還活著、容器不會重啟——
    # 結果是這幾台機台的 Queue 沒人抽、模擬器被背壓卡死，表面上卻一切正常。
    conn = None
    buffer: list[tuple] = []  # 本地緩衝，累積尚未寫進資料庫的 (ts, device_id, sensor_id, value) tuple
    last_flush = time.time()  # 上一次呼叫 flush() 的時間，用來判斷 DB_BATCH_SECONDS 是否已到
    last_report = time.time()  # 上一次印出吞吐量統計的時間
    written_since_report = 0  # 距離上次印出統計以來，實際成功寫入資料庫的筆數
    connection_ok = False  # 目前資料庫連線是否正常；False 代表尚未連上或正在斷線重連
    last_reconnect_attempt = 0.0  # 上一次嘗試連線的時間戳，用來計算 RECONNECT_COOLDOWN
    ever_connected = False  # 只用來決定 log 要寫「連線成功」還是「重新連線成功」

    def ensure_connection() -> bool:
        """確保有可用連線；沒有就嘗試（重新）連線，並遵守 RECONNECT_COOLDOWN 避免瘋狂重試。"""
        nonlocal conn, connection_ok, last_reconnect_attempt, ever_connected
        if connection_ok:
            return True
        now = time.time()
        if now - last_reconnect_attempt < RECONNECT_COOLDOWN:
            return False  # 還在冷卻時間內，資料留在 buffer 裡等下一輪再嘗試
        last_reconnect_attempt = now
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            conn = None
        try:
            conn = db_writer.connect(TIMESCALEDB_DSN)
        except psycopg2.Error as exc:
            print(
                f"[db_writer-{writer_id}] 資料庫連線失敗（{exc.__class__.__name__}），"
                f"{RECONNECT_COOLDOWN:.0f} 秒後再試"
            )
            return False
        connection_ok = True
        print(
            f"[db_writer-{writer_id}] "
            + ("重新連線成功，繼續補寫累積的資料" if ever_connected else "資料庫連線成功")
        )
        ever_connected = True
        return True

    def flush():
        """把 buffer 裡目前累積的資料整批寫入資料庫；失敗的話保留資料並嘗試重連，不拋出例外。"""
        nonlocal last_flush, written_since_report, connection_ok
        if not buffer:
            last_flush = time.time()
            return

        if not ensure_connection():
            return

        try:
            t0 = time.perf_counter()
            # 遇到「資料本身有問題」的批次時，壞的那幾筆會被剔除回傳、其餘照寫——
            # 不能整批保留重試，否則同一筆壞資料會讓這條 thread 永遠卡在重試迴圈裡。
            rejected = db_writer.insert_rows_isolating_bad(conn, buffer)
            latency_ms = (time.perf_counter() - t0) * 1000
            n_written = len(buffer) - len(rejected)
            _record_write_stats(n_written, latency_ms)  # 回報給 Phase4 系統健康監控
            written_since_report += n_written
            buffer.clear()
            if rejected:
                sample = ", ".join(repr(r) for r in rejected[:3])
                print(
                    f"[db_writer-{writer_id}] 剔除 {len(rejected)} 筆資料庫拒收的資料"
                    f"（資料格式/數值不合法，重試也不會成功），例如：{sample}"
                )
        except psycopg2.Error as exc:
            # 寫入失敗（最常見原因：資料庫斷線/重啟中）。不重新拋出例外，
            # 避免這條 thread 因為未捕捉的例外而永久死掉、之後再也沒人抽這幾台機台的 Queue。
            # 下一次 flush() 會透過 ensure_connection() 重連。
            print(
                f"[db_writer-{writer_id}] 寫入失敗（{exc.__class__.__name__}），"
                f"保留 {len(buffer)} 筆待重試，嘗試重新連線..."
            )
            connection_ok = False
        last_flush = time.time()

    ensure_connection()  # 先試一次；失敗也沒關係，之後每次 flush() 都會再試

    try:
        while not stop_event.is_set():
            got_any = False  # 這一輪迴圈有沒有真的撈到任何新資料，沒有的話最後會小睡一下避免空轉燒 CPU

            # buffer 已經累積太多還沒寫進去（例如資料庫斷線中），就先不再從 Queue 撈新資料，
            # 讓機台自己的 Queue（有 maxsize 保護）去頂住背壓，而不是在這裡的 buffer 無限吃記憶體。
            if len(buffer) < MAX_BUFFER_SIZE:
                for device_id, machine_queue in queues.items():
                    try:
                        batch = machine_queue.get(timeout=0.05)  # 逾時就跳過，換下一台機台
                    except Exception:
                        continue
                    got_any = True
                    buffer.extend(explode_batch_raw(batch))  # 把一批 NumPy 矩陣攤平成一筆筆 tuple

                if mqtt_queue is not None:
                    # MQTT 進線是低頻資料，直接用 get_nowait() 把目前有的全部撈乾，不用逐條等逾時。
                    while True:
                        try:
                            row = mqtt_queue.get_nowait()
                        except queue.Empty:
                            break
                        got_any = True
                        buffer.append(row)

            now = time.time()
            if len(buffer) >= DB_BATCH_SIZE or (buffer and now - last_flush >= DB_BATCH_SECONDS):
                flush()
            if now - last_report >= report_interval:
                pending = f"（另有 {len(buffer)} 筆待重試中）" if not connection_ok and buffer else ""
                print(
                    f"[db_writer-{writer_id}] 過去 {report_interval:.0f} 秒寫入 TimescaleDB "
                    f"{written_since_report} 筆{pending}"
                )
                written_since_report = 0
                last_report = now
            if not got_any:
                time.sleep(0.02)  # 完全沒資料可撈時稍微睡一下，避免無意義地忙碌輪詢
    finally:
        # 不管是正常收到 stop_event 還是發生例外，結束前都盡量把剩餘資料寫掉、關閉連線。
        try:
            flush()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass


def process_monitor_loop(processes: list, stop_event, check_interval: float = 2.0, writer_threads=None):
    """背景健康檢查：定期確認 10 台機台的進程、以及各條資料庫寫入 thread 是不是還活著。

    死掉的機台只會印出一次警示（機台編號、PID、結束代碼），不會自動重啟——
    自動重啟牽涉到要重建該機台的 Queue、danger_flag，並讓對應的 db_writer thread
    重新指到新的 Queue 上，複雜度較高，這裡先用警示訊息讓人工決定要不要處理。

    寫入 thread 正常情況下不會死（連線錯誤、壞資料都在 db_writer_loop 內部處理掉了），
    真的死了代表遇到程式 bug——它負責的機台會因為 Queue 沒人抽而被背壓卡住，
    外表看起來卻一切正常，所以一樣要警示出來。

    參數：
        processes: build_workers() 回傳的 10 個 Process 物件清單。
        stop_event: 全域停止旗標，設定後這個檢查迴圈就會結束。
        check_interval: 每隔幾秒檢查一次。
        writer_threads: db_writer_loop 的 threading.Thread 清單（可省略）。
    """
    known_dead: set[str] = set()  # 已經警示過的機台編號/thread 名稱，避免重複洗版
    while not stop_event.is_set():
        for process in processes:
            if not process.is_alive() and process.name not in known_dead:
                known_dead.add(process.name)
                print(
                    f"[process_monitor] 警示：機台 {process.name} 的進程已停止運作"
                    f"（PID={process.pid}, exitcode={process.exitcode}），不會自動重啟，需要人工檢查"
                )
        for thread in writer_threads or ():
            if not thread.is_alive() and thread.name not in known_dead and not stop_event.is_set():
                known_dead.add(thread.name)
                print(
                    f"[process_monitor] 警示：資料庫寫入 thread {thread.name} 已意外結束"
                    f"（上方應有 traceback），它負責的機台資料不會再寫入，需要人工檢查/重啟服務"
                )
        stop_event.wait(check_interval)


def system_health_loop(stop_event, interval: float = 2.0):
    """Phase4 系統健康監控：定期把 edge_pipeline 這個進程整體的 CPU%/記憶體用量、
    近似即時 TPS（來自 _health_stats）、最近一次批次寫入延遲，寫進 system_health 表，
    給 Grafana 的「系統健康監控」面板讀取，用來證明系統在高負載下依然健康平穩。

    參數：
        stop_event: 全域停止旗標，設定後這個監控迴圈就會結束。
        interval: 每隔幾秒量測並寫入一次。
    """
    process = psutil.Process()  # 代表「目前這個 Python 進程」本身的資源用量
    process.cpu_percent()  # 第一次呼叫只是建立量測基準點，回傳值沒有意義、故意丟棄

    conn = None  # 這個監控迴圈自己專用的資料庫連線，跟 db_writer_loop 的連線互相獨立
    connection_ok = False
    last_reconnect_attempt = 0.0

    def ensure_connected():
        """確保 conn 是可用的連線；斷線時嘗試重連，並遵守 RECONNECT_COOLDOWN 避免瘋狂重試。"""
        nonlocal conn, connection_ok, last_reconnect_attempt
        if connection_ok:
            return True
        now = time.time()
        if now - last_reconnect_attempt < RECONNECT_COOLDOWN:
            return False
        last_reconnect_attempt = now
        try:
            conn = psycopg2.connect(TIMESCALEDB_DSN)
            conn.autocommit = True
            connection_ok = True
            return True
        except psycopg2.Error:
            return False

    ensure_connected()

    while not stop_event.is_set():
        time.sleep(interval)

        # cpu_percent() 這次呼叫回傳的是「距離上次呼叫以來」這段時間的平均使用率。
        cpu_percent = process.cpu_percent()
        memory_mb = process.memory_info().rss / (1024 * 1024)  # RSS：實際佔用的實體記憶體

        with _health_lock:
            rows = _health_stats["rows_since_sample"]
            latency_ms = _health_stats["last_latency_ms"]
            _health_stats["rows_since_sample"] = 0  # 讀取後歸零，下一輪重新累計
        tps = rows / interval  # 這段取樣期間的平均每秒寫入筆數

        if not ensure_connected():
            continue  # 資料庫連不上，這一輪的監控數值就先不寫（下一輪再試）
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO system_health (ts, cpu_percent, memory_mb, tps, write_latency_ms) "
                    "VALUES (now(), %s, %s, %s, %s)",
                    (cpu_percent, memory_mb, tps, latency_ms),
                )
        except psycopg2.Error:
            connection_ok = False
            try:
                conn.close()
            except Exception:
                pass

    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass


def control_loop(danger_flags: dict, stop_event):
    """讀取使用者從標準輸入打的指令，藉此手動切換機台的危險模式，或結束整個程式。

    支援的指令：
        SCT-05 DANGER   把 SCT-05 切成危險模式
        SCT-05 NORMAL   把 SCT-05 切回正常模式
        STATUS          印出所有機台目前是正常還是危險模式
        QUIT            設定 stop_event，結束整個程式

    這個函式跑在背景 thread（見 main()）：`for line in sys.stdin` 會一直阻塞到下一行輸入，
    如果放在主執行緒，HTTP 的 POST /quit 設了 stop_event 主執行緒也醒不過來、程式不會結束。
    stdin 遇到 EOF（例如容器沒開 stdin）只代表命令列介面不能用，系統照常運作。

    參數：
        danger_flags: build_workers() 回傳的 {device_id: multiprocessing.Value} 對照表。
        stop_event: 全域停止旗標，收到 QUIT 指令時會被設定。
    """
    print("輸入指令切換危險模式，例如: SCT-05 DANGER / SCT-05 NORMAL / STATUS / QUIT")
    for line in sys.stdin:
        if stop_event.is_set():
            break
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0].upper() == "QUIT":
            stop_event.set()
            break
        if parts[0].upper() == "STATUS":
            print({device_id: bool(flag.value) for device_id, flag in danger_flags.items()})
            continue
        if len(parts) != 2 or parts[1].upper() not in ("DANGER", "NORMAL"):
            print("格式錯誤，範例: SCT-05 DANGER")
            continue
        device_id, mode = parts[0].upper(), parts[1].upper()
        if device_id not in danger_flags:
            print(f"未知機台: {device_id}")
            continue
        danger_flags[device_id].value = 1 if mode == "DANGER" else 0
        print(f"{device_id} -> {mode}")


def _make_control_request_handler(danger_flags: dict, degrade_flags: dict, stop_event, token: str = ""):
    """建立一個綁定了 danger_flags/degrade_flags/stop_event 的 HTTP handler 類別。

    用 closure 把共享狀態帶進去，是因為 BaseHTTPRequestHandler 的子類別
    是由 HTTPServer 自己在每次請求時 instantiate，沒辦法直接在建構子塞自訂參數，
    所以改成用外層函式包一層，讓內層類別透過 closure 拿到需要的共享狀態。

    token：POST 類操作要求的 Bearer token；空字串代表沒設定，POST 一律拒絕。
    """

    def _authorized(header_value: "str | None") -> bool:
        if not token:
            return False
        scheme, _, presented = (header_value or "").partition(" ")
        if scheme.lower() != "bearer":
            return False
        # compare_digest：比對時間不隨「前幾個字元對不對」而變，避免被逐字元猜出 token
        return hmac.compare_digest(presented.strip().encode("utf-8"), token.encode("utf-8"))

    class ControlRequestHandler(BaseHTTPRequestHandler):
        def _send_json(self, status_code: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            # GET /status -> 回傳所有機台的危險模式與漸進劣化狀態
            if self.path == "/status":
                self._send_json(200, {
                    device_id: {
                        "danger": bool(danger_flags[device_id].value),
                        "degrading": degrade_flags[device_id].value >= 0.0,
                    }
                    for device_id in danger_flags
                })
            else:
                self._send_json(404, {"error": "not found", "hint": "try GET /status"})

        def do_POST(self):
            # POST /danger/SCT-05  -> 切成危險模式（瞬間拉高振幅）
            # POST /degrade/SCT-05 -> 觸發漸進劣化（頻譜諧波隨時間成長，供 AI 驗證用）
            # POST /normal/SCT-05  -> 切回正常（同時解除危險模式與漸進劣化）
            # POST /quit           -> 結束整個程式（等同下 QUIT 指令；compose 底下會被自動重啟）
            # 以上都必須帶 Authorization: Bearer <CONTROL_API_TOKEN>
            if not token:
                self._send_json(403, {"error": "控制 API 未設定 CONTROL_API_TOKEN，POST 操作已停用"})
                return
            if not _authorized(self.headers.get("Authorization")):
                self.send_response(401)
                self.send_header("WWW-Authenticate", "Bearer")
                body = json.dumps({"error": "未授權：需要 Authorization: Bearer <CONTROL_API_TOKEN>"},
                                  ensure_ascii=False).encode("utf-8")
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path == "/quit":
                stop_event.set()
                self._send_json(200, {"status": "stopping"})
                return

            parts = self.path.strip("/").split("/")
            if len(parts) == 2 and parts[0] in ("danger", "normal", "degrade"):
                device_id = parts[1].upper()
                if device_id not in danger_flags:
                    self._send_json(404, {"error": f"未知機台: {device_id}"})
                    return
                if parts[0] == "danger":
                    danger_flags[device_id].value = 1
                elif parts[0] == "degrade":
                    degrade_flags[device_id].value = time.time()  # 記下劣化起始時間，開始線性成長
                else:  # normal：兩種異常都解除
                    danger_flags[device_id].value = 0
                    degrade_flags[device_id].value = -1.0
                self._send_json(200, {device_id: parts[0].upper()})
            else:
                self._send_json(404, {"error": "not found", "hint": "try POST /danger|degrade|normal/<device_id>"})

        def log_message(self, format, *args):
            pass  # 靜音：預設會把每個 request 都印一行到 stdout，量一多會洗掉其他 log

    return ControlRequestHandler


def http_control_loop(danger_flags: dict, degrade_flags: dict, stop_event, port: int = HTTP_CONTROL_PORT):
    """背景 HTTP API：讓 `docker compose up -d`（背景執行、沒有互動式 stdin 可用）
    也能用 curl/瀏覽器切換異常模式，功能跟 control_loop() 的 stdin 介面等價、共用同一份旗標。

    端點：
        GET  /status               回傳所有機台的危險模式與漸進劣化狀態
        POST /danger/<device_id>   把指定機台切成危險模式（瞬間拉高振幅）
        POST /degrade/<device_id>  觸發漸進劣化（頻譜諧波隨時間成長，供 Phase3 AI 驗證）
        POST /normal/<device_id>   解除該機台的危險模式與漸進劣化
        POST /quit                 結束整個程式
    """
    handler_cls = _make_control_request_handler(danger_flags, degrade_flags, stop_event, CONTROL_API_TOKEN)
    server = ThreadingHTTPServer(("0.0.0.0", port), handler_cls)
    server.timeout = 0.5  # 讓 handle_request() 最多等 0.5 秒就回來，才能定期檢查 stop_event
    auth_note = "POST 需 Bearer token" if CONTROL_API_TOKEN else "未設定 CONTROL_API_TOKEN，POST 操作停用"
    print(
        f"[http_control] 控制 API 啟動於 0.0.0.0:{port}"
        f"（GET /status, POST /danger|degrade|normal/<device>；{auth_note}）"
    )
    while not stop_event.is_set():
        server.handle_request()
    server.server_close()


def main():
    """程式進入點：組裝所有元件並啟動整套系統。

    啟動順序：
    1. 讀取設定、建立 10 台機台的進程/Queue/危險旗標，並實際 start() 這些進程（Phase1）。
    2. 啟動背景監控 thread：機台健康檢查、系統資源監控（Phase4）。
    3. 啟動危險模式控制用的 HTTP API thread（背景執行時不方便用 stdin，改用 curl/瀏覽器）。
    4. 啟動 MQTT 進線 thread。
    5. 依 NUM_DB_WRITERS 啟動對應數量的平行資料庫寫入 thread，把 10 台機台的 Queue
       平均分配給它們，其中第 0 條順便處理 MQTT 進線的資料（Phase2）。
    6. 命令列介面 control_loop() 放到背景 thread；主執行緒只負責等 stop_event
       （來源：stdin 的 QUIT、HTTP 的 POST /quit、Ctrl+C、docker stop 送來的 SIGTERM）。
    7. 結束時先等寫入 thread 把手上的資料 flush 完，再確保 10 台機台的進程都真的被清乾淨。

    注意：docker-compose.yml 設了 restart: unless-stopped，程式自己結束（包括 POST /quit）
    之後 Docker 會把容器重新啟動，所以在 compose 底下 /quit 的效果等於「重啟」。
    要真正停掉請用 docker compose stop edge_pipeline。
    """
    config = machine_config.load_config()
    stop_event = mp.Event()  # 唯一的全域停止旗標，所有進程/thread 都靠檢查這個來知道該收尾了
    processes, queues, danger_flags, degrade_flags = build_workers(config, stop_event)

    for process in processes:
        process.start()

    # SIGTERM（docker stop / compose down）改成走正常收尾流程。必須在 start() 機台進程之後才設，
    # 子進程才不會繼承這個 handler（子進程由下面的 terminate() 負責收）。
    # 另外 Python 在容器裡是 PID 1，PID 1 對「沒有自訂 handler」的 SIGTERM 會直接忽略，
    # 不設的話 docker stop 只能等 10 秒逾時後 SIGKILL，緩衝裡的資料全部丟掉。
    signal.signal(signal.SIGTERM, lambda signum, frame: stop_event.set())

    writer_threads: list[threading.Thread] = []
    threading.Thread(
        target=process_monitor_loop,
        args=(processes, stop_event),
        kwargs={"writer_threads": writer_threads},  # 同一個 list，下面建好寫入 thread 後就會看得到
        daemon=True,
    ).start()
    threading.Thread(target=system_health_loop, args=(stop_event,), daemon=True).start()
    threading.Thread(
        target=http_control_loop, args=(danger_flags, degrade_flags, stop_event), daemon=True
    ).start()

    # 執行緒安全的一般 Queue（不是 multiprocessing.Queue），因為 MQTT client 的 callback
    # 是在同一個進程內的背景 thread 執行，不需要跨進程 IPC。上限見 MQTT_QUEUE_MAXSIZE 的說明。
    mqtt_queue: "queue.Queue" = queue.Queue(maxsize=MQTT_QUEUE_MAXSIZE)
    threading.Thread(
        target=run_mqtt_ingest,
        args=(MQTT_BROKER_HOST, MQTT_BROKER_PORT, MQTT_TOPIC_PREFIX, mqtt_queue, stop_event),
        kwargs={"username": MQTT_USERNAME, "password": MQTT_PASSWORD},
        daemon=True,
    ).start()

    for writer_id, partition in enumerate(partition_queues(queues, NUM_DB_WRITERS)):
        # 只有第 0 條寫入 thread 會拿到 mqtt_queue，負責順便處理 MQTT 進線；其餘傳 None。
        writer_mqtt_queue = mqtt_queue if writer_id == 0 else None
        writer_thread = threading.Thread(
            target=db_writer_loop,
            args=(writer_id, partition, writer_mqtt_queue, stop_event),
            name=f"db_writer-{writer_id}",
            daemon=True,
        )
        writer_thread.start()
        writer_threads.append(writer_thread)

    # stdin 命令列介面放背景 thread（原因見 control_loop 的說明）。daemon=True：
    # 它可能永遠卡在讀 stdin，不能讓它擋住程式結束。
    threading.Thread(target=control_loop, args=(danger_flags, stop_event), daemon=True).start()

    try:
        # 用帶逾時的 wait 輪詢，而不是無限期 wait()：這樣 Ctrl+C 才能即時打斷主執行緒。
        while not stop_event.wait(0.5):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        print("[main] 收到停止訊號，等待寫入 thread 把緩衝資料寫完...")
        # 先讓寫入 thread 跑完它們 finally 裡的最後一次 flush，再處理機台進程。
        # 資料庫正常時一次 flush 不到一秒；斷線中的話就放棄，不無限期卡住關機流程。
        # 所有寫入 thread 共用一個總期限（不是每條各等 5 秒），整體要落在 docker stop 預設的
        # 10 秒寬限期內，否則會被 SIGKILL 打斷。
        join_deadline = time.time() + 5.0
        for writer_thread in writer_threads:
            writer_thread.join(timeout=max(0.0, join_deadline - time.time()))
        for process in processes:
            # 機台進程若還在等自己的 Queue 有空間（put() 被阻塞），
            # 一旦消費端（db_writer_loop）也因為 stop_event 停止抽乾 Queue，
            # 就可能卡住不會自己結束（這是 multiprocessing.Queue 的已知行為：
            # 子進程只要 put() 過東西，join() 前必須等內部 feeder thread 把資料撢乾淨，
            # 消費端一停，這個等待就可能失敗）。所以先短暫等待，還沒結束就直接 terminate()，
            # 對純模擬資料來說，犧牲最後不到 1 秒的資料不會有實質影響。
            process.join(timeout=0.2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)


if __name__ == "__main__":
    main()
