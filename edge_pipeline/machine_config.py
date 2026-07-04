"""機台與物理參數設定。

這些數值原本對應 config/machine_config.xlsx 的內容，但改成直接寫在程式碼裡、
不在執行期讀取 Excel——Excel 檔案只保留作為需求對照參考，要調整參數就直接改這個檔案。
"""

MACHINES = [
    {"device_id": "SCT-01", "size_class": "I"},
    {"device_id": "SCT-02", "size_class": "I"},
    {"device_id": "SCT-03", "size_class": "I"},
    {"device_id": "SCT-04", "size_class": "I"},
    {"device_id": "SCT-05", "size_class": "I"},
    {"device_id": "SCT-06", "size_class": "I"},
    {"device_id": "SCT-07", "size_class": "II"},
    {"device_id": "SCT-08", "size_class": "II"},
    {"device_id": "SCT-09", "size_class": "III"},
    {"device_id": "SCT-10", "size_class": "III"},
]

TEMPERATURE = {"upper": 26.5, "setpoint": 26.0, "lower": 25.5}

# 百分比數字（6.0 代表 6.00%），對應原 Excel 電流欄位的小數（0.06 = 6.00%）換算後的值。
CURRENT = {"upper": 6.5, "setpoint": 6.0, "lower": 5.5}

# 振動速度規格表（mm/s RMS），依機台 Class 分級；"危險" 只有下限（None 代表無上限）。
VIBRATION_ISO = {
    "I": {
        "優良": (0.0, 0.71),
        "可接受": (0.71, 1.80),
        "警示": (1.80, 4.50),
        "危險": (4.50, None),
    },
    "II": {
        "優良": (0.0, 1.12),
        "可接受": (1.12, 2.80),
        "警示": (2.80, 7.10),
        "危險": (7.10, None),
    },
    "III": {
        "優良": (0.0, 1.80),
        "可接受": (1.80, 4.50),
        "警示": (4.50, 11.20),
        "危險": (11.20, None),
    },
}


def load_config() -> dict:
    return {
        "machines": MACHINES,
        "temperature": TEMPERATURE,
        "current": CURRENT,
        "vibration_iso": VIBRATION_ISO,
    }


def danger_threshold(config: dict, size_class: str) -> float:
    """回傳該 Class 的危險(D)門檻（mm/s），也就是切到 DANGER 模式時振動值必須超過的數字。"""
    lower, _ = config["vibration_iso"][size_class]["危險"]
    return lower


def good_zone_upper(config: dict, size_class: str) -> float:
    """回傳該 Class 的優良(A)區間上限，正常模式下振動值應落在這個數字以下。"""
    _, upper = config["vibration_iso"][size_class]["優良"]
    return upper


def acceptable_zone_upper(config: dict, size_class: str) -> float:
    """回傳該 Class 的可接受(B)區間上限。超過這個值代表進入警示(C)或危險(D)，
    用於 anomaly_extractor.py 判斷「異常」（警示+危險都算異常）。"""
    _, upper = config["vibration_iso"][size_class]["可接受"]
    return upper


def size_class_of(config: dict, device_id: str) -> str:
    for machine in config["machines"]:
        if machine["device_id"] == device_id:
            return machine["size_class"]
    raise KeyError(f"未知機台: {device_id}")
