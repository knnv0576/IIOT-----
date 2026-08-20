"""訓練好的模型的儲存層：把「訓練」與「推論」之間唯一的介面收在這裡。

設計目的（為什麼要獨立出來）：
原本 SVM 模型只活在 ai_service 的記憶體裡，服務每次啟動都重訓一次。這造成兩個問題：
1. 每次重啟都要等 300 秒累積資料 + 訓練，期間無法偵測。
2. **重訓出來的模型分數尺度會變**——實測 decision_function 的離散度在重訓前後差 1.5~3 倍。
   而 CUSUM 的基線（mu0/sigma0）是「機台 × 該次模型」的屬性，模型一換基線就失效，
   導致 slack 與門檻一起縮水、健康通道大量誤報。

把模型持久化之後，這兩個問題同時消失：模型不變 → 分數尺度不變 → 基線可以安全沿用。
模型指紋（fingerprint）就是用來保證這件事：基線只有在「模型完全相同」時才會被取用。

儲存位置由 MODEL_STORE_DIR 指定，預設 /app/model_store（docker volume，跨容器重建保留）。
注意與 /app/models 區分：那是 image 內建的 RUL RandomForest，唯讀、跟著程式碼走。
"""

import hashlib
import json
import os
import pickle
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

MODEL_STORE_DIR = Path(os.environ.get("MODEL_STORE_DIR", "/app/model_store"))
SVM_SUBDIR = "svm"
# 模型檔格式版本。日後若改了特徵定義或 bundle 結構，把這個數字加一，
# 舊模型就會被視為不相容而自動重訓，不會拿舊特徵的模型去評新特徵的分數。
SCHEMA_VERSION = 1


def _channel_dir() -> Path:
    d = MODEL_STORE_DIR / SVM_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _paths(device_id: str, sensor_id: str) -> tuple[Path, Path]:
    base = _channel_dir() / f"{device_id}_{sensor_id}"
    return base.with_suffix(".pkl"), base.with_suffix(".json")


def model_fingerprint(bundle: dict) -> str:
    """由模型與 scaler 的實際參數算出指紋。相同訓練結果 → 相同指紋。

    用途：CUSUM 基線只有在指紋相符時才會被取用。這樣「模型換了、基線卻沿用」這個
    實測會造成大量誤報的情況，在結構上就不可能發生。
    """
    model, scaler = bundle["model"], bundle["scaler"]
    h = hashlib.sha256()
    h.update(str(SCHEMA_VERSION).encode())
    h.update(str(bundle.get("fundamental_hz", "")).encode())
    for arr in (
        model.dual_coef_, model.support_vectors_, np.atleast_1d(model.intercept_),
        scaler.mean_, scaler.scale_,
    ):
        h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    return h.hexdigest()[:16]


def save(device_id: str, sensor_id: str, bundle: dict, meta_extra: dict | None = None) -> str:
    """存一個通道的模型，回傳指紋。

    先寫暫存檔再 os.replace（同目錄內為原子操作），避免推論端剛好讀到寫到一半的檔案。
    """
    pkl_path, meta_path = _paths(device_id, sensor_id)
    fingerprint = model_fingerprint(bundle)

    fd, tmp = tempfile.mkstemp(dir=str(pkl_path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, pkl_path)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise

    meta = {
        "device_id": device_id,
        "sensor_id": sensor_id,
        "fingerprint": fingerprint,
        "schema_version": SCHEMA_VERSION,
        "fundamental_hz": bundle.get("fundamental_hz"),
        "n_train_windows": bundle.get("n_train_windows"),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        **(meta_extra or {}),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return fingerprint


def load(device_id: str, sensor_id: str) -> tuple[dict, dict] | None:
    """載入一個通道的模型，回傳 (bundle, meta)；不存在或不相容回傳 None。

    不相容的情況（schema_version 不符、檔案損壞）一律當成「沒有模型」，讓呼叫端去重訓，
    不讓壞掉的模型悄悄拿去評分。
    """
    pkl_path, meta_path = _paths(device_id, sensor_id)
    if not pkl_path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("schema_version") != SCHEMA_VERSION:
            return None
        with open(pkl_path, "rb") as f:
            bundle = pickle.load(f)
    except Exception:
        return None
    return bundle, meta


def list_meta() -> list[dict]:
    """列出所有已存模型的中繼資料（給 CLI/巡檢用）。"""
    out = []
    for meta_path in sorted(_channel_dir().glob("*.json")):
        try:
            out.append(json.loads(meta_path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return out


def delete(device_id: str, sensor_id: str) -> bool:
    """刪掉一個通道的模型（強制重訓用）。"""
    pkl_path, meta_path = _paths(device_id, sensor_id)
    existed = pkl_path.exists()
    pkl_path.unlink(missing_ok=True)
    meta_path.unlink(missing_ok=True)
    return existed
