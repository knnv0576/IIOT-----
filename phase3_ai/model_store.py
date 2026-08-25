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


# ── 訓練特徵集的存取（可版控的訓練基線）─────────────────────────────────
# 為什麼存特徵而不是原始波形：SVM 實際訓練的就是特徵矩陣。原始波形在 2kHz × 70 感測器下
# 是 ~1GB/5分鐘，且 sensor_readings 只保留 20 分鐘，不可能長期保存；特徵矩陣整個機隊
# 只有幾百 KB，小到可以進 git 版控。存下來就能「事後完全重現當初的訓練」——同一份特徵
# 永遠擬合出同一個模型（指紋相同），對稽核與比較很有用。
# 限制：特徵是用「當時的特徵定義」算的。若日後改了 FEATURE_NAMES 或 SCHEMA_VERSION，
# 舊的特徵集就作廢（load_features 會擋下來），那時只能靠原始波形快照重算
# （見 tools/export_baseline_snapshot.py）。
FEATURESET_MANIFEST = "manifest.json"


def save_features(out_dir, device_id: str, sensor_id: str, feature_vectors,
                  fundamental_hz: float) -> Path:
    """把一個通道的訓練特徵矩陣存成 .npz。out_dir 由呼叫端指定（通常是可版控的目錄）。"""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{device_id}_{sensor_id}.npz"
    np.savez_compressed(
        path,
        X=np.asarray(feature_vectors, dtype=np.float64),
        fundamental_hz=np.float64(fundamental_hz),
    )
    return path


def write_featureset_manifest(out_dir, extra: dict | None = None) -> Path:
    """寫整組特徵集的 manifest：記錄特徵定義與版本，載入時用來擋不相容的舊特徵集。"""
    from spectral_features import FEATURE_NAMES  # 延遲載入，避免模組層級的循環相依
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / FEATURESET_MANIFEST
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "created_at": datetime.now(timezone.utc).isoformat(),
        **(extra or {}),
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_features(in_dir, device_id: str, sensor_id: str):
    """載入一個通道的訓練特徵矩陣，回傳 (X, fundamental_hz)；不存在或不相容回傳 None。

    相容性檢查是刻意嚴格的：特徵定義一旦改變，用舊特徵集訓練出來的模型會學到錯的東西，
    而且不會有任何明顯症狀——寧可擋下來要求重新產生，也不要靜默地訓練出錯的模型。
    """
    d = Path(in_dir)
    path = d / f"{device_id}_{sensor_id}.npz"
    if not path.exists():
        return None
    mpath = d / FEATURESET_MANIFEST
    if mpath.exists():
        try:
            from spectral_features import FEATURE_NAMES
            m = json.loads(mpath.read_text(encoding="utf-8"))
            if (m.get("schema_version") != SCHEMA_VERSION
                    or list(m.get("feature_names", [])) != list(FEATURE_NAMES)):
                return None
        except Exception:
            return None
    try:
        z = np.load(path)
        return z["X"], float(z["fundamental_hz"])
    except Exception:
        return None


def delete(device_id: str, sensor_id: str) -> bool:
    """刪掉一個通道的模型（強制重訓用）。"""
    pkl_path, meta_path = _paths(device_id, sensor_id)
    existed = pkl_path.exists()
    pkl_path.unlink(missing_ok=True)
    meta_path.unlink(missing_ok=True)
    return existed
