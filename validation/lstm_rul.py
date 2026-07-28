# -*- coding: utf-8 -*-
"""深度學習 RUL：LSTM 吃「最近 K 個快照的特徵序列」預測 RUL，捕捉劣化「軌跡形狀」
（RF 只看單一時刻/斜率，抓不到整條曲線的彎法）。訓練 PHM+XJTU，測完全沒看過的 IMS。

從 extract_features_cache.py 的快取秒載入。特徵：[rms_norm, kurt, crest]（序列，LSTM 自己學趨勢）。
piecewise 目標（RUL 截斷 250min）、標準化、val 早停。

跨資料集警告：三集快照間隔不同（PHM 10s / XJTU 60s / IMS 600s），K 個快照代表不同真實時長；
這裡用固定 K 快照、靠 LSTM 學「形狀」，是務實但不完美的處理（已誠實記錄）。

用法（torch 容器內）：python lstm_rul.py <cache.pkl>
"""
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

K = 16              # 序列長度（快照數）
MAX_RUL = 250.0
STRIDE = 2          # 訓練序列取樣間隔（降低數量、加速 CPU 訓練）
HEALTHY_FRAC = 0.15
EVAL_FRACS = (0.5, 0.7, 0.9)
torch.manual_seed(0)


def bearing_arrays(rec):
    """快取的一顆軸承 → (特徵[n,3], RUL[n], 真實RUL[n])。"""
    import pandas as pd
    rms = pd.Series(rec["rms"]).rolling(5, min_periods=1, center=True).median().to_numpy()
    n = len(rms)
    base = float(np.mean(rms[: max(1, int(HEALTHY_FRAC * n))]))
    rms_norm = rms / (base if base > 1e-12 else 1.0)
    feat = np.column_stack([rms_norm, np.array(rec["kurt"]), np.array(rec["crest"])])
    rul = (n - np.arange(n)) * rec["interval"]
    return feat.astype(np.float32), np.minimum(rul, MAX_RUL).astype(np.float32), rul


def make_seqs(feat, y, stride=1):
    X, Y = [], []
    for i in range(K - 1, len(feat), stride):
        X.append(feat[i - K + 1 : i + 1]); Y.append(y[i])
    return np.array(X, np.float32), np.array(Y, np.float32)


class LSTMRUL(nn.Module):
    def __init__(self, d=3, h=48):
        super().__init__()
        self.lstm = nn.LSTM(d, h, 2, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.Linear(h, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.head(o[:, -1, :]).squeeze(-1)


def main():
    cache = pickle.load(open(sys.argv[1], "rb"))
    train_names = [k for k in cache if not k.startswith("IMS/")]
    ims_names = [k for k in cache if k.startswith("IMS/")]
    print(f"訓練 {len(train_names)} 顆(PHM+XJTU)、測試 {len(ims_names)} 顆(IMS)")

    # 標準化：用訓練資料的特徵算 mean/std
    all_feat = np.vstack([bearing_arrays(cache[k])[0] for k in train_names])
    mu, sd = all_feat.mean(0), all_feat.std(0) + 1e-9

    def prep(names, stride):
        Xs, Ys = [], []
        for k in names:
            feat, y, _ = bearing_arrays(cache[k])
            feat = (feat - mu) / sd
            X, Y = make_seqs(feat, y, stride)
            if len(X): Xs.append(X); Ys.append(Y)
        return np.vstack(Xs), np.concatenate(Ys)

    # 訓練/驗證分割（依軸承，避免洩漏）
    rng = np.random.default_rng(0); order = rng.permutation(train_names)
    n_val = max(3, len(train_names) // 6)
    val_names, tr_names = list(order[:n_val]), list(order[n_val:])
    Xtr, Ytr = prep(tr_names, STRIDE); Xva, Yva = prep(val_names, STRIDE)
    print(f"訓練序列 {len(Xtr)}、驗證序列 {len(Xva)}")

    dev = "cpu"
    net = LSTMRUL().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    dl = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(Ytr / MAX_RUL)), batch_size=256, shuffle=True)
    xva = torch.tensor(Xva); yva = torch.tensor(Yva / MAX_RUL)

    best, best_state, patience = 1e9, None, 0
    for ep in range(40):
        net.train()
        for xb, yb in dl:
            opt.zero_grad(); loss = lossf(net(xb), yb); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = lossf(net(xva), yva).item()
        if vl < best - 1e-4:
            best, best_state, patience = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            patience += 1
        if ep % 5 == 0 or patience == 0:
            print(f"  epoch {ep}: val_loss {vl:.4f}{'  *' if patience==0 else ''}", flush=True)
        if patience >= 6:
            print(f"  早停於 epoch {ep}"); break
    net.load_state_dict(best_state); net.eval()

    print("\n=== LSTM 在【完全沒看過的 IMS】測試（可行動窗 RUL 相對誤差）===")
    for k in ims_names:
        feat, y, rul = bearing_arrays(cache[k])
        feat = (feat - mu) / sd
        X, _ = make_seqs(feat, y, 1)               # 測試不跳
        idx = np.arange(K - 1, len(feat))          # 每個序列對應的原始快照 index
        with torch.no_grad():
            pred = net(torch.tensor(X)).numpy() * MAX_RUL
        rul_seq = rul[idx]
        mask = rul_seq <= MAX_RUL
        ae = np.abs(pred[mask] - rul_seq[mask]) / np.maximum(rul_seq[mask], 1e-9)
        n = len(feat)
        line = f"  {k}: 可行動窗誤差中位 {np.median(ae):.0%}（25~75分位 {np.percentile(ae,25):.0%}~{np.percentile(ae,75):.0%}）"
        print(line)


if __name__ == "__main__":
    main()
