# -*- coding: utf-8 -*-
"""CUSUM 地板取值的假警重放：拿健康分數餵進「正式服務用的同一個 CusumTracker」
（含 8σ→地板 的時間衰減），掃不同地板值，數健康通道多久誤報一次，用來把地板定案。

理念：理論(Siegmund)定起點、實測定案。預設用高斯健康分數當「保守上界」（真實分數更緊、
假警更少）；DB 恢復後可用 --scores 餵真實 anomaly_scores.score 做定論測試。

用法：
  # 蒙地卡羅（保守估計）：60 通道各跑 1 週，掃地板 3~8σ
  python cusum_false_alarm_replay.py --channels 60 --hours 168 --floors 3,4,5,6,8

  # 用真實分數（每行一個 score，或每通道一檔用 --scores-glob）
  python cusum_false_alarm_replay.py --scores healthy_scores.txt --floors 5,6
"""
import argparse
import sys
from pathlib import Path

import numpy as np

# 匯入正式服務的 CusumTracker（測到的就是真實衰減邏輯）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase3_ai"))
import anomaly_detector as ad  # noqa: E402
from anomaly_detector import CusumTracker, CUSUM_WARMUP_SAMPLES  # noqa: E402


def theoretical_arl(floor_mult: float, k: float = 0.5) -> float:
    """單邊 CUSUM 在無偏移下的近似假警平均間隔（Siegmund，樣本數）。"""
    b = floor_mult + 1.166
    return 2.0 * (np.exp(b) - b - 1.0)


def gen_healthy(rng, n: int, mu0: float, sigma0: float, dist: str) -> np.ndarray:
    """產生 n 筆健康分數。gauss=常態（保守上界）；tight=下行尾巴更輕，貼近實測有界分布。"""
    if dist == "tight":
        # 實測健康分數又緊又有界（99% 在 -0.37 以上、最差 -0.72）：向下的偏移比常態罕見。
        # 用「向下偏移只取常態的一半幅度」近似這種輕下尾。
        z = rng.standard_normal(n)
        z = np.where(z < 0, z * 0.5, z)
        return mu0 + sigma0 * z
    return rng.normal(mu0, sigma0, size=n)


def replay_channel(scores: np.ndarray) -> tuple[int, int, int | None]:
    """把一條健康分數序列餵進 CusumTracker，回傳 (校準後樣本數, 假警事件數, 首次穿越樣本idx)。

    兩個量測慣例（都不改動正式程式，只在此腳本量測）：
    - 首次穿越：onset 在正式服務只觸發一次，故「這條健康通道整個生命週期會不會誤報 onset」
      看的是第一次 is_degrading=True 的時刻。
    - 事件數/ARL：CUSUM 跨門檻後 value 不會自己歸零、會連續多輪都 True，直接數樣本會爆。
      故每次警報後把 value 重置為 0（標準 CUSUM 警報後重啟），才能算出正確的假警「速率」。"""
    t = CusumTracker()
    post_cal = 0
    events = 0
    first_passage = None
    for s in scores:
        _, degrading = t.update(float(s))
        if not t.calibrated:
            continue
        post_cal += 1
        if degrading:
            events += 1
            if first_passage is None:
                first_passage = post_cal
            t.value = 0.0  # 警報後重啟，量測 ARL 用
    return post_cal, events, first_passage


def run_sweep(floors, channels, samples, mu0, sigma0, dist, interval, seed, score_streams):
    rng = np.random.default_rng(seed)
    print(f"{'地板':>5} | {'窗內誤報onset通道':>16} | {'中位首次穿越':>12} | "
          f"{'實測ARL':>9} | {'單通道假警間隔':>14} | {'理論ARL':>8}")
    print("-" * 84)
    for f in floors:
        ad.CUSUM_H_FLOOR_MULT = f  # monkeypatch：tracker.update() 會在校準時讀這個模組全域
        total_post, total_events, ch_onset = 0, 0, 0
        first_passages = []
        if score_streams is not None:
            streams = score_streams
        else:
            streams = (gen_healthy(rng, CUSUM_WARMUP_SAMPLES + samples, mu0, sigma0, dist)
                       for _ in range(channels))
        n_ch = 0
        for s in streams:
            post, events, fp = replay_channel(np.asarray(s, dtype=float))
            total_post += post
            total_events += events
            if fp is not None:
                ch_onset += 1
                first_passages.append(fp)
            n_ch += 1
        emp_arl = (total_post / total_events) if total_events else float("inf")
        secs = emp_arl * interval if not np.isinf(emp_arl) else None
        interval_txt = ("無假警" if secs is None
                        else (f"{secs/3600:.1f} 小時" if secs < 3 * 86400 else f"{secs/86400:.1f} 天"))
        arl_txt = f"{emp_arl:.0f}" if not np.isinf(emp_arl) else "∞"
        fp_txt = f"{int(np.median(first_passages))} 樣本" if first_passages else "—"
        print(f"{f:>4.0f}σ | {ch_onset:>6}/{n_ch:<8} | {fp_txt:>12} | {arl_txt:>9} | "
              f"{interval_txt:>14} | {theoretical_arl(f):>8.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floors", default="3,4,5,6,8", help="逗號分隔的候選地板倍率")
    ap.add_argument("--channels", type=int, default=60, help="模擬通道數（機隊 6×10=60）")
    ap.add_argument("--hours", type=float, default=168.0, help="每通道模擬時長（小時）")
    ap.add_argument("--interval", type=float, default=30.0, help="每筆 CUSUM 樣本間隔（秒）")
    ap.add_argument("--mu0", type=float, default=-0.2, help="健康分數均值")
    ap.add_argument("--sigma0", type=float, default=0.10, help="健康分數標準差")
    ap.add_argument("--dist", choices=["gauss", "tight"], default="gauss",
                    help="gauss=常態(保守上界)、tight=輕下尾(貼近實測)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scores", help="真實分數檔（每行一個 score，整條當單通道；不指定則蒙地卡羅）")
    args = ap.parse_args()

    floors = [float(x) for x in args.floors.split(",")]
    samples = int(args.hours * 3600 / args.interval)

    score_streams = None
    if args.scores:
        vals = [float(x) for x in Path(args.scores).read_text().split() if x.strip()]
        score_streams = [vals]
        print(f"用真實分數 {len(vals)} 筆（單通道）；衰減週期 {ad.CUSUM_DECAY_PERIOD_SAMPLES} 樣本、"
              f"起始 {ad.CUSUM_H_MULT:.0f}σ")
    else:
        print(f"蒙地卡羅：{args.channels} 通道 × {args.hours:.0f}h（{samples} 樣本/通道 @{args.interval:.0f}s）、"
              f"健康分數 N({args.mu0},{args.sigma0}) [{args.dist}]、衰減 {ad.CUSUM_H_MULT:.0f}σ→地板")
    print(f"衰減：每 {ad.CUSUM_DECAY_PERIOD_SAMPLES} 樣本 ×{ad.CUSUM_DECAY_FACTOR}"
          f"（8σ→地板約 {ad.CUSUM_DECAY_PERIOD_SAMPLES*3*args.interval/3600:.1f}h）\n")
    run_sweep(floors, args.channels, samples, args.mu0, args.sigma0, args.dist,
              args.interval, args.seed, score_streams)


if __name__ == "__main__":
    main()
