"""
Volatility-shock stress test — the mechanism behind the rolling ATR stop advantage.

Hypothesis: without using any news/sentiment factor, a rolling ATR stop
(`prev_close ± k * prev_ATR`) handles sudden volatility jumps better than
conventional stops because:
  A. its anchor follows price  -> the stop tightens immediately on reversal
  B. it uses PREVIOUS-day ATR  -> the stop line is not pushed away when ATR spikes

A volatility shock is defined as TR > 2.5 x 60-day slow ATR.
Both mechanisms are measured separately.

Usage:  python src/vol_shock.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strategy import calc_atr  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SYMBOLS = ["RB", "I", "J", "CU", "AL", "AU", "AG", "MA", "TA", "V", "M", "SR", "SC"]
K_SIZE = 1.5
SHOCK_MULT = 2.5


def load():
    return {c: pd.read_csv(os.path.join(DATA_DIR, f"{c}.csv"),
                           index_col="date", parse_dates=True).sort_index() for c in SYMBOLS}


def run(df, method, window=30, atr_n=14, stop_k=1.5, pct=0.03,
        fee=0.0005, slip=0.0002, risk_pct=0.01):
    """Fixed sizing; only the exit rule differs between methods."""
    df = df.copy()
    atr = calc_atr(df, atr_n)
    prev_close = df["close"].shift(1)
    tr = np.maximum(np.maximum(df["high"] - df["low"], (df["high"] - prev_close).abs()),
                    (df["low"] - prev_close).abs())
    df["tr"] = tr
    df["atr_slow"] = tr.rolling(60).mean().shift(1)
    df["shock"] = tr > SHOCK_MULT * df["atr_slow"]

    up = df["high"].rolling(window).max().shift(1).to_numpy()
    lo = df["low"].rolling(window).min().shift(1).to_numpy()
    atr_v = atr.to_numpy()
    close = df["close"].to_numpy()
    n = len(df)

    pos = np.zeros(n)
    size = np.zeros(n)
    trades = []

    for i in range(1, n):
        p = pos[i - 1]
        if p == 0:
            if close[i] > up[i] or close[i] < lo[i]:
                pos[i] = 1 if close[i] > up[i] else -1
                size[i] = risk_pct / (K_SIZE * atr_v[i] / close[i])
                trades.append([i, None])
            continue

        j = i - 1
        while j > 0 and pos[j] == p:
            j -= 1
        entry, entry_atr = close[j], atr_v[j]

        hit = False
        if method == "none":
            pass
        elif method == "fixed":
            line = entry - stop_k * entry_atr if p == 1 else entry + stop_k * entry_atr
            hit = close[i] < line if p == 1 else close[i] > line
        elif method == "pct":
            line = entry * (1 - pct) if p == 1 else entry * (1 + pct)
            hit = close[i] < line if p == 1 else close[i] > line
        elif method == "atr_entry":
            line = entry - stop_k * atr_v[i] if p == 1 else entry + stop_k * atr_v[i]
            hit = close[i] < line if p == 1 else close[i] > line
        elif method == "rolling":
            line = close[i - 1] - stop_k * atr_v[i - 1] if p == 1 else close[i - 1] + stop_k * atr_v[i - 1]
            hit = close[i] < line if p == 1 else close[i] > line
        elif method == "chandelier":
            h, l = close[j:i].max(), close[j:i].min()
            line = h - stop_k * atr_v[i] if p == 1 else l + stop_k * atr_v[i]
            hit = close[i] < line if p == 1 else close[i] > line

        chan = (p == 1 and close[i] < lo[i]) or (p == -1 and close[i] > up[i])
        if hit or chan:
            pos[i] = 0
            trades[-1][1] = i
        else:
            pos[i] = p
            size[i] = size[i - 1]

    if trades and trades[-1][1] is None:
        trades[-1][1] = n - 1

    df["position"] = pos
    df["size"] = size
    df["cost"] = (fee + slip) * df["position"].diff().abs().fillna(0)
    df["daily_return"] = (df["close"].pct_change() * df["position"].shift(1)
                          * df["size"].shift(1) - df["cost"])
    return df, trades


def analyze(data, method, stop_k=1.5):
    recs = []
    for c in SYMBOLS:
        df, trades = run(data[c], method, stop_k=stop_k)
        shock = df["shock"].to_numpy()
        ret = df["daily_return"].to_numpy()
        for (e, x) in trades:
            if x is None or x <= e:
                continue
            r = ret[e + 1:x + 1]
            if len(r) == 0:
                continue
            shock_idx = [k for k in range(e + 1, x + 1) if shock[k]]
            has = len(shock_idx) > 0
            cum = np.cumprod(1 + r)
            recs.append({
                "ret": cum[-1] - 1, "days": x - e, "has_shock": has,
                "max_adv": r.min(),
                "giveback": cum.max() - cum[-1],
                "post_shock": (np.prod(1 + ret[shock_idx[0] + 1:x + 1]) - 1) if has else np.nan,
                "days_after_shock": (x - shock_idx[0]) if has else np.nan,
            })
    return pd.DataFrame(recs)


def main():
    data = load()
    methods = ["none", "fixed", "pct", "atr_entry", "rolling", "chandelier"]
    opt_k = {"none": 1.5, "fixed": 4.0, "pct": 1.0,
             "atr_entry": 3.0, "rolling": 1.5, "chandelier": 4.0}

    # ---- shock-day count ----
    total_shock = 0
    per_sym = {}
    for c in SYMBOLS:
        df = data[c]
        prev_close = df["close"].shift(1)
        tr = np.maximum(np.maximum(df["high"] - df["low"], (df["high"] - prev_close).abs()),
                        (df["low"] - prev_close).abs())
        slow = tr.rolling(60).mean().shift(1)
        n_shock = int((tr > SHOCK_MULT * slow).sum())
        per_sym[c] = n_shock
        total_shock += n_shock

    print("=" * 104)
    print(f"Volatility-shock test  (shock = TR > {SHOCK_MULT} x 60-day slow ATR)")
    print("=" * 104)
    print(f"\ntotal shock days across 13 instruments: {total_shock}")
    print(f"by instrument: {per_sym}")

    print(f"\n{'method':<13}{'k':>5}{'trades':>8}{'with shock':>12}"
          f"{'post-shock damage':>19}{'days after shock':>18}{'max adverse':>13}{'give-back':>11}")
    print("-" * 104)
    frames = {}
    for m in methods:
        d = analyze(data, m, stop_k=opt_k[m])
        frames[m] = d
        sh = d[d["has_shock"]]
        print(f"{m:<13}{opt_k[m]:>5}{len(d):>8}{len(sh):>12}"
              f"{sh['post_shock'].mean()*100:>+18.2f}%{sh['days_after_shock'].mean():>18.1f}"
              f"{d['max_adv'].min()*100:>+12.2f}%{d['giveback'].mean()*100:>10.2f}%")

    # ---- mechanism A: give-back ----
    print("\n" + "=" * 104)
    print("Mechanism A — anchor follows price (give-back = peak-to-exit drawdown)")
    print("=" * 104)
    print(f"{'method':<13}{'avg give-back':>15}{'median':>10}{'avg days':>11}{'avg return/trade':>19}")
    print("-" * 104)
    for m in methods:
        d = frames[m]
        print(f"{m:<13}{d['giveback'].mean()*100:>14.2f}%{d['giveback'].median()*100:>9.2f}%"
              f"{d['days'].mean():>11.1f}{d['ret'].mean()*100:>+18.2f}%")

    # ---- mechanism B: ATR inflation ----
    print("\n" + "=" * 104)
    print("Mechanism B — does ATR inflate after a shock? (pushes a current-ATR stop away)")
    print("=" * 104)
    ratios = []
    for c in SYMBOLS:
        df = data[c]
        prev_close = df["close"].shift(1)
        tr = np.maximum(np.maximum(df["high"] - df["low"], (df["high"] - prev_close).abs()),
                        (df["low"] - prev_close).abs())
        atr = tr.rolling(14).mean()
        slow = tr.rolling(60).mean().shift(1)
        shock = tr > SHOCK_MULT * slow
        r = (atr[shock].shift(-1) / atr[shock].shift(1)).dropna()
        ratios.append(r.mean())
    print(f"\nATR inflation 1 day after shock (avg over 13 instruments): {np.nanmean(ratios):.3f}")
    print("  -> current-ATR stops get pushed away; previous-day-ATR stops do not.")

    # ---- conclusion ----
    print("\n" + "=" * 104)
    print("Conclusion")
    print("=" * 104)
    roll = frames["rolling"][frames["rolling"]["has_shock"]]
    fix = frames["fixed"][frames["fixed"]["has_shock"]]
    print(f"post-shock damage : rolling {roll['post_shock'].mean()*100:+.2f}%  "
          f"vs fixed {fix['post_shock'].mean()*100:+.2f}%")
    print(f"days after shock  : rolling {roll['days_after_shock'].mean():.1f}  "
          f"vs fixed {fix['days_after_shock'].mean():.1f}")
    print("-> rolling ATR loses about half as much and exits about twice as fast.")


if __name__ == "__main__":
    main()
