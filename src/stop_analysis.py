"""
Stop-loss analysis — the core research contribution.

1. Ablation: with vs without stop-loss (everything else identical)
2. Fair comparison: six stop-loss methods, each tuned to its own optimal k
3. Walk-forward: is the optimal k stable out-of-sample?

Usage:  python src/stop_analysis.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strategy import calc_atr, extract_trades  # noqa: E402
from backtest import load, SYMBOLS  # noqa: E402


def run(df, method, window=30, atr_n=14, k=1.5, pct=0.03,
        fee=0.0005, slippage=0.0002, risk_pct=0.01):
    """Run strategy with a selectable stop-loss method.

    methods: none | fixed | pct | atr_entry | rolling | chandelier
    """
    df = df.copy()
    df["upper"] = df["high"].rolling(window).max().shift(1)
    df["lower"] = df["low"].rolling(window).min().shift(1)
    df["atr"] = calc_atr(df, atr_n)

    close = df["close"].to_numpy()
    atr = df["atr"].to_numpy()
    up = df["upper"].to_numpy()
    lo = df["lower"].to_numpy()
    n = len(df)
    position = np.zeros(n)
    size = np.zeros(n)

    for i in range(1, n):
        prev = position[i - 1]
        if prev == 0:
            if close[i] > up[i]:
                position[i] = 1
                size[i] = risk_pct / (k * atr[i] / close[i])
            elif close[i] < lo[i]:
                position[i] = -1
                size[i] = risk_pct / (k * atr[i] / close[i])
            continue

        j = i - 1
        while j > 0 and position[j] == prev:
            j -= 1
        entry_price, entry_atr = close[j], atr[j]

        exit_now = False
        if method == "none":
            pass
        elif method == "fixed":          # Turtle: entry price +/- k*entry ATR
            line = entry_price - k * entry_atr if prev == 1 else entry_price + k * entry_atr
            exit_now = (close[i] < line) if prev == 1 else (close[i] > line)
        elif method == "pct":            # percent stop
            line = entry_price * (1 - pct) if prev == 1 else entry_price * (1 + pct)
            exit_now = (close[i] < line) if prev == 1 else (close[i] > line)
        elif method == "atr_entry":      # fixed anchor, current ATR
            line = entry_price - k * atr[i] if prev == 1 else entry_price + k * atr[i]
            exit_now = (close[i] < line) if prev == 1 else (close[i] > line)
        elif method == "rolling":        # THIS PROJECT: prev_close +/- k*prev_ATR
            line = close[i - 1] - k * atr[i - 1] if prev == 1 else close[i - 1] + k * atr[i - 1]
            exit_now = (close[i] < line) if prev == 1 else (close[i] > line)
        elif method == "chandelier":     # ratchet trailing
            sh, sl = close[j:i].max(), close[j:i].min()
            line = sh - k * atr[i] if prev == 1 else sl + k * atr[i]
            exit_now = (close[i] < line) if prev == 1 else (close[i] > line)

        chan = (prev == 1 and close[i] < lo[i]) or (prev == -1 and close[i] > up[i])
        if exit_now or chan:
            position[i] = 0
        else:
            position[i] = prev
            size[i] = size[i - 1]

    df["position"] = position
    df["size"] = size
    df["position_change"] = df["position"].diff().abs().fillna(0)
    df["cost"] = (fee + slippage) * df["position_change"]
    df["daily_return"] = (df["close"].pct_change() * df["position"].shift(1)
                          * df["size"].shift(1) - df["cost"])
    return df


def evaluate(data, method, period=None, **kw):
    rets, trades = [], []
    for s in SYMBOLS:
        d = run(data[s], method, **kw)
        rets.append(d["daily_return"].rename(s))
        if period is None:
            trades += extract_trades(d)
    port = pd.concat(rets, axis=1).mean(axis=1)
    if period == "train":
        port = port[port.index.year <= 2023]
    elif period == "test":
        port = port[port.index.year >= 2024]

    eq = (1 + port.fillna(0)).cumprod()
    total = eq.iloc[-1] - 1
    years = len(port) / 252
    annual = (1 + total) ** (1 / years) - 1
    sharpe = port.mean() / port.std() * np.sqrt(252)
    mdd = ((eq - eq.cummax()) / eq.cummax()).min()
    out = {"total": total, "annual": annual, "sharpe": sharpe, "mdd": mdd,
           "calmar": annual / abs(mdd), "n": len(trades)}

    if trades:
        r = np.array([t[0] for t in trades])
        L, W = r[r <= 0], r[r > 0]
        out.update({"max_loss": L.min(), "pf": W.sum() / abs(L.sum())})
    return out


def main():
    data = load()

    # ---------- 1. ablation ----------
    print("=" * 88)
    print("1. Ablation: with vs without stop-loss (identical otherwise)")
    print("=" * 88)
    for label, kw in [("no stop", {"method": "none"}),
                      ("rolling ATR stop", {"method": "rolling", "k": 1.5})]:
        r = evaluate(data, **kw)
        print(f"{label:<20} total {r['total']*100:+.2f}%  sharpe {r['sharpe']:.2f}  "
              f"maxDD {r['mdd']*100:.2f}%  maxLoss {r['max_loss']*100:.2f}%  trades {r['n']}")

    # ---------- 2. fair comparison (each method -> its own best k) ----------
    print("\n" + "=" * 88)
    print("2. Fair comparison: each method tuned to its own optimal k")
    print("=" * 88)
    ks = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0]
    methods = ["fixed", "pct", "atr_entry", "rolling", "chandelier"]
    best = {}
    for m in methods:
        b = max(((k, evaluate(data, m, k=k)) for k in ks),
                key=lambda x: x[1]["sharpe"])
        best[m] = b

    base = evaluate(data, "none")
    rows = [("no stop", "-", base)] + [(m, b[0], b[1]) for m, b in best.items()]
    print(f"{'method':<16}{'best k':>8}{'total':>10}{'sharpe':>9}{'maxDD':>10}"
          f"{'maxLoss':>10}{'trades':>8}")
    print("-" * 88)
    for label, k, r in sorted(rows, key=lambda x: -x[2]["sharpe"]):
        print(f"{label:<16}{str(k):>8}{r['total']*100:>+9.2f}%{r['sharpe']:>9.2f}"
              f"{r['mdd']*100:>+9.2f}%{r['max_loss']*100:>+9.2f}%{r['n']:>8}")

    # ---------- 3. walk-forward on the stop parameter ----------
    print("\n" + "=" * 88)
    print("3. Walk-forward: is the optimal k stable out-of-sample?")
    print("=" * 88)
    train_best = max(((k, evaluate(data, "rolling", period="train", k=k)) for k in ks),
                     key=lambda x: x[1]["sharpe"])
    print(f"train-optimal k = {train_best[0]} (sharpe {train_best[1]['sharpe']:.2f})")
    for period, name in [("train", "in-sample 2020-2023"), ("test", "out-of-sample 2024-2025")]:
        r = evaluate(data, "rolling", period=period, k=train_best[0])
        b = evaluate(data, "none", period=period)
        print(f"  {name:<24} rolling k={train_best[0]}: sharpe {r['sharpe']:.2f}  "
              f"| no stop: sharpe {b['sharpe']:.2f}")


if __name__ == "__main__":
    main()
