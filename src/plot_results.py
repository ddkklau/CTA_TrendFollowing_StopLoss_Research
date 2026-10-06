"""
Generate the two charts used in the README:
  1. stop_k_scan.png     — Sharpe vs stop parameter k (single-peak evidence)
  2. vol_shock.png       — post-shock damage / exit speed by stop method

Usage:  python src/plot_results.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strategy import donchian  # noqa: E402
from backtest import load, portfolio_returns  # noqa: E402
from vol_shock import run as run_stop, analyze, SYMBOLS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
os.makedirs(DOCS, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 10, "axes.grid": True,
    "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
})


def main():
    data = load()

    # ============ 1. Sharpe vs k ============
    ks = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0]
    sharpes = []
    for k in ks:
        p = portfolio_returns(data, k=k)
        sharpes.append(p.mean() / p.std() * np.sqrt(252))
    best = int(np.argmax(sharpes))

    # no-stop baseline (fixed sizing)
    rets_ns = [run_stop(data[s], "none")[0]["daily_return"].rename(s) for s in SYMBOLS]
    p_ns = pd.concat(rets_ns, axis=1).mean(axis=1)
    shp_ns = p_ns.mean() / p_ns.std() * np.sqrt(252)

    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(ks, sharpes, "o-", color="#1f77b4", lw=1.9, ms=5.5)
    ax.plot(ks[best], sharpes[best], "o", color="#d62728", ms=12,
            label=f"optimum k={ks[best]}  (Sharpe {sharpes[best]:.2f})")
    ax.axhline(shp_ns, color="#999999", ls="--", lw=1.1,
               label=f"no stop-loss ({shp_ns:.2f})")
    ax.set_xlabel("Stop parameter k   (stop distance = k × ATR)")
    ax.set_ylabel("Sharpe ratio")
    ax.set_title("Rolling ATR stop-loss: Sharpe vs parameter k\n"
                 "single peak at k=1.5 — too small over-trades, too large gives no protection")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "stop_k_scan.png"), bbox_inches="tight")
    plt.close(fig)
    print("saved docs/stop_k_scan.png")

    # ============ 2. volatility-shock comparison ============
    methods = ["none", "fixed", "atr_entry", "pct", "rolling"]
    opt_k = {"none": 1.5, "fixed": 4.0, "atr_entry": 3.0, "pct": 1.0, "rolling": 1.5}
    labels = {"none": "no stop", "fixed": "fixed (Turtle)", "atr_entry": "entry ± ATR",
              "pct": "percent 3%", "rolling": "rolling ATR\n(this project)"}

    damage, days = [], []
    for m in methods:
        d = analyze(data, m, stop_k=opt_k[m])
        sh = d[d["has_shock"]]
        damage.append(sh["post_shock"].mean() * 100)
        days.append(sh["days_after_shock"].mean())

    order = np.argsort(damage)
    names = [labels[methods[i]] for i in order]
    dmg = [damage[i] for i in order]
    dys = [days[i] for i in order]
    colors = ["#d62728" if "this project" in n else "#7f7f7f" for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    b1 = ax1.barh(names, dmg, color=colors, alpha=0.88)
    ax1.axvline(0, color="black", lw=0.8)
    for b, v in zip(b1, dmg):
        ax1.annotate(f"{v:+.2f}%", (v, b.get_y() + b.get_height() / 2),
                     va="center", ha="left", fontsize=9, xytext=(4, 0),
                     textcoords="offset points")
    ax1.set_xlabel("Post-shock damage (%)")
    ax1.set_title("Loss after a volatility shock\n(lower is better)")

    b2 = ax2.barh(names, dys, color=colors, alpha=0.88)
    for b, v in zip(b2, dys):
        ax2.annotate(f"{v:.1f}d", (v, b.get_y() + b.get_height() / 2),
                     va="center", ha="left", fontsize=9, xytext=(4, 0),
                     textcoords="offset points")
    ax2.set_xlabel("Days held after shock")
    ax2.set_title("Exit speed after a shock\n(faster is better)")

    fig.suptitle("Volatility-shock stress test "
                 "(shock = TR > 2.5 × 60-day ATR)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "vol_shock.png"), bbox_inches="tight")
    plt.close(fig)
    print("saved docs/vol_shock.png")

    print("\ncharts written to docs/")


if __name__ == "__main__":
    main()
