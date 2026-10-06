"""
Backtest engine + performance metrics for the multi-asset CTA portfolio.

Usage:  python src/backtest.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strategy import donchian, extract_trades  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

SYMBOLS = ["RB", "I", "J", "CU", "AL", "AU", "AG", "MA", "TA", "V", "M", "SR", "SC"]


def load(symbols=SYMBOLS) -> dict:
    data = {}
    for s in symbols:
        path = os.path.join(DATA_DIR, f"{s}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found — run download_data.py first")
        df = pd.read_csv(path, index_col="date", parse_dates=True).sort_index()
        data[s] = df
    return data


def portfolio_returns(data: dict, symbols=SYMBOLS, **kwargs) -> pd.Series:
    """Equal-weight average of daily returns across instruments."""
    rets = [donchian(data[s], **kwargs)["daily_return"].rename(s) for s in symbols]
    return pd.concat(rets, axis=1).mean(axis=1)


def performance(port: pd.Series, name: str = "portfolio") -> dict:
    """Compute and print standard performance metrics."""
    equity = (1 + port.fillna(0)).cumprod()
    total = equity.iloc[-1] - 1
    years = len(port) / 252
    annual = (1 + total) ** (1 / years) - 1
    sharpe = port.mean() / port.std() * np.sqrt(252)
    dd = (equity - equity.cummax()) / equity.cummax()
    max_dd = dd.min()
    calmar = annual / abs(max_dd)

    print(f"\n===== {name} =====")
    print(f"  total return   : {total * 100:+.2f}%")
    print(f"  annualized     : {annual * 100:+.2f}%")
    print(f"  sharpe         : {sharpe:.2f}")
    print(f"  max drawdown   : {max_dd * 100:.2f}%")
    print(f"  calmar         : {calmar:.2f}")

    return {"total": total, "annual": annual, "sharpe": sharpe,
            "max_dd": max_dd, "calmar": calmar}


def trade_stats(data: dict, symbols=SYMBOLS, **kwargs):
    """Aggregate trade-level statistics across all instruments."""
    all_trades = []
    for s in symbols:
        all_trades += extract_trades(donchian(data[s], **kwargs))

    rets = np.array([t[0] for t in all_trades])
    wins, losses = rets[rets > 0], rets[rets <= 0]

    print(f"\n  trades         : {len(all_trades)}")
    print(f"  win rate       : {len(wins) / len(all_trades) * 100:.1f}%")
    print(f"  payoff ratio   : {wins.mean() / abs(losses.mean()):.2f}")
    print(f"  profit factor  : {wins.sum() / abs(losses.sum()):.2f}")
    print(f"  max trade loss : {losses.min() * 100:.2f}%")
    print(f"  max trade gain : {wins.max() * 100:.2f}%")


def main():
    data = load()
    print(f"loaded {len(data)} instruments")

    port = portfolio_returns(data)
    performance(port, "CTA trend-following (2020-2025, 13 assets)")
    trade_stats(data)


if __name__ == "__main__":
    main()
