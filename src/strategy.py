"""
Core strategy: dual-direction Donchian breakout + ATR stop-loss + ATR position sizing.

Strategy logic
--------------
Entry      : close > upper (go long) / close < lower (go short)
             upper/lower = N-day rolling max/min, shifted by 1 (no look-ahead)
Stop-loss  : long  stop = prev_close - k * prev_ATR
             short stop = prev_close + k * prev_ATR
Sizing     : size = risk_pct / (k * ATR / close)   -> each trade risks risk_pct of account
Costs      : commission + slippage charged on every position change
"""

import numpy as np
import pandas as pd


def calc_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range (includes overnight gaps).

    TR = max(high-low, |high-prev_close|, |low-prev_close|)
    """
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    return tr.rolling(window).mean()


def donchian(
    df: pd.DataFrame,
    window: int = 30,
    atr_n: int = 14,
    k: float = 1.5,
    fee: float = 0.0005,
    slippage: float = 0.0002,
    risk_pct: float = 0.01,
) -> pd.DataFrame:
    """Run the strategy on a single-instrument OHLCV DataFrame.

    Parameters
    ----------
    df       : must contain columns open/high/low/close/volume, DatetimeIndex
    window   : Donchian channel lookback
    atr_n    : ATR lookback
    k        : ATR stop multiple
    fee      : commission rate per position change
    slippage : slippage rate per position change
    risk_pct : max loss per trade as fraction of account

    Returns
    -------
    df with added columns: upper, lower, atr, position, size, cost, daily_return
    position is 1 (long) / -1 (short) / 0 (flat)
    """
    df = df.copy()
    df["upper"] = df["high"].rolling(window).max().shift(1)
    df["lower"] = df["low"].rolling(window).min().shift(1)
    df["atr"] = calc_atr(df, atr_n)

    # use numpy arrays in the loop: much faster than df.loc[df.index[i]]
    close = df["close"].to_numpy()
    atr = df["atr"].to_numpy()
    up = df["upper"].to_numpy()
    lo = df["lower"].to_numpy()
    n = len(df)

    position = np.zeros(n)
    size = np.zeros(n)

    for i in range(1, n):
        prev = position[i - 1]

        # ---- entry ----
        if prev == 0:
            if close[i] > up[i]:
                position[i] = 1
                size[i] = risk_pct / (k * atr[i] / close[i])
            elif close[i] < lo[i]:
                position[i] = -1
                size[i] = risk_pct / (k * atr[i] / close[i])
            continue

        # ---- exit: ATR stop (rolling anchor) or channel reversal ----
        stop_hit = False
        if prev == 1 and close[i] < close[i - 1] - k * atr[i - 1]:
            stop_hit = True
        elif prev == -1 and close[i] > close[i - 1] + k * atr[i - 1]:
            stop_hit = True

        channel_exit = (prev == 1 and close[i] < lo[i]) or (prev == -1 and close[i] > up[i])

        if stop_hit or channel_exit:
            position[i] = 0
        else:
            position[i] = prev
            size[i] = size[i - 1]  # size fixed at entry

    df["position"] = position
    df["size"] = size
    df["position_change"] = df["position"].diff().abs().fillna(0)
    df["cost"] = (fee + slippage) * df["position_change"]
    df["daily_return"] = (
        df["close"].pct_change() * df["position"].shift(1) * df["size"].shift(1)
        - df["cost"]
    )
    return df


def extract_trades(df: pd.DataFrame) -> list:
    """Return [(return, holding_days), ...] for each completed trade.

    Includes the final open trade (marked to the last close).
    """
    trades = []
    entry_idx = None
    pos = df["position"].to_numpy()

    for i in range(len(df)):
        prev = pos[i - 1] if i > 0 else 0
        if pos[i] != 0 and prev == 0:
            entry_idx = i
        elif pos[i] == 0 and prev != 0:
            seg = df["daily_return"].iloc[entry_idx:i + 1]
            trades.append(((1 + seg).prod() - 1, i - entry_idx))
            entry_idx = None

    if entry_idx is not None:  # still holding at the end
        seg = df["daily_return"].iloc[entry_idx:]
        trades.append(((1 + seg).prod() - 1, len(df) - entry_idx))

    return trades
