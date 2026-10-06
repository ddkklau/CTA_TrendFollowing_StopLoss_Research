"""
Download main-continuous futures data from AKShare (Sina Finance).

Universe: 13 Chinese commodity futures across 6 industry chains, 2020-2025.

Requirement: Python 3.11 (AKShare fails to build on 3.14).
Usage:  python src/download_data.py
Output: data/{SYMBOL}.csv  (columns: open, high, low, close, volume; DatetimeIndex)
"""

import os
import time
import pandas as pd
import akshare as ak

# symbol -> (name, industry chain)
UNIVERSE = {
    "RB": ("rebar",         "ferrous"),
    "I":  ("iron ore",      "ferrous"),
    "J":  ("coke",          "ferrous"),
    "CU": ("copper",        "base metals"),
    "AL": ("aluminum",      "base metals"),
    "AU": ("gold",          "precious metals"),
    "AG": ("silver",        "precious metals"),
    "MA": ("methanol",      "chemicals"),
    "TA": ("PTA",           "chemicals"),
    "V":  ("PVC",           "chemicals"),
    "M":  ("soybean meal",  "agriculture"),
    "SR": ("sugar",         "agriculture"),
    "SC": ("crude oil",     "energy"),
}

START = "20200101"
END = "20251231"
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def download(code: str, retry: int = 3) -> pd.DataFrame | None:
    """Download one symbol's main-continuous daily data, with retries."""
    for attempt in range(retry):
        try:
            df = ak.futures_main_sina(symbol=f"{code}0", start_date=START, end_date=END)
            df = df.rename(columns={
                "日期": "date", "开盘价": "open", "最高价": "high",
                "最低价": "low", "收盘价": "close", "成交量": "volume",
            })
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            df = df[["open", "high", "low", "close", "volume"]]
            return df
        except Exception as e:
            print(f"  {code} attempt {attempt + 1} failed: {type(e).__name__}")
            time.sleep(1)
    return None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ok, failed = [], []

    for code, (name, chain) in UNIVERSE.items():
        print(f"downloading {code} ({name}, {chain}) ...")
        df = download(code)
        if df is None:
            failed.append(code)
            continue
        df.to_csv(os.path.join(OUT_DIR, f"{code}.csv"))
        ok.append(code)
        print(f"  {code}: {len(df)} days, {df.index[0].date()} ~ {df.index[-1].date()}")

    print(f"\ndone. ok={len(ok)} {ok}")
    if failed:
        print(f"failed: {failed}")


if __name__ == "__main__":
    main()
