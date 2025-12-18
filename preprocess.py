# preprocess.py
# pip install yfinance pandas numpy
#
# Outputs:
#   data_processed/prices_adjclose.csv
#   data_processed/log_returns.csv

import os
import time
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

TICKERS = ["SPY", "QQQ", "AAPL", "JPM", "XOM"]
START = "2010-01-01"
INTERVAL = "1d"
OUTDIR = Path("data_processed")

# Disable any sqlite-based caching that can cause "database is locked" in some setups
os.environ.setdefault("YFINANCE_CACHE_DISABLE", "1")


def download_adj_close(tickers=TICKERS, start=START, interval=INTERVAL, retries=3, sleep_s=2.0) -> pd.DataFrame:
    """
    Download Adj Close from Yahoo Finance using yfinance's internal curl_cffi session.
    Retries on transient failures. Drops tickers that fail completely (all-NaN).
    """
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                tickers=tickers,
                start=start,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,     # reduce concurrency issues
                group_by="column",
            )
            if df is None or df.empty:
                raise RuntimeError("Empty response from yfinance.")

            px = df["Adj Close"]
            px = px.to_frame() if isinstance(px, pd.Series) else px
            px = px.sort_index()

            # Clean obvious bad values
            px = px.replace([np.inf, -np.inf], np.nan)
            px = px.where(px > 0)

            # Drop columns that are entirely NaN (failed tickers)
            bad = [c for c in px.columns if px[c].isna().all()]
            if bad:
                print(f"WARNING: Dropping tickers with no data: {bad}")
                px = px.drop(columns=bad)

            if px.shape[1] == 0:
                raise RuntimeError("All tickers failed (all columns were NaN).")

            return px

        except Exception as e:
            last_err = e
            print(f"Download attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(sleep_s)

    raise RuntimeError(f"Failed to download after {retries} attempts. Last error: {last_err}")


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log returns from Adj Close prices.
    Forward-fill small gaps; does not create leading history.
    """
    prices = prices.sort_index().ffill()
    return np.log(prices).diff()


def print_quality(prices: pd.DataFrame, log_r: pd.DataFrame) -> None:
    print("\n=== Data Quality (Adj Close Prices) ===")
    print("Tickers:", list(prices.columns))
    print("Date range:", prices.index.min().date(), "→", prices.index.max().date())
    print("\nMissing % by ticker (prices):")
    print((prices.isna().mean() * 100).round(3).to_string())
    print("Rows with any NA in prices:", int(prices.isna().any(axis=1).sum()), f"out of {len(prices)}")

    print("\n=== Data Quality (Log Returns) ===")
    print("\nMissing % by ticker (log returns):")
    print((log_r.isna().mean() * 100).round(3).to_string())
    print("Rows with any NA in log returns:", int(log_r.isna().any(axis=1).sum()), f"out of {len(log_r)}")
    print("Note: the first row is NA for log returns by construction (diff).\n")

    # Optional but useful: how many rows remain if you require all tickers present
    common_rows = len(log_r.dropna(how="any"))
    print(f"Rows with complete data across ALL tickers (log returns): {common_rows} / {len(log_r)}\n")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    prices = download_adj_close()
    log_r = compute_log_returns(prices)

    print_quality(prices, log_r)

    prices_path = OUTDIR / "prices_adjclose.csv"
    log_path = OUTDIR / "log_returns.csv"

    prices.to_csv(prices_path, index=True)
    log_r.to_csv(log_path, index=True)

    print("Saved:")
    print(" -", prices_path.resolve())
    print(" -", log_path.resolve())
    print("")


if __name__ == "__main__":
    main()
