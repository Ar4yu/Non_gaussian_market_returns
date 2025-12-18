import numpy as np
import pandas as pd
import yfinance as yf
import requests
from pathlib import Path

TICKERS = ["SPY", "QQQ", "AAPL", "JPM", "XOM"]
START = "2010-01-01"
OUTDIR = Path("data_processed")
TEST_SIZE = 0.20

def load_adj_close(tickers=TICKERS, start=START) -> pd.DataFrame:
    # Force a plain session (avoids sqlite cache locking issues)
    session = requests.Session()

    df = yf.download(
        tickers=tickers,
        start=start,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,     # reduce concurrency = fewer locking weirdness
        session=session,   # key line
    )

    px = df["Adj Close"]
    px = px.to_frame() if isinstance(px, pd.Series) else px
    px = px.sort_index().replace([np.inf, -np.inf], np.nan).where(lambda d: d > 0)

    # Drop tickers that completely failed (all NaN column)
    all_nan_cols = [c for c in px.columns if px[c].isna().all()]
    if all_nan_cols:
        print(f"WARNING: Dropping tickers with no data: {all_nan_cols}")
        px = px.drop(columns=all_nan_cols)

    return px

def make_returns(px: pd.DataFrame):
    px = px.ffill()

    log_r = np.log(px).diff()

    # avoid future warning explicitly
    r = px.pct_change(fill_method=None)

    return log_r, r
