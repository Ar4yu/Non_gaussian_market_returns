# eda.py
# pip install pandas numpy matplotlib scipy statsmodels
#
# Reads:
#   data_processed/prices_adjclose.csv
#   data_processed/log_returns.csv
# Writes figures to:
#   visualizations/

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.tsa.stattools import acf

DATA_DIR = Path("data_processed")
VIZ_DIR = Path("visualizations")

PRICES_CSV = DATA_DIR / "prices_adjclose.csv"
LOGRET_CSV = DATA_DIR / "log_returns.csv"

# --- split controls ---
TEST_SIZE = 0.20           # used if SPLIT_DATE is None
SPLIT_DATE = None          # e.g. "2020-01-01" to split by date instead of fraction

# --- EDA controls ---
ROLLING_WINDOW = 63        # ~3 months of trading days
BINS = 60                  # histogram bins


def load_data():
    prices = pd.read_csv(PRICES_CSV, index_col=0, parse_dates=True)
    log_r = pd.read_csv(LOGRET_CSV, index_col=0, parse_dates=True).dropna(how="any")
    # align on common dates just in case
    common = prices.index.intersection(log_r.index)
    return prices.loc[common].sort_index(), log_r.loc[common].sort_index()


def time_split(df: pd.DataFrame, test_size=TEST_SIZE, split_date=SPLIT_DATE):
    df = df.sort_index()
    if split_date:
        d = pd.to_datetime(split_date)
        train = df.loc[df.index <= d].copy()
        test = df.loc[df.index > d].copy()
        return train, test
    k = int(np.floor((1 - test_size) * len(df)))
    return df.iloc[:k].copy(), df.iloc[k:].copy()


def savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_prices(prices: pd.DataFrame):
    # normalize so you can compare on one chart
    norm = prices / prices.iloc[0]
    plt.figure()
    plt.plot(norm.index, norm.values)
    plt.title("Normalized Prices (Adj Close, start = 1.0)")
    plt.xlabel("Date")
    plt.ylabel("Normalized price")
    plt.legend(norm.columns, fontsize=8)
    savefig(VIZ_DIR / "01_normalized_prices.png")


def plot_log_returns_timeseries(log_r: pd.DataFrame):
    plt.figure()
    plt.plot(log_r.index, log_r.values, linewidth=0.7)
    plt.title("Daily Log Returns (Time Series)")
    plt.xlabel("Date")
    plt.ylabel("Log return")
    plt.legend(log_r.columns, fontsize=8)
    savefig(VIZ_DIR / "02_log_returns_timeseries.png")


def plot_hist_and_qq(log_r: pd.DataFrame):
    # Do per-asset: histogram + normal overlay; QQ plot vs Normal
    for col in log_r.columns:
        x = log_r[col].dropna().values
        mu, sig = x.mean(), x.std(ddof=1)

        # Histogram with normal overlay
        plt.figure()
        plt.hist(x, bins=BINS, density=True, alpha=0.9)
        grid = np.linspace(x.min(), x.max(), 400)
        plt.plot(grid, stats.norm.pdf(grid, loc=mu, scale=sig), linewidth=2)
        plt.title(f"{col}: Log Returns Histogram + Normal Overlay")
        plt.xlabel("Log return")
        plt.ylabel("Density")
        savefig(VIZ_DIR / f"03_hist_normal_overlay_{col}.png")

        # QQ plot vs Normal
        plt.figure()
        stats.probplot(x, dist="norm", sparams=(mu, sig), plot=plt)
        plt.title(f"{col}: QQ Plot vs Normal")
        savefig(VIZ_DIR / f"04_qqplot_normal_{col}.png")


def plot_rolling_vol(log_r: pd.DataFrame, window=ROLLING_WINDOW):
    # Rolling volatility (std of daily log returns)
    roll = log_r.rolling(window).std()
    plt.figure()
    plt.plot(roll.index, roll.values)
    plt.title(f"Rolling Volatility of Log Returns (window = {window} days)")
    plt.xlabel("Date")
    plt.ylabel("Rolling std dev")
    plt.legend(log_r.columns, fontsize=8)
    savefig(VIZ_DIR / "05_rolling_volatility.png")


def plot_acf_each_asset(log_r: pd.DataFrame, max_lag=30):
    # ACF of returns (should be small if close to iid)
    lags = np.arange(max_lag + 1)
    for col in log_r.columns:
        x = log_r[col].dropna().values
        acf_vals = acf(x, nlags=max_lag, fft=True)

        plt.figure()
        plt.stem(lags, acf_vals, basefmt=" ")
        plt.title(f"{col}: ACF of Daily Log Returns (0..{max_lag} lags)")
        plt.xlabel("Lag")
        plt.ylabel("ACF")
        savefig(VIZ_DIR / f"06_acf_returns_{col}.png")


def plot_tail_focus(log_r: pd.DataFrame):
    # Visual “fat tails”: compare empirical quantiles to normal quantiles (tail plot)
    # Simple: plot sorted returns vs normal quantiles for each asset.
    for col in log_r.columns:
        x = np.sort(log_r[col].dropna().values)
        n = len(x)
        p = (np.arange(1, n + 1) - 0.5) / n
        mu, sig = x.mean(), x.std(ddof=1)
        qn = stats.norm.ppf(p, loc=mu, scale=sig)

        plt.figure()
        plt.plot(qn, x, linewidth=1)
        # 45-degree reference line using endpoints
        lo = min(qn.min(), x.min())
        hi = max(qn.max(), x.max())
        plt.plot([lo, hi], [lo, hi], linewidth=1)
        plt.title(f"{col}: Empirical Quantiles vs Normal Quantiles (Tail Check)")
        plt.xlabel("Normal quantiles (fitted μ,σ)")
        plt.ylabel("Empirical quantiles")
        savefig(VIZ_DIR / f"07_quantile_compare_{col}.png")


def write_split_info(train: pd.DataFrame, test: pd.DataFrame):
    # Simple split summary saved as txt for your paper notes
    txt = (
        f"Split method: {'date' if SPLIT_DATE else 'fraction'}\n"
        f"SPLIT_DATE: {SPLIT_DATE}\n"
        f"TEST_SIZE: {TEST_SIZE}\n\n"
        f"Train: {train.index.min().date()} → {train.index.max().date()} ({len(train)} obs)\n"
        f"Test:  {test.index.min().date()} → {test.index.max().date()} ({len(test)} obs)\n"
        f"Tickers: {list(train.columns)}\n"
    )
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    (VIZ_DIR / "split_info.txt").write_text(txt)


def main():
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    prices, log_r = load_data()
    train, test = time_split(log_r)

    # Save split info for reference
    write_split_info(train, test)

    # Core exploratory plots
    plot_prices(prices)
    plot_log_returns_timeseries(log_r)
    plot_hist_and_qq(train)          # use TRAIN for model fitting diagnostics
    plot_rolling_vol(log_r)
    plot_acf_each_asset(train)
    plot_tail_focus(train)

    print(f"Saved figures to: {VIZ_DIR.resolve()}")
    print(f"Train/Test sizes: {len(train)} / {len(test)}")
    if SPLIT_DATE:
        print(f"Split date: {SPLIT_DATE}")


if __name__ == "__main__":
    main()
