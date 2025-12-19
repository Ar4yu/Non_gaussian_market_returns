# Non-Gaussian Market Returns: Normal vs Student-t (ECON H324 Final Project)

**Authors:** Lyla Saigal & Aaryaman Jaising  
**Project title:** _Non-Gaussian Market Returns: Evidence from Normal and Student-t Models_  
**Repository:** https://github.com/Ar4yu/Non_gaussian_market_returns

## Overview

A common simplifying assumption in finance is that daily asset returns are approximately Gaussian. Under Normality, large negative moves are extremely rare, yet real markets appear to experience “crashes” more often than a Normal model would predict. This project tests the Gaussian benchmark against a heavy-tailed alternative using **maximum likelihood estimation (MLE)**.

We estimate and compare:

- **Model 1:** Normal distribution for daily log returns
- **Model 2:** Student-t distribution for daily log returns (degrees of freedom selected via grid search)

We evaluate models using:

- In-sample log-likelihood
- Information criteria (**AIC**, **BIC**)
- Out-of-sample log-likelihood (chronological train/test split)
- Out-of-sample **Value at Risk (VaR)** backtests (Kupiec and Christoffersen tests)
- A two-year **rolling-window VaR** robustness check

The main empirical focus in the code is **QQQ** (Nasdaq-100 ETF), with additional tickers downloaded for context and replication.

---

## Data

### Data source

- Downloaded via Python package **`yfinance`** (not affiliated with Yahoo).
- Uses Yahoo Finance’s historical price series.

### Instruments

Tickers included in the dataset:

- `SPY`, `QQQ`, `AAPL`, `JPM`, `XOM`

### Price field

- Uses **Adjusted Close** (adjusted for splits and dividends).

### Returns

Daily **log returns**:
\[
r*t = \log(P_t) - \log(P*{t-1})
\]

### Sample period

- **2010-01-01** to **2025-12-17** (trading days)

### Train/Test split (chronological)

- **Train:** 2010-01-05 → 2022-10-05 (**3211 obs**)
- **Test:** 2022-10-06 → 2025-12-17 (**803 obs**)

A time-based split preserves a forecasting interpretation and avoids look-ahead bias.

---

## Repository Structure

```text
.
├── data_processed/
│   ├── prices_adjclose.csv
│   └── log_returns.csv
├── outputs/
│   ├── summary_qqq.csv
│   ├── report_qqq.txt
│   ├── var_backtest_qqq.csv
│   ├── rolling_var_qqq_daily.csv
│   ├── rolling_var_qqq_summary.csv
│   └── visualizations/
│       ├── hist_overlay_qqq.png
│       ├── qq_plots_qqq.png
│       └── ll_vs_df_qqq.png
├── preprocess.py
├── exploratory_analysis.py
├── analysis.py
├── var_backtest.py
├── rolling_var.py
└── README.md
```
