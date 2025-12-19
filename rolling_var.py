# rolling_var_fixed.py
# pip install pandas numpy scipy
#
# OUTPUTS:
#   outputs/rolling_var_qqq_daily.csv        (daily VaR + exceedances)
#   outputs/rolling_var_qqq_summary.csv      (one-row summary per model)

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR = Path("data_processed")
OUT_DIR = Path("outputs")

LOGRET_CSV = DATA_DIR / "log_returns.csv"
TICKER = "QQQ"

WINDOW = 504
ALPHA = 0.01          # VaR level
SIG_LEVEL = 0.05      # significance level for tests
DF_T = 3
START_TEST_DATE = "2022-10-06"


def normal_mle(x):
    mu = np.mean(x)
    sigma = np.sqrt(np.mean((x - mu) ** 2))
    return mu, sigma


def kupiec_uc_test(exceed, alpha):
    n = len(exceed)
    x = int(np.sum(exceed))
    phat = x / n
    eps = 1e-12

    ll0 = (n - x) * np.log(max(1 - alpha, eps)) + x * np.log(max(alpha, eps))
    ll1 = (n - x) * np.log(max(1 - phat, eps)) + x * np.log(max(phat, eps))

    lr = -2 * (ll0 - ll1)
    pval = 1 - stats.chi2.cdf(lr, df=1)
    return float(lr), float(pval), x, n


def christoffersen_ind_test(exceed):
    e = exceed.astype(int)
    if len(e) < 2:
        return np.nan, np.nan

    n00 = np.sum((e[:-1] == 0) & (e[1:] == 0))
    n01 = np.sum((e[:-1] == 0) & (e[1:] == 1))
    n10 = np.sum((e[:-1] == 1) & (e[1:] == 0))
    n11 = np.sum((e[:-1] == 1) & (e[1:] == 1))

    eps = 1e-12
    p01 = n01 / max(n00 + n01, 1)
    p11 = n11 / max(n10 + n11, 1)
    p = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    ll0 = (n00 + n10) * np.log(max(1 - p, eps)) + (n01 + n11) * np.log(max(p, eps))
    ll1 = (
        n00 * np.log(max(1 - p01, eps)) +
        n01 * np.log(max(p01, eps)) +
        n10 * np.log(max(1 - p11, eps)) +
        n11 * np.log(max(p11, eps))
    )

    lr = -2 * (ll0 - ll1)
    pval = 1 - stats.chi2.cdf(lr, df=1)
    return float(lr), float(pval)


def conditional_coverage_test(exceed, alpha):
    lr_uc, p_uc, x, n = kupiec_uc_test(exceed, alpha)
    lr_ind, p_ind = christoffersen_ind_test(exceed)
    lr_cc = lr_uc + lr_ind
    p_cc = 1 - stats.chi2.cdf(lr_cc, df=2)
    return {
        "LR_uc": lr_uc, "p_uc": p_uc,
        "LR_ind": lr_ind, "p_ind": p_ind,
        "LR_cc": float(lr_cc), "p_cc": float(p_cc),
        "x": x, "n": n,
        "exceed_rate": x / n
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(LOGRET_CSV, index_col=0, parse_dates=True).sort_index()
    r = df[TICKER].dropna()

    rows = []
    for t in range(WINDOW, len(r)):
        date = r.index[t]
        if date < pd.to_datetime(START_TEST_DATE):
            continue

        train = r.iloc[t - WINDOW:t].values
        ret = float(r.iloc[t])

        # Normal VaR
        mu_n, sig_n = normal_mle(train)
        var_n = mu_n + sig_n * stats.norm.ppf(ALPHA)
        exc_n = int(ret < var_n)

        # Student-t VaR (df fixed)
        mu_t, sig_t = normal_mle(train)
        var_t = mu_t + sig_t * stats.t.ppf(ALPHA, DF_T)
        exc_t = int(ret < var_t)

        rows.append({
            "date": date,
            "return": ret,
            "VaR_normal": float(var_n),
            "VaR_t": float(var_t),
            "exceed_normal": exc_n,
            "exceed_t": exc_t,
        })

    daily = pd.DataFrame(rows).set_index("date")
    daily_path = OUT_DIR / "rolling_var_qqq_daily.csv"
    daily.reset_index().to_csv(daily_path, index=False)

    # Overall backtests (one p-value per model)
    summaries = []
    for model in ["normal", "t"]:
        exceed = daily[f"exceed_{model}"].values.astype(int)
        bt = conditional_coverage_test(exceed, ALPHA)
        summaries.append({
            "ticker": TICKER,
            "model": model,
            "alpha": ALPHA,
            "window_days": WINDOW,
            "start_test_date": START_TEST_DATE,
            **bt,
            "reject_uc_5pct": int(bt["p_uc"] < SIG_LEVEL),
            "reject_ind_5pct": int(bt["p_ind"] < SIG_LEVEL),
            "reject_cc_5pct": int(bt["p_cc"] < SIG_LEVEL),
        })

    summary = pd.DataFrame(summaries)
    summary_path = OUT_DIR / "rolling_var_qqq_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Saved daily rolling VaR to: {daily_path.resolve()}")
    print(f"Saved rolling VaR summary to: {summary_path.resolve()}\n")
    print(summary[["model", "alpha", "x", "n", "exceed_rate", "p_uc", "p_ind", "p_cc",
                   "reject_uc_5pct", "reject_cc_5pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
