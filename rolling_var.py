# rolling_var_with_tests.py
# pip install pandas numpy scipy
#
# OUTPUT:
#   outputs/rolling_var_backtest_qqq.csv

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR = Path("data_processed")
OUT_DIR = Path("outputs")

LOGRET_CSV = DATA_DIR / "log_returns.csv"
TICKER = "QQQ"

WINDOW = 504              # ~2 years
ALPHA = 0.01              # 1% VaR
SIG_LEVEL = 0.05          # 5% test level
DF_T = 3                  # fixed df from Student-t MLE
START_TEST_DATE = "2022-10-06"


def normal_mle(x):
    mu = np.mean(x)
    sigma = np.sqrt(np.mean((x - mu) ** 2))
    return mu, sigma


def kupiec_uc_test(exceed, alpha):
    n = len(exceed)
    x = np.sum(exceed)
    phat = x / n
    eps = 1e-12

    ll0 = (n - x) * np.log(max(1 - alpha, eps)) + x * np.log(max(alpha, eps))
    ll1 = (n - x) * np.log(max(1 - phat, eps)) + x * np.log(max(phat, eps))

    lr = -2 * (ll0 - ll1)
    pval = 1 - stats.chi2.cdf(lr, df=1)
    return lr, pval


def christoffersen_ind_test(exceed):
    e = exceed.astype(int)
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
    return lr, pval


def conditional_coverage_test(exceed, alpha):
    lr_uc, p_uc = kupiec_uc_test(exceed, alpha)
    lr_ind, p_ind = christoffersen_ind_test(exceed)
    lr_cc = lr_uc + lr_ind
    p_cc = 1 - stats.chi2.cdf(lr_cc, df=2)
    return lr_uc, p_uc, lr_ind, p_ind, lr_cc, p_cc


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(LOGRET_CSV, index_col=0, parse_dates=True).sort_index()
    r = df[TICKER].dropna()

    records = []

    for t in range(WINDOW, len(r)):
        date = r.index[t]
        if date < pd.to_datetime(START_TEST_DATE):
            continue

        train = r.iloc[t - WINDOW:t].values
        test_ret = r.iloc[t]

        # --- Normal VaR ---
        mu_n, sig_n = normal_mle(train)
        var_n = mu_n + sig_n * stats.norm.ppf(ALPHA)
        exc_n = int(test_ret < var_n)

        # --- Student-t VaR ---
        mu_t = np.mean(train)
        sig_t = np.sqrt(np.mean((train - mu_t) ** 2))
        var_t = mu_t + sig_t * stats.t.ppf(ALPHA, DF_T)
        exc_t = int(test_ret < var_t)

        records.append({
            "date": date,
            "return": test_ret,
            "VaR_normal": var_n,
            "VaR_t": var_t,
            "exceed_normal": exc_n,
            "exceed_t": exc_t,
        })

    out = pd.DataFrame(records).set_index("date")

    # --- Backtests ---
    for model in ["normal", "t"]:
        exceed = out[f"exceed_{model}"].values
        lr_uc, p_uc, lr_ind, p_ind, lr_cc, p_cc = conditional_coverage_test(exceed, ALPHA)

        out[f"{model}_LR_uc"] = lr_uc
        out[f"{model}_p_uc"] = p_uc
        out[f"{model}_reject_uc"] = int(p_uc < SIG_LEVEL)

        out[f"{model}_LR_ind"] = lr_ind
        out[f"{model}_p_ind"] = p_ind
        out[f"{model}_reject_ind"] = int(p_ind < SIG_LEVEL)

        out[f"{model}_LR_cc"] = lr_cc
        out[f"{model}_p_cc"] = p_cc
        out[f"{model}_reject_cc"] = int(p_cc < SIG_LEVEL)

    path = OUT_DIR / "rolling_var_backtest_qqq.csv"
    out.reset_index().to_csv(path, index=False)

    print(f"Saved rolling VaR backtest with p-values to: {path.resolve()}")
    print("\nRejection summary (5% level):")
    print(out[[
        "normal_reject_uc", "normal_reject_cc",
        "t_reject_uc", "t_reject_cc"
    ]].max())


if __name__ == "__main__":
    main()
