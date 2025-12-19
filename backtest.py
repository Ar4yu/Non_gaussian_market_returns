# var_backtest.py
# pip install pandas numpy scipy matplotlib
#
# INPUT:
#   data_processed/log_returns.csv
#   outputs/summary_qqq.csv
# OUTPUT:
#   outputs/var_backtest_qqq.csv
#   outputs/visualizations/var_timeseries_qqq.png
#   outputs/visualizations/exceedance_rates_qqq.png

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

DATA_DIR = Path("data_processed")
OUT_DIR = Path("outputs")
VIZ_DIR = OUT_DIR / "visualizations"

LOGRET_CSV = DATA_DIR / "log_returns.csv"
SUMMARY_CSV = OUT_DIR / "summary_qqq.csv"
TICKER = "QQQ"
TRAIN_END_DATE = "2022-10-05"  # same as your analysis split

ALPHAS = [0.01, 0.05]  # VaR levels


def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)


def load_test_returns():
    df = pd.read_csv(LOGRET_CSV, index_col=0, parse_dates=True).sort_index()
    x = df[TICKER].dropna()
    train_end = pd.to_datetime(TRAIN_END_DATE)
    test = x.loc[x.index > train_end].copy()
    if len(test) < 50:
        raise RuntimeError(f"Test set too small: {len(test)} observations.")
    return test


def load_fitted_params():
    summ = pd.read_csv(SUMMARY_CSV)
    # Expect rows: Normal, Student-t
    row_n = summ.loc[summ["model"] == "Normal"].iloc[0]
    row_t = summ.loc[summ["model"] == "Student-t"].iloc[0]

    mu_n, sig_n = float(row_n["mu_hat"]), float(row_n["sigma_hat"])
    df_t = int(row_t["df_hat"])
    mu_t, sig_t = float(row_t["mu_hat"]), float(row_t["sigma_hat"])

    return (mu_n, sig_n), (df_t, mu_t, sig_t)


def var_threshold(dist: str, alpha: float, params):
    """
    Returns the return-threshold r_alpha such that P(R <= r_alpha) = alpha.
    Exceedance event is: R_t < r_alpha (left tail).
    """
    if dist == "normal":
        mu, sig = params
        return mu + sig * stats.norm.ppf(alpha)
    if dist == "t":
        df, mu, sig = params
        return mu + sig * stats.t.ppf(alpha, df=df, loc=0, scale=1) * sig + (mu - mu)  # keep structure explicit


def var_threshold_t(alpha: float, df: int, mu: float, sig: float) -> float:
    return mu + sig * stats.t.ppf(alpha, df=df)


def var_threshold_normal(alpha: float, mu: float, sig: float) -> float:
    return mu + sig * stats.norm.ppf(alpha)


def kupiec_uc_test(exceed: np.ndarray, alpha: float):
    """
    Kupiec (1995) unconditional coverage test.
    exceed: boolean array where True indicates VaR exceedance.
    H0: P(exceed) = alpha.
    Returns: (LR_uc, p_value, x, n)
    """
    n = int(len(exceed))
    x = int(np.sum(exceed))
    # Handle boundary cases safely
    phat = x / n if n > 0 else np.nan
    eps = 1e-12

    # log-likelihood under H0
    ll0 = (n - x) * np.log(max(1 - alpha, eps)) + x * np.log(max(alpha, eps))
    # log-likelihood under MLE phat
    ll1 = (n - x) * np.log(max(1 - phat, eps)) + x * np.log(max(phat, eps))

    lr = -2.0 * (ll0 - ll1)
    p = 1.0 - stats.chi2.cdf(lr, df=1)
    return float(lr), float(p), x, n


def christoffersen_ind_test(exceed: np.ndarray):
    """
    Christoffersen (1998) independence test.
    H0: exceedances are independent over time (first-order Markov with p01 = p11).
    Returns: (LR_ind, p_value, transition_counts)
    """
    e = exceed.astype(int)
    if len(e) < 2:
        return np.nan, np.nan, {"n00": 0, "n01": 0, "n10": 0, "n11": 0}

    n00 = int(np.sum((e[:-1] == 0) & (e[1:] == 0)))
    n01 = int(np.sum((e[:-1] == 0) & (e[1:] == 1)))
    n10 = int(np.sum((e[:-1] == 1) & (e[1:] == 0)))
    n11 = int(np.sum((e[:-1] == 1) & (e[1:] == 1)))

    eps = 1e-12
    # transition probs
    p01 = n01 / max(n00 + n01, 1)
    p11 = n11 / max(n10 + n11, 1)
    p   = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    # log-likelihood under independence (single p)
    ll0 = (n00 + n10) * np.log(max(1 - p, eps)) + (n01 + n11) * np.log(max(p, eps))
    # log-likelihood under Markov (p01, p11)
    ll1 = n00 * np.log(max(1 - p01, eps)) + n01 * np.log(max(p01, eps)) \
        + n10 * np.log(max(1 - p11, eps)) + n11 * np.log(max(p11, eps))

    lr = -2.0 * (ll0 - ll1)
    pval = 1.0 - stats.chi2.cdf(lr, df=1)
    counts = {"n00": n00, "n01": n01, "n10": n10, "n11": n11}
    return float(lr), float(pval), counts


def conditional_coverage_test(exceed: np.ndarray, alpha: float):
    """
    Christoffersen conditional coverage: LR_cc = LR_uc + LR_ind (df=2).
    """
    lr_uc, p_uc, x, n = kupiec_uc_test(exceed, alpha)
    lr_ind, p_ind, counts = christoffersen_ind_test(exceed)
    lr_cc = lr_uc + lr_ind
    p_cc = 1.0 - stats.chi2.cdf(lr_cc, df=2)
    return {
        "LR_uc": lr_uc, "p_uc": p_uc,
        "LR_ind": lr_ind, "p_ind": p_ind,
        "LR_cc": float(lr_cc), "p_cc": float(p_cc),
        "x": x, "n": n, **counts
    }


def savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_var_timeseries(test: pd.Series, var_lines: dict):
    """
    One plot: test returns + VaR lines (normal and t) for alpha=1% (cleanest).
    """
    alpha = 0.01
    plt.figure(figsize=(11, 4.2))
    plt.plot(test.index, test.values, linewidth=0.8, label="QQQ log return (test)")
    plt.axhline(var_lines[("normal", alpha)], linewidth=2, label="Normal VaR (1%) threshold")
    plt.axhline(var_lines[("t", alpha)], linewidth=2, label="Student-t VaR (1%) threshold")
    plt.title("Out-of-sample: QQQ log returns with 1% VaR thresholds")
    plt.xlabel("Date")
    plt.ylabel("Log return")
    plt.legend(fontsize=9)
    savefig(VIZ_DIR / "var_timeseries_qqq.png")


def plot_exceedance_rates(results_rows):
    """
    One plot: exceedance rate vs expected alpha for each model/alpha.
    """
    labels = []
    observed = []
    expected = []
    for r in results_rows:
        labels.append(f"{r['model']} α={r['alpha']}")
        observed.append(r["exceed_rate"])
        expected.append(r["alpha"])

    x = np.arange(len(labels))
    plt.figure(figsize=(10, 4.2))
    plt.bar(x - 0.2, observed, width=0.4, label="Observed exceedance rate")
    plt.bar(x + 0.2, expected, width=0.4, label="Expected rate (α)")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.title("VaR exceedance rates (test set): observed vs expected")
    plt.ylabel("Rate")
    plt.legend(fontsize=9)
    savefig(VIZ_DIR / "exceedance_rates_qqq.png")


def main():
    ensure_dirs()

    test = load_test_returns()
    (mu_n, sig_n), (df_t, mu_t, sig_t) = load_fitted_params()

    results = []
    var_lines = {}

    for alpha in ALPHAS:
        # VaR thresholds (left-tail return quantiles)
        thr_n = var_threshold_normal(alpha, mu_n, sig_n)
        thr_t = var_threshold_t(alpha, df_t, mu_t, sig_t)
        var_lines[("normal", alpha)] = thr_n
        var_lines[("t", alpha)] = thr_t

        # Exceedance indicators on TEST set: return < threshold
        exc_n = (test.values < thr_n)
        exc_t = (test.values < thr_t)

        # Backtests
        bt_n = conditional_coverage_test(exc_n, alpha)
        bt_t = conditional_coverage_test(exc_t, alpha)

        # Store rows (compact + paper friendly)
        for model, thr, exc, bt in [
            ("Normal", thr_n, exc_n, bt_n),
            ("Student-t", thr_t, exc_t, bt_t),
        ]:
            x = bt["x"]
            n = bt["n"]
            results.append({
                "ticker": TICKER,
                "model": model,
                "alpha": alpha,
                "VaR_threshold_return": float(thr),
                "VaR_threshold_loss": float(-thr),  # often reported as positive loss
                "exceed_count": int(x),
                "n_test": int(n),
                "exceed_rate": float(x / n),
                "LR_uc": bt["LR_uc"],
                "p_uc": bt["p_uc"],
                "LR_ind": bt["LR_ind"],
                "p_ind": bt["p_ind"],
                "LR_cc": bt["LR_cc"],
                "p_cc": bt["p_cc"],
                "n00": bt["n00"], "n01": bt["n01"], "n10": bt["n10"], "n11": bt["n11"],
            })

    out_df = pd.DataFrame(results)
    out_path = OUT_DIR / "var_backtest_qqq.csv"
    out_df.to_csv(out_path, index=False)

    # Minimal visuals
    plot_var_timeseries(test, var_lines)
    plot_exceedance_rates(results)

    # Console summary (short)
    print(f"Saved VaR backtest results to: {out_path.resolve()}")
    print(f"Saved plots to: {VIZ_DIR.resolve()}\n")

    # Print a compact table to screen
    show_cols = ["model", "alpha", "VaR_threshold_loss", "exceed_count", "n_test", "exceed_rate", "p_uc", "p_ind", "p_cc"]
    print(out_df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
