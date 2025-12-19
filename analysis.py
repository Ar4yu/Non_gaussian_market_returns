# analysis.py
# pip install pandas numpy matplotlib scipy
#
# INPUT:
#   data_processed/log_returns.csv
# OUTPUT:
#   outputs/summary_qqq.csv
#   outputs/report_qqq.txt
#   outputs/visualizations/hist_overlay_qqq.png
#   outputs/visualizations/qq_plots_qqq.png
#   outputs/visualizations/ll_vs_df_qqq.png

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize

# -------------------- Config --------------------
DATA_DIR = Path("data_processed")
OUT_DIR = Path("outputs")
VIZ_DIR = OUT_DIR /"visualizations"

LOGRET_CSV = DATA_DIR / "log_returns.csv"
TICKER = "QQQ"

# Your paper's split (train inclusive of end date)
TRAIN_END_DATE = "2022-10-05"  # Train: <= this date
DF_MIN, DF_MAX = 1, 150        # grid search df in [1,150]
# ------------------------------------------------


def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)


def load_series() -> pd.Series:
    df = pd.read_csv(LOGRET_CSV, index_col=0, parse_dates=True).sort_index()
    if TICKER not in df.columns:
        raise KeyError(f"{TICKER} not found in {LOGRET_CSV}. Columns: {list(df.columns)}")
    x = df[TICKER].dropna()
    x.name = TICKER
    return x


def split_train_test(x: pd.Series, train_end_date: str):
    train_end = pd.to_datetime(train_end_date)
    train = x.loc[x.index <= train_end].copy()
    test = x.loc[x.index > train_end].copy()
    if len(train) < 50 or len(test) < 20:
        raise RuntimeError(f"Train/test too small. Train={len(train)}, Test={len(test)}. Check split date.")
    return train, test


# -------- Normal MLE (closed-form) --------
def normal_mle(x: np.ndarray):
    mu = float(np.mean(x))
    sigma = float(np.sqrt(np.mean((x - mu) ** 2)))  # MLE uses n in denominator
    return mu, sigma


def normal_loglik(x: np.ndarray, mu: float, sigma: float):
    return float(np.sum(stats.norm.logpdf(x, loc=mu, scale=sigma)))


# -------- Student-t MLE with df grid search --------
def t_neg_loglik_params(theta, x: np.ndarray, df: int):
    # theta = [mu, log_sigma] -> sigma = exp(log_sigma) ensures positivity
    mu = theta[0]
    sigma = np.exp(theta[1])
    # if sigma too small, penalize
    if not np.isfinite(sigma) or sigma <= 0:
        return 1e100
    ll = np.sum(stats.t.logpdf(x, df=df, loc=mu, scale=sigma))
    return -float(ll)


def fit_t_for_df(x: np.ndarray, df: int, mu0: float, sigma0: float):
    # initialize at Normal MLE, optimize mu + log_sigma
    theta0 = np.array([mu0, np.log(max(sigma0, 1e-8))], dtype=float)

    res = minimize(
        t_neg_loglik_params,
        theta0,
        args=(x, df),
        method="L-BFGS-B",
        # mild bounds for numerical stability
        bounds=[(mu0 - 1.0, mu0 + 1.0), (np.log(1e-8), np.log(1.0))],
    )

    mu_hat = float(res.x[0])
    sigma_hat = float(np.exp(res.x[1]))
    ll = -float(res.fun)
    return mu_hat, sigma_hat, ll, res.success


def grid_search_t_mle(x: np.ndarray, df_min: int, df_max: int):
    mu0, sigma0 = normal_mle(x)

    rows = []
    best = None

    for df in range(df_min, df_max + 1):
        mu_hat, sigma_hat, ll, ok = fit_t_for_df(x, df, mu0, sigma0)
        rows.append({"df": df, "mu": mu_hat, "sigma": sigma_hat, "ll_train": ll, "ok": ok})

        if ok and (best is None or ll > best["ll_train"]):
            best = rows[-1]

    if best is None:
        raise RuntimeError("t MLE grid search failed for all df values.")

    results = pd.DataFrame(rows)
    return best, results


def aic_bic(ll: float, n: int, k: int):
    # AIC = 2k - 2LL
    # BIC = ln(n)k - 2LL
    aic = 2 * k - 2 * ll
    bic = np.log(n) * k - 2 * ll
    return float(aic), float(bic)


# -------- Visuals --------
def plot_hist_overlay(x_train: np.ndarray, normal_params, t_params, outpath: Path):
    mu_n, sig_n = normal_params
    df_t, mu_t, sig_t = t_params

    plt.figure()
    plt.hist(x_train, bins=60, density=True, alpha=0.9)

    grid = np.linspace(np.percentile(x_train, 0.5), np.percentile(x_train, 99.5), 600)
    plt.plot(grid, stats.norm.pdf(grid, loc=mu_n, scale=sig_n), linewidth=2, label="Normal (MLE)")
    plt.plot(grid, stats.t.pdf(grid, df=df_t, loc=mu_t, scale=sig_t), linewidth=2, label=f"Student-t (df={df_t}, MLE)")

    plt.title("QQQ Daily Log Returns (Train): Histogram + Fitted Densities")
    plt.xlabel("Log return")
    plt.ylabel("Density")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_qq_plots(x_train: np.ndarray, normal_params, t_params, outpath: Path):
    mu_n, sig_n = normal_params
    df_t, mu_t, sig_t = t_params

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    # QQ vs Normal
    stats.probplot(x_train, dist="norm", sparams=(mu_n, sig_n), plot=axes[0])
    axes[0].set_title("QQ Plot vs Normal (fitted μ,σ)")
    axes[0].set_xlabel("Normal theoretical quantiles")
    axes[0].set_ylabel("Empirical quantiles")

    # QQ vs Student-t
    # Use scipy's probplot with dist=stats.t and pass df via sparams + scale/loc via sparams ordering:
    # For 't', sparams expects (df, loc, scale)
    stats.probplot(x_train, dist=stats.t, sparams=(df_t, mu_t, sig_t), plot=axes[1])
    axes[1].set_title(f"QQ Plot vs Student-t (df={df_t})")
    axes[1].set_xlabel("t theoretical quantiles")
    axes[1].set_ylabel("Empirical quantiles")

    fig.suptitle("QQQ (Train) QQ Diagnostics", y=1.02)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


def plot_ll_vs_df(grid_results: pd.DataFrame, outpath: Path):
    plt.figure()
    plt.plot(grid_results["df"], grid_results["ll_train"].astype(float), linewidth=2)
    plt.title("QQQ (Train): Student-t Log-Likelihood vs Degrees of Freedom")
    plt.xlabel("Degrees of freedom (df)")
    plt.ylabel("Log-likelihood (train)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# -------- Report writing --------
def write_report(path: Path, text: str):
    path.write_text(text, encoding="utf-8")


def main():
    ensure_dirs()

    x = load_series()
    train, test = split_train_test(x, TRAIN_END_DATE)

    x_train = train.values.astype(float)
    x_test = test.values.astype(float)

    # --- Normal MLE ---
    mu_n, sig_n = normal_mle(x_train)
    ll_n_train = normal_loglik(x_train, mu_n, sig_n)
    ll_n_test = normal_loglik(x_test, mu_n, sig_n)
    aic_n, bic_n = aic_bic(ll_n_train, n=len(x_train), k=2)

    # --- Student-t MLE via df grid search ---
    best_t, grid = grid_search_t_mle(x_train, DF_MIN, DF_MAX)
    df_t = int(best_t["df"])
    mu_t = float(best_t["mu"])
    sig_t = float(best_t["sigma"])
    ll_t_train = float(best_t["ll_train"])
    ll_t_test = float(np.sum(stats.t.logpdf(x_test, df=df_t, loc=mu_t, scale=sig_t)))
    aic_t, bic_t = aic_bic(ll_t_train, n=len(x_train), k=3)

    # --- Approx LR statistic (note: not exact chi-square because df→∞ boundary) ---
    lr_stat = 2.0 * (ll_t_train - ll_n_train)

    # --- Summary table (single CSV) ---
    summary = pd.DataFrame([
        {
            "model": "Normal",
            "ticker": TICKER,
            "train_start": str(train.index.min().date()),
            "train_end": str(train.index.max().date()),
            "test_start": str(test.index.min().date()),
            "test_end": str(test.index.max().date()),
            "n_train": len(train),
            "n_test": len(test),
            "mu_hat": mu_n,
            "sigma_hat": sig_n,
            "df_hat": np.inf,
            "ll_train": ll_n_train,
            "ll_test": ll_n_test,
            "AIC_train": aic_n,
            "BIC_train": bic_n,
        },
        {
            "model": "Student-t",
            "ticker": TICKER,
            "train_start": str(train.index.min().date()),
            "train_end": str(train.index.max().date()),
            "test_start": str(test.index.min().date()),
            "test_end": str(test.index.max().date()),
            "n_train": len(train),
            "n_test": len(test),
            "mu_hat": mu_t,
            "sigma_hat": sig_t,
            "df_hat": df_t,
            "ll_train": ll_t_train,
            "ll_test": ll_t_test,
            "AIC_train": aic_t,
            "BIC_train": bic_t,
        },
    ])

    summary_path = OUT_DIR / "summary_qqq.csv"
    summary.to_csv(summary_path, index=False)

    # --- Visuals (few) ---
    plot_hist_overlay(
        x_train,
        normal_params=(mu_n, sig_n),
        t_params=(df_t, mu_t, sig_t),
        outpath=VIZ_DIR / "hist_overlay_qqq.png",
    )

    plot_qq_plots(
        x_train,
        normal_params=(mu_n, sig_n),
        t_params=(df_t, mu_t, sig_t),
        outpath=VIZ_DIR / "qq_plots_qqq.png",
    )

    plot_ll_vs_df(grid, VIZ_DIR / "ll_vs_df_qqq.png")

    # --- Short TXT report (paper-friendly) ---
    report = []
    report.append("Advanced Econometrics Final Paper: Non-Gaussian Market Returns")
    report.append(f"Ticker analyzed: {TICKER}")
    report.append("")
    report.append(f"Train: {train.index.min().date()} → {train.index.max().date()} ({len(train)} obs)")
    report.append(f"Test:  {test.index.min().date()} → {test.index.max().date()} ({len(test)} obs)")
    report.append("")
    report.append("Normal MLE (train):")
    report.append(f"  mu_hat = {mu_n:.8f}, sigma_hat = {sig_n:.8f}")
    report.append(f"  LL_train = {ll_n_train:.2f}, LL_test = {ll_n_test:.2f}")
    report.append(f"  AIC_train = {aic_n:.2f}, BIC_train = {bic_n:.2f}")
    report.append("")
    report.append("Student-t MLE (train) with df grid search 1..150:")
    report.append(f"  df_hat = {df_t}, mu_hat = {mu_t:.8f}, sigma_hat = {sig_t:.8f}")
    report.append(f"  LL_train = {ll_t_train:.2f}, LL_test = {ll_t_test:.2f}")
    report.append(f"  AIC_train = {aic_t:.2f}, BIC_train = {bic_t:.2f}")
    report.append("")
    report.append("Model comparison (train):")
    report.append(f"  LL improvement (t - normal) = {ll_t_train - ll_n_train:.2f}")
    report.append(f"  LR statistic = 2*(LL_t - LL_n) = {lr_stat:.2f}")
    report.append("  Note: Normal is the df→∞ limit of Student-t, so LR χ² reference is only approximate.")
    report.append("")
    report.append("Saved:")
    report.append(f"  {summary_path}")
    report.append(f"  {VIZ_DIR / 'hist_overlay_qqq.png'}")
    report.append(f"  {VIZ_DIR / 'qq_plots_qqq.png'}")
    report.append(f"  {VIZ_DIR / 'll_vs_df_qqq.png'}")
    report_text = "\n".join(report)

    report_path = OUT_DIR / "report_qqq.txt"
    write_report(report_path, report_text)

    # Console prints (short)
    print(report_text)


if __name__ == "__main__":
    main()
