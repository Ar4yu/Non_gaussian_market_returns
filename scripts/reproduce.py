"""Recompute the paper results offline without overwriting the archived outputs."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import analysis
import backtest
import rolling_var


def verify_data():
    manifest = json.loads((ROOT / 'data_processed/manifest.json').read_text())
    for name, digest in manifest['sha256'].items():
        if hashlib.sha256((ROOT / 'data_processed' / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Frozen data changed: {name}')
    prices = pd.read_csv(ROOT / 'data_processed/prices_adjclose.csv', index_col=0, parse_dates=True)
    returns = pd.read_csv(ROOT / 'data_processed/log_returns.csv', index_col=0, parse_dates=True)
    assert prices.index.is_unique and prices.index.is_monotonic_increasing
    assert returns.index.equals(prices.index)
    np.testing.assert_allclose(np.log(prices.ffill()).diff(), returns, atol=1e-12, equal_nan=True)
    return returns


def compare_csv(actual, reference, tolerance=1e-6):
    a, b = pd.read_csv(actual), pd.read_csv(reference)
    assert list(a.columns) == list(b.columns)
    assert a.shape == b.shape
    for col in a:
        if pd.api.types.is_numeric_dtype(b[col]):
            np.testing.assert_allclose(a[col], b[col], rtol=tolerance, atol=tolerance, equal_nan=True, err_msg=col)
        else:
            assert a[col].equals(b[col]), col


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'reproduced')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out == ROOT or out == ROOT / 'outputs' or ROOT / 'outputs' in out.parents:
        raise ValueError('Choose a separate directory to preserve original outputs')
    verify_data()
    out.mkdir(parents=True, exist_ok=True)
    for module in [analysis, backtest, rolling_var]:
        module.LOGRET_CSV = ROOT / 'data_processed/log_returns.csv'
        module.OUT_DIR = out
        module.VIZ_DIR = out / 'visualizations'
    backtest.SUMMARY_CSV = out / 'summary_qqq.csv'
    analysis.main()
    backtest.main()
    rolling_var.main()
    # Refit optimizer values may differ slightly across SciPy versions.
    compare_csv(out / 'summary_qqq.csv', ROOT / 'outputs/summary_qqq.csv', tolerance=2e-5)
    for filename in ['var_backtest_qqq.csv', 'rolling_var_qqq_daily.csv', 'rolling_var_qqq_summary.csv']:
        compare_csv(out / filename, ROOT / 'outputs' / filename)
    (out / 'verification.json').write_text(json.dumps({
        'status': 'passed', 'data_checksums': 'passed', 'saved_tables_recomputed': 4,
        'rolling_method': 'historical mean/std plug-in t(3), not Student-t MLE',
        'test_observations': 803,
    }, indent=2) + '\n')
    print('VERIFIED: frozen prices/returns and all four paper result tables.')


if __name__ == '__main__':
    main()
