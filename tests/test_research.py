import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import analysis
import backtest
from scripts.reproduce import verify_data

ROOT = Path(__file__).resolve().parents[1]


class ResearchTests(unittest.TestCase):
    def test_data_reconstructs_from_prices(self):
        r = verify_data()['QQQ'].dropna()
        train, test = analysis.split_train_test(r, '2022-10-05')
        self.assertEqual((len(train), len(test)), (3211, 803))
        self.assertLess(train.index.max(), test.index.min())

    def test_normal_mle_uses_population_variance(self):
        mu, sigma = analysis.normal_mle(np.array([-1., 0., 1.]))
        self.assertEqual(mu, 0)
        self.assertAlmostEqual(sigma, np.sqrt(2/3))

    def test_generic_t_quantile_has_one_scale_factor(self):
        expected = stats.t.ppf(.01, df=3, loc=.001, scale=.008)
        self.assertAlmostEqual(backtest.var_threshold('t', .01, (3, .001, .008)), expected)

    def test_invalid_distribution_rejected(self):
        with self.assertRaises(ValueError):
            backtest.var_threshold('unknown', .01, ())

    def test_kupiec_matches_analytic_bernoulli_likelihood(self):
        e = np.array([1]*15 + [0]*788)
        phat = 15/803
        expected = 2*(15*np.log(phat/.01) + 788*np.log((1-phat)/.99))
        lr, p, x, n = backtest.kupiec_uc_test(e, .01)
        self.assertAlmostEqual(lr, expected)
        self.assertEqual((x, n), (15, 803))
        self.assertLess(p, .05)

    def test_boundary_counts_finite(self):
        for e in [np.zeros(100), np.ones(100)]:
            lr, p, _, _ = backtest.kupiec_uc_test(e, .01)
            self.assertTrue(np.isfinite(lr))
            self.assertTrue(0 <= p <= 1)

    def test_independence_transition_counts(self):
        _, _, counts = backtest.christoffersen_ind_test(np.array([0, 0, 1, 1, 0]))
        self.assertEqual(counts, dict(n00=1, n01=1, n10=1, n11=1))

    def test_saved_fixed_var_counts_recompute(self):
        returns = verify_data()['QQQ'].dropna().loc['2022-10-06':]
        table = pd.read_csv(ROOT / 'outputs/var_backtest_qqq.csv')
        for row in table.itertuples():
            result = backtest.conditional_coverage_test(returns.to_numpy() < row.VaR_threshold_return, row.alpha)
            self.assertEqual(result['x'], row.exceed_count)
            self.assertAlmostEqual(result['p_cc'], row.p_cc)


if __name__ == '__main__':
    unittest.main()
