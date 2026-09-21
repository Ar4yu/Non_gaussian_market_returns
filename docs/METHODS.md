# Methods and interpretation notes

## Estimation and evaluation

We compute `log(P[t]) - log(P[t-1])` from the frozen adjusted-close series. The split date is October 5, 2022 inclusive for training. The first price observation has no return. The resulting QQQ sample has 3,211 training and 803 test returns.

Normal MLE uses the sample mean and variance with denominator n. Student-t estimation searches integer df from 1 to 150, optimizing location and log-scale for each candidate with L-BFGS-B. Selection uses training likelihood only. The holdout is evaluated without refitting the static models. AIC/BIC use two parameters for Normal and three for Student-t, counting the selected df. They remain model-comparison diagnostics, with the usual regularity caveats for discrete search and dependent financial observations.

A Student-t scale parameter is not its standard deviation. For df > 2, standard deviation equals scale times `sqrt(df / (df - 2))`. At df = 3, the fourth moment is not finite. Describing that fitted model as merely having a particular finite excess kurtosis would be inaccurate. See the [SciPy distribution definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html).

## Reading the tests correctly

A breach occurs when the observed log return is below the model's left-tail quantile. The levels 1% and 5% denote tail probabilities, corresponding to 99% and 95% loss-VaR confidence levels. A positive reported loss threshold is the negated log-return threshold, not an exact simple-return percentage.

Kupiec's unconditional coverage test asks whether the breach rate matches the nominal probability. Christoffersen's independence test asks whether successive breach indicators cluster under a first-order Markov comparison. The joint statistic adds those two likelihood ratios. Failure to reject is not proof that a model is correct, and the independence test can reject even when the joint test does not because the joint reference distribution has a different degree of freedom.

## Clarifications to the submitted paper

The original PDF remains unchanged. These notes accompany it so readers can distinguish the submission from the updated interpretation.

1. **5% Student-t calibration:** the paper calls 61 breaches “over-conservatism.” The threshold is −1.781% in log-return units, versus −2.074% for Normal, and the observed rate is 7.60%. It is insufficiently conservative at this level: losses breach it too frequently.
2. **1% Student-t dependence:** coverage and joint tests do not reject, but independence has p = 0.0460 and rejects at 5%. A blanket statement that all backtests pass would be wrong.
3. **Likelihood-ratio inference:** the reported 693.22 is a descriptive likelihood improvement statistic. Normal occurs as df tends to infinity; a standard finite-parameter nested chi-square cutoff is not justified here. Do not use the paper's informal “threshold of 10” as a formal test.
4. **Log returns:** computing log returns does not imply that those returns are lognormally distributed. The modeled variables here are log returns under Normal or location-scale Student-t distributions.
5. **Rolling method:** the stored rolling outputs use the sample mean and standard deviation as t location/scale. This expands the implied t standard deviation by `sqrt(3)` relative to the Normal estimate. It is not a rolling Student-t MLE comparison.

## Historical rolling sensitivity experiment

`rolling_var.py` forecasts each test date from the preceding 504 observations, excluding that date. It keeps df = 3 fixed and uses a 1% tail. The Normal experiment records 16/803 breaches; the plug-in t experiment records 2/803. Both reject unconditional and joint coverage at 5% significance. The t result is overly conservative in this particular experiment, unlike the static model's 5% result.

`outputs/legacy/rolling_var_backtest_qqq.csv` is an older combined daily export. Its rejection flags do not use the same 5% convention as the canonical summary, and it is excluded from the evidence path. Use `rolling_var_qqq_daily.csv` and `rolling_var_qqq_summary.csv`, which the reproduction command regenerates and checks.

A future rolling comparison should fit t location/scale in each window or explicitly variance-match distributions, then distinguish tail-shape effects from volatility-scale effects. That would be a new experiment and should not silently replace the submitted results.

## Portfolio code correction

The generic `backtest.var_threshold('t', ...)` helper previously applied scale twice. The published main backtest already used the separate, correct `var_threshold_t` helper, so the paper tables were unaffected. The generic helper now delegates to that implementation and has a regression test against SciPy's location-scale quantile.

The reproduction runner redirects the original modules to a separate output directory. The archived input and result files stay unchanged. No fresh market-data request is part of verification.
