# Presenting the returns study

## Short introduction

“Lyla Saigal and I studied whether a Student-t distribution describes QQQ daily log returns better than a Normal model. We fit both on a historical training period and evaluated them on 803 later trading days. Student-t with three degrees of freedom improved held-out likelihood and extreme-tail coverage, but the results at the 5% threshold went the other way. The project shows why better overall fit and better risk calibration are related but different questions.”

## Five-minute walkthrough

1. Open the README figure: contrast the 1% and 5% breach rates before discussing the model machinery.
2. Show paper pages 2–4: prices, log returns, chronological split, and training-only df selection.
3. Show the density figure and page 6: explain the location/scale fit, likelihood, and complexity penalty.
4. Show the four-row coverage table: discuss why the 1% independence rejection matters even though the joint test does not reject.
5. Run `python scripts/reproduce.py`: show that the original result tables can be reconstructed without a live download.
6. Close with the rolling-method limitation and what a conditional-volatility model or correctly refitted rolling comparison could investigate next.

## Resume bullet

“Coauthored a QQQ tail-risk study comparing Normal and Student-t models with MLE and chronological holdout evaluation; selected df = 3 and analyzed 1%/5% VaR using coverage and independence tests, with a reproducible Python pipeline.”

Optional evidence for interviews: the Normal model had 15 breaches against 8.03 expected at the 1% tail; Student-t had 7, but still rejected independence at 5% significance. Student-t's held-out log-likelihood was 53.20 higher. These are risk-model diagnostics, not trading profits.

## Attribution and research integrity

Research authorship is shared between Lyla Saigal and Aaryaman Jaising. Describe individual contributions only from your own recollection and records. The September 2026 packaging and code checks used Codex assistance. The original submitted PDF remains intact, with interpretive corrections documented separately in METHODS.md.
