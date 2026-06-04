# TODO

This file summarizes the next research and engineering improvements discussed
for the mutual fund selection project.

## Priority 1: Evaluation Improvements

- Add subperiod performance tables:
  - 1991-2000
  - 2001-2010
  - 2011-2020
  - 2021-2025
- Add Newey-West adjusted alpha t-statistics for monthly portfolio returns.
- Report both ordinary OLS alpha t-stat and Newey-West alpha t-stat.
- Add factor beta output from the FF5+MOM portfolio alpha regressions:
  - MKT
  - SMB
  - HML
  - RMW
  - CMA
  - MOM
- Separate long-only and long-short performance interpretation:
  - Long-only can be compared with `equal_weight_all`.
  - Long-short should be evaluated as a zero-cost spread strategy, not against
    the long-only equal-weight benchmark curve.

## Priority 2: Long-Short Diagnostics

- Decompose long-short performance into:
  - long-leg return
  - short-leg return
  - long-short spread return
- Add cumulative plots for each leg separately.
- Check whether alpha comes from:
  - long leg outperforming
  - short leg underperforming
  - both legs contributing
- Add average number of holdings per leg by year.
- Add a minimum holdings filter for long-short runs:
  - `MIN_HOLDINGS_PER_LEG = 30` or `50`
  - skip years with too few funds in either leg
  - this avoids early-sample results being dominated by very small portfolios

## Priority 3: Decile and Strategy Sensitivity

- Run a small decile grid:
  - `LONG_DECILE = 0.05`, `SHORT_DECILE = 0`
  - `LONG_DECILE = 0.10`, `SHORT_DECILE = 0`
  - `LONG_DECILE = 0.20`, `SHORT_DECILE = 0`
  - `LONG_DECILE = 0.10`, `SHORT_DECILE = 0.10`
  - `LONG_DECILE = 0.20`, `SHORT_DECILE = 0.20`
  - `LONG_DECILE = 0.10`, `SHORT_DECILE = 0.20`
- Compare each setting by:
  - annualized alpha
  - alpha t-stat
  - Newey-West t-stat
  - annualized volatility
  - information ratio
  - max drawdown
  - average holdings
  - turnover
- Save the decile-grid output to a separate results file.

## Priority 4: Turnover and Implementation Feasibility

- Add yearly turnover calculations for each strategy.
- Estimate how many funds enter and exit the portfolio each rebalance year.
- Add implementation notes for long-short portfolios:
  - long-short mutual fund portfolios are useful for testing ranking signal
    quality
  - they are not directly tradable in most real-world settings because mutual
    fund shares are generally difficult or impossible to short
- For practical strategy evaluation, also report:
  - long-only top-decile portfolio
  - long-only top-quintile portfolio
  - avoid-bottom-decile analysis

## Priority 5: Robustness Checks

- Compare results across factor sources:
  - downloaded FF5+MOM factors
  - paper replication factor files
- Compare sample filter modes:
  - loose load screen
  - strict load screen
  - possible hybrid load screen
- Investigate the post-2011 candidate pool expansion:
  - new funds by CRSP objective code
  - load field missingness
  - passive/index flag missingness
  - variable-account or insurance-related fund names
- Check whether results are sensitive to:
  - minimum TNA threshold
  - no-load definition
  - domestic-equity objective-code definition
  - active/passive filtering rules

## Priority 6: Model Extensions

- Test alpha-based hyperparameter tuning instead of MSE-only tuning:
  - current `GridSearchCV` uses `neg_mean_squared_error`, which can prefer
    conservative predictions that reduce error but do not improve top-decile
    ranking
  - first experiment: tune Random Forest and XGBoost by validation top-decile
    `target_alpha`, using the average realized `target_alpha` of funds with the
    highest predicted alpha in each validation fold
  - stronger version: compute the score by validation year, then average across
    years, so large candidate-pool years do not dominate the tuning objective
  - possible extension: tune on validation long-short spread alpha, defined as
    top-decile realized `target_alpha` minus bottom-decile realized
    `target_alpha`
  - defer full monthly portfolio alpha tuning for now because it would require
    constructing CV portfolios and running factor-adjusted portfolio alpha
    inside each hyperparameter search
  - if this improves portfolio alpha, consider replacing `ElasticNetCV` with
    `GridSearchCV(ElasticNet(...))` so Elastic Net uses the same alpha-based
    tuning objective
- Build an ensemble ranking signal:
  - average rank across OLS, Elastic Net, Random Forest, and XGBoost
  - compare against individual model rankings
- Test whether rank-based predictions are more stable than raw predicted alpha.
- Add feature importance diagnostics for tree-based models.
- Track selected hyperparameters over time.
- Consider adding a simpler benchmark model:
  - past alpha only
  - past excess return only
  - equal-weighted characteristic score

## Priority 7: Documentation and Presentation

- Add a concise results table to `README.md` after the next stable run.
- Add a separate section explaining why long-short results are signal tests, not
  directly investable mutual fund strategies.
- Add clear notes on data availability:
  - raw CRSP/WRDS data is not included
  - generated outputs depend on licensed data access
- Keep `workflow.md` as the technical pipeline reference.
- Keep `README.md` focused on idea, insights, and result highlights.
