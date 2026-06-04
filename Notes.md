# Data Missingness Notes

# Paper-style sample filter

`src/sample_filter.py` now builds the monthly sample used by `src/features.py`.

The script applies the main sample restrictions from the paper before feature construction:

1. Use the full date range available in the raw monthly return file.
2. Keep no-load share classes. The default is `no_load=True`, and a share-class-month is set to `False` if either `front_load > 0` or `rear_load > 0` after filling load schedules within each share class.
3. Exclude ETFs and passive/index funds. A fund is treated as passive/index if `index_fund_flag` is non-missing. When `index_fund_flag` is missing, the fund name is checked for index-like keywords such as `index`, `S&P`, `Russell`, `Nasdaq`, `Dow Jones`, `Wilshire`, and `MSCI`. ETFs are excluded when `et_flag` is non-missing.
4. Keep U.S. domestic-equity observations using CRSP objective codes that start with `ED`.
5. Keep observations with `per_com >= 70`.
6. Exclude observations before the first month with at least $5 million TNA.
7. Mark observations where the share class has reached 36 months of age.

Important timing detail: `sf_monthly_returns.csv` is filtered on `in_alpha_history`, which excludes the age >= 36 months requirement. This keeps the first 36 months of eligible return history available for FF5+MOM rolling-beta estimation. After `features.py` computes rolling alpha and targets, it keeps only `in_paper_sample == True`, where `in_paper_sample` includes `age_ge_36m == True`.

The outputs are:

- `data/processed/sf_monthly_returns.csv`
- `data/processed/sf_sample_flags.csv`

The filter-count summary is printed at the end of `sample_filter.py`; it is not saved as a CSV.

Current filter counts:

```text
            filter    rows  unique_share_classes
  raw_monthly_rows 9703080                 74448
           no_load 6067329                 52804
    active_managed 8631841                 63706
     per_com_ge_70 4758505                 40315
        age_ge_36m 7279799                 60913
after_first_5m_tna 7808735                 56919
   in_paper_sample 1624173                 15537
```

The filter-count summary below is from an earlier run before the U.S. domestic-equity and alpha-history changes. Rerun `src/sample_filter.py` to refresh the current counts.

## `fund_summary.per_com` gap

The raw December `per_com` coverage in `fund_summary.csv` is normal through 1997, collapses in 1998-2001, remains weak in 2002, and recovers from 2003 onward.

This creates an artificial break in the `per_com >= 70` equity screen used by the sample-filter pipeline. Many share classes have usable `per_com` before 1998 and again after 2003, but have missing December `per_com` during the gap years.

## Cleaning rule

The cleaned December `per_com` series is built at the `crsp_fundno`-year level with the following rules:

1. If December `per_com` exists, keep the actual December value.
2. If December is missing but the same year has other `per_com` observations, use that year's mean `per_com`.
3. Main clean version: if the entire year is missing, use only the nearest previous year's mean `per_com`.
4. If no previous year with usable `per_com` exists, leave `per_com` missing and let the observation fail the `per_com >= 70` screen.

This is controlled by `PER_COM_FILL_MODE = 'backward'` in `src/sample_filter.py`. This is the default setting for main results because it is backward-looking only and avoids using future equity-allocation information.

Robustness version:

Set `PER_COM_FILL_MODE = 'two_sided'` to run the older two-sided fill as a sensitivity check. Under this mode, if the entire year is missing, the code uses the nearest year on either side and averages the previous and next years when they are equally distant. This version is not the main specification because it can use future `per_com` information.

Example with available yearly means in 1997 and 2003:

- Main `backward`: 1998-2002 use the 1997 yearly mean.
- Robustness `two_sided`: 1998 and 1999 use 1997, 2000 uses the average of 1997 and 2003, and 2001-2002 use 2003.

## Output

The cleaned December `per_com` series is an internal intermediate in `src/sample_filter.py`; it is not saved as a standalone CSV. The output files include `per_com_fill_source` so the fill method can be audited later. The `per_com >= 70` screen is applied upstream in `src/sample_filter.py`, and `src/features.py` builds `f_panel.csv` from that already-filtered monthly sample.

# Random forest tuning runtime

The original RF grid was too expensive inside the expanding-window walk-forward loop with 5-fold cross-validation. As the training sample grew over time, RF became the main runtime bottleneck and eventually took well over one hour for a single test year.

This happened because RF tuning was repeated every year on a relatively large grid while the estimation window kept expanding.

Evidence from `data/processed/log_model.rtf`:

- RF runtime increased from about 4.8 minutes in 1993 to about 111.5 minutes in 2009.
- The mean RF runtime across completed years was about 45.8 minutes per test year.
- By the late sample, RF was the clear bottleneck in the full walk-forward run.

## RF grid revision

The RF grid was reduced using two principles:

1. Keep the specification consistent with the paper's random-forest setup.
2. Narrow the search around the parameter region that was repeatedly selected in the tuning logs.

The goal of the revision was to keep the model defensible while making the yearly walk-forward run computationally feasible.

The tuning logs showed that the selected RF parameters were already stabilizing in a narrow region:

- `max_features=4` was selected most often.
- `min_samples_leaf=25` became common in the later years.
- `max_depth=4` also became common in the later years.
- `n_estimators` still switched between 500 and 1000, so both candidates were kept.

This is why the revised grid keeps `n_estimators` flexible but fixes the other RF parameters at the values that were repeatedly selected once the training window became large.

Original RF grid:

```python
RF_PARAM_GRID = {
    'n_estimators': [500, 1000],
    'max_depth': [4, 8],
    'min_samples_leaf': [5, 25],
    'max_features': [4, 8, 12],
    'bootstrap': [True],
}
```

Updated RF grid:

```python
RF_PARAM_GRID = {
    'n_estimators': [500, 1000],
    'max_depth': [4],
    'min_samples_leaf': [25],
    'max_features': [4],
    'bootstrap': [True],
}
```

# Monthly return outliers

The raw `mret` series contains a small number of extreme observations that are large enough to distort equal-weight benchmarks and other portfolio aggregates.

The most important example found in the raw data is:

- `crsp_fundno=22169.0`
- `caldt=2007-12-31`
- `mret=12012.417728472221`
- `mtna=3.5`

This single observation was enough to push the equal-weight benchmark return on `2007-12-31` to about `0.5115`. Excluding that one fund brings the equal-weight mean for the same month back to about `-0.0050`.

More generally:

- the raw monthly return file contains repeated extreme outliers
- some are associated with very small `mtna`
- some appear in the final observed month of a share class
- some funds, such as `93941.0`, show repeated extreme positive returns over multiple dates

## Chosen cleaning rule

To keep the cleaning minimal and explicit, the raw monthly return download step now applies the following rules in `src/data_download.py` before writing `monthly_returns.csv`:

1. If `mret <= -1`, set `mret = 0`.
2. Else, if `mtna` is missing, set `mret = NaN`.
3. Else, if the observation is the final month of the share class and `mret > 10`, set `mret = NaN`.

This rule is implemented in `clean_ret()` and is applied immediately before saving:

- `data/raw/monthly_returns.csv`

# Time-series cross-validation

The hyperparameter tuning step in `model.py` was changed from ordinary five-fold CV to a time-series cross-validation method.

## Why ordinary CV was not appropriate

This is a financial prediction problem with a clear time ordering:

- predictors are formed in year `t`
- the model is used to predict outcomes in year `t+1`

Using ordinary cross-validation inside the training window can mix earlier and later years across training and validation folds. That makes the validation step too optimistic, because a model may be validated on earlier observations after being trained on later observations from the same historical window.

The outer walk-forward test was already out-of-sample, but the inner hyperparameter tuning step was not fully time-consistent under ordinary CV.

## Chosen method: year-block time-series CV

The revised tuning step uses year-based expanding-window splits.

The fold unit is a full calendar year block, not an individual `fund-year` observation. This matters because the model is trained and evaluated on annual cross-sections, and splitting a single year across train and validation would leak information from the same market environment into both sides of the fold.

The procedure is:

1. Sort the unique training years in time order.
2. Split those years into ordered year blocks.
3. For each fold, use all earlier blocks as the training set.
4. Use the next block as the validation set.

So each fold satisfies:

- training years < validation years
- no year appears in both train and validation
- all share classes from the same year stay in the same fold

Example for the first feasible test year (`1993`):

- train `1983-1984`, validate `1985-1986`
- train `1983-1986`, validate `1987-1988`
- train `1983-1988`, validate `1989-1990`
- train `1983-1990`, validate `1991`
- train `1983-1991`, validate `1992`

This year-block design is stricter and more defensible for financial data than ordinary CV, while still keeping the walk-forward framework unchanged.
