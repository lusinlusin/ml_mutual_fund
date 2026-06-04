# Workflow

## `sample_filter.py`

### Inputs
- [data/raw/monthly_returns.csv](/Users/allison/Documents/Portfolio/ml_mutual_fund/data/raw/monthly_returns.csv)
- [data/raw/front_load.csv](/Users/allison/Documents/Portfolio/ml_mutual_fund/data/raw/front_load.csv)
- [data/raw/rear_load.csv](/Users/allison/Documents/Portfolio/ml_mutual_fund/data/raw/rear_load.csv)
- [data/raw/fund_hdr.csv](/Users/allison/Documents/Portfolio/ml_mutual_fund/data/raw/fund_hdr.csv)
- [data/raw/fund_hdr_hist.csv](/Users/allison/Documents/Portfolio/ml_mutual_fund/data/raw/fund_hdr_hist.csv)
- [data/raw/fund_style.csv](/Users/allison/Documents/Portfolio/ml_mutual_fund/data/raw/fund_style.csv)
- [data/raw/fund_summary.csv](/Users/allison/Documents/Portfolio/ml_mutual_fund/data/raw/fund_summary.csv)

### Outputs
- [data/processed/sf_monthly_returns.csv](/Users/allison/Documents/Portfolio/ml_mutual_fund/data/processed/sf_monthly_returns.csv)
- [data/processed/sf_sample_flags.csv](/Users/allison/Documents/Portfolio/ml_mutual_fund/data/processed/sf_sample_flags.csv)

Notes:
- This script applies the paper-style sample filters needed before feature construction, but keeps pre-36-month return history for rolling alpha estimation.
- It uses the full date range available in the raw monthly return file.
- It has three top-level mode switches:
- `LOAD_FILTER_MODE = 'strict'` keeps a share-class-month only when both `front_load == 0` and `rear_load == 0` are observed after backward-filling load schedules within each share class. Missing load values fail the no-load screen. `LOAD_FILTER_MODE = 'loose'` starts from `no_load=True` and excludes only observations with `front_load > 0` or `rear_load > 0`.
- `DOMESTIC_EQUITY_MODE = 'strict'` keeps only the stricter ED domestic-equity objective-code list defined in `STRICT_DOMESTIC_EQUITY_CODES`. `DOMESTIC_EQUITY_MODE = 'ed_prefix'` keeps any observation whose `crsp_obj_cd` starts with `ED`.
- `PER_COM_FILL_MODE = 'backward'` fills missing full-year `per_com` values only from past annual `per_com`. `PER_COM_FILL_MODE = 'two_sided'` uses the nearest year on either side and should be used only for sensitivity checks because it can use future `per_com`.
- It excludes ETFs and passive/index funds using `index_fund_flag`, `et_flag`, and fund-name keywords when `index_fund_flag` is missing.
- It applies the `per_com >= 70` and post-first-$5M-TNA filters to `sf_monthly_returns.csv`.
- It writes `age_ge_36m` and `in_paper_sample` flags, but `sf_monthly_returns.csv` is filtered on `in_alpha_history`, not on `age_ge_36m`.
- It prints filter-count diagnostics at the end, but does not save those diagnostics as a CSV.

## `features.py`

### Inputs
- [data/processed/sf_monthly_returns.csv](/Users/allison/Documents/Portfolio/ml_mutual_fund/data/processed/sf_monthly_returns.csv)
- [data/raw/fund_fees.csv](/Users/allison/Documents/Portfolio/ml_mutual_fund/data/raw/fund_fees.csv)
- [data/raw/fund_hdr_hist.csv](/Users/allison/Documents/Portfolio/ml_mutual_fund/data/raw/fund_hdr_hist.csv)
- [data/raw/ff5_mom_factors.csv](/Users/allison/Documents/Portfolio/ml_mutual_fund/data/raw/ff5_mom_factors.csv)

### Outputs
- [data/processed/f_panel.csv](/Users/allison/Documents/Portfolio/ml_mutual_fund/data/processed/f_panel.csv)
- [data/processed/f_corr_monthly.csv](/Users/allison/Documents/Portfolio/ml_mutual_fund/data/processed/f_corr_monthly.csv)

Notes:
- `f_panel.csv` now includes both `excess_return` and `target_excess_return` in addition to `alpha` and `target_alpha`.
- `features.py` no longer constructs the sample from raw monthly returns directly; it uses `sf_monthly_returns.csv` from `sample_filter.py`.
- Rolling alpha is computed on the broader `in_alpha_history` sample. After alpha and targets are built, `features.py` keeps only `in_paper_sample == True` observations for `f_panel.csv`.

## `model.py`

### Inputs
- [data/processed/f_panel.csv](/Users/allison/Documents/Portfolio/ml_mutual_fund/data/processed/f_panel.csv)

### Outputs
- [data/processed/m_corr_annual.csv](/Users/allison/Documents/Portfolio/ml_mutual_fund/data/processed/m_corr_annual.csv)
- Progress log files in [data/processed](/Users/allison/Documents/Portfolio/ml_mutual_fund/data/processed), such as `m_model_progress_*.log`
- `alpha` mode:
  - [data/processed/m_predictions_alpha.csv](/Users/allison/Documents/Portfolio/ml_mutual_fund/data/processed/m_predictions_alpha.csv)
  - [data/processed/m_metrics_alpha.csv](/Users/allison/Documents/Portfolio/ml_mutual_fund/data/processed/m_metrics_alpha.csv)
- `excess_return` mode:
  - [data/processed/m_predictions_return.csv](/Users/allison/Documents/Portfolio/ml_mutual_fund/data/processed/m_predictions_return.csv)
  - [data/processed/m_metrics_return.csv](/Users/allison/Documents/Portfolio/ml_mutual_fund/data/processed/m_metrics_return.csv)

Notes:
- `model.py` has a top-level switch:
  - `TARGET_MODE = 'alpha'`
  - `TARGET_MODE = 'excess_return'`
- In `alpha` mode, the model uses `target_alpha` as the target and includes `alpha` in `FEATURE_COLS` but excludes `excess_return`.
- In `excess_return` mode, the model uses `target_excess_return` as the target and includes `excess_return` in `FEATURE_COLS` but excludes `alpha`.
- `model.py` also writes the annual correlation matrix from `prepare_annual_panel(panel)` before the walk-forward run starts.
- `model.py` no longer reads `fund_summary` or applies a second `per_com >= 70` screen; this screen is handled upstream by `sample_filter.py`.
- During the walk-forward run, predictions are written incrementally by year, so interrupting the script does not discard already completed years.
- The metrics files store yearly IC by model plus the bottom summary rows `IC_mean`, `ICIR`, and `IC>0`.

Quick guide:
1. Run [sample_filter.py](/Users/allison/Documents/Portfolio/ml_mutual_fund/src/sample_filter.py) to regenerate the paper-style monthly sample.
2. Run [features.py](/Users/allison/Documents/Portfolio/ml_mutual_fund/src/features.py) to regenerate [f_panel.csv](/Users/allison/Documents/Portfolio/ml_mutual_fund/data/processed/f_panel.csv).
3. Set `TARGET_MODE = 'alpha'` in [model.py](/Users/allison/Documents/Portfolio/ml_mutual_fund/src/model.py) and run it once.
4. Set `TARGET_MODE = 'excess_return'` in [model.py](/Users/allison/Documents/Portfolio/ml_mutual_fund/src/model.py) and run it again.
5. Compare:
   - [data/processed/m_metrics_alpha.csv](/Users/allison/Documents/Portfolio/ml_mutual_fund/data/processed/m_metrics_alpha.csv)
   - [data/processed/m_metrics_return.csv](/Users/allison/Documents/Portfolio/ml_mutual_fund/data/processed/m_metrics_return.csv)

## `backtest.py`

### Inputs
- [data/processed/sf_monthly_returns.csv](/Users/allison/Documents/Portfolio/ml_mutual_fund/data/processed/sf_monthly_returns.csv)
- [data/raw/ff5_mom_factors.csv](/Users/allison/Documents/Portfolio/ml_mutual_fund/data/raw/ff5_mom_factors.csv)
- `alpha` mode:
  - [data/processed/m_predictions_alpha.csv](/Users/allison/Documents/Portfolio/ml_mutual_fund/data/processed/m_predictions_alpha.csv)
- `excess_return` mode:
  - [data/processed/m_predictions_return.csv](/Users/allison/Documents/Portfolio/ml_mutual_fund/data/processed/m_predictions_return.csv)

### Outputs
- `alpha` mode:
  - [results/portfolio_returns_alpha.csv](/Users/allison/Documents/Portfolio/ml_mutual_fund/results/portfolio_returns_alpha.csv)
  - [results/metrics_alpha.csv](/Users/allison/Documents/Portfolio/ml_mutual_fund/results/metrics_alpha.csv)
  - [results/plots/performance_alpha.png](/Users/allison/Documents/Portfolio/ml_mutual_fund/results/plots/performance_alpha.png)
- `excess_return` mode:
  - [results/portfolio_returns_return.csv](/Users/allison/Documents/Portfolio/ml_mutual_fund/results/portfolio_returns_return.csv)
  - [results/metrics_return.csv](/Users/allison/Documents/Portfolio/ml_mutual_fund/results/metrics_return.csv)
  - [results/plots/performance_return.png](/Users/allison/Documents/Portfolio/ml_mutual_fund/results/plots/performance_return.png)

Notes:
- `backtest.py` has a top-level switch:
  - `TARGET_MODE = 'alpha'`
  - `TARGET_MODE = 'excess_return'`
- `portfolio_returns_*.csv` includes both strategy portfolios and the `equal_weight_all` benchmark.
- `portfolio_returns_*.csv` contains both monthly portfolio returns (`mret`) and monthly portfolio alpha (`portfolio_alpha`).
- `portfolio_alpha` is computed from monthly portfolio excess returns minus the product of monthly FF5+MOM factor realizations and whole out-of-sample portfolio betas estimated separately for each portfolio.
- `performance_*.png` plots cumulative return, drawdown, and cumulative alpha in one figure.
- `metrics_*.csv` includes FF5+MOM out-of-sample alpha estimated from a time-series regression of monthly portfolio excess returns on contemporaneous factor returns.
