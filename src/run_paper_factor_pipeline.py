# src/run_paper_factor_pipeline.py
from datetime import datetime
import importlib
from pathlib import Path
import time

import pandas as pd

try:
    from .config import TARGET_MODE
except ImportError:
    from config import TARGET_MODE


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
PAPER_DATA_DIR = (
    BASE_DIR /
    'Machine Learning and Fund Characteristics Help to Select Mutual Funds with Positive Alpha' /
    'replication_files' /
    'data_sets'
)
OUTPUT_DIR = PROCESSED_DIR / 'paper_factors'
RESULTS_DIR = BASE_DIR / 'results' / 'paper_factors'
PLOTS_DIR = RESULTS_DIR / 'plots'

# TARGET_MODE comes from config.py (single source of truth).
TEST_YEAR = None       # Set to one year for debugging; None runs all years.

RUN_FEATURES = True
RUN_MODEL = True
RUN_BACKTEST = True

# The authors' R files use the first 690 FF5 rows, ending in 2020-12.
MATCH_AUTHOR_ROW_LIMITS = True
FF5_MONTHLY_ROWS = 690
MOM_MONTHLY_ROWS = 1128

FACTOR_COLS = ['mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'mom']


def import_local_module(module_name):
    if __package__:
        return importlib.import_module(f'.{module_name}', package=__package__)
    return importlib.import_module(module_name)


def _read_ken_french_monthly_csv(path):
    df = pd.read_csv(path, dtype={0: str})
    df.columns = [str(col).strip() for col in df.columns]
    df = df.rename(columns={df.columns[0]: 'yyyymm'})
    df = df[df['yyyymm'].astype(str).str.fullmatch(r'\d{6}')].copy()
    df['yyyymm'] = df['yyyymm'].astype(int)
    return df


def load_paper_ff5_mom_factors():
    ff5_path = PAPER_DATA_DIR / 'F-F_Research_Data_5_Factors_2x3.CSV'
    mom_path = PAPER_DATA_DIR / 'F-F_Momentum_Factor.CSV'

    ff5 = _read_ken_french_monthly_csv(ff5_path)
    mom = _read_ken_french_monthly_csv(mom_path)

    if MATCH_AUTHOR_ROW_LIMITS:
        ff5 = ff5.head(FF5_MONTHLY_ROWS)
        mom = mom.head(MOM_MONTHLY_ROWS)

    ff5 = ff5.rename(
        columns={
            'Mkt-RF': 'mkt_rf',
            'SMB': 'smb',
            'HML': 'hml',
            'RMW': 'rmw',
            'CMA': 'cma',
            'RF': 'rf',
        }
    )
    mom = mom.rename(columns={mom.columns[1]: 'mom'})

    factors = ff5.merge(mom[['yyyymm', 'mom']], on='yyyymm', how='inner')
    factors['date'] = (
        pd.to_datetime(factors['yyyymm'].astype(str), format='%Y%m') +
        pd.offsets.MonthEnd(0)
    )

    keep_cols = ['date', 'mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'rf', 'mom']
    factors = factors[keep_cols].copy()
    for col in keep_cols[1:]:
        factors[col] = pd.to_numeric(factors[col], errors='coerce') / 100

    return factors


def build_feature_panel_with_paper_factors():
    features = import_local_module('features')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading paper FF5+MOM factors...")
    factors = load_paper_ff5_mom_factors()
    factors_path = OUTPUT_DIR / 'ff5_mom_factors.csv'
    factors.to_csv(factors_path, index=False)
    print(
        f"Saved paper factors: {factors_path} "
        f"({factors['date'].min().date()} -> {factors['date'].max().date()})"
    )

    print("Loading current filtered sample and raw feature inputs...")
    ret = pd.read_csv(
        PROCESSED_DIR / 'sf_monthly_returns.csv',
        parse_dates=['caldt', 'first_offer_dt'],
    )
    fees = pd.read_csv(RAW_DIR / 'fund_fees.csv', parse_dates=['begdt', 'enddt'])
    hdr_hist = pd.read_csv(
        RAW_DIR / 'fund_hdr_hist.csv',
        parse_dates=['chgdt', 'chgenddt', 'mgr_dt'],
    )

    print("Building feature panel with paper factors...")
    panel = features.build_features(ret, fees, hdr_hist, factors)

    panel_path = OUTPUT_DIR / 'f_panel.csv'
    panel.to_csv(panel_path, index=False)

    corr_cols = [col for col in panel.columns if col not in ['crsp_fundno', 'caldt']]
    corr_path = OUTPUT_DIR / 'f_corr_monthly.csv'
    panel[corr_cols].corr().to_csv(corr_path)

    print(f"Saved feature panel: {panel_path}")
    print(f"Saved monthly correlation matrix: {corr_path}")
    print(f"Panel shape: {panel.shape}")
    print(f"Panel date range: {panel['caldt'].min()} -> {panel['caldt'].max()}")
    print(f"Panel share classes: {panel['crsp_fundno'].nunique():,}")
    return panel


def configure_model_module(model, target_mode):
    if target_mode not in {'alpha', 'excess_return'}:
        raise ValueError("TARGET_MODE must be 'alpha' or 'excess_return'")

    model.TARGET_MODE = target_mode
    model.TARGET_COL = 'target_alpha' if target_mode == 'alpha' else 'target_excess_return'
    model.FEATURE_COLS = (
        model.COMMON_FEATURE_COLS + ['alpha']
        if target_mode == 'alpha'
        else model.COMMON_FEATURE_COLS + ['excess_return']
    )
    model.PREDICTION_COLS = [
        'crsp_fundno',
        'year',
        model.TARGET_COL,
        'model',
        'pred_alpha',
    ]
    model.TEST_YEAR = TEST_YEAR
    model.PROCESSED_DIR = OUTPUT_DIR


def run_model_with_paper_factor_panel():
    model = import_local_module('model')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_model_module(model, TARGET_MODE)

    panel_path = OUTPUT_DIR / 'f_panel.csv'
    if not panel_path.exists():
        raise FileNotFoundError(
            f"{panel_path} does not exist. Set RUN_FEATURES = True first."
        )

    if TARGET_MODE == 'alpha':
        predictions_path = OUTPUT_DIR / 'm_predictions_alpha.csv'
        metrics_path = OUTPUT_DIR / 'm_metrics_alpha.csv'
    else:
        predictions_path = OUTPUT_DIR / 'm_predictions_return.csv'
        metrics_path = OUTPUT_DIR / 'm_metrics_return.csv'

    log_path = OUTPUT_DIR / f"m_model_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log = model.make_logger(log_path)
    start_time = time.time()

    log(f"Progress log: {log_path}")
    log(f"Target mode: {model.TARGET_MODE} ({model.TARGET_COL})")
    log(f"Loading paper-factor feature panel: {panel_path}")
    panel = pd.read_csv(panel_path, parse_dates=['caldt'])
    log(f"Panel shape: {panel.shape}")

    log("Converting to annual panel...")
    annual = model.prepare_annual_panel(panel)
    log(f"Annual panel: {annual.shape}, years: {annual['year'].min()}-{annual['year'].max()}")
    if TEST_YEAR is not None:
        log(f"Requested test year: {TEST_YEAR}")

    log("")
    log("Starting walk-forward prediction...")
    predictions = model.walk_forward_predict(
        annual,
        run_start_time=start_time,
        test_year=TEST_YEAR,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        log_fn=log,
    )

    if predictions_path.exists():
        stored_predictions = pd.read_csv(predictions_path)
    else:
        stored_predictions = pd.DataFrame(columns=model.PREDICTION_COLS)

    if stored_predictions.empty:
        log("")
        log("No predictions were generated. Skipping evaluation.")
        return stored_predictions

    log("")
    log(f"Predictions generated this run: {predictions.shape}")
    log(f"Combined predictions on disk: {stored_predictions.shape}")

    log("")
    log("Model evaluation (yearly IC / summary):")
    metrics = model.evaluate_predictions(stored_predictions)
    log(metrics.tail(3).to_string())
    metrics.to_csv(metrics_path)

    elapsed = time.time() - start_time
    log(f"\nTotal runtime: {elapsed:.1f} seconds ({elapsed / 60:.2f} minutes)")
    return stored_predictions


def run_backtest_with_paper_factors():
    backtest = import_local_module('backtest')
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    factors = pd.read_csv(OUTPUT_DIR / 'ff5_mom_factors.csv', parse_dates=['date'])
    monthly_ret = pd.read_csv(PROCESSED_DIR / 'sf_monthly_returns.csv', parse_dates=['caldt'])

    if TARGET_MODE == 'alpha':
        predictions_path = OUTPUT_DIR / 'm_predictions_alpha.csv'
        portfolio_returns_path = RESULTS_DIR / 'portfolio_returns_alpha.csv'
        metrics_path = RESULTS_DIR / 'metrics_alpha.csv'
        plot_path = PLOTS_DIR / 'performance_alpha.png'
    else:
        predictions_path = OUTPUT_DIR / 'm_predictions_return.csv'
        portfolio_returns_path = RESULTS_DIR / 'portfolio_returns_return.csv'
        metrics_path = RESULTS_DIR / 'metrics_return.csv'
        plot_path = PLOTS_DIR / 'performance_return.png'

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"{predictions_path} does not exist. Set RUN_MODEL = True first."
        )

    predictions = pd.read_csv(predictions_path)
    portfolio_returns = backtest.construct_portfolios(predictions, monthly_ret)
    strategy_dates = portfolio_returns['caldt'].drop_duplicates()
    benchmark = backtest.compute_benchmarks(
        monthly_ret,
        factors,
        start_date=portfolio_returns['caldt'].min(),
        end_date=portfolio_returns['caldt'].max(),
        strategy_dates=strategy_dates,
    )
    all_returns = pd.concat([portfolio_returns, benchmark], ignore_index=True)

    portfolio_alpha = backtest.construct_portfolio_alpha(all_returns, factors)
    portfolio_results = all_returns.merge(
        portfolio_alpha,
        on=['caldt', 'model'],
        how='left',
    )
    portfolio_results.to_csv(portfolio_returns_path, index=False)

    metrics = backtest.compute_metrics(portfolio_results, factors)
    metrics.to_csv(metrics_path)
    backtest.plot_results(portfolio_results, plot_path)

    print(f"Saved backtest portfolio returns: {portfolio_returns_path}")
    print(f"Saved backtest metrics: {metrics_path}")
    print(metrics.round(4))


def main():
    if RUN_FEATURES:
        build_feature_panel_with_paper_factors()
    if RUN_MODEL:
        run_model_with_paper_factor_panel()
    if RUN_BACKTEST:
        run_backtest_with_paper_factors()


if __name__ == '__main__':
    main()
