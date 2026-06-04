# src/features.py
import numpy as np
import pandas as pd
import time
from pathlib import Path

try:
    from .sample_filter import merge_interval_columns_to_monthly
except ImportError:
    from sample_filter import merge_interval_columns_to_monthly

try:
    from .config import TARGET_MODE
except ImportError:
    from config import TARGET_MODE

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'

# TARGET_MODE comes from config.py (single source of truth). Here it only
# controls which target the final feature panel is filtered on; both targets
# are always built.

FACTORS = ['mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'mom']
ROLLING_WINDOW = 36
ROLLING_OUTPUT_COLUMNS = [
    'crsp_fundno',
    'caldt',
    'alpha',
    'alpha_tstat',
    'r2',
    'mkt_rf_tstat',
    'smb_tstat',
    'hml_tstat',
    'rmw_tstat',
    'cma_tstat',
    'mom_tstat',
]

def load_raw_data():
    ret = pd.read_csv(PROCESSED_DIR / 'sf_monthly_returns.csv', parse_dates=['caldt', 'first_offer_dt'])
    fees = pd.read_csv(RAW_DIR / 'fund_fees.csv', parse_dates=['begdt', 'enddt'])
    hdr_hist = pd.read_csv( RAW_DIR / 'fund_hdr_hist.csv',  parse_dates=['chgdt', 'chgenddt', 'mgr_dt'])
    factors = pd.read_csv(RAW_DIR / 'ff5_mom_factors.csv', parse_dates=['date'])
    return ret, fees, hdr_hist, factors


def merge_manager_history_to_monthly(ret, hdr_hist):
    manager_intervals = hdr_hist.rename(
        columns={'chgdt': 'begdt', 'chgenddt': 'enddt'}
    )
    return merge_interval_columns_to_monthly(
        ret,
        manager_intervals,
        ['mgr_name', 'mgr_dt'],
    )

def compute_flow(df):
    """Compute fund flow as (TNA_t - TNA_{t-1} * (1 + ret_t)) / TNA_{t-1}."""
    df = df.sort_values(['crsp_fundno', 'caldt'])
    df['tna_lag'] = df.groupby('crsp_fundno')['mtna'].shift(1)
    df['flow'] = (
        (df['mtna'] - df['tna_lag'] * (1 + df['mret'])) / df['tna_lag']
    )
    return df


def winsorize_flow_cross_section(df):
    """Winsorize monthly flow cross-sectionally at the 1st and 99th percentiles."""
    df = df.copy()
    flow_p01 = df.groupby('caldt')['flow'].transform(lambda x: x.quantile(0.01))
    flow_p99 = df.groupby('caldt')['flow'].transform(lambda x: x.quantile(0.99))
    df['flow'] = df['flow'].where(df['flow'] >= flow_p01, flow_p01)
    df['flow'] = df['flow'].where(df['flow'] <= flow_p99, flow_p99)
    return df


def winsorize_alpha_by_year(df):
    """Winsorize monthly realized alpha within calendar year at the 1st and 99th percentiles."""
    df = df.copy()
    year = df['caldt'].dt.year
    alpha_p01 = df.groupby(year)['alpha'].transform(lambda x: x.quantile(0.01))
    alpha_p99 = df.groupby(year)['alpha'].transform(lambda x: x.quantile(0.99))
    df['alpha'] = df['alpha'].where(df['alpha'] >= alpha_p01, alpha_p01)
    df['alpha'] = df['alpha'].where(df['alpha'] <= alpha_p99, alpha_p99)
    return df

def compute_rolling_alphas(df, factors):
    """
    Run a 36-month rolling FF5+MOM regression for each share class.
    Betas are estimated using the 36-month window ending in month m-1,
    and monthly realized alpha in month m is computed using those lagged betas.
    Output realized alpha, alpha t-stat, factor-beta t-stats, and R-squared.
    """
    start_time = time.perf_counter()
    df = df.copy()
    df['month'] = df['caldt'].dt.to_period('M')

    factors = factors.copy()
    factors['month'] = factors['date'].dt.to_period('M')

    df = df.merge(
        factors.drop(columns=['date']),
        on='month',
        how='left',
    )
    df = df.drop(columns=['month'])
    df['excess_ret'] = df['mret'] - df['rf']

    results = []
    total_funds = df['crsp_fundno'].nunique()
    progress_interval = max(1, total_funds // 20)

    print(f"Running rolling regressions for {total_funds:,} share classes...")

    for idx, (fundno, grp) in enumerate(df.groupby('crsp_fundno'), start=1):
        grp = grp.sort_values('caldt').reset_index(drop=True)
        for i in range(ROLLING_WINDOW, len(grp)):
            window = grp.iloc[i - ROLLING_WINDOW:i]
            current = grp.iloc[i]
            y = window['excess_ret'].values
            X = window[FACTORS].values
            X = np.column_stack([np.ones(len(X)), X])
            current_excess_ret = current['excess_ret']
            current_factors = current[FACTORS].to_numpy(dtype=float)

            if (
                np.isnan(y).any()
                or np.isnan(X).any()
                or np.isnan(current_excess_ret)
                or np.isnan(current_factors).any()
            ):
                continue

            try:
                coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                resid  = y - X @ coef
                sse    = resid @ resid
                sst    = ((y - y.mean()) ** 2).sum()
                r2     = 1 - sse / sst if sst > 0 else np.nan
                n, k   = X.shape
                sigma2 = sse / (n - k)
                se     = np.sqrt(np.diag(sigma2 * np.linalg.pinv(X.T @ X)))
                realized_alpha = current_excess_ret - np.dot(current_factors, coef[1:])

                row = {
                    'crsp_fundno': fundno,
                    'caldt':       current['caldt'],
                    'alpha':       realized_alpha,
                    'alpha_tstat': coef[0] / se[0] if se[0] > 0 else np.nan,
                    'r2':          r2,
                }
                for j, f in enumerate(FACTORS):
                    row[f'{f}_tstat'] = coef[j+1] / se[j+1] if se[j+1] > 0 else np.nan

                results.append(row)
            except Exception:
                continue

        if idx % progress_interval == 0 or idx == total_funds:
            elapsed = time.perf_counter() - start_time
            timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            print(
                f"  [{timestamp}] Progress: {idx:,}/{total_funds:,} share classes "
                f"({idx / total_funds:.1%}), elapsed {elapsed:.1f}s"
            )

    total_elapsed = time.perf_counter() - start_time
    print(f"Rolling regressions finished in {total_elapsed:.1f}s.")

    if not results:
        return pd.DataFrame(columns=ROLLING_OUTPUT_COLUMNS)

    return pd.DataFrame(results, columns=ROLLING_OUTPUT_COLUMNS)

def compute_annual_alpha(df):
    """Target variable: annualized realized alpha as mean monthly alpha times 12."""
    df = df.copy()
    df['year'] = df['caldt'].dt.year
    annual = (
        df.groupby(['crsp_fundno', 'year'])['alpha']
          .mean()
          .mul(12)
          .reset_index()
          .rename(columns={'alpha': 'annual_alpha'})
    )
    return annual


def compute_annual_excess_return(df):
    """Return-experiment target: annualized realized excess return as mean
    monthly excess return times 12. Mirrors compute_annual_alpha so that the
    'excess_return' mode has the same one-year horizon as the 'alpha' mode."""
    df = df.copy()
    df['year'] = df['caldt'].dt.year
    annual = (
        df.groupby(['crsp_fundno', 'year'])['excess_return']
          .mean()
          .mul(12)
          .reset_index()
          .rename(columns={'excess_return': 'annual_excess_return'})
    )
    return annual


def add_excess_return(ret, factors):
    """Add the monthly excess-return feature (mret - rf).

    Note: the supervised target_excess_return is NOT built here. It is
    constructed later in build_features() as a next-year annualized series so
    that it matches target_alpha's one-year horizon and the 12-month portfolio
    holding period, instead of a one-month-ahead return.
    """
    ret = ret.copy()
    ret['month'] = ret['caldt'].dt.to_period('M')

    rf = factors[['date', 'rf']].copy()
    rf['month'] = rf['date'].dt.to_period('M')

    ret = ret.merge(rf[['month', 'rf']], on='month', how='left')
    ret = ret.drop(columns=['month'])
    ret['excess_return'] = ret['mret'] - ret['rf']
    return ret

def build_features(ret, fees, hdr_hist, factors):
    # Step 1: merge fee data into the monthly panel
    print("Merging fee data...")
    ret = merge_interval_columns_to_monthly(ret, fees, ['exp_ratio', 'turn_ratio'])

    # Step 2: compute and winsorize monthly flow
    print("Computing and winsorizing flow...")
    ret = compute_flow(ret)
    ret = winsorize_flow_cross_section(ret)

    # Step 3: compute fund age and manager tenure
    print("Computing fund age and manager tenure...")
    ret = merge_manager_history_to_monthly(ret, hdr_hist)
    ret['manager_tenure'] = (
        (ret['caldt'].dt.year - ret['mgr_dt'].dt.year) +
        (ret['caldt'].dt.month - ret['mgr_dt'].dt.month) / 12
    ).clip(lower=0)

    # Step 4: run rolling FF5+MOM regressions
    print("Computing rolling regressions. This may take a few minutes...")
    rolling = compute_rolling_alphas(ret, factors)
    ret = ret.merge(rolling, on=['crsp_fundno', 'caldt'], how='left')
    # print("Winsorizing monthly realized alpha by year...")
    # ret = winsorize_alpha_by_year(ret)

    # Step 5: add the monthly excess-return feature (target built in Step 8)
    ret = add_excess_return(ret, factors)

    # Step 6: compute value added
    ret['value_added'] = (
        (ret['alpha'] + ret['exp_ratio'] / 12) * ret['tna_lag']
    )

    # Step 7: monthly flow_vol is kept as a rolling diagnostic series.
    # The annual model panel recomputes flow and flow_vol from monthly
    # winsorized flow using calendar-year aggregation rules.
    ret = ret.sort_values(['crsp_fundno', 'caldt'])
    ret['flow_vol'] = (
        ret.groupby('crsp_fundno')['flow']
           .transform(lambda x: x.rolling(12).std())
    )

    # Step 8: build both supervised targets as next-year annualized series.
    # For feature-year t, the target is year t+1's annualized value
    # (mean monthly value * 12). Both targets share this one-year horizon, which
    # matches the 12-month portfolio holding period in backtest.py.
    print("Building target variables...")
    ret['year'] = ret['caldt'].dt.year

    # target_alpha: next-year annualized realized alpha.
    annual_alpha = compute_annual_alpha(ret[['crsp_fundno', 'caldt', 'alpha']].dropna())
    annual_alpha['label_year'] = annual_alpha['year'] - 1  # Attach year t+1 alpha to year t

    ret = ret.merge(
        annual_alpha[['crsp_fundno', 'label_year', 'annual_alpha']].rename(
            columns={'annual_alpha': 'target_alpha'}
        ),
        left_on=['crsp_fundno', 'year'],
        right_on=['crsp_fundno', 'label_year'],
        how='left'
    )

    # target_excess_return: next-year annualized excess return (return experiment).
    # Built the same way as target_alpha, so it is constant within a fund-year
    # and is picked up correctly by the December snapshot in model.py.
    annual_excess = compute_annual_excess_return(
        ret[['crsp_fundno', 'caldt', 'excess_return']].dropna()
    )
    annual_excess['label_year_er'] = annual_excess['year'] - 1  # year t+1 excess return -> year t

    ret = ret.merge(
        annual_excess[['crsp_fundno', 'label_year_er', 'annual_excess_return']].rename(
            columns={'annual_excess_return': 'target_excess_return'}
        ),
        left_on=['crsp_fundno', 'year'],
        right_on=['crsp_fundno', 'label_year_er'],
        how='left'
    ).drop(columns=['label_year_er'])

    # Step 9: after using the broader history for alpha estimation, keep
    # only observations eligible for the final paper-style panel.
    if 'in_paper_sample' in ret.columns:
        before_filter = len(ret)
        ret = ret[ret['in_paper_sample'].eq(True)].copy()
        print(
            "Keeping final eligible sample after alpha estimation: "
            f"{len(ret):,}/{before_filter:,} rows"
        )

    # Step 10: select and return the final output columns
    feature_cols = [
        'crsp_fundno', 'caldt',
        # Fund characteristics
        'mtna', 'exp_ratio', 'turn_ratio', 'age_months', 'manager_tenure',
        # Load status
        'front_load', 'rear_load', 'no_load',
        # Flow
        'flow', 'flow_vol',
        # Value added
        'value_added',
        # Rolling regression
        'alpha', 'alpha_tstat', 'r2',
        'mkt_rf_tstat', 'smb_tstat', 'hml_tstat',
        'rmw_tstat', 'cma_tstat', 'mom_tstat',
        # Monthly excess return target experiment
        'excess_return', 'target_excess_return',
        # Target
        'target_alpha',
    ]

    panel = ret[[c for c in feature_cols if c in ret.columns]].copy()

    # Drop on the ACTIVE target so the return experiment is not silently
    # filtered by target_alpha availability. Both modes also require 'alpha' so
    # the rolling-regression-derived features (alpha_tstat, r2, factor t-stats,
    # value_added) are real rather than zero-filled.
    if TARGET_MODE == 'alpha':
        panel = panel.dropna(subset=['alpha', 'target_alpha'])
    else:
        panel = panel.dropna(subset=['alpha', 'excess_return', 'target_excess_return'])

    return panel

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw data...")
    ret, fees, hdr_hist, factors = load_raw_data()

    print(f"Raw data: {ret.shape[0]:,} rows, {ret['crsp_fundno'].nunique():,} share classes")

    panel = build_features(
        ret,
        fees,
        hdr_hist,
        factors,
    )

    panel.to_csv(PROCESSED_DIR / 'f_panel.csv', index=False)
    print(f"\nFeature panel: {panel.shape}")
    print(f"Date range: {panel['caldt'].min()} -> {panel['caldt'].max()}")
    print(f"Share classes: {panel['crsp_fundno'].nunique():,}")
    print("\nMissing value counts:")
    print(panel.isnull().sum())

    correlation_cols = [c for c in panel.columns if c not in ['crsp_fundno', 'caldt']]
    correlation_matrix = panel[correlation_cols].corr()
    correlation_path = PROCESSED_DIR / 'f_corr_monthly.csv'
    correlation_matrix.to_csv(correlation_path)

    print("\nFeature correlation matrix:")
    print(correlation_matrix.round(3))
    print(f"\nSaved feature correlation matrix to: {correlation_path}")

if __name__ == '__main__':
    main()
