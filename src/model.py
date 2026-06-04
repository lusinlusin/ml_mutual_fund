# src/model.py
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import time
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

try:
    from .config import TARGET_MODE
except ImportError:
    from config import TARGET_MODE

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'

# TARGET_MODE is defined once in config.py. TARGET_COL is derived from it.
TARGET_COL = 'target_alpha' if TARGET_MODE == 'alpha' else 'target_excess_return'

COMMON_FEATURE_COLS = [
    'mtna', 'exp_ratio', 'turn_ratio', 'age_months', 'manager_tenure',
    'flow', 'flow_vol', 'value_added',
    'alpha_tstat', 'r2',
    'mkt_rf_tstat', 'smb_tstat', 'hml_tstat',
    'rmw_tstat', 'cma_tstat', 'mom_tstat',
]
FEATURE_COLS = (
    COMMON_FEATURE_COLS + ['alpha']
    if TARGET_MODE == 'alpha'
    else COMMON_FEATURE_COLS + ['excess_return']
)

MIN_TRAIN_YEARS = 10   # Minimum number of years required for training.
RETRAIN_FREQ    = 1    # Retrain every year.
CV_FOLDS        = 5    # Tune hyperparameters with five-fold cross-validation.
TEST_YEAR       = None # Set to a single year such as 2010 for debugging; use None to run all years.

RF_PARAM_GRID = {
    'n_estimators': [500, 1000],
    'max_depth': [3, 6],
    'min_samples_leaf': [10, 50],
    'max_features': [3, 6],
    'bootstrap': [True],
}

XGB_PARAM_GRID = {
    'n_estimators': [300, 800],
    'learning_rate': [0.01, 0.05],
    'max_depth': [2, 4],
    'min_child_weight': [5, 20],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
}

PREDICTION_COLS = ['crsp_fundno', 'year', TARGET_COL, 'model', 'pred_alpha']


def initialize_output_files(predictions_path, metrics_path, log_fn=print):
    """
    Start a full model run from clean output files.
    Single-year debug runs keep existing files and only replace that year.
    """
    if TEST_YEAR is not None:
        log_fn("Single-year run: keeping existing prediction file.")
        return

    pd.DataFrame(columns=PREDICTION_COLS).to_csv(predictions_path, index=False)
    if metrics_path.exists():
        metrics_path.unlink()
    log_fn(f"Initialized prediction file: {predictions_path}")


def make_logger(log_path):
    def log(message=""):
        text = str(message)
        print(text)
        with log_path.open('a', encoding='utf-8') as f:
            f.write(text + '\n')
    return log


def build_year_cv_splits(years, n_splits=CV_FOLDS):
    """
    Build year-based expanding-window CV splits.
    Each validation fold is a contiguous block of years, and each training fold
    uses only earlier years.
    """
    years = np.asarray(years)
    unique_years = np.sort(np.unique(years))

    if len(unique_years) < 2:
        return []

    n_splits = min(n_splits, len(unique_years) - 1)
    year_blocks = [block for block in np.array_split(unique_years, n_splits + 1) if len(block) > 0]

    splits = []
    for i in range(len(year_blocks) - 1):
        train_years = np.concatenate(year_blocks[: i + 1])
        val_years = year_blocks[i + 1]

        train_idx = np.flatnonzero(np.isin(years, train_years))
        val_idx = np.flatnonzero(np.isin(years, val_years))

        if len(train_idx) == 0 or len(val_idx) == 0:
            continue

        splits.append((train_idx, val_idx))

    return splits


def get_models(cv_splits):
    return {
        'ols': LinearRegression(),
        'elastic_net': ElasticNetCV(
            l1_ratio=[0.001, 0.01, 0.05, 0.1, 0.3, 0.5],
            alphas=np.logspace(-8, -2, 13),
            cv=cv_splits,
            max_iter=10000,
            random_state=42,
        ),
        'random_forest': GridSearchCV(
            estimator=RandomForestRegressor(
                random_state=42,
                n_jobs=1,
            ),
            param_grid=RF_PARAM_GRID,
            cv=cv_splits,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            refit=True,
        ),
        'xgboost': GridSearchCV(
            estimator=XGBRegressor(
                objective='reg:squarederror',
                random_state=42,
                n_jobs=1,
                verbosity=0,
            ),
            param_grid=XGB_PARAM_GRID,
            cv=cv_splits,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            refit=True,
        ),
    }

def prepare_annual_panel(panel):
    """
    Convert the monthly panel to annual frequency, following the paper:
    1. alpha, flow, and value_added use annualized monthly averages
    2. flow_vol uses the calendar-year standard deviation of monthly flow,
       annualized by sqrt(12), and requires at least 10 monthly observations
    3. all other characteristics use December values
    4. each feature is standardized within year
    5. missing standardized features are set to 0
    6. rows with missing target_alpha are dropped
    """
    df = panel.copy()
    df['year'] = df['caldt'].dt.year
    df['month'] = df['caldt'].dt.month

    annualized_mean_cols = ['flow', 'value_added']
    if TARGET_MODE == 'alpha':
        annualized_mean_cols.append('alpha')
    else:
        annualized_mean_cols.append('excess_return')

    december_cols = [
        'crsp_fundno', 'year',
        'mtna', 'exp_ratio', 'turn_ratio', 'age_months', 'manager_tenure',
        'alpha_tstat', 'r2',
        'mkt_rf_tstat', 'smb_tstat', 'hml_tstat',
        'rmw_tstat', 'cma_tstat', 'mom_tstat',
        'no_load',
        TARGET_COL,
    ]

    # Use December snapshots for characteristics that should stay at year-end values.
    december = (
        df[df['month'] == 12][december_cols]
        .copy()
    )

    # Annualize monthly alpha, flow, and value_added using mean * 12.
    annualized_means = (
        df.groupby(['crsp_fundno', 'year'])[annualized_mean_cols]
          .mean()
          .mul(12)
          .reset_index()
    )

    annual_flow_vol = (
        df.groupby(['crsp_fundno', 'year'])['flow']
          .agg(flow_obs='count', flow_vol='std')
          .reset_index()
    )
    annual_flow_vol.loc[annual_flow_vol['flow_obs'] < 10, 'flow_vol'] = np.nan
    annual_flow_vol['flow_vol'] = annual_flow_vol['flow_vol'] * np.sqrt(12)

    annual = december.merge(
        annualized_means,
        on=['crsp_fundno', 'year'],
        how='left',
    )
    annual = annual.merge(
        annual_flow_vol[['crsp_fundno', 'year', 'flow_vol']],
        on=['crsp_fundno', 'year'],
        how='left',
    )

    # Standardize each feature within year so every annual cross-section
    # has mean 0 and standard deviation 1.
    feature_means = annual.groupby('year')[FEATURE_COLS].transform('mean')
    feature_stds = annual.groupby('year')[FEATURE_COLS].transform(
        lambda x: x.std(ddof=0)
    ).replace(0, np.nan)

    annual[FEATURE_COLS] = (annual[FEATURE_COLS] - feature_means) / feature_stds

    # After standardization, fill missing feature values with the
    # cross-sectional mean, which is 0 by construction.
    annual[FEATURE_COLS] = annual[FEATURE_COLS].fillna(0.0)

    # Keep only rows with an observed target.
    annual = annual.dropna(subset=[TARGET_COL])

    corr_cols = [col for col in FEATURE_COLS + [TARGET_COL] if col in annual.columns]
    annual_corr = annual[corr_cols].corr()
    annual_corr.to_csv(PROCESSED_DIR / 'm_corr_annual.csv')

    return annual.sort_values(['year', 'crsp_fundno']).reset_index(drop=True)

def walk_forward_predict(
    annual,
    run_start_time=None,
    test_year=None,
    predictions_path=None,
    metrics_path=None,
    log_fn=print,
):
    all_years = sorted(annual['year'].unique())
    years = [test_year] if test_year is not None else all_years

    if not years:
        log_fn("No matching test years to run.")
        return pd.DataFrame(columns=PREDICTION_COLS)

    run_predictions = []

    for i, test_year in enumerate(years):
        train_years = [y for y in all_years if y < test_year]
        if len(train_years) < MIN_TRAIN_YEARS:
            log_fn(
                f"  Skipping {test_year}: only {len(train_years)} prior years "
                f"available"
            )
            continue

        train = annual[annual['year'].isin(train_years)]
        test  = annual[
            (annual['year'] == test_year) &
            (annual['no_load'] == True)  # Safety check; sample_filter.py already screens no-load share classes.
        ].copy()

        if test.empty:
            log_fn(f"  Skipping {test_year}: no eligible funds")
            continue

        X_train = train[FEATURE_COLS].values
        y_train = train[TARGET_COL].values
        X_test  = test[FEATURE_COLS].values
        train_year_labels = train['year'].to_numpy()

        valid_mask = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
        X_train = X_train[valid_mask]
        y_train = y_train[valid_mask]
        train_year_labels = train_year_labels[valid_mask]

        if len(X_train) < 100:
            log_fn(f"  Skipping {test_year}: fewer than 100 valid training rows")
            continue

        cv_splits = build_year_cv_splits(train_year_labels, CV_FOLDS)
        if not cv_splits:
            log_fn(f"  Skipping {test_year}: unable to build year-based CV splits")
            continue

        log_fn(
            f"  Predicting {test_year}: {len(train_years)} training years, "
            f"{len(X_train):,} training samples, {len(test):,} candidate funds"
        )

        models = get_models(cv_splits)
        year_predictions = []
        for model_idx, (model_name, model) in enumerate(models.items(), start=1):
            try:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                elapsed_minutes = (
                    (time.time() - run_start_time) / 60
                    if run_start_time is not None
                    else np.nan
                )
                log_fn(
                    f"    [{timestamp}] elapsed {elapsed_minutes:.1f}m | "
                    f"{test_year} {model_idx}/{len(models)} {model_name}"
                )
                model.fit(X_train, y_train)
                if hasattr(model, 'best_params_'):
                    log_fn(f"      Best params: {model.best_params_}")
                preds = model.predict(X_test)
                result = test[['crsp_fundno', 'year', TARGET_COL]].copy()
                result['model']      = model_name
                result['pred_alpha'] = preds
                year_predictions.append(result)
            except Exception as e:
                log_fn(f"    {model_name} failed: {e}")
                continue

        if not year_predictions:
            log_fn(f"  No predictions generated for {test_year}")
            continue

        year_predictions = pd.concat(year_predictions, ignore_index=True)
        run_predictions.append(year_predictions)

        if predictions_path is not None:
            if predictions_path.exists():
                stored_predictions = pd.read_csv(predictions_path)
                stored_predictions = stored_predictions[
                    stored_predictions['year'] != test_year
                ].copy()
            else:
                stored_predictions = pd.DataFrame(columns=PREDICTION_COLS)

            stored_predictions = pd.concat(
                [stored_predictions, year_predictions],
                ignore_index=True,
            )
            stored_predictions = stored_predictions.sort_values(
                ['year', 'model', 'crsp_fundno']
            ).reset_index(drop=True)
            stored_predictions.to_csv(predictions_path, index=False)
            log_fn(
                f"  Saved {test_year}: {len(year_predictions):,} rows -> "
                f"{predictions_path}"
            )

            if metrics_path is not None and not stored_predictions.empty:
                metrics = evaluate_predictions(stored_predictions)
                metrics.to_csv(metrics_path)

    if not run_predictions:
        return pd.DataFrame(columns=PREDICTION_COLS)

    return pd.concat(run_predictions, ignore_index=True)



def evaluate_predictions(predictions):
    """Build a yearly IC table plus summary rows for each model."""
    yearly_ic = (
        predictions.groupby(['year', 'model'])[['pred_alpha', TARGET_COL]]
        .apply(lambda x: x['pred_alpha'].corr(x[TARGET_COL]))
        .reset_index(name='ic')
        .pivot(index='year', columns='model', values='ic')
        .sort_index()
    )

    summary = pd.DataFrame(index=['IC_mean', 'ICIR', 'IC>0'], columns=yearly_ic.columns, dtype=float)
    for model_name in yearly_ic.columns:
        ic_series = yearly_ic[model_name]
        summary.loc['IC_mean', model_name] = ic_series.mean()
        summary.loc['ICIR', model_name] = ic_series.mean() / ic_series.std()
        summary.loc['IC>0', model_name] = (ic_series > 0).mean()

    metrics = pd.concat([yearly_ic, summary])
    metrics.index.name = 'year'
    return metrics.round(3)

def main():
    start_time = time.time()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if TARGET_MODE == 'alpha':
        predictions_path = PROCESSED_DIR / 'm_predictions_alpha.csv'
        metrics_path = PROCESSED_DIR / 'm_metrics_alpha.csv'
    else:
        predictions_path = PROCESSED_DIR / 'm_predictions_return.csv'
        metrics_path = PROCESSED_DIR / 'm_metrics_return.csv'
    log_path = PROCESSED_DIR / f"m_model_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log = make_logger(log_path)

    log(f"Progress log: {log_path}")
    log(f"Target mode: {TARGET_MODE} ({TARGET_COL})")
    initialize_output_files(predictions_path, metrics_path, log_fn=log)

    log("Loading feature panel...")
    panel = pd.read_csv(PROCESSED_DIR / 'f_panel.csv', parse_dates=['caldt'])
    log(f"Panel shape: {panel.shape}")

    log("Converting to annual panel...")
    annual = prepare_annual_panel(panel)
    log(f"Annual panel: {annual.shape}, years: {annual['year'].min()}-{annual['year'].max()}")
    if TEST_YEAR is not None:
        log(f"Requested test year: {TEST_YEAR}")

    log("")
    log("Starting walk-forward prediction...")
    predictions = walk_forward_predict(
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
        stored_predictions = pd.DataFrame(columns=PREDICTION_COLS)

    if stored_predictions.empty:
        log("")
        log("No predictions were generated. Skipping evaluation.")
        return

    log("")
    log(f"Predictions generated this run: {predictions.shape}")
    log(f"Combined predictions on disk: {stored_predictions.shape}")

    log("")
    log("Model evaluation (yearly IC / summary):")
    metrics = evaluate_predictions(stored_predictions)
    log(metrics.tail(3).to_string())
    metrics.to_csv(metrics_path)

    elapsed_seconds = time.time() - start_time
    elapsed_minutes = elapsed_seconds / 60
    log(
        f"\nTotal runtime: {elapsed_seconds:.1f} seconds "
        f"({elapsed_minutes:.2f} minutes)"
    )

if __name__ == '__main__':
    main()
