# src/backtest.py
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

try:
    from .config import TARGET_MODE
except ImportError:
    from config import TARGET_MODE


LONG_DECILE   = 0.10   # Long leg percentile by predicted alpha
SHORT_DECILE  = 0    # Short leg percentile; set to 0 for long-only
FACTORS       = ['mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'mom']
# TARGET_MODE comes from config.py (single source of truth).
COMPOUND_ALPHA = True  # True: (1 + alpha).cumprod() - 1; False: cumulative sum
BACKTEST_START =  None # Set to None to use the full available period. '2006-01-01'
BACKTEST_END   =  None # Set to None to use the full available period. '2025-12-31'
BASE_DIR      = Path(__file__).resolve().parent.parent
DATA_DIR      = BASE_DIR / 'data'
RAW_DIR       = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
RESULTS_DIR   = BASE_DIR / 'results'
PLOTS_DIR     = RESULTS_DIR / 'plots'

def load_data():
    if TARGET_MODE == 'alpha':
        predictions_path = PROCESSED_DIR / 'm_predictions_alpha.csv'
    else:
        predictions_path = PROCESSED_DIR / 'm_predictions_return.csv'

    predictions = pd.read_csv(predictions_path)
    monthly_ret = pd.read_csv(PROCESSED_DIR / 'sf_monthly_returns.csv', parse_dates=['caldt'])
    factors     = pd.read_csv(RAW_DIR / 'ff5_mom_factors.csv', parse_dates=['date'])
    return predictions, monthly_ret, factors


def compute_excess_return(df):
    excess_ret = df['mret'] - df['rf']
    if SHORT_DECILE > 0:
        strategy_rows = df['model'] != 'equal_weight_all'
        excess_ret.loc[strategy_rows] = df.loc[strategy_rows, 'mret']
    return excess_ret


def strategy_label():
    period = ''
    if BACKTEST_START is not None or BACKTEST_END is not None:
        start = BACKTEST_START if BACKTEST_START is not None else 'start'
        end = BACKTEST_END if BACKTEST_END is not None else 'end'
        period = f", {start} to {end}"
    if SHORT_DECILE > 0:
        return f"Long {LONG_DECILE:.2f} - Short {SHORT_DECILE:.2f}{period}"
    return f"Long {LONG_DECILE:.2f}{period}"


def strategy_file_tag():
    return f"l{int(LONG_DECILE * 100)}_s{int(SHORT_DECILE * 100)}"


def period_file_tag():
    if BACKTEST_START is None and BACKTEST_END is None:
        return 'full'
    start = pd.Timestamp(BACKTEST_START).strftime('%Y%m') if BACKTEST_START else 'start'
    end = pd.Timestamp(BACKTEST_END).strftime('%Y%m') if BACKTEST_END else 'end'
    return f'{start}_{end}'


def filter_backtest_period(portfolio_returns):
    df = portfolio_returns.copy()
    if BACKTEST_START is not None:
        df = df[df['caldt'] >= pd.Timestamp(BACKTEST_START)]
    if BACKTEST_END is not None:
        df = df[df['caldt'] <= pd.Timestamp(BACKTEST_END)]
    return df


def paper_style_holding_returns(holding_ret, selected, year):
    selected = pd.Index(selected.dropna().drop_duplicates(), name='crsp_fundno')
    if len(selected) == 0:
        return pd.DataFrame()

    hold_start = pd.Timestamp(f'{year + 1}-01-01')
    hold_end   = pd.Timestamp(f'{year + 1}-12-31')

    hold_data = (
        holding_ret[
            holding_ret['crsp_fundno'].isin(selected) &
            holding_ret['caldt'].between(hold_start, hold_end)
        ][['crsp_fundno', 'caldt', 'mret']]
        .sort_values(['caldt', 'crsp_fundno'])
    )
    if hold_data.empty:
        return pd.DataFrame()

    weights = pd.Series(1 / len(selected), index=selected)
    year_returns = []

    for caldt, month_data in hold_data.groupby('caldt', sort=True):
        month_ret = month_data.set_index('crsp_fundno')['mret']
        present = weights.index.intersection(month_ret.index)
        if len(present) == 0:
            break

        # Treat missing funds as disappeared for the rest of the holding year.
        missing_weight = weights.drop(index=present).sum()
        weights = weights.loc[present].copy()
        if missing_weight != 0:
            weights += missing_weight / len(weights)
        weights /= weights.sum()

        returns = month_ret.loc[weights.index]
        portfolio_return = float((weights * returns).sum())
        year_returns.append({
            'caldt': caldt,
            'mret': portfolio_return,
        })

        updated_weights = weights * (1 + returns)
        total_value = updated_weights.sum()
        if total_value <= 0:
            break
        weights = updated_weights / total_value

    return pd.DataFrame(year_returns)


def construct_portfolios(predictions, monthly_ret):
    """
    每年按预测值选long/short组合，下一年年初等权持有。
    年内不主动再平衡；如果基金当月没有可用收益，则把它的权重平分给剩余基金。
    """
    all_returns = []
    holding_ret = monthly_ret
    if 'in_paper_sample' in holding_ret.columns:
        holding_ret = holding_ret[holding_ret['in_paper_sample'].eq(True)].copy()
    holding_ret = holding_ret.dropna(subset=['mret']).copy()

    for model_name, pred_grp in predictions.groupby('model'):
        for year, year_grp in pred_grp.groupby('year'):
            long_threshold = year_grp['pred_alpha'].quantile(1 - LONG_DECILE)
            long_selected = (
                year_grp[year_grp['pred_alpha'] >= long_threshold]['crsp_fundno']
                .dropna()
                .drop_duplicates()
            )
            long_returns = paper_style_holding_returns(holding_ret, long_selected, year)
            if long_returns.empty:
                continue

            if SHORT_DECILE > 0:
                short_threshold = year_grp['pred_alpha'].quantile(SHORT_DECILE)
                short_selected = (
                    year_grp[year_grp['pred_alpha'] <= short_threshold]['crsp_fundno']
                    .dropna()
                    .drop_duplicates()
                )
                short_returns = paper_style_holding_returns(holding_ret, short_selected, year)
                if short_returns.empty:
                    continue

                year_returns = long_returns.merge(
                    short_returns,
                    on='caldt',
                    suffixes=('_long', '_short'),
                    how='inner',
                )
                year_returns['mret'] = (
                    year_returns['mret_long'] -
                    year_returns['mret_short']
                )
                year_returns = year_returns[['caldt', 'mret']]
            else:
                year_returns = long_returns

            year_returns['model'] = model_name
            year_returns['year'] = year
            all_returns.append(year_returns)

    if not all_returns:
        return pd.DataFrame()

    return pd.concat(all_returns, ignore_index=True)


def construct_portfolio_alpha(portfolio_returns, factors):
    """
    Compute monthly net alpha using whole out-of-sample FF5+MOM betas:
    monthly alpha = portfolio excess return - beta_hat' * factor realization.
    """
    df = portfolio_returns.copy()
    df['month'] = df['caldt'].dt.to_period('M')
    factors_monthly = factors.copy()
    factors_monthly['month'] = factors_monthly['date'].dt.to_period('M')
    df = df.merge(
        factors_monthly[['month', 'rf'] + FACTORS],
        on='month',
        how='left',
    )
    df['excess_ret'] = compute_excess_return(df)

    all_alpha = []
    for model_name, grp in df.groupby('model'):
        grp = grp.dropna(subset=['excess_ret'] + FACTORS).sort_values('caldt').copy()
        if len(grp) < 24:
            continue

        y = grp['excess_ret'].values
        X = np.column_stack([np.ones(len(grp)), grp[FACTORS].values])
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        betas = coef[1:]

        grp['portfolio_alpha'] = grp['excess_ret'] - grp[FACTORS].values @ betas
        all_alpha.append(grp[['caldt', 'model', 'portfolio_alpha']])

    if not all_alpha:
        return pd.DataFrame()

    return pd.concat(all_alpha, ignore_index=True)

def compute_metrics(portfolio_returns, factors):
    """
    计算各策略的业绩指标和 FF5+MOM out-of-sample alpha
    """
    df = portfolio_returns.copy()
    df['month'] = df['caldt'].dt.to_period('M')
    factors_monthly = factors[['date', 'rf'] + FACTORS].copy()
    factors_monthly['month'] = factors_monthly['date'].dt.to_period('M')
    df = df.merge(factors_monthly[['month', 'rf'] + FACTORS], on='month', how='left')
    df['excess_ret'] = compute_excess_return(df)

    results = []
    for model_name, grp in df.groupby('model'):
        grp = grp.dropna(subset=['mret', 'excess_ret'] + FACTORS).sort_values('caldt').copy()
        r   = grp['mret'].values
        er  = grp['excess_ret'].values

        # 累计收益
        cum_ret = (1 + r).prod() - 1
        n_months = len(r)
        n_years  = n_months / 12

        ann_ret    = (1 + cum_ret) ** (1 / n_years) - 1
        ann_vol    = er.std() * np.sqrt(12)
        sharpe     = er.mean() / er.std() * np.sqrt(12) if er.std() > 0 else np.nan

        # 最大回撤
        cum_curve  = (1 + r).cumprod()
        rolling_max = np.maximum.accumulate(cum_curve)
        drawdowns   = cum_curve / rolling_max - 1
        max_dd      = drawdowns.min()

        # FF5+MOM out-of-sample alpha
        X = np.column_stack([np.ones(len(grp)), grp[FACTORS].values])
        coef, _, _, _ = np.linalg.lstsq(X, er, rcond=None)
        resid = er - X @ coef
        dof = len(grp) - X.shape[1]
        if dof > 0:
            sigma2 = (resid @ resid) / dof
            xtx_inv = np.linalg.pinv(X.T @ X)
            se = np.sqrt(np.diag(sigma2 * xtx_inv))
            alpha_tstat = coef[0] / se[0] if se[0] > 0 else np.nan
        else:
            alpha_tstat = np.nan

        results.append({
            'model':        model_name,
            'cum_return':   cum_ret,
            'cum_alpha':    (1 + grp['portfolio_alpha'].fillna(0)).prod() - 1
                            if 'portfolio_alpha' in grp.columns
                            else np.nan,
            'ann_return':   ann_ret,
            'ann_vol':      ann_vol,
            'sharpe':       sharpe,
            'max_drawdown': max_dd,
            'oos_alpha_monthly': coef[0],
            'oos_alpha_annual':  coef[0] * 12,
            'oos_alpha_tstat':   alpha_tstat,
            'n_months':     n_months,
        })

    return pd.DataFrame(results).set_index('model')

def compute_benchmarks(monthly_ret, factors, start_date=None, end_date=None, strategy_dates=None):
    """
    Benchmark：等权 filtered sample funds from sample_filter.py.
    """
    benchmark_ret = monthly_ret
    if 'in_paper_sample' in benchmark_ret.columns:
        benchmark_ret = benchmark_ret[benchmark_ret['in_paper_sample'].eq(True)]

    if strategy_dates is not None:
        benchmark_ret = benchmark_ret[
            benchmark_ret['caldt'].isin(pd.to_datetime(strategy_dates))
        ]
    elif start_date is not None and end_date is not None:
        benchmark_ret = benchmark_ret[
            benchmark_ret['caldt'].between(start_date, end_date)
        ]

    benchmark_ret = benchmark_ret.copy()
    df = (
        benchmark_ret.groupby('caldt')['mret']
                   .mean()
                   .reset_index()
                   .rename(columns={'mret': 'mret'})
    )
    df['year'] = df['caldt'].dt.year - 1
    df['model'] = 'equal_weight_all'
    return df

def plot_results(portfolio_results, plot_path):
    """画累计收益、回撤和累计alpha图"""
    df = portfolio_results.copy()
    df['caldt'] = pd.to_datetime(df['caldt'])

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)

    colors = {
        'elastic_net':      '#2196F3',
        'ols':              '#7E57C2',
        'random_forest':    '#4CAF50',
        'xgboost':          '#FF5722',
        'equal_weight_all': '#9E9E9E',
    }

    for model_name, grp in df.groupby('model'):
        grp  = grp.sort_values('caldt')
        r    = grp['mret'].values
        dates = grp['caldt'].values
        alpha = grp['portfolio_alpha'].values if 'portfolio_alpha' in grp.columns else None

        # 累计收益
        cum = (1 + r).cumprod()

        # 回撤
        rolling_max = np.maximum.accumulate(cum)
        drawdown    = cum / rolling_max - 1

        lw    = 1.5 if model_name != 'equal_weight_all' else 1.0
        ls    = '-'  if model_name != 'equal_weight_all' else '--'
        color = colors.get(model_name, 'black')
        label = model_name.replace('_', ' ').title()

        axes[0].plot(dates, cum,      color=color, lw=lw, ls=ls, label=label)
        axes[1].plot(dates, drawdown, color=color, lw=lw, ls=ls, label=label)
        if alpha is not None:
            if COMPOUND_ALPHA:
                cum_alpha = (1 + pd.Series(alpha)).cumprod().to_numpy() - 1
            else:
                cum_alpha = np.nancumsum(alpha)
            axes[2].plot(dates, cum_alpha, color=color, lw=lw, ls=ls, label=label)

    # 累计收益图
    axes[0].set_ylabel('Cumulative Return')
    axes[0].set_title(f'Cumulative Returns: {strategy_label()} ({TARGET_MODE})')
    axes[0].legend(loc='upper left', fontsize=9)
    axes[0].axhline(1, color='black', lw=0.5, ls=':')
    axes[0].grid(alpha=0.3)

    # 回撤图
    axes[1].set_ylabel('Drawdown')
    axes[1].set_title(f'Drawdown: {strategy_label()}')
    axes[1].fill_between(
        df[df['model'] == 'random_forest']['caldt'],
        df[df['model'] == 'random_forest']
            .sort_values('caldt')
            .assign(cum=lambda x: (1 + x['mret']).cumprod())
            .assign(dd=lambda x: x['cum'] / x['cum'].cummax() - 1)['dd'],
        0,
        alpha=0.1,
        color=colors['random_forest'],
    )
    axes[1].axhline(0, color='black', lw=0.5, ls=':')
    axes[1].grid(alpha=0.3)

    # 累计alpha图
    axes[2].set_ylabel('Cumulative Alpha')
    alpha_title = 'Compounded Cumulative Portfolio Alpha' if COMPOUND_ALPHA else 'Cumulative Portfolio Alpha'
    axes[2].set_title(f'{alpha_title}: {strategy_label()}')
    axes[2].axhline(0, color='black', lw=0.5, ls=':')
    axes[2].grid(alpha=0.3)
    axes[2].legend(loc='upper left', fontsize=9)
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[2].xaxis.set_major_locator(mdates.YearLocator(5))

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 图表保存到 {plot_path}")

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    tag = f'{strategy_file_tag()}_{period_file_tag()}'
    portfolio_returns_path = RESULTS_DIR / f'portfolio_returns_{tag}.csv'
    metrics_path = RESULTS_DIR / f'metrics_{tag}.csv'
    plot_path = PLOTS_DIR / f'performance_{tag}.png'

    print(f"Target mode: {TARGET_MODE}")
    print("读取数据...")
    predictions, monthly_ret, factors = load_data()

    print("构建组合...")
    portfolio_returns = construct_portfolios(predictions, monthly_ret)
    portfolio_returns = filter_backtest_period(portfolio_returns)
    if portfolio_returns.empty:
        raise ValueError("No portfolio returns remain after applying BACKTEST_START/BACKTEST_END.")

    if SHORT_DECILE > 0:
        all_returns = portfolio_returns
    else:
        strategy_start = portfolio_returns['caldt'].min()
        strategy_end = portfolio_returns['caldt'].max()
        benchmark = compute_benchmarks(
            monthly_ret,
            factors,
            start_date=strategy_start,
            end_date=strategy_end,
            strategy_dates=portfolio_returns['caldt'].drop_duplicates(),
        )
        all_returns = pd.concat([portfolio_returns, benchmark], ignore_index=True)

    print("计算组合alpha...")
    portfolio_alpha = construct_portfolio_alpha(all_returns, factors)
    portfolio_results = all_returns.merge(
        portfolio_alpha,
        on=['caldt', 'model'],
        how='left',
    )
    portfolio_results['target_mode'] = TARGET_MODE
    portfolio_results['long_decile'] = LONG_DECILE
    portfolio_results['short_decile'] = SHORT_DECILE
    portfolio_results['backtest_start'] = BACKTEST_START
    portfolio_results['backtest_end'] = BACKTEST_END
    portfolio_results.to_csv(portfolio_returns_path, index=False)

    print("\n计算业绩指标...")
    metrics = compute_metrics(portfolio_results, factors)
    metrics.insert(0, 'target_mode', TARGET_MODE)
    metrics.insert(1, 'long_decile', LONG_DECILE)
    metrics.insert(2, 'short_decile', SHORT_DECILE)
    metrics.insert(3, 'backtest_start', BACKTEST_START)
    metrics.insert(4, 'backtest_end', BACKTEST_END)
    print(metrics.round(4))
    metrics.to_csv(metrics_path)

    print("画图...")
    plot_results(portfolio_results, plot_path)

    print("\n✅ 回测完成")

if __name__ == '__main__':
    main()
