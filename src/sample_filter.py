# src/sample_filter.py
import re
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'

LOAD_FILTER_MODE = 'strict'  # 'loose' or 'strict'
DOMESTIC_EQUITY_MODE = 'strict'  # 'ed_prefix' or 'strict'
PER_COM_FILL_MODE = 'backward'  # 'backward' for main results; 'two_sided' for sensitivity.

STRICT_DOMESTIC_EQUITY_CODES = {
    'EDCI', 'EDCL', 'EDCM', 'EDCS',
    'EDSA', 'EDSC', 'EDSF', 'EDSG', 'EDSH', 'EDSI',
    'EDSM', 'EDSN', 'EDSR', 'EDSS', 'EDST', 'EDSU',
    'EDYB', 'EDYG', 'EDYI',
}

INDEX_NAME_PATTERN = re.compile(
    r'\b(?:index|idx|indx|s\s*&\s*p|s&p|russell|nasdaq|dow\s+jones|djia|wilshire|msci)\b',
    flags=re.IGNORECASE,
)


def merge_interval_columns_to_monthly(ret, interval_df, value_cols):
    """
    Attach interval-based attributes to monthly returns.
    Steps: align by fund and latest interval start date, then blank values
    whose interval end date is before the monthly return date.
    """
    ret = ret.copy()
    interval_df = interval_df.copy()

    ret['crsp_fundno'] = ret['crsp_fundno'].astype(float)
    interval_df['crsp_fundno'] = interval_df['crsp_fundno'].astype(float)

    keep_cols = ['crsp_fundno', 'begdt', 'enddt'] + value_cols
    interval_df = (
        interval_df[keep_cols]
        .dropna(subset=['begdt'])
        .sort_values('begdt')
    )
    ret_sorted = ret.sort_values('caldt')

    merged = pd.merge_asof(
        ret_sorted,
        interval_df,
        left_on='caldt',
        right_on='begdt',
        by='crsp_fundno',
        direction='backward',
    )

    expired = merged['enddt'].notna() & (merged['caldt'] > merged['enddt'])
    merged.loc[expired, value_cols] = np.nan
    return merged.drop(columns=['begdt', 'enddt'], errors='ignore')


def clean_per_com_dec(summary, fill_mode=PER_COM_FILL_MODE):
    """
    Build a cleaned December per_com series for each share class-year.

    Rules:
    1. If December per_com exists, keep it.
    2. If December is missing but the year has other per_com observations,
       use that year's mean per_com.
    3. If the whole year is missing, use `fill_mode`:
       - backward: use only the nearest previous year's annual mean.
       - two_sided: use the nearest year on either side; average if tied.
    """
    if fill_mode not in {'backward', 'two_sided'}:
        raise ValueError("PER_COM_FILL_MODE must be 'backward' or 'two_sided'")

    df = summary.copy()
    df['year'] = df['caldt'].dt.year
    df['month'] = df['caldt'].dt.month

    annual_mean = df.groupby(['crsp_fundno', 'year'])['per_com'].mean()
    december_mean = (
        df[df['month'] == 12]
        .groupby(['crsp_fundno', 'year'])['per_com']
        .mean()
    )
    fund_span = (
        df.groupby('crsp_fundno')['year']
          .agg(['min', 'max'])
          .rename(columns={'min': 'start_year', 'max': 'end_year'})
    )
    annual_mean_by_fund = {
        fundno: grp.droplevel(0).to_dict()
        for fundno, grp in annual_mean.groupby(level=0)
    }
    december_mean_by_fund = {
        fundno: grp.droplevel(0).to_dict()
        for fundno, grp in december_mean.groupby(level=0)
    }

    cleaned_rows = []
    for fundno, span in fund_span.iterrows():
        years = np.arange(int(span['start_year']), int(span['end_year']) + 1)
        annual_mean_map = annual_mean_by_fund.get(fundno, {})
        december_mean_map = december_mean_by_fund.get(fundno, {})

        available_items = sorted(
            (int(year), float(value))
            for year, value in annual_mean_map.items()
            if pd.notna(value)
        )
        available_years = np.array([year for year, _ in available_items], dtype=int)
        available_values = np.array([value for _, value in available_items], dtype=float)

        cleaned_values = []
        fill_sources = []
        for year in years:
            december_value = december_mean_map.get(year, np.nan)
            if pd.notna(december_value):
                cleaned_values.append(december_value)
                fill_sources.append('december_actual')
                continue

            annual_value = annual_mean_map.get(year, np.nan)
            if pd.notna(annual_value):
                cleaned_values.append(annual_value)
                fill_sources.append('same_year_mean')
                continue

            if len(available_years) == 0:
                cleaned_values.append(np.nan)
                fill_sources.append('no_available_per_com')
                continue

            insert_idx = np.searchsorted(available_years, year)
            if fill_mode == 'backward':
                if insert_idx == 0:
                    cleaned_values.append(np.nan)
                    fill_sources.append('no_past_per_com')
                else:
                    prev_year = available_years[insert_idx - 1]
                    cleaned_values.append(annual_mean_map[prev_year])
                    fill_sources.append('nearest_prev_year_mean')
                continue

            if insert_idx == 0:
                cleaned_values.append(available_values[0])
                fill_sources.append('nearest_next_year_mean')
            elif insert_idx == len(available_years):
                cleaned_values.append(available_values[-1])
                fill_sources.append('nearest_prev_year_mean')
            else:
                prev_year = available_years[insert_idx - 1]
                next_year = available_years[insert_idx]
                prev_val = annual_mean_map[prev_year]
                next_val = annual_mean_map[next_year]
                prev_dist = year - prev_year
                next_dist = next_year - year

                if prev_dist < next_dist:
                    cleaned_values.append(prev_val)
                    fill_sources.append('nearest_prev_year_mean')
                elif next_dist < prev_dist:
                    cleaned_values.append(next_val)
                    fill_sources.append('nearest_next_year_mean')
                else:
                    cleaned_values.append((prev_val + next_val) / 2)
                    fill_sources.append('nearest_years_average')

        fund_years = pd.DataFrame({
            'crsp_fundno': fundno,
            'year': years,
            'per_com': cleaned_values,
            'fill_source': fill_sources,
        })
        fund_years['caldt'] = pd.to_datetime(fund_years['year'].astype(str) + '-12-31')
        cleaned_rows.append(fund_years[['crsp_fundno', 'caldt', 'per_com', 'fill_source']])

    return pd.concat(cleaned_rows, ignore_index=True)


def add_load_flags(ret, front_load, rear_load):
    """
    Add front-load, rear-load, and no-load flags to monthly returns.
    Steps: collapse load schedules to fund-date intervals, merge them to
    monthly observations, fill within fund, then flag positive loads.
    """
    front_interval = (
        front_load.groupby(['crsp_fundno', 'begdt', 'enddt'], dropna=False)['front_load']
        .max()
        .reset_index()
    )
    rear_interval = (
        rear_load.groupby(['crsp_fundno', 'begdt', 'enddt'], dropna=False)['rear_load']
        .max()
        .reset_index()
    )

    ret = merge_interval_columns_to_monthly(ret, front_interval, ['front_load'])
    ret = merge_interval_columns_to_monthly(ret, rear_interval, ['rear_load'])
    ret = ret.sort_values(['crsp_fundno', 'caldt']).copy()

    for col in ['front_load', 'rear_load']:
        ret[col] = (
            ret.groupby('crsp_fundno')[col]
               .transform(lambda x: x.ffill())
        )

    if LOAD_FILTER_MODE == 'loose':
        ret['no_load'] = pd.Series(True, index=ret.index, dtype='boolean')
        has_positive_load = (
            ret['front_load'].fillna(0).gt(0) |
            ret['rear_load'].fillna(0).gt(0)
        )
        ret.loc[has_positive_load, 'no_load'] = False
    elif LOAD_FILTER_MODE == 'strict':
        ret['no_load'] = (
            ret['front_load'].eq(0) &
            ret['rear_load'].eq(0)
        ).astype('boolean')
    else:
        raise ValueError("LOAD_FILTER_MODE must be 'loose' or 'strict'")

    return ret


def add_header_flags(ret, hdr, hdr_hist):
    """
    Add active/passive and ETF indicators from CRSP header data.
    Steps: merge historical header intervals, fall back to static headers,
    then identify index/passive funds from CRSP flags or fund-name keywords.
    """
    hist = hdr_hist.rename(columns={'chgdt': 'begdt', 'chgenddt': 'enddt'})
    hist_cols = ['fund_name', 'first_offer_dt', 'index_fund_flag', 'et_flag']
    ret = merge_interval_columns_to_monthly(ret, hist, hist_cols)

    hdr_fallback = hdr[
        ['crsp_fundno', 'fund_name', 'first_offer_dt', 'index_fund_flag', 'et_flag']
    ].rename(
        columns={
            'fund_name': 'fund_name_hdr',
            'first_offer_dt': 'first_offer_dt_hdr',
            'index_fund_flag': 'index_fund_flag_hdr',
            'et_flag': 'et_flag_hdr',
        }
    )
    ret = ret.merge(hdr_fallback, on='crsp_fundno', how='left')
    for col in ['fund_name', 'first_offer_dt', 'index_fund_flag', 'et_flag']:
        ret[col] = ret[col].combine_first(ret[f'{col}_hdr'])
        ret = ret.drop(columns=[f'{col}_hdr'])

    fund_name = ret['fund_name'].fillna('')
    index_flag_present = ret['index_fund_flag'].notna()
    ret['is_index_from_flag'] = index_flag_present
    ret['is_index_from_name'] = (~index_flag_present) & fund_name.str.contains(INDEX_NAME_PATTERN)
    ret['is_passive'] = ret['is_index_from_flag'] | ret['is_index_from_name']
    ret['is_etf'] = ret['et_flag'].notna()
    ret['is_active'] = ~(ret['is_passive'] | ret['is_etf'])
    return ret


def add_domestic_equity_flags(ret, style):
    """
    Add the CRSP domestic-equity objective-code screen.
    Steps: merge fund_style intervals to monthly observations, then keep
    codes whose CRSP objective code starts with ED.
    """
    ret = merge_interval_columns_to_monthly(ret, style, ['crsp_obj_cd'])
    crsp_obj_cd = ret['crsp_obj_cd'].astype('string')

    if DOMESTIC_EQUITY_MODE == 'ed_prefix':
        ret['is_us_domestic_equity'] = crsp_obj_cd.str.startswith('ED').fillna(False)
    elif DOMESTIC_EQUITY_MODE == 'strict':
        ret['is_us_domestic_equity'] = crsp_obj_cd.isin(STRICT_DOMESTIC_EQUITY_CODES).fillna(False)
    else:
        raise ValueError("DOMESTIC_EQUITY_MODE must be 'ed_prefix' or 'strict'")

    return ret


def add_age_and_tna_flags(ret):
    """
    Add fund-age and first-$5M-TNA eligibility flags.
    Steps: compute age from first offer date, find each fund's first month
    with TNA at least $5M, then mark observations after that month.
    """
    ret = ret.copy()
    ret['age_months'] = (
        (ret['caldt'].dt.year - ret['first_offer_dt'].dt.year) * 12 +
        (ret['caldt'].dt.month - ret['first_offer_dt'].dt.month)
    )
    ret.loc[ret['age_months'] < 0, 'age_months'] = np.nan
    ret['age_ge_36m'] = ret['age_months'] >= 36

    first_5m_tna = (
        ret.loc[ret['mtna'] >= 5]
        .groupby('crsp_fundno')['caldt']
        .min()
        .rename('first_5m_tna_dt')
        .reset_index()
    )
    ret = ret.merge(first_5m_tna, on='crsp_fundno', how='left')
    ret['after_first_5m_tna'] = (
        ret['first_5m_tna_dt'].notna() &
        (ret['caldt'] >= ret['first_5m_tna_dt'])
    )
    return ret


def add_per_com_flags(ret, per_com_dec):
    """
    Add the annual equity-allocation screen to monthly returns.
    Steps: match each monthly observation to its cleaned fund-year
    December per_com value, then flag observations with per_com >= 70.
    """
    ret = ret.copy()
    ret['year'] = ret['caldt'].dt.year
    per_com_dec = per_com_dec.copy()
    per_com_dec['year'] = per_com_dec['caldt'].dt.year

    ret = ret.merge(
        per_com_dec[['crsp_fundno', 'year', 'per_com', 'fill_source']],
        on=['crsp_fundno', 'year'],
        how='left',
    )
    ret = ret.rename(columns={'fill_source': 'per_com_fill_source'})
    ret['per_com_ge_70'] = ret['per_com'] >= 70
    return ret


def build_sample():
    """
    Construct the paper-style monthly analysis sample.
    Steps: load raw files, add load/header/style/age/TNA/per_com filters,
    then build a broader alpha-history sample plus the final
    in_paper_sample indicator.
    """
    ret = pd.read_csv(RAW_DIR / 'monthly_returns.csv', parse_dates=['caldt'])
    front_load = pd.read_csv(RAW_DIR / 'front_load.csv', parse_dates=['begdt', 'enddt'])
    rear_load = pd.read_csv(RAW_DIR / 'rear_load.csv', parse_dates=['begdt', 'enddt'])
    hdr = pd.read_csv(RAW_DIR / 'fund_hdr.csv', parse_dates=['first_offer_dt'])
    hdr_hist = pd.read_csv(
        RAW_DIR / 'fund_hdr_hist.csv',
        parse_dates=['chgdt', 'chgenddt', 'first_offer_dt'],
    )
    style = pd.read_csv(RAW_DIR / 'fund_style.csv', parse_dates=['begdt', 'enddt'])
    summary = pd.read_csv(RAW_DIR / 'fund_summary.csv', parse_dates=['caldt'])

    print(f"Load filter mode: {LOAD_FILTER_MODE}")
    print(f"Domestic equity mode: {DOMESTIC_EQUITY_MODE}")
    print(f"per_com fill mode: {PER_COM_FILL_MODE}")

    print("Cleaning December per_com...")
    per_com_dec = clean_per_com_dec(summary, fill_mode=PER_COM_FILL_MODE)

    print("Adding load flags...")
    ret = add_load_flags(ret, front_load, rear_load)

    print("Adding active/passive and ETF flags...")
    ret = add_header_flags(ret, hdr, hdr_hist)

    print("Adding U.S. domestic-equity flags...")
    ret = add_domestic_equity_flags(ret, style)

    print("Adding age and TNA flags...")
    ret = add_age_and_tna_flags(ret)

    print("Adding equity allocation flags...")
    ret = add_per_com_flags(ret, per_com_dec)

    ret['in_paper_sample'] = (
        ret['no_load'].eq(True) &
        ret['is_active'].eq(True) &
        ret['is_us_domestic_equity'].eq(True) &
        ret['per_com_ge_70'].eq(True) &
        ret['age_ge_36m'].eq(True) &
        ret['after_first_5m_tna'].eq(True)
    )
    ret['in_alpha_history'] = (
        ret['no_load'].eq(True) &
        ret['is_active'].eq(True) &
        ret['is_us_domestic_equity'].eq(True) &
        ret['per_com_ge_70'].eq(True) &
        ret['after_first_5m_tna'].eq(True)
    )

    return ret


def summarize_filters(ret):
    """
    Summarize how many rows and share classes pass each sample filter.
    Steps: evaluate each filter mask separately and count observations and
    unique crsp_fundno values that satisfy the mask.
    """
    filters = [
        ('raw_monthly_rows', pd.Series(True, index=ret.index)),
        ('no_load', ret['no_load'].eq(True)),
        ('active_managed', ret['is_active'].eq(True)),
        ('us_domestic_equity', ret['is_us_domestic_equity'].eq(True)),
        ('per_com_ge_70', ret['per_com_ge_70'].eq(True)),
        ('age_ge_36m', ret['age_ge_36m'].eq(True)),
        ('after_first_5m_tna', ret['after_first_5m_tna'].eq(True)),
        ('alpha_history_sample', ret['in_alpha_history'].eq(True)),
        ('in_paper_sample', ret['in_paper_sample'].eq(True)),
    ]

    rows = []
    for name, mask in filters:
        rows.append({
            'filter': name,
            'rows': int(mask.sum()),
            'unique_share_classes': int(ret.loc[mask, 'crsp_fundno'].nunique()),
        })
    return pd.DataFrame(rows)


def main():
    """
    Run the sample-filter pipeline and write processed CSV outputs.
    Steps: build the filtered sample, save diagnostic flags and final
    monthly returns, then print the filter-count summary.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ret = build_sample()

    sample_flags_cols = [
        'crsp_fundno', 'caldt', 'front_load', 'rear_load', 'no_load',
        'index_fund_flag', 'et_flag', 'fund_name',
        'is_index_from_flag', 'is_index_from_name', 'is_passive', 'is_etf',
        'is_active', 'crsp_obj_cd', 'is_us_domestic_equity',
        'per_com', 'per_com_fill_source', 'per_com_ge_70',
        'age_months', 'age_ge_36m',
        'first_5m_tna_dt', 'after_first_5m_tna',
        'in_alpha_history', 'in_paper_sample',
    ]
    monthly_cols = [
        'crsp_fundno', 'caldt', 'mret', 'mtna',
        'front_load', 'rear_load', 'no_load',
        'per_com', 'per_com_fill_source', 'per_com_ge_70',
        'age_months', 'age_ge_36m',
        'first_offer_dt', 'first_5m_tna_dt', 'after_first_5m_tna',
        'fund_name', 'index_fund_flag', 'et_flag',
        'crsp_obj_cd', 'is_active', 'is_us_domestic_equity',
        'in_alpha_history', 'in_paper_sample',
    ]

    sample_flags = ret[sample_flags_cols].copy()
    sample = ret.loc[ret['in_alpha_history'], monthly_cols].copy()
    filter_counts = summarize_filters(ret)

    flags_path = PROCESSED_DIR / 'sf_sample_flags.csv'
    sample_path = PROCESSED_DIR / 'sf_monthly_returns.csv'

    sample_flags.to_csv(flags_path, index=False)
    sample.to_csv(sample_path, index=False)

    print(f"Saved {flags_path}")
    print(f"Saved {sample_path}")
    print("\nFilter counts:")
    print(filter_counts.to_string(index=False))


if __name__ == '__main__':
    main()
