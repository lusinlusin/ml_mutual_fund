# src/data_loader.py
import wrds
import pandas as pd
import pandas_datareader.data as web
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'

START = '1977-01-01'
END   = '2025-12-31'
OUT   = DATA_DIR / 'raw'


def clean_ret(ret):
    ret = ret.sort_values(['crsp_fundno', 'caldt']).copy()
    is_last_month = ret.groupby('crsp_fundno')['caldt'].transform('max').eq(ret['caldt'])

    # CRSP delisting/liquidation-like bad returns
    ret.loc[ret['mret'] <= -1, 'mret'] = 0

    # Extreme positive returns are suspicious, especially at fund's last month.
    ret.loc[is_last_month & ret['mret'].gt(10), 'mret'] = pd.NA

    # General extreme returns, independent of mtna availability.
    ret.loc[ret['mret'].abs().gt(5), 'mret'] = pd.NA

    return ret

def download_ff5_factors(start=START):
    print("下载FF5+MOM因子...")
    ff5 = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start=start)[0]
    mom = web.DataReader('F-F_Momentum_Factor',              'famafrench', start=start)[0]
    ff5.columns = [c.strip() for c in ff5.columns]
    mom.columns = ['mom']
    factors = ff5.join(mom) / 100
    factors.index = factors.index.to_timestamp('M')
    factors = factors.reset_index().rename(columns={'Date': 'date'})
    factors.columns = [c.lower().replace('-', '_') for c in factors.columns]
    factors.to_csv(OUT / 'ff5_mom_factors.csv', index=False)
    print(f"✅ FF5+MOM因子: {factors.shape}")

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    db = wrds.Connection(wrds_username='yuanj_cuhk')

    # ── 1. 月度收益 + TNA（合并表）──────────────────────────
    ret = db.raw_sql(f"""
        SELECT crsp_fundno, caldt, mret, mtna
        FROM crsp_q_mutualfunds.monthly_tna_ret_nav
        WHERE caldt BETWEEN '{START}' AND '{END}'
        AND mret IS NOT NULL
    """, date_cols=['caldt'])
    ret = clean_ret(ret)
    ret.to_csv(OUT / 'monthly_returns.csv', index=False)
    print(f"returns: {ret.shape}")

    # ── 2. 基金分类 ──────────────────────────────────────────
    style = db.raw_sql("""
        SELECT crsp_fundno, begdt, enddt,
               crsp_obj_cd, si_obj_cd, lipper_class, lipper_class_name,  wbrger_obj_cd, policy
        FROM crsp_q_mutualfunds.fund_style
    """, date_cols=['begdt', 'enddt'])
    style.to_csv(OUT / 'fund_style.csv', index=False)
    print(f"style: {style.shape}")

    # ── 3. 费率、换手率 ──────────────────────────────────────
    fees = db.raw_sql("""
        SELECT crsp_fundno, begdt, enddt, exp_ratio, turn_ratio
        FROM crsp_q_mutualfunds.fund_fees
    """, date_cols=['begdt', 'enddt'])
    fees.to_csv(OUT / 'fund_fees.csv', index=False)
    print(f"fees: {fees.shape}")

    # ── 4. 基金基本信息 ──────────────────────────────────────
    hdr = db.raw_sql("""
        SELECT crsp_fundno,crsp_portno, fund_name, ticker,first_offer_dt,
               mgmt_name, mgmt_cd, mgr_name, mgr_dt, index_fund_flag, et_flag,end_dt, dead_flag
        FROM crsp_q_mutualfunds.fund_hdr
    """, date_cols=['end_dt'])
    hdr.to_csv(OUT / 'fund_hdr.csv', index=False)
    print(f"hdr: {hdr.shape}")

    hdr_hist = db.raw_sql("""
                     SELECT crsp_fundno,crsp_portno, chgdt, chgenddt, fund_name, ticker,first_offer_dt,
               mgmt_name, mgmt_cd, mgr_name, mgr_dt, index_fund_flag, et_flag
                     FROM crsp_q_mutualfunds.fund_hdr_hist
                     """, date_cols=['chgdt', 'chgenddt'])
    hdr_hist.to_csv(OUT / 'fund_hdr_hist.csv', index=False)
    print(f"hdr_hist: {hdr_hist.shape}")


    # ── 5. 资金流（直接有表，不需要自己算）─────────────────
    flows = db.raw_sql(f"""
        SELECT *
        FROM crsp_q_mutualfunds.fund_flows
        WHERE report_dt BETWEEN '{START}' AND '{END}'
    """, date_cols=['report_dt'])
    flows.to_csv(OUT / 'fund_flows.csv', index=False)
    print(f"flows: {flows.shape}")

    # ── 6. Share class → fund level映射 ─────────────────────
    mflinks = db.raw_sql("""
        SELECT crsp_fundno, wficn
        FROM mfl.mflink1
        WHERE wficn IS NOT NULL
    """)
    mflinks.to_csv(OUT / 'mflinks.csv', index=False)
    print(f"mflinks: {mflinks.shape}")

    # ── 7. 股票仓位（用于筛选>70%权益）────────────────────────
    summary = db.raw_sql("""
                     SELECT crsp_fundno, caldt,asset_dt,per_com,per_pref
                     FROM crsp_q_mutualfunds.fund_summary
                     """, date_cols=['caldt'])
    summary .to_csv(OUT / 'fund_summary.csv', index=False)
    print(f"summary: {summary.shape}")

    download_ff5_factors()
    db.close()


    print("\n✅ finished downloading.")


if __name__ == '__main__':
    main()
