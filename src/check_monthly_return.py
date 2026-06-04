import pandas as pd

df = pd.read_csv(
    '/Users/allison/Documents/Portfolio/ml_mutual_fund/data/raw/monthly_returns.csv',
    parse_dates=['caldt'],
)

result = df[df['crsp_fundno'] == 22169.0].sort_values('caldt')
print(result.to_string(index=False))


df = pd.read_csv(
    '/Users/allison/Documents/Portfolio/ml_mutual_fund/data/processed/f_panel.csv',
    parse_dates=['caldt'],
)

result = df[(df['no_load'] == False) & (df['exp_ratio'].notna())].sort_values('caldt')
print(result.to_string(index=False))
