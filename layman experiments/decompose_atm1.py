import pandas as pd
import numpy as np

# Detailed analysis for ATM 1
df = pd.read_csv('simulated/maharashtra/data/atm_daily.csv')
atm1 = df[df['terminal_id'] == 1].copy()
atm1['date'] = pd.to_datetime(atm1['date'])
atm1 = atm1.sort_values('date').reset_index(drop=True)

y1 = atm1[(atm1['date'] >= '2024-09-01') & (atm1['date'] <= '2025-08-31')].copy().reset_index(drop=True)
y2 = atm1[(atm1['date'] >= '2025-09-01') & (atm1['date'] <= '2026-08-31')].copy().reset_index(drop=True)

y1_mean = y1['demand'].mean()
y2_mean = y2['demand'].mean()

# Day of month multipliers in Y1 vs Y2
dom1 = y1.groupby('dom')['demand'].mean() / y1_mean
dom2 = y2.groupby('dom')['demand'].mean() / y2_mean
dom_df = pd.DataFrame({'Y1_Multiplier': dom1, 'Y2_Multiplier': dom2})
dom_df['Diff'] = dom_df['Y2_Multiplier'] - dom_df['Y1_Multiplier']
print("=== DAY OF MONTH PATTERN (MULTIPLIERS) ===")
print(dom_df.round(3).to_string())

# Day of week multipliers in Y1 vs Y2
dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
dow1 = y1.groupby('dow')['demand'].mean() / y1_mean
dow2 = y2.groupby('dow')['demand'].mean() / y2_mean
dow_df = pd.DataFrame({'Day': dow_names, 'Y1_Multiplier': dow1.values, 'Y2_Multiplier': dow2.values})
print("\n=== DAY OF WEEK PATTERN (MULTIPLIERS) ===")
print(dow_df.round(3).to_string(index=False))

# Monthly seasonality
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
m1 = y1.groupby('month')['demand'].mean() / y1_mean
m2 = y2.groupby('month')['demand'].mean() / y2_mean
m_df = pd.DataFrame({'Month': [month_names[i-1] for i in m1.index], 'Y1_Multiplier': m1.values, 'Y2_Multiplier': m2.values})
print("\n=== MONTHLY SEASONALITY (MULTIPLIERS) ===")
print(m_df.round(3).to_string(index=False))

# Variance Decomposition (ANOVA style R2 on Y2)
ss_total = np.sum((y2['demand'] - y2_mean)**2)

# Residual SS for various models on Y2
def get_ss_res(pred):
    return np.sum((y2['demand'] - pred)**2)

pred_flat = np.full(len(y2), y1_mean)
pred_dom = y2['dom'].map(dom1).values * y1_mean
pred_dow = y2['dow'].map(dow1).values * y1_mean

# DOM + DOW
pred_dom_dow = (y2['dom'].map(dom1).values * y2['dow'].map(dow1).values) * y1_mean

# DOM + Month
m_map1 = y1.groupby('month')['demand'].mean() / y1_mean
pred_dom_month = (y2['dom'].map(dom1).values * y2['month'].map(m_map1).values) * y1_mean

# DOM + DOW + Month
pred_full_calendar = (y2['dom'].map(dom1).values * y2['dow'].map(dow1).values * y2['month'].map(m_map1).values) * y1_mean

print("\n=== EXPLAINED VARIANCE ON YEAR 2 (OUT-OF-SAMPLE 1-YEAR AHEAD) ===")
models = {
    'Flat Y1 Mean': pred_flat,
    'Y1 DOW Profile': pred_dow,
    'Y1 DOM Profile': pred_dom,
    'Y1 DOM * DOW Profile': pred_dom_dow,
    'Y1 DOM * Month Profile': pred_dom_month,
    'Y1 DOM * DOW * Month Profile': pred_full_calendar,
}

for name, pred in models.items():
    ss_res = get_ss_res(pred)
    r2 = 1 - ss_res / ss_total
    wape = np.sum(np.abs(y2['demand'] - pred)) / y2['demand'].sum()
    print(f"{name:30s} | R2: {r2:7.4f} | WAPE: {wape*100:5.2f}%")
