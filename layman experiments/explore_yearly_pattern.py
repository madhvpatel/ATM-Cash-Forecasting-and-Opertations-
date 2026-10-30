import pandas as pd
import numpy as np

# Load ATM 1 data
df = pd.read_csv('simulated/maharashtra/data/atm_daily.csv')
atm1 = df[df['terminal_id'] == 1].copy()
atm1['date'] = pd.to_datetime(atm1['date'])
atm1 = atm1.sort_values('date').reset_index(drop=True)

# Define Year 1 and Year 2
# Year 1: 2024-09-01 to 2025-08-31 (365 days)
# Year 2: 2025-09-01 to 2026-08-31 (365 days)
y1 = atm1[(atm1['date'] >= '2024-09-01') & (atm1['date'] <= '2025-08-31')].copy().reset_index(drop=True)
y2 = atm1[(atm1['date'] >= '2025-09-01') & (atm1['date'] <= '2026-08-31')].copy().reset_index(drop=True)

print("=== BASIC SUMMARY FOR ATM 1 (SYNATM0001 - Mumbai Railway Concourse) ===")
print(f"Total rows in dataset: {len(atm1)}")
print(f"Year 1 days: {len(y1)} ({y1['date'].min().date()} to {y1['date'].max().date()})")
print(f"Year 2 days: {len(y2)} ({y2['date'].min().date()} to {y2['date'].max().date()})")

# Annual macro stats
y1_total = y1['demand'].sum()
y2_total = y2['demand'].sum()
y1_mean = y1['demand'].mean()
y2_mean = y2['demand'].mean()
y1_std = y1['demand'].std()
y2_std = y2['demand'].std()
y1_cv = y1_std / y1_mean
y2_cv = y2_std / y2_mean

print(f"\nYear 1 Total Demand: ₹{y1_total:,.0f} (₹{y1_total/1e5:.2f} Lakh)")
print(f"Year 2 Total Demand: ₹{y2_total:,.0f} (₹{y2_total/1e5:.2f} Lakh)")
print(f"Annual Drift / Growth: {((y2_total - y1_total) / y1_total) * 100:.2f}%")
print(f"Year 1 Daily Mean: ₹{y1_mean:,.0f} (Std: ₹{y1_std:,.0f}, CV: {y1_cv:.4f})")
print(f"Year 2 Daily Mean: ₹{y2_mean:,.0f} (Std: ₹{y2_std:,.0f}, CV: {y2_cv:.4f})")

# Correlation between Year 1 and Year 2
# 1. Matching exact same calendar date (month and day)
y1['month_day'] = y1['date'].dt.strftime('%m-%d')
y2['month_day'] = y2['date'].dt.strftime('%m-%d')
merged_calendar = pd.merge(y1[['month_day', 'demand', 'dow', 'dom']], 
                           y2[['month_day', 'demand', 'dow', 'dom']], 
                           on='month_day', suffixes=('_y1', '_y2'))
corr_cal = merged_calendar['demand_y1'].corr(merged_calendar['demand_y2'])
print(f"\nCorrelation between Year 1 and Year 2 by same calendar date (MM-DD): {corr_cal:.4f}")

# 2. Matching Day of Year index (1 to 365)
corr_doy = np.corrcoef(y1['demand'].values, y2['demand'].values)[0, 1]
print(f"Correlation between Year 1 and Year 2 by Day of Year (day 1..365): {corr_doy:.4f}")

# 3. Monthly totals
m1 = y1.groupby(y1['date'].dt.month)['demand'].sum()
m2 = y2.groupby(y2['date'].dt.month)['demand'].sum()
m_comp = pd.DataFrame({'Year1_Total': m1, 'Year2_Total': m2})
m_comp['Diff_%'] = (m_comp['Year2_Total'] - m_comp['Year1_Total']) / m_comp['Year1_Total'] * 100
print("\n--- Monthly Total Comparison ---")
print(m_comp)
print(f"Monthly totals correlation: {m1.corr(m2):.4f}")

# 4. Day of Month pattern
dom1 = y1.groupby('dom')['demand'].mean()
dom2 = y2.groupby('dom')['demand'].mean()
print(f"\nDay of Month profile correlation (Y1 vs Y2): {dom1.corr(dom2):.4f}")

# 5. Day of Week pattern
dow1 = y1.groupby('dow')['demand'].mean()
dow2 = y2.groupby('dow')['demand'].mean()
print(f"Day of Week profile correlation (Y1 vs Y2): {dow1.corr(dow2):.4f}")

# Forecasting tests: using Year 1 pattern to predict Year 2
# Metrics: WAPE = sum(|actual - pred|) / sum(actual), RMSE, R2, MAPE
def eval_metrics(actual, pred):
    wape = np.sum(np.abs(actual - pred)) / np.sum(actual)
    rmse = np.sqrt(np.mean((actual - pred)**2))
    mape = np.mean(np.abs(actual - pred) / actual)
    r2 = 1 - np.sum((actual - pred)**2) / np.sum((actual - np.mean(actual))**2)
    bias = (np.sum(pred) - np.sum(actual)) / np.sum(actual)
    return {'WAPE': wape, 'RMSE': rmse, 'MAPE': mape, 'R2': r2, 'Bias': bias}

# Model 1: Flat Year 1 Mean
pred_flat = np.full(len(y2), y1_mean)
m1_res = eval_metrics(y2['demand'].values, pred_flat)

# Model 2: Naive YoY (exact same calendar day last year)
pred_yoy_cal = merged_calendar['demand_y1'].values
actual_y2_cal = merged_calendar['demand_y2'].values
m2_res = eval_metrics(actual_y2_cal, pred_yoy_cal)

# Model 3: Year 1 Day of Month (DOM) factor * Year 1 Mean
dom_index_y1 = y1.groupby('dom')['demand'].mean() / y1_mean
pred_dom = y2['dom'].map(dom_index_y1).values * y1_mean
m3_res = eval_metrics(y2['demand'].values, pred_dom)

# Model 4: Year 1 Day of Week (DOW) factor * Year 1 Mean
dow_index_y1 = y1.groupby('dow')['demand'].mean() / y1_mean
pred_dow = y2['dow'].map(dow_index_y1).values * y1_mean
m4_res = eval_metrics(y2['demand'].values, pred_dow)

# Model 5: Year 1 DOM * DOW
dom_dow_y1 = y1.groupby(['dom', 'dow'])['demand'].mean().to_dict()
# fallback to dom_index * dow_index * mean if combination missing
pred_dom_dow = []
for idx, row in y2.iterrows():
    val = dom_dow_y1.get((row['dom'], row['dow']), dom_index_y1[row['dom']] * dow_index_y1[row['dow']] * y1_mean)
    pred_dom_dow.append(val)
pred_dom_dow = np.array(pred_dom_dow)
m5_res = eval_metrics(y2['demand'].values, pred_dom_dow)

# Model 6: Year 1 DOM * Month
dom_month_y1 = y1.groupby(['month', 'dom'])['demand'].mean().to_dict()
pred_dom_month = np.array([dom_month_y1.get((r['month'], r['dom']), dom_index_y1[r['dom']] * y1_mean) for _, r in y2.iterrows()])
m6_res = eval_metrics(y2['demand'].values, pred_dom_month)

results = pd.DataFrame([
    {'Model': '1. Flat Year 1 Mean', **m1_res},
    {'Model': '2. Exact Same Day Last Year (Naive YoY)', **m2_res},
    {'Model': '3. Year 1 Day-of-Month Profile', **m3_res},
    {'Model': '4. Year 1 Day-of-Week Profile', **m4_res},
    {'Model': '5. Year 1 DOM + DOW Profile', **m5_res},
    {'Model': '6. Year 1 DOM + Month Profile', **m6_res},
])

print("\n=== DAILY FORECAST PERFORMANCE ON ENTIRE YEAR 2 (USING ONLY YEAR 1 PATTERN) ===")
print(results.to_string(index=False))

# Now let's test aggregated horizons: 7-day replenishment cycles and Monthly totals
# 7-day cycle WAPE
def weekly_wape(actual, pred):
    n_weeks = len(actual) // 7
    act_w = actual[:n_weeks*7].reshape(n_weeks, 7).sum(axis=1)
    pred_w = pred[:n_weeks*7].reshape(n_weeks, 7).sum(axis=1)
    wape = np.sum(np.abs(act_w - pred_w)) / np.sum(act_w)
    r2 = 1 - np.sum((act_w - pred_w)**2) / np.sum((act_w - np.mean(act_w))**2)
    return wape, r2

w_flat, r2_w_flat = weekly_wape(y2['demand'].values, pred_flat)
w_yoy, r2_w_yoy = weekly_wape(actual_y2_cal, pred_yoy_cal)
w_dom, r2_w_dom = weekly_wape(y2['demand'].values, pred_dom)
w_dom_month, r2_w_dom_month = weekly_wape(y2['demand'].values, pred_dom_month)

print("\n=== 7-DAY AGGREGATED HORIZON (52 WEEKS IN YEAR 2) ===")
print(f"Flat Y1 Mean: Weekly WAPE = {w_flat*100:.2f}%, R2 = {r2_w_flat:.4f}")
print(f"Naive YoY:    Weekly WAPE = {w_yoy*100:.2f}%, R2 = {r2_w_yoy:.4f}")
print(f"Y1 DOM:       Weekly WAPE = {w_dom*100:.2f}%, R2 = {r2_w_dom:.4f}")
print(f"Y1 DOM+Month: Weekly WAPE = {w_dom_month*100:.2f}%, R2 = {r2_w_dom_month:.4f}")

# Monthly WAPE
y2_m = y2.copy()
y2_m['pred_dom'] = pred_dom
y2_m['pred_flat'] = pred_flat
y2_m['pred_dom_month'] = pred_dom_month
y2_m_agg = y2_m.groupby('month')[['demand', 'pred_flat', 'pred_dom', 'pred_dom_month']].sum()
m_wape_dom = np.sum(np.abs(y2_m_agg['demand'] - y2_m_agg['pred_dom'])) / np.sum(y2_m_agg['demand'])
m_wape_dom_month = np.sum(np.abs(y2_m_agg['demand'] - y2_m_agg['pred_dom_month'])) / np.sum(y2_m_agg['demand'])
m_wape_flat = np.sum(np.abs(y2_m_agg['demand'] - y2_m_agg['pred_flat'])) / np.sum(y2_m_agg['demand'])

print("\n=== MONTHLY AGGREGATED HORIZON (12 MONTHS IN YEAR 2) ===")
print(f"Flat Y1 Mean: Monthly WAPE = {m_wape_flat*100:.2f}%")
print(f"Y1 DOM:       Monthly WAPE = {m_wape_dom*100:.2f}%")
print(f"Y1 DOM+Month: Monthly WAPE = {m_wape_dom_month*100:.2f}%")
print(f"Total Annual Forecast Error: {abs(y1_total - y2_total) / y2_total * 100:.2f}%")
