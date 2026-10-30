import pandas as pd
import numpy as np

# Load full daily_cash_out to check across ATMs
df_all = pd.read_csv('data/processed/daily_cash_out.csv')
meta = pd.read_csv('data/master/cash_files/terminal_metadata.csv')

def analyze_atm(tid):
    atm = df_all[df_all['terminal_id'] == tid].copy()
    atm['date'] = pd.to_datetime(atm['transaction_date'])
    atm = atm.sort_values('date').reset_index(drop=True)
    
    y1 = atm[(atm['date'] >= '2024-09-01') & (atm['date'] <= '2025-08-31')].copy().reset_index(drop=True)
    y2 = atm[(atm['date'] >= '2025-09-01') & (atm['date'] <= '2026-08-31')].copy().reset_index(drop=True)
    
    # Add calendar features
    for y in [y1, y2]:
        y['dom'] = y['date'].dt.day
        y['dow'] = y['date'].dt.dayofweek
        y['month'] = y['date'].dt.month
        y['week'] = y['date'].dt.isocalendar().week
    
    y1_mean = y1['net_demand'].mean()
    y2_mean = y2['net_demand'].mean()
    y1_std = y1['net_demand'].std()
    y2_std = y2['net_demand'].std()
    y1_cv = y1_std / y1_mean
    
    annual_drift = (y2['net_demand'].sum() - y1['net_demand'].sum()) / y1['net_demand'].sum() * 100
    
    # Correlations
    dom1 = y1.groupby('dom')['net_demand'].mean()
    dom2 = y2.groupby('dom')['net_demand'].mean()
    corr_dom = dom1.corr(dom2)
    
    dow1 = y1.groupby('dow')['net_demand'].mean()
    dow2 = y2.groupby('dow')['net_demand'].mean()
    corr_dow = dow1.corr(dow2)
    
    m1 = y1.groupby('month')['net_demand'].sum()
    m2 = y2.groupby('month')['net_demand'].sum()
    corr_month = m1.corr(m2)
    
    # Models evaluated on Year 2
    # 1. Flat mean
    pred_flat = np.full(len(y2), y1_mean)
    wape_flat = np.sum(np.abs(y2['net_demand'] - pred_flat)) / y2['net_demand'].sum()
    
    # 2. Naive same calendar date last year
    y1['mm_dd'] = y1['date'].dt.strftime('%m-%d')
    y2['mm_dd'] = y2['date'].dt.strftime('%m-%d')
    merged_cal = pd.merge(y2[['mm_dd', 'net_demand']], y1[['mm_dd', 'net_demand']], on='mm_dd', suffixes=('', '_y1'))
    wape_yoy_cal = np.sum(np.abs(merged_cal['net_demand'] - merged_cal['net_demand_y1'])) / merged_cal['net_demand'].sum()
    
    # 3. DOM model fit on Y1
    dom_idx = y1.groupby('dom')['net_demand'].mean() / y1_mean
    pred_dom = y2['dom'].map(dom_idx).values * y1_mean
    wape_dom = np.sum(np.abs(y2['net_demand'] - pred_dom)) / y2['net_demand'].sum()
    r2_dom = 1 - np.sum((y2['net_demand'] - pred_dom)**2) / np.sum((y2['net_demand'] - y2_mean)**2)
    
    # 4. Weekly aggregation (7-day cycles)
    act_7 = y2['net_demand'].values[:52*7].reshape(52, 7).sum(axis=1)
    pred_dom_7 = pred_dom[:52*7].reshape(52, 7).sum(axis=1)
    wape_7 = np.sum(np.abs(act_7 - pred_dom_7)) / np.sum(act_7)
    r2_7 = 1 - np.sum((act_7 - pred_dom_7)**2) / np.sum((act_7 - np.mean(act_7))**2)
    
    # 5. Monthly aggregation
    y2_copy = y2.copy()
    y2_copy['pred_dom'] = pred_dom
    y2_m = y2_copy.groupby('month')[['net_demand', 'pred_dom']].sum()
    wape_month = np.sum(np.abs(y2_m['net_demand'] - y2_m['pred_dom'])) / y2_m['net_demand'].sum()
    
    loc = meta[meta['terminal_id'] == tid]['location'].values[0] if len(meta[meta['terminal_id'] == tid]) > 0 else 'Unknown'
    
    return {
        'TID': tid,
        'Location': loc[:32],
        'Mean_Lakh': round(y1_mean / 1e5, 1),
        'CV': round(y1_cv, 2),
        'Ann_Drift_%': round(annual_drift, 1),
        'Corr_DOM': round(corr_dom, 3),
        'Corr_Month': round(corr_month, 3),
        'Daily_WAPE_Flat': round(wape_flat * 100, 1),
        'Daily_WAPE_DOM': round(wape_dom * 100, 1),
        'Daily_R2_DOM': round(r2_dom, 3),
        'Weekly_WAPE_DOM': round(wape_7 * 100, 1),
        'Weekly_R2_DOM': round(r2_7, 3),
        'Monthly_WAPE': round(wape_month * 100, 1)
    }

# Test sample of ATMs: Maharashtra ATMs 1-8 plus ATM 22 and others
tids_to_check = [1, 2, 3, 4, 5, 6, 7, 8, 22, 50, 75, 100]
results = [analyze_atm(t) for t in tids_to_check]
df_res = pd.DataFrame(results)
print(df_res.to_string(index=False))
