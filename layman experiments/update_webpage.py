"""Build the enhanced Features and Master Data Explorer web page.

Outputs:
- layman experiments/features_and_masters.html
- layman experiments/index.html
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
CMS_ROOT = HERE.parent
DATA_DIR = CMS_ROOT / "data"

def build():
    # 1. Feature Registry
    df_reg = pd.read_csv(DATA_DIR / "feature_registry.csv")
    
    # 2. Input features definition
    history_feats = [
        ("terminal_id", "Terminal unique identifier", "ATM ID", "History & Context", "A, B, C, D, F", "Available"),
        ("loc_type", "Physical site typology (Concourse, Airport, Mall, etc.)", "Categorical", "History & Context", "A, B, C, D, F", "Available"),
        ("horizon", "Forecast lead step in days (h=1..7)", "Integer (1..7)", "History & Context", "A, B, C, D, F", "Available"),
        ("lag_1", "Withdrawal demand on the forecast origin cutoff day", "INR Cash", "History & Context", "A, B, C, D, F", "Derived"),
        ("lag_7", "Withdrawal demand 7 days prior to target date (<= cutoff)", "INR Cash", "History & Context", "A, B, C, D, F", "Derived"),
        ("lag_14", "Withdrawal demand 14 days prior to target date (<= cutoff)", "INR Cash", "History & Context", "A, B, C, D, F", "Derived"),
        ("lag_28", "Withdrawal demand 28 days prior to target date (<= cutoff)", "INR Cash", "History & Context", "A, B, C, D, F", "Derived"),
        ("roll_mean_7", "7-day trailing mean demand strictly ending on cutoff", "INR Cash", "History & Context", "A, B, C, D, F", "Derived"),
        ("roll_std_7", "7-day trailing demand standard deviation ending on cutoff", "INR Cash", "History & Context", "A, B, C, D, F", "Derived"),
        ("roll_mean_14", "14-day trailing mean demand ending on cutoff", "INR Cash", "History & Context", "A, B, C, D, F", "Derived"),
        ("roll_std_14", "14-day trailing demand standard deviation ending on cutoff", "INR Cash", "History & Context", "A, B, C, D, F", "Derived"),
        ("roll_mean_28", "28-day trailing mean demand ending on cutoff", "INR Cash", "History & Context", "A, B, C, D, F", "Derived"),
        ("roll_std_28", "28-day trailing demand standard deviation ending on cutoff", "INR Cash", "History & Context", "A, B, C, D, F", "Derived"),
    ]
    
    calendar_feats = [
        ("dom", "Day of Month (1..31); primary monthly salary wave driver", "Integer (1..31)", "Calendar Dynamics", "B, C, D, F, Exp3", "Available"),
        ("dow", "Day of Week (Monday=0 to Sunday=6)", "Integer (0..6)", "Calendar Dynamics", "B, C, D, F", "Available"),
        ("month", "Calendar month (1..12); annual seasonal envelope", "Integer (1..12)", "Calendar Dynamics", "B, C, D, F", "Available"),
        ("is_weekend", "Indicator for Saturday or Sunday", "Binary (0/1)", "Calendar Dynamics", "B, C, D, F", "Available"),
        ("is_2_4_sat", "Indian bank closure Saturdays (2nd & 4th Saturday)", "Binary (0/1)", "Calendar Dynamics", "B, C, D, F", "Available"),
        ("is_month_end", "Indicator for last 3 days of the month", "Binary (0/1)", "Calendar Dynamics", "B, C, D, F", "Available"),
    ]
    
    holiday_feats = [
        ("is_payday", "Assumed salary disbursement window (1st, 15th, last weekday)", "Binary (0/1)", "Holidays & Events", "C, F", "Available"),
        ("is_regional_holiday", "RBI-notified bank holiday for regional office jurisdiction", "Binary (0/1)", "Holidays & Events", "C, F", "Available"),
        ("pre_holiday", "Day immediately preceding an RBI regional bank holiday", "Binary (0/1)", "Holidays & Events", "C, F", "Available"),
    ]
    
    weather_feats = [
        ("rain_mm", "Open-Meteo ERA5 daily precipitation sum", "Float (mm)", "Weather Conditions", "D, F", "Exogenous"),
        ("rain_ge_10", "Moderate rain threshold flag (>= 10 mm/day)", "Binary (0/1)", "Weather Conditions", "D, F", "Exogenous"),
        ("rain_ge_25", "Heavy monsoon rain threshold flag (>= 25 mm/day)", "Binary (0/1)", "Weather Conditions", "D, F", "Exogenous"),
        ("tmax_c", "Daily maximum 2-meter temperature in Celsius", "Float (°C)", "Weather Conditions", "D, F", "Exogenous"),
        ("tmean_c", "Daily mean 2-meter temperature in Celsius", "Float (°C)", "Weather Conditions", "D, F", "Exogenous"),
        ("tmin_c", "Daily minimum 2-meter temperature in Celsius", "Float (°C)", "Weather Conditions", "D, F", "Exogenous"),
        ("tmax_ge_35", "Extreme heat alert threshold flag (>= 35 °C)", "Binary (0/1)", "Weather Conditions", "D, F", "Exogenous"),
    ]
    
    uncensored_feats = [
        ("zero_balance_snapshot_count", "Number of 15-minute cash inventory snapshots hitting ₹0 balance", "Integer (0..96)", "Demand Uncensoring", "Evaluated in Study", "Available"),
        ("hours_empty", "Calculated hours ATM spent completely empty during operating day (snapshots × 0.25h)", "Float (hours)", "Demand Uncensoring", "Evaluated in Study", "Derived"),
        ("active_fraction", "Operating fraction of day with cash in vault (max(1 - empty/96, 0.20))", "Float (0.2..1.0)", "Demand Uncensoring", "Evaluated in Study", "Derived"),
        ("is_cashout_day", "Binary indicator if machine experienced stockout on target date", "Binary (0/1)", "Demand Uncensoring", "Evaluated in Study", "Derived"),
        ("uncensored_demand", "Run-rate reconstructed true customer demand (Demand / Active Fraction)", "INR Cash", "Demand Uncensoring", "Uncensored Target", "Derived Target"),
    ]
    
    uncollected_feats = [
        ("footfall", "Real-time or forecasted station/mall pedestrian foot traffic counter", "Counts", "Uncollected Signals", "None (Excluded)", "Unavailable"),
        ("local_event", "Verified city stadium matches, major concerts, local expos", "Categorical/Flag", "Uncollected Signals", "None (Excluded)", "Unavailable"),
        ("capacity_amount", "Physical cassette rupee storage capacity limit", "INR Cash", "Uncollected Signals", "None (Excluded)", "Unavailable"),
        ("device_type", "Machine hardware model, CRM recycler vs standard dispenser", "Categorical", "Uncollected Signals", "None (Excluded)", "Unavailable"),
        ("uptime_pct", "Daily network connectivity and machine dispensing availability %", "Float (0..100)", "Uncollected Signals", "None (Excluded)", "Unavailable"),
        ("failed_amount", "Rupee volume of declined/stuck dispensing attempts", "INR Cash", "Uncollected Signals", "None (Excluded)", "Unavailable"),
        ("denomination", "Observed banknote denomination split (constant 100 unit in raw)", "INR Unit", "Uncollected Signals", "Context Only", "Static Unit"),
        ("is_recycler", "Cash-in deposit capability (placeholder in primary export)", "Binary", "Uncollected Signals", "Context Only", "Placeholder"),
    ]
    
    all_features = []
    for f in history_feats + calendar_feats + holiday_feats + weather_feats + uncensored_feats + uncollected_feats:
        all_features.append({
            "name": f[0],
            "definition": f[1],
            "type": f[2],
            "group": f[3],
            "models": f[4],
            "availability": f[5],
        })

    html_code = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="description" content="Interactive Master Data and Input Features Explorer for Cash Demand Forecasting.">
  <title>Cash Intelligence · Features Registry & Master Data Explorer</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
  <script src="master_data_bundle.js"></script>
  <style>
    :root {{
      --bg-base: #0a0d14;
      --bg-surface: #111622;
      --bg-surface-elevated: #161d2d;
      --bg-card: rgba(22, 29, 45, 0.7);
      --border-subtle: rgba(255, 255, 255, 0.08);
      --border-hover: rgba(56, 189, 248, 0.4);
      --text-main: #f1f5f9;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      --primary: #38bdf8;
      --primary-glow: rgba(56, 189, 248, 0.15);
      --emerald: #10b981;
      --emerald-glow: rgba(16, 185, 129, 0.15);
      --amber: #f59e0b;
      --indigo: #6366f1;
      --rose: #f43f5e;
      --purple: #a855f7;
      --font-sans: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }}

    * {{ box-sizing: border-box; margin: 0; padding: 0; }}

    body {{
      background-color: var(--bg-base);
      color: var(--text-main);
      font-family: var(--font-sans);
      line-height: 1.5;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }}

    .glow-bg {{
      position: fixed;
      top: -200px;
      left: 50%;
      transform: translateX(-50%);
      width: 1000px;
      height: 500px;
      background: radial-gradient(circle, rgba(56, 189, 248, 0.08) 0%, rgba(99, 102, 241, 0.04) 50%, transparent 70%);
      pointer-events: none;
      z-index: 0;
    }}

    header {{
      position: sticky;
      top: 0;
      z-index: 50;
      background: rgba(10, 13, 20, 0.88);
      backdrop-filter: blur(16px);
      border-bottom: 1px solid var(--border-subtle);
      padding: 14px 32px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}

    .brand {{
      display: flex;
      align-items: center;
      gap: 12px;
      text-decoration: none;
    }}

    .brand-icon {{
      width: 38px;
      height: 38px;
      border-radius: 10px;
      background: linear-gradient(135deg, var(--primary), var(--indigo));
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 4px 16px var(--primary-glow);
    }}

    .brand-icon svg {{ width: 22px; height: 22px; color: #fff; }}

    .brand-text h1 {{
      font-size: 1.15rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      color: #fff;
    }}

    .brand-text span {{
      font-size: 0.75rem;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-weight: 600;
    }}

    .header-stats {{
      display: flex;
      gap: 16px;
    }}

    .stat-pill {{
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 9999px;
      padding: 6px 14px;
      font-size: 0.8rem;
      display: flex;
      align-items: center;
      gap: 8px;
    }}

    .stat-pill .dot {{
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--emerald);
      box-shadow: 0 0 8px var(--emerald);
    }}

    .container {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 28px 32px;
      position: relative;
      z-index: 10;
      flex: 1;
      width: 100%;
    }}

    /* Tabs Bar */
    .tabs-wrapper {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid var(--border-subtle);
      margin-bottom: 24px;
      gap: 16px;
      flex-wrap: wrap;
    }}

    .tabs {{
      display: flex;
      gap: 6px;
    }}

    .tab-btn {{
      background: transparent;
      border: none;
      color: var(--text-muted);
      font-family: var(--font-sans);
      font-size: 0.95rem;
      font-weight: 600;
      padding: 12px 18px;
      cursor: pointer;
      border-radius: 8px 8px 0 0;
      position: relative;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      gap: 8px;
    }}

    .tab-btn:hover {{ color: #fff; }}
    .tab-btn.active {{ color: var(--primary); }}

    .tab-btn.active::after {{
      content: '';
      position: absolute;
      bottom: -1px;
      left: 0;
      right: 0;
      height: 2px;
      background: var(--primary);
      box-shadow: 0 -2px 8px var(--primary);
    }}

    .badge-count {{
      font-size: 0.72rem;
      padding: 2px 7px;
      border-radius: 9999px;
      background: var(--bg-surface-elevated);
      color: var(--text-muted);
      font-family: var(--font-mono);
    }}

    .tab-btn.active .badge-count {{
      background: var(--primary-glow);
      color: var(--primary);
    }}

    /* Global Search Box */
    .search-box {{
      position: relative;
      min-width: 300px;
    }}

    .search-box input {{
      width: 100%;
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 10px;
      padding: 10px 14px 10px 38px;
      color: #fff;
      font-family: var(--font-sans);
      font-size: 0.9rem;
      outline: none;
      transition: all 0.2s;
    }}

    .search-box input:focus {{
      border-color: var(--primary);
      box-shadow: 0 0 0 3px var(--primary-glow);
    }}

    .search-icon {{
      position: absolute;
      left: 12px;
      top: 50%;
      transform: translateY(-50%);
      color: var(--text-dim);
      width: 16px;
      height: 16px;
      pointer-events: none;
    }}

    /* Master Data Explorer Layout */
    .master-layout {{
      display: grid;
      grid-template-columns: 310px 1fr;
      gap: 24px;
      align-items: start;
    }}

    @media (max-width: 1024px) {{
      .master-layout {{ grid-template-columns: 1fr; }}
    }}

    .sidebar-menu {{
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 14px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 6px;
      max-height: calc(100vh - 180px);
      overflow-y: auto;
      position: sticky;
      top: 90px;
    }}

    .sidebar-title {{
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--text-dim);
      font-weight: 700;
      padding: 8px 12px 4px 12px;
    }}

    .menu-item {{
      background: transparent;
      border: 1px solid transparent;
      color: var(--text-muted);
      border-radius: 9px;
      padding: 10px 14px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      cursor: pointer;
      font-family: var(--font-sans);
      font-size: 0.88rem;
      font-weight: 500;
      transition: all 0.15s ease;
      text-align: left;
    }}

    .menu-item:hover {{
      background: var(--bg-surface-elevated);
      color: #fff;
    }}

    .menu-item.active {{
      background: rgba(56, 189, 248, 0.12);
      border-color: rgba(56, 189, 248, 0.35);
      color: var(--primary);
      font-weight: 700;
    }}

    .menu-item-left {{
      display: flex;
      align-items: center;
      gap: 10px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}

    .menu-item-count {{
      font-family: var(--font-mono);
      font-size: 0.75rem;
      color: var(--text-dim);
      background: var(--bg-base);
      padding: 2px 7px;
      border-radius: 9999px;
    }}

    .menu-item.active .menu-item-count {{
      background: var(--primary);
      color: #000;
      font-weight: 700;
    }}

    /* Main Table Panel */
    .master-view-panel {{
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 14px;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 18px;
      min-width: 0;
    }}

    .panel-header {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 16px;
      border-bottom: 1px solid var(--border-subtle);
      padding-bottom: 18px;
    }}

    .panel-title-area h3 {{
      font-size: 1.4rem;
      font-weight: 700;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 10px;
    }}

    .panel-title-area p {{
      color: var(--text-muted);
      font-size: 0.9rem;
      margin-top: 4px;
      max-width: 800px;
    }}

    .panel-path-tag {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-family: var(--font-mono);
      font-size: 0.78rem;
      color: var(--primary);
      background: rgba(56, 189, 248, 0.08);
      border: 1px solid rgba(56, 189, 248, 0.2);
      padding: 3px 10px;
      border-radius: 6px;
      margin-top: 8px;
    }}

    .panel-actions {{
      display: flex;
      gap: 10px;
    }}

    .btn-action {{
      background: var(--bg-surface-elevated);
      border: 1px solid var(--border-subtle);
      color: var(--text-main);
      padding: 8px 14px;
      border-radius: 8px;
      font-size: 0.82rem;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: all 0.2s;
    }}

    .btn-action:hover {{
      border-color: var(--primary);
      color: var(--primary);
      background: rgba(56, 189, 248, 0.05);
    }}

    /* Table Toolbar */
    .table-toolbar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
    }}

    .table-search {{
      position: relative;
      flex: 1;
      max-width: 400px;
    }}

    .table-search input {{
      width: 100%;
      background: var(--bg-base);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 8px 12px 8px 34px;
      color: #fff;
      font-size: 0.85rem;
      outline: none;
    }}

    .table-search input:focus {{
      border-color: var(--primary);
    }}

    .table-search svg {{
      position: absolute;
      left: 10px;
      top: 50%;
      transform: translateY(-50%);
      color: var(--text-dim);
      width: 14px;
      height: 14px;
    }}

    .table-controls {{
      display: flex;
      align-items: center;
      gap: 12px;
      font-size: 0.82rem;
      color: var(--text-muted);
    }}

    .table-controls select {{
      background: var(--bg-base);
      border: 1px solid var(--border-subtle);
      color: #fff;
      padding: 6px 10px;
      border-radius: 6px;
      outline: none;
      font-size: 0.82rem;
    }}

    /* Data Table */
    .data-table-wrapper {{
      border: 1px solid var(--border-subtle);
      border-radius: 10px;
      overflow-x: auto;
      max-height: 600px;
      position: relative;
      background: var(--bg-base);
    }}

    table.master-grid {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.84rem;
      white-space: nowrap;
    }}

    table.master-grid th {{
      position: sticky;
      top: 0;
      z-index: 20;
      background: var(--bg-surface-elevated);
      color: var(--text-muted);
      font-weight: 700;
      padding: 10px 14px;
      text-align: left;
      border-bottom: 1px solid var(--border-subtle);
      cursor: pointer;
      user-select: none;
    }}

    table.master-grid th:hover {{
      color: #fff;
      background: #1c263a;
    }}

    table.master-grid th .sort-icon {{
      display: inline-block;
      margin-left: 6px;
      color: var(--primary);
    }}

    table.master-grid td {{
      padding: 9px 14px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
      color: #cbd5e1;
      font-family: var(--font-mono);
      font-size: 0.8rem;
    }}

    table.master-grid tbody tr:hover td {{
      background: rgba(56, 189, 248, 0.04);
      color: #fff;
      cursor: pointer;
    }}

    /* Pagination Footer */
    .pagination-bar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 8px 0;
      font-size: 0.82rem;
      color: var(--text-muted);
      flex-wrap: wrap;
      gap: 12px;
    }}

    .pagination-btns {{
      display: flex;
      gap: 6px;
    }}

    .btn-page {{
      background: var(--bg-base);
      border: 1px solid var(--border-subtle);
      color: var(--text-main);
      padding: 5px 12px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 0.82rem;
      transition: all 0.15s;
    }}

    .btn-page:hover:not(:disabled) {{
      border-color: var(--primary);
      color: var(--primary);
    }}

    .btn-page:disabled {{
      opacity: 0.4;
      cursor: not-allowed;
    }}

    /* Row Details Modal */
    .modal-backdrop {{
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.7);
      backdrop-filter: blur(8px);
      z-index: 100;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 24px;
    }}

    .modal-card {{
      background: var(--bg-surface);
      border: 1px solid var(--border-hover);
      border-radius: 16px;
      width: 100%;
      max-width: 680px;
      max-height: 85vh;
      overflow-y: auto;
      padding: 24px;
      box-shadow: 0 20px 50px rgba(0,0,0,0.6);
    }}

    .modal-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid var(--border-subtle);
      padding-bottom: 14px;
      margin-bottom: 16px;
    }}

    .modal-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }}

    .modal-field {{
      background: var(--bg-base);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 10px 14px;
    }}

    .modal-field label {{
      display: block;
      font-size: 0.72rem;
      text-transform: uppercase;
      color: var(--text-dim);
      font-weight: 700;
      margin-bottom: 4px;
    }}

    .modal-field span {{
      font-family: var(--font-mono);
      font-size: 0.88rem;
      color: #fff;
      word-break: break-all;
    }}

    /* Card items & tags */
    .grid-cards {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
      gap: 18px;
      margin-bottom: 32px;
    }}

    .item-card {{
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 14px;
      padding: 18px;
      backdrop-filter: blur(12px);
      transition: all 0.2s ease;
      display: flex;
      flex-direction: column;
    }}

    .item-card:hover {{
      border-color: var(--border-hover);
      transform: translateY(-2px);
      box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4);
    }}

    .card-top {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      margin-bottom: 10px;
      gap: 10px;
    }}

    .card-title {{
      font-family: var(--font-mono);
      font-size: 1rem;
      font-weight: 700;
      color: #fff;
    }}

    .tag-badge {{
      font-size: 0.7rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      padding: 3px 8px;
      border-radius: 6px;
      white-space: nowrap;
    }}

    .tag-history {{ background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); }}
    .tag-calendar {{ background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }}
    .tag-holidays {{ background: rgba(244, 63, 94, 0.15); color: #f43f5e; border: 1px solid rgba(244, 63, 94, 0.3); }}
    .tag-weather {{ background: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); }}
    .tag-uncollected {{ background: rgba(148, 163, 184, 0.15); color: #94a3b8; border: 1px solid rgba(148, 163, 184, 0.3); }}
    .tag-master {{ background: rgba(168, 85, 247, 0.15); color: #a855f7; border: 1px solid rgba(168, 85, 247, 0.3); }}

    .card-desc {{
      color: var(--text-muted);
      font-size: 0.85rem;
      margin-bottom: 14px;
      flex-grow: 1;
    }}

    .card-meta-list {{
      display: flex;
      flex-direction: column;
      gap: 5px;
      border-top: 1px solid var(--border-subtle);
      padding-top: 10px;
      font-size: 0.78rem;
    }}

    .card-meta-row {{
      display: flex;
      justify-content: space-between;
      color: var(--text-dim);
    }}

    .card-meta-row span:last-child {{
      color: var(--text-main);
      font-family: var(--font-mono);
      font-weight: 500;
    }}

    .filter-pills {{
      display: flex;
      gap: 8px;
      overflow-x: auto;
      padding-bottom: 6px;
      margin-bottom: 20px;
    }}

    .pill-btn {{
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      color: var(--text-muted);
      border-radius: 8px;
      padding: 7px 14px;
      font-size: 0.82rem;
      font-weight: 600;
      cursor: pointer;
      white-space: nowrap;
      transition: all 0.2s;
    }}

    .pill-btn:hover {{ background: var(--bg-surface-elevated); color: #fff; }}
    .pill-btn.active {{ background: var(--primary); color: #000; border-color: var(--primary); font-weight: 700; }}

    footer {{
      border-top: 1px solid var(--border-subtle);
      padding: 20px 32px;
      text-align: center;
      color: var(--text-dim);
      font-size: 0.82rem;
    }}

    footer a {{ color: var(--primary); text-decoration: none; }}
  </style>
</head>
<body>
  <div class="glow-bg"></div>

  <header>
    <a href="#" class="brand">
      <div class="brand-icon">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
          <ellipse cx="12" cy="5" rx="9" ry="3"></ellipse>
          <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path>
          <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path>
        </svg>
      </div>
      <div class="brand-text">
        <h1>Cash Intelligence</h1>
        <span>Master Data & Feature Governance Portal</span>
      </div>
    </a>

    <div class="header-stats">
      <div class="stat-pill">
        <div class="dot"></div>
        <span><strong>13</strong> Master Tables Loaded</span>
      </div>
      <div class="stat-pill">
        <div class="dot"></div>
        <span><strong>29</strong> Model Inputs</span>
      </div>
      <div class="stat-pill">
        <div class="dot"></div>
        <span><strong>100</strong> Primary ATMs</span>
      </div>
    </div>
  </header>

  <main class="container">
    <div class="tabs-wrapper">
      <div class="tabs" role="tablist">
        <button id="tab-explorer-btn" class="tab-btn active" onclick="switchMainTab('explorer')">
          Master Data Explorer (Live Rows)
          <span class="badge-count" id="badge-total-tables">13 Tables</span>
        </button>
        <button id="tab-features-btn" class="tab-btn" onclick="switchMainTab('features')">
          Input Features Catalog
          <span class="badge-count" id="badge-features">37 Features</span>
        </button>
        <button id="tab-matrix-btn" class="tab-btn" onclick="switchMainTab('matrix')">
          Model Feature Matrix
          <span class="badge-count">6 Models</span>
        </button>
      </div>

      <div class="search-box">
        <svg class="search-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="11" cy="11" r="8"></circle>
          <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
        </svg>
        <input type="text" id="globalSearchInput" placeholder="Quick search..." oninput="handleGlobalSearch()">
      </div>
    </div>

    <!-- TAB 1: MASTER DATA LIVE EXPLORER -->
    <div id="section-explorer">
      <div class="master-layout">
        <!-- Sidebar Menu of Tables -->
        <div class="sidebar-menu" id="sidebarMenu">
          <div class="sidebar-title">Select Master Table</div>
          <!-- Menu items rendered by JS -->
        </div>

        <!-- Main View Panel -->
        <div class="master-view-panel" id="masterPanel">
          <div class="panel-header">
            <div class="panel-title-area">
              <h3 id="panelTableTitle">
                <span id="panelTableIcon">🏢</span>
                <span id="panelTableName">terminal_metadata.csv</span>
              </h3>
              <p id="panelTableDesc">Primary register of all 100 ATM terminals in the cash estate.</p>
              <div class="panel-path-tag" id="panelTablePath">data/master/cash_files/terminal_metadata.csv</div>
            </div>

            <div class="panel-actions">
              <button class="btn-action" onclick="exportCurrentTable('csv')">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
                Export CSV
              </button>
              <button class="btn-action" onclick="copyCurrentTableJSON()">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
                Copy JSON
              </button>
            </div>
          </div>

          <!-- Table Toolbar -->
          <div class="table-toolbar">
            <div class="table-search">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
              <input type="text" id="tableFilterInput" placeholder="Filter rows in this table..." oninput="filterCurrentTable()">
            </div>

            <div class="table-controls">
              <span>Rows per page:</span>
              <select id="pageSizeSelect" onchange="changePageSize()">
                <option value="15">15</option>
                <option value="25" selected>25</option>
                <option value="50">50</option>
                <option value="100">100</option>
                <option value="all">All</option>
              </select>
            </div>
          </div>

          <!-- Data Grid -->
          <div class="data-table-wrapper" id="gridWrapper">
            <table class="master-grid" id="currentMasterGrid">
              <thead id="gridHead">
                <!-- Rendered by JS -->
              </thead>
              <tbody id="gridBody">
                <!-- Rendered by JS -->
              </tbody>
            </table>
          </div>

          <!-- Pagination Footer -->
          <div class="pagination-bar">
            <div id="paginationInfo">Showing 1 to 25 of 100 rows</div>
            <div class="pagination-btns">
              <button class="btn-page" id="btnPrevPage" onclick="prevPage()">Previous</button>
              <button class="btn-page" id="btnNextPage" onclick="nextPage()">Next</button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 2: INPUT FEATURES CATALOG -->
    <div id="section-features" style="display: none;">
      <div class="filter-pills" id="featurePills">
        <button class="pill-btn active" onclick="filterFeatures('all')">All Groups (42)</button>
        <button class="pill-btn" onclick="filterFeatures('History & Context')">Withdrawal History (13)</button>
        <button class="pill-btn" onclick="filterFeatures('Calendar Dynamics')">Calendar Dynamics (6)</button>
        <button class="pill-btn" onclick="filterFeatures('Holidays & Events')">Holidays & Events (3)</button>
        <button class="pill-btn" onclick="filterFeatures('Weather Conditions')">Weather Signals (7)</button>
        <button class="pill-btn" onclick="filterFeatures('Demand Uncensoring')">Demand Uncensoring (5)</button>
        <button class="pill-btn" onclick="filterFeatures('Uncollected Signals')">Uncollected Signals (8)</button>
      </div>

      <div class="grid-cards" id="featuresGrid">
        <!-- Rendered by JS -->
      </div>
    </div>

    <!-- TAB 3: MODEL FEATURE MATRIX -->
    <div id="section-matrix" style="display: none;">
      <div style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: 14px; overflow-x: auto;">
        <table class="master-grid" style="width: 100%;">
          <thead>
            <tr>
              <th style="padding: 14px 18px;">Feature Name</th>
              <th>Information Tier</th>
              <th style="text-align: center;">Model A<br><small style="color:var(--text-dim)">History Only</small></th>
              <th style="text-align: center;">Model B<br><small style="color:var(--text-dim)">+Calendar</small></th>
              <th style="text-align: center;">Model C<br><small style="color:var(--text-dim)">+Holidays</small></th>
              <th style="text-align: center;">Model D<br><small style="color:var(--text-dim)">+Weather</small></th>
              <th style="text-align: center; background: rgba(56, 189, 248, 0.1);">Model F<br><small style="color:var(--primary)">Full Stack</small></th>
              <th style="text-align: center;">Exp 3<br><small style="color:var(--text-dim)">Seasonal</small></th>
            </tr>
          </thead>
          <tbody id="matrixTableBody">
            <!-- Rendered by JS -->
          </tbody>
        </table>
      </div>
    </div>
  </main>

  <!-- Row Details Modal -->
  <div class="modal-backdrop" id="rowModal" onclick="closeModal(event)">
    <div class="modal-card" onclick="event.stopPropagation()">
      <div class="modal-header">
        <h4 style="color: #fff; font-size: 1.1rem;" id="modalTitle">Record Details</h4>
        <button class="btn-action" style="padding: 4px 10px;" onclick="closeModalDirect()">✕</button>
      </div>
      <div class="modal-grid" id="modalGrid">
        <!-- Rendered by JS -->
      </div>
    </div>
  </div>

  <footer>
    <p>Cash Intelligence Platform · CMS2 Evaluation Environment · Live Master Explorer loaded from <a href="master_data_bundle.js">master_data_bundle.js</a></p>
  </footer>

  <script>
    const FEATURES_DATA = {json.dumps(all_features)};
    const BUNDLE = window.MASTER_DATA_BUNDLE || {{}};
    
    let activeMainTab = 'explorer';
    let currentTableKey = 'terminal_metadata';
    let currentFilteredRecords = [];
    let currentPage = 1;
    let pageSize = 25;
    let sortColumn = null;
    let sortAsc = true;
    let currentFeatureGroup = 'all';

    function init() {{
      renderSidebar();
      loadTable(currentTableKey);
      renderFeatures();
      renderMatrix();
    }}

    function switchMainTab(tab) {{
      activeMainTab = tab;
      document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
      document.getElementById('tab-' + tab + '-btn').classList.add('active');

      document.getElementById('section-explorer').style.display = tab === 'explorer' ? 'block' : 'none';
      document.getElementById('section-features').style.display = tab === 'features' ? 'block' : 'none';
      document.getElementById('section-matrix').style.display = tab === 'matrix' ? 'block' : 'none';
    }}

    function renderSidebar() {{
      const menu = document.getElementById('sidebarMenu');
      menu.innerHTML = '<div class="sidebar-title">Select Master Table</div>';

      Object.keys(BUNDLE).forEach(key => {{
        const t = BUNDLE[key];
        const btn = document.createElement('button');
        btn.className = `menu-item ${{key === currentTableKey ? 'active' : ''}}`;
        btn.id = `sidebar-btn-${{key}}`;
        btn.onclick = () => loadTable(key);
        btn.innerHTML = `
          <div class="menu-item-left">
            <span>${{t.icon}}</span>
            <span style="overflow: hidden; text-overflow: ellipsis;">${{t.name}}</span>
          </div>
          <span class="menu-item-count">${{t.total_rows.toLocaleString()}}</span>
        `;
        menu.appendChild(btn);
      }});
    }}

    function loadTable(key) {{
      currentTableKey = key;
      const t = BUNDLE[key];
      if (!t) return;

      document.querySelectorAll('.menu-item').forEach(el => el.classList.remove('active'));
      const activeBtn = document.getElementById(`sidebar-btn-${{key}}`);
      if (activeBtn) activeBtn.classList.add('active');

      document.getElementById('panelTableIcon').innerText = t.icon;
      document.getElementById('panelTableName').innerText = t.name;
      document.getElementById('panelTableDesc').innerText = t.description;
      document.getElementById('panelTablePath').innerText = `${{t.path}} · ${{t.total_rows.toLocaleString()}} rows · ${{t.columns.length}} cols · Grain: ${{t.grain}}`;

      document.getElementById('tableFilterInput').value = '';
      currentPage = 1;
      sortColumn = null;
      sortAsc = true;

      currentFilteredRecords = [...t.records];
      renderTableHeaders(t.columns);
      renderTableBody();
    }}

    function renderTableHeaders(cols) {{
      const thead = document.getElementById('gridHead');
      thead.innerHTML = '';
      const tr = document.createElement('tr');

      cols.forEach(col => {{
        const th = document.createElement('th');
        th.onclick = () => handleSort(col);
        const sortIcon = sortColumn === col ? (sortAsc ? ' ▲' : ' ▼') : '';
        th.innerHTML = `${{col}}<span class="sort-icon">${{sortIcon}}</span>`;
        tr.appendChild(th);
      }});
      thead.appendChild(tr);
    }}

    function renderTableBody() {{
      const tbody = document.getElementById('gridBody');
      tbody.innerHTML = '';

      const t = BUNDLE[currentTableKey];
      const total = currentFilteredRecords.length;

      const size = pageSize === 'all' ? total : parseInt(pageSize);
      const totalPages = Math.ceil(total / size) || 1;
      if (currentPage > totalPages) currentPage = totalPages;

      const start = (currentPage - 1) * size;
      const end = Math.min(start + size, total);
      const pageRecords = currentFilteredRecords.slice(start, end);

      pageRecords.forEach(row => {{
        const tr = document.createElement('tr');
        tr.onclick = () => showRowDetails(row);
        t.columns.forEach(col => {{
          const td = document.createElement('td');
          const val = row[col];
          td.innerText = (val !== null && val !== undefined) ? val : '';
          tr.appendChild(td);
        }});
        tbody.appendChild(tr);
      }});

      document.getElementById('paginationInfo').innerText = `Showing ${{total === 0 ? 0 : start + 1}} to ${{end}} of ${{total.toLocaleString()}} rows ${{t.sample_only ? '(sample of 75.4k)' : ''}}`;
      document.getElementById('btnPrevPage').disabled = currentPage <= 1;
      document.getElementById('btnNextPage').disabled = currentPage >= totalPages;
    }}

    function handleSort(col) {{
      if (sortColumn === col) {{
        sortAsc = !sortAsc;
      }} else {{
        sortColumn = col;
        sortAsc = true;
      }}

      currentFilteredRecords.sort((a, b) => {{
        let vA = a[col];
        let vB = b[col];
        if (vA === null || vA === undefined) vA = '';
        if (vB === null || vB === undefined) vB = '';
        if (typeof vA === 'number' && typeof vB === 'number') {{
          return sortAsc ? vA - vB : vB - vA;
        }}
        return sortAsc ? String(vA).localeCompare(String(vB)) : String(vB).localeCompare(String(vA));
      }});

      const t = BUNDLE[currentTableKey];
      renderTableHeaders(t.columns);
      renderTableBody();
    }}

    function filterCurrentTable() {{
      const query = document.getElementById('tableFilterInput').value.toLowerCase();
      const t = BUNDLE[currentTableKey];

      if (!query) {{
        currentFilteredRecords = [...t.records];
      }} else {{
        currentFilteredRecords = t.records.filter(row => {{
          return t.columns.some(col => String(row[col]).toLowerCase().includes(query));
        }});
      }}
      currentPage = 1;
      renderTableBody();
    }}

    function changePageSize() {{
      pageSize = document.getElementById('pageSizeSelect').value;
      currentPage = 1;
      renderTableBody();
    }}

    function prevPage() {{
      if (currentPage > 1) {{
        currentPage--;
        renderTableBody();
      }}
    }}

    function nextPage() {{
      const total = currentFilteredRecords.length;
      const size = pageSize === 'all' ? total : parseInt(pageSize);
      const totalPages = Math.ceil(total / size) || 1;
      if (currentPage < totalPages) {{
        currentPage++;
        renderTableBody();
      }}
    }}

    function showRowDetails(row) {{
      const t = BUNDLE[currentTableKey];
      document.getElementById('modalTitle').innerText = `${{t.name}} Record Details`;
      const grid = document.getElementById('modalGrid');
      grid.innerHTML = '';

      t.columns.forEach(col => {{
        const div = document.createElement('div');
        div.className = 'modal-field';
        div.innerHTML = `
          <label>${{col}}</label>
          <span>${{row[col] !== '' ? row[col] : '<span style="color:var(--text-dim)">(empty)</span>'}}</span>
        `;
        grid.appendChild(div);
      }});

      document.getElementById('rowModal').style.display = 'flex';
    }}

    function closeModal(e) {{
      if (e.target.id === 'rowModal') {{
        document.getElementById('rowModal').style.display = 'none';
      }}
    }}

    function closeModalDirect() {{
      document.getElementById('rowModal').style.display = 'none';
    }}

    function exportCurrentTable(format) {{
      const t = BUNDLE[currentTableKey];
      if (format === 'csv') {{
        let csv = t.columns.join(',') + '\\n';
        currentFilteredRecords.forEach(r => {{
          const rowVals = t.columns.map(c => {{
            let v = String(r[c] || '');
            if (v.includes(',') || v.includes('"') || v.includes('\\n')) {{
              v = '"' + v.replace(/"/g, '""') + '"';
            }}
            return v;
          }});
          csv += rowVals.join(',') + '\\n';
        }});

        const blob = new Blob([csv], {{ type: 'text/csv' }});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = t.name.replace('.csv', '') + '_filtered.csv';
        a.click();
      }}
    }}

    function copyCurrentTableJSON() {{
      navigator.clipboard.writeText(JSON.stringify(currentFilteredRecords, null, 2)).then(() => {{
        alert(`Copied ${{currentFilteredRecords.length}} rows of JSON to clipboard!`);
      }});
    }}

    // Features tab functions
    function filterFeatures(group) {{
      currentFeatureGroup = group;
      document.querySelectorAll('#featurePills .pill-btn').forEach(btn => {{
        if (btn.innerText.toLowerCase().includes(group.toLowerCase()) || (group === 'all' && btn.innerText.includes('All'))) {{
          btn.classList.add('active');
        }} else {{
          btn.classList.remove('active');
        }}
      }});
      renderFeatures();
    }}

    function getTagClass(group) {{
      if (group.includes('History')) return 'tag-history';
      if (group.includes('Calendar')) return 'tag-calendar';
      if (group.includes('Holiday')) return 'tag-holidays';
      if (group.includes('Weather')) return 'tag-weather';
      if (group.includes('Uncollected')) return 'tag-uncollected';
      return 'tag-master';
    }}

    function renderFeatures() {{
      const query = document.getElementById('globalSearchInput').value.toLowerCase();
      const grid = document.getElementById('featuresGrid');
      grid.innerHTML = '';

      const filtered = FEATURES_DATA.filter(item => {{
        const matchGroup = currentFeatureGroup === 'all' || item.group === currentFeatureGroup;
        const matchQuery = !query || item.name.toLowerCase().includes(query) || 
                            item.definition.toLowerCase().includes(query) || 
                            item.group.toLowerCase().includes(query) ||
                            item.models.toLowerCase().includes(query);
        return matchGroup && matchQuery;
      }});

      filtered.forEach(item => {{
        const card = document.createElement('div');
        card.className = 'item-card';
        card.innerHTML = `
          <div class="card-top">
            <span class="card-title">${{item.name}}</span>
            <span class="tag-badge ${{getTagClass(item.group)}}">${{item.group}}</span>
          </div>
          <p class="card-desc">${{item.definition}}</p>
          <div class="card-meta-list">
            <div class="card-meta-row">
              <span>Data Type</span>
              <span>${{item.type}}</span>
            </div>
            <div class="card-meta-row">
              <span>Model Coverage</span>
              <span>${{item.models}}</span>
            </div>
            <div class="card-meta-row">
              <span>Availability</span>
              <span style="color: ${{item.availability === 'Available' ? 'var(--emerald)' : (item.availability === 'Derived' ? 'var(--primary)' : 'var(--text-dim)')}}">${{item.availability}}</span>
            </div>
          </div>
        `;
        grid.appendChild(card);
      }});

      document.getElementById('badge-features').innerText = `${{filtered.length}} Features`;
    }}

    function renderMatrix() {{
      const tbody = document.getElementById('matrixTableBody');
      tbody.innerHTML = '';

      FEATURES_DATA.forEach(item => {{
        const m = item.models;
        const inA = m.includes('A');
        const inB = m.includes('B');
        const inC = m.includes('C');
        const inD = m.includes('D');
        const inF = m.includes('F');
        const inExp3 = m.includes('Exp3');

        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td style="color: #fff; font-weight: 600;">${{item.name}}</td>
          <td><span class="tag-badge ${{getTagClass(item.group)}}">${{item.group}}</span></td>
          <td style="text-align: center;"><span style="color: ${{inA ? 'var(--emerald)' : 'var(--text-dim)'}}; font-weight: 700;">${{inA ? '✓' : '—'}}</span></td>
          <td style="text-align: center;"><span style="color: ${{inB ? 'var(--emerald)' : 'var(--text-dim)'}}; font-weight: 700;">${{inB ? '✓' : '—'}}</span></td>
          <td style="text-align: center;"><span style="color: ${{inC ? 'var(--emerald)' : 'var(--text-dim)'}}; font-weight: 700;">${{inC ? '✓' : '—'}}</span></td>
          <td style="text-align: center;"><span style="color: ${{inD ? 'var(--emerald)' : 'var(--text-dim)'}}; font-weight: 700;">${{inD ? '✓' : '—'}}</span></td>
          <td style="text-align: center; background: rgba(56, 189, 248, 0.05);"><span style="color: ${{inF ? 'var(--primary)' : 'var(--text-dim)'}}; font-size: 1.1rem; font-weight: 800;">${{inF ? '✓' : '—'}}</span></td>
          <td style="text-align: center;"><span style="color: ${{inExp3 ? 'var(--emerald)' : 'var(--text-dim)'}}; font-weight: 700;">${{inExp3 ? '✓' : '—'}}</span></td>
        `;
        tbody.appendChild(tr);
      }});
    }}

    function handleGlobalSearch() {{
      if (activeMainTab === 'features') {{
        renderFeatures();
      }} else if (activeMainTab === 'explorer') {{
        document.getElementById('tableFilterInput').value = document.getElementById('globalSearchInput').value;
        filterCurrentTable();
      }}
    }}

    window.onload = init;
  </script>
</body>
</html>
"""
    
    out1 = HERE / "features_and_masters.html"
    out2 = HERE / "index.html"
    out1.write_text(html_code, encoding="utf-8")
    out2.write_text(html_code, encoding="utf-8")
    print(f"Generated: {out1} ({out1.stat().st_size / 1024:.1f} KB)")
    print(f"Generated: {out2} ({out2.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    build()
