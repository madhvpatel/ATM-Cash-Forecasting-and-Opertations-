"""Build the enhanced Features, Masters, and Cash Replenishment & Optimization Portal.

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
OUTPUT_DIR = HERE / "output"


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
    
    uncensoring_feats = [
        ("zero_balance_snapshot_count", "Number of 15-minute cash inventory snapshots hitting ₹0 balance", "Integer (0..96)", "Demand Uncensoring", "Evaluated in Study", "Available"),
        ("hours_empty", "Calculated hours ATM spent completely empty during operating day (snapshots × 0.25h)", "Float (hours)", "Demand Uncensoring", "Evaluated in Study", "Derived"),
        ("active_fraction", "Operating fraction of day with cash in vault (max(1 - empty/96, 0.20))", "Float (0.2..1.0)", "Demand Uncensoring", "Evaluated in Study", "Derived"),
        ("is_cashout_day", "Binary indicator if machine experienced stockout on target date", "Binary (0/1)", "Demand Uncensoring", "Evaluated in Study", "Derived"),
        ("uncensored_demand", "Run-rate reconstructed true customer demand (Demand / Active Fraction)", "INR Cash", "Demand Uncensoring", "Uncensored Target", "Derived Target"),
    ]

    missing_feats = [
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
    for f in (history_feats + calendar_feats + holiday_feats + weather_feats + uncensoring_feats + missing_feats):
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
  <meta name="description" content="Cash Intelligence Platform: Features Registry, Master Data Explorer, and Automated Cash Replenishment Optimizer.">
  <title>Cash Intelligence · Optimization & Replenishment Portal</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
  <script src="master_data_bundle.js"></script>
  <script src="output/replenishment_portal_bundle.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    :root {{
      --bg-base: #0a0d14;
      --bg-surface: #111622;
      --bg-surface-elevated: #161d2d;
      --bg-card: rgba(22, 29, 45, 0.75);
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
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: radial-gradient(circle at 15% 15%, rgba(56, 189, 248, 0.05), transparent 45%),
                  radial-gradient(circle at 85% 85%, rgba(16, 185, 129, 0.04), transparent 45%);
      pointer-events: none;
      z-index: 0;
    }}

    header {{
      background: rgba(17, 22, 34, 0.85);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--border-subtle);
      padding: 16px 32px;
      position: sticky;
      top: 0;
      z-index: 50;
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
      background: linear-gradient(135deg, #0284c7, #0d9488);
      border-radius: 10px;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 0 16px rgba(56, 189, 248, 0.3);
    }}

    .brand-icon svg {{ width: 22px; height: 22px; color: #fff; }}

    .brand-text h1 {{
      font-size: 1.15rem;
      font-weight: 700;
      color: #fff;
      letter-spacing: -0.02em;
    }}

    .brand-text span {{
      font-size: 0.78rem;
      color: var(--text-muted);
      display: block;
    }}

    .header-stats {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}

    .stat-pill {{
      background: var(--bg-surface-elevated);
      border: 1px solid var(--border-subtle);
      border-radius: 9999px;
      padding: 6px 14px;
      font-size: 0.82rem;
      color: var(--text-muted);
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
      padding: 24px 32px;
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
      font-size: 0.92rem;
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
      min-width: 280px;
    }}

    .search-box input {{
      width: 100%;
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 10px;
      padding: 9px 14px 9px 38px;
      color: #fff;
      font-family: var(--font-sans);
      font-size: 0.88rem;
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

    /* Common Card & Grid Styles */
    .glass-card {{
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 14px;
      padding: 22px;
      backdrop-filter: blur(10px);
    }}

    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
      margin-bottom: 24px;
    }}

    @media (max-width: 1024px) {{
      .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
    }}
    @media (max-width: 640px) {{
      .kpi-grid {{ grid-template-columns: 1fr; }}
    }}

    .kpi-card {{
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 12px;
      padding: 18px 20px;
      display: flex;
      flex-direction: column;
      gap: 6px;
      position: relative;
      overflow: hidden;
    }}

    .kpi-card::before {{
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      width: 4px;
      height: 100%;
      background: var(--primary);
    }}

    .kpi-card.emerald::before {{ background: var(--emerald); }}
    .kpi-card.amber::before {{ background: var(--amber); }}
    .kpi-card.indigo::before {{ background: var(--indigo); }}

    .kpi-label {{
      font-size: 0.78rem;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      font-weight: 600;
    }}

    .kpi-value {{
      font-size: 1.65rem;
      font-weight: 800;
      color: #fff;
      font-family: var(--font-mono);
      letter-spacing: -0.02em;
    }}

    .kpi-subtext {{
      font-size: 0.78rem;
      color: var(--emerald);
      display: flex;
      align-items: center;
      gap: 4px;
    }}

    /* Control Panel (Sliders) */
    .control-panel {{
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 14px;
      padding: 20px 24px;
      margin-bottom: 24px;
    }}

    .control-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 18px;
    }}

    .control-title {{
      font-size: 1rem;
      font-weight: 700;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 8px;
    }}

    .sliders-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 20px;
      align-items: end;
    }}

    .slider-item {{
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}

    .slider-label-row {{
      display: flex;
      justify-content: space-between;
      font-size: 0.82rem;
      font-weight: 600;
      color: var(--text-muted);
    }}

    .slider-val {{
      color: var(--primary);
      font-family: var(--font-mono);
      font-weight: 700;
    }}

    input[type=range] {{
      -webkit-appearance: none;
      width: 100%;
      height: 6px;
      border-radius: 9999px;
      background: var(--bg-surface-elevated);
      outline: none;
    }}

    input[type=range]::-webkit-slider-thumb {{
      -webkit-appearance: none;
      appearance: none;
      width: 18px;
      height: 18px;
      border-radius: 50%;
      background: var(--primary);
      cursor: pointer;
      box-shadow: 0 0 10px var(--primary);
      border: 2px solid #fff;
    }}

    .custom-select {{
      background: var(--bg-surface-elevated);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 8px 12px;
      color: #fff;
      font-family: var(--font-sans);
      font-size: 0.85rem;
      outline: none;
      cursor: pointer;
      width: 100%;
    }}

    /* Table Styles */
    .master-grid {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
    }}

    .master-grid th {{
      background: var(--bg-surface-elevated);
      color: var(--text-muted);
      font-weight: 600;
      text-align: left;
      padding: 12px 16px;
      border-bottom: 1px solid var(--border-subtle);
      white-space: nowrap;
    }}

    .master-grid td {{
      padding: 12px 16px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
      color: var(--text-main);
    }}

    .master-grid tr:hover td {{
      background: rgba(255, 255, 255, 0.02);
    }}

    .badge-tag {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 6px;
      font-size: 0.74rem;
      font-weight: 600;
    }}

    .tag-emerald {{ background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }}
    .tag-primary {{ background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); }}
    .tag-amber {{ background: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); }}
    .tag-rose {{ background: rgba(244, 63, 94, 0.15); color: #f43f5e; border: 1px solid rgba(244, 63, 94, 0.3); }}

    /* Layout for charts side-by-side */
    .two-col-grid {{
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 24px;
      margin-bottom: 24px;
    }}

    @media (max-width: 1024px) {{
      .two-col-grid {{ grid-template-columns: 1fr; }}
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
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 4px;
      max-height: 780px;
      overflow-y: auto;
    }}

    .sidebar-title {{
      font-size: 0.75rem;
      font-weight: 700;
      color: var(--text-dim);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      padding: 8px 12px;
    }}

    .menu-item {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 10px 14px;
      border-radius: 8px;
      color: var(--text-muted);
      cursor: pointer;
      transition: all 0.15s;
      border: 1px solid transparent;
      background: transparent;
      text-align: left;
      font-family: var(--font-sans);
      font-size: 0.86rem;
      font-weight: 500;
    }}

    .menu-item:hover {{
      background: var(--bg-surface-elevated);
      color: #fff;
    }}

    .menu-item.active {{
      background: var(--primary-glow);
      color: var(--primary);
      border-color: rgba(56, 189, 248, 0.3);
      font-weight: 600;
    }}

    .menu-item-left {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}

    .menu-item-count {{
      font-size: 0.72rem;
      color: var(--text-dim);
      font-family: var(--font-mono);
    }}

    .master-view-panel {{
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 14px;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }}

    .panel-header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      border-bottom: 1px solid var(--border-subtle);
      padding-bottom: 18px;
      gap: 16px;
      flex-wrap: wrap;
    }}

    .panel-title-area h3 {{
      font-size: 1.25rem;
      font-weight: 700;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 8px;
    }}

    .panel-title-area p {{
      color: var(--text-muted);
      font-size: 0.88rem;
      margin-top: 4px;
    }}

    .panel-path-tag {{
      display: inline-block;
      font-family: var(--font-mono);
      font-size: 0.74rem;
      padding: 3px 8px;
      background: var(--bg-surface-elevated);
      border: 1px solid var(--border-subtle);
      border-radius: 6px;
      color: var(--text-dim);
      margin-top: 8px;
    }}

    .panel-actions {{
      display: flex;
      gap: 8px;
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
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.2s;
    }}

    .btn-action:hover {{
      border-color: var(--primary);
      color: var(--primary);
    }}

    .table-toolbar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      flex-wrap: wrap;
    }}

    .table-search {{
      position: relative;
      flex: 1;
      min-width: 240px;
    }}

    .table-search input {{
      width: 100%;
      background: var(--bg-surface-elevated);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 8px 12px 8px 34px;
      color: #fff;
      font-size: 0.84rem;
      outline: none;
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

    .data-table-wrapper {{
      overflow-x: auto;
      border: 1px solid var(--border-subtle);
      border-radius: 10px;
      max-height: 540px;
    }}

    .pagination-bar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 0.82rem;
      color: var(--text-muted);
    }}

    .pagination-btns {{
      display: flex;
      gap: 6px;
    }}

    .btn-page {{
      background: var(--bg-surface-elevated);
      border: 1px solid var(--border-subtle);
      color: #fff;
      padding: 5px 12px;
      border-radius: 6px;
      font-size: 0.8rem;
      cursor: pointer;
    }}

    /* Grid cards for catalog */
    .filter-pills {{
      display: flex;
      gap: 8px;
      margin-bottom: 20px;
      flex-wrap: wrap;
    }}

    .pill-btn {{
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      color: var(--text-muted);
      padding: 8px 14px;
      border-radius: 9999px;
      font-size: 0.82rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s;
    }}

    .pill-btn:hover {{ color: #fff; border-color: var(--border-hover); }}
    .pill-btn.active {{
      background: var(--primary-glow);
      color: var(--primary);
      border-color: rgba(56, 189, 248, 0.4);
    }}

    .grid-cards {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(290px, 1fr));
      gap: 16px;
    }}

    .item-card {{
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 12px;
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 12px;
      transition: all 0.2s;
    }}

    .item-card:hover {{
      border-color: var(--border-hover);
      transform: translateY(-2px);
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }}

    .card-top {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 8px;
    }}

    .card-title {{
      font-size: 0.95rem;
      font-weight: 700;
      color: #fff;
      font-family: var(--font-mono);
    }}

    .card-desc {{
      font-size: 0.82rem;
      color: var(--text-muted);
      line-height: 1.45;
      flex: 1;
    }}

    .card-meta-list {{
      display: flex;
      flex-direction: column;
      gap: 6px;
      border-top: 1px solid rgba(255, 255, 255, 0.05);
      padding-top: 10px;
      font-size: 0.76rem;
    }}

    .card-meta-row {{
      display: flex;
      justify-content: space-between;
      color: var(--text-dim);
    }}

    .card-meta-row span:last-child {{
      color: var(--text-main);
      font-weight: 500;
    }}

    footer {{
      background: var(--bg-surface);
      border-top: 1px solid var(--border-subtle);
      padding: 20px 32px;
      text-align: center;
      font-size: 0.82rem;
      color: var(--text-dim);
      margin-top: 40px;
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
        <span>Master Data & Automated Replenishment Optimizer</span>
      </div>
    </a>

    <div class="header-stats">
      <div class="stat-pill">
        <div class="dot"></div>
        <span><strong>13</strong> Master Tables</span>
      </div>
      <div class="stat-pill">
        <div class="dot"></div>
        <span><strong>42</strong> Features</span>
      </div>
      <div class="stat-pill">
        <div class="dot"></div>
        <span><strong>440</strong> Sim Cycles</span>
      </div>
    </div>
  </header>

  <main class="container">
    <div class="tabs-wrapper">
      <div class="tabs" role="tablist">
        <button id="tab-replenishment-btn" class="tab-btn active" onclick="switchMainTab('replenishment')">
          ⚡ Cash Replenishment & Optimization
          <span class="badge-count" style="background: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3);">Engine Ready</span>
        </button>
        <button id="tab-explorer-btn" class="tab-btn" onclick="switchMainTab('explorer')">
          Master Data Explorer (Live Rows)
          <span class="badge-count" id="badge-total-tables">13 Tables</span>
        </button>
        <button id="tab-features-btn" class="tab-btn" onclick="switchMainTab('features')">
          Input Features Catalog
          <span class="badge-count" id="badge-features">42 Features</span>
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

    <!-- TAB 1: REPLENISHMENT & CASH OPTIMIZATION ENGINE -->
    <div id="section-replenishment">
      <!-- Executive KPI Cards -->
      <div class="kpi-grid">
        <div class="kpi-card">
          <span class="kpi-label">Simulated Portfolio Cost</span>
          <span class="kpi-value" id="kpiTotalCost">₹30.57 L</span>
          <span class="kpi-subtext" style="color: var(--text-muted);">Across 440 Replenishment Cycles</span>
        </div>
        <div class="kpi-card emerald">
          <span class="kpi-label">Net Rupee Savings vs Naive</span>
          <span class="kpi-value" id="kpiSavings" style="color: var(--emerald);">+₹6.76 L</span>
          <span class="kpi-subtext" id="kpiSavingsPct">↑ 18.11% lower logistics & penalty cost</span>
        </div>
        <div class="kpi-card indigo">
          <span class="kpi-label">Stockout Outage Rate</span>
          <span class="kpi-value" id="kpiStockoutRate">9.55%</span>
          <span class="kpi-subtext" style="color: var(--primary);">vs 12.95% on naive baseline</span>
        </div>
        <div class="kpi-card amber">
          <span class="kpi-label">Realized Capital CER</span>
          <span class="kpi-value" id="kpiCER">86.8%</span>
          <span class="kpi-subtext" style="color: var(--amber);">Optimal vault capital utilization</span>
        </div>
      </div>

      <!-- Interactive Scenario Modeler (Sliders) -->
      <div class="control-panel">
        <div class="control-header">
          <div class="control-title">
            <span>⚙️</span> Interactive Financial Scenario Modeler (Live Rupee Recalculation)
          </div>
          <button class="btn-action" onclick="resetCostSliders()">
            ↺ Reset Defaults
          </button>
        </div>
        <div class="sliders-grid">
          <div class="slider-item">
            <div class="slider-label-row">
              <span>Annual Holding Rate (% p.a.)</span>
              <span class="slider-val" id="valHoldingRate">6.5%</span>
            </div>
            <input type="range" id="sliderHoldingRate" min="3.0" max="15.0" step="0.5" value="6.5" oninput="updateScenarioModel()">
          </div>

          <div class="slider-item">
            <div class="slider-label-row">
              <span>CIT Trip Cost (₹/visit)</span>
              <span class="slider-val" id="valCitCost">₹2,000</span>
            </div>
            <input type="range" id="sliderCitCost" min="500" max="5000" step="250" value="2000" oninput="updateScenarioModel()">
          </div>

          <div class="slider-item">
            <div class="slider-label-row">
              <span>RBI Stockout Penalty (₹/event)</span>
              <span class="slider-val" id="valPenaltyCost">₹10,000</span>
            </div>
            <input type="range" id="sliderPenaltyCost" min="2000" max="25000" step="1000" value="10000" oninput="updateScenarioModel()">
          </div>

          <div class="slider-item">
            <div class="slider-label-row">
              <span>Emergency Refill Premium (₹)</span>
              <span class="slider-val" id="valEmergencyCost">₹3,500</span>
            </div>
            <input type="range" id="sliderEmergencyCost" min="1000" max="8000" step="500" value="3500" oninput="updateScenarioModel()">
          </div>

          <div class="slider-item">
            <div class="slider-label-row">
              <span>Target Service Level SLA</span>
              <span class="slider-val" id="valServiceLevel">99.0%</span>
            </div>
            <input type="range" id="sliderServiceLevel" min="90.0" max="99.9" step="0.1" value="99.0" oninput="updateScenarioModel()">
          </div>

          <div class="slider-item">
            <div class="slider-label-row">
              <span>Select ATM Terminal</span>
            </div>
            <select class="custom-select" id="selectAtmFilter" onchange="changeAtmScheduleView()">
              <option value="all">Combined 8-ATM Portfolio</option>
              <option value="1">ATM 1 · Mumbai Railway Concourse (₹9.1L/day)</option>
              <option value="2">ATM 2 · Mumbai Airport Terminal (₹7.9L/day)</option>
              <option value="3">ATM 3 · Pune City Centre Mall (₹7.8L/day)</option>
              <option value="4">ATM 4 · Pune Education Complex (₹2.7L/day)</option>
              <option value="5">ATM 5 · Nagpur Fuel Plaza (₹4.9L/day)</option>
              <option value="6">ATM 6 · Nagpur Tourist Zone (₹7.4L/day)</option>
              <option value="7" selected>ATM 7 · Nashik Metro Concourse (₹10.7L/day)</option>
              <option value="8">ATM 8 · Nashik Hospital (₹3.1L/day)</option>
            </select>
          </div>
        </div>
      </div>

      <!-- Side-by-Side Model Benchmark Table -->
      <div class="glass-card" style="margin-bottom: 24px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
          <div>
            <h3 style="font-size: 1.1rem; color: #fff; font-weight: 700;">Side-by-Side Model Financial Benchmark</h3>
            <p style="font-size: 0.82rem; color: var(--text-muted);">440 rolling replenishment cycles across 8 Maharashtra typologies. Recalculates dynamically with slider parameters.</p>
          </div>
          <span class="badge-tag tag-emerald">Audited Backtest Engine</span>
        </div>
        <div class="data-table-wrapper">
          <table class="master-grid" id="benchmarkTable">
            <thead>
              <tr>
                <th>Replenishment Model</th>
                <th>7-Day WAPE</th>
                <th>Stockout Cycles</th>
                <th>Outage Rate</th>
                <th>Unmet Shortfall</th>
                <th>Avg Load</th>
                <th>Holding Cost</th>
                <th>CIT Logistics</th>
                <th>Penalties & Emergencies</th>
                <th style="background: rgba(56, 189, 248, 0.1);">Total Simulated Cost</th>
                <th>Net Rupee Savings</th>
              </tr>
            </thead>
            <tbody id="benchmarkTableBody">
              <!-- Rendered by JS -->
            </tbody>
          </table>
        </div>
      </div>

      <!-- Charts Two-Column Grid -->
      <div class="two-col-grid">
        <!-- 7-Day Schedule & Drawdown Chart -->
        <div class="glass-card">
          <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 14px;">
            <div>
              <h3 style="font-size: 1.05rem; color: #fff; font-weight: 700;" id="scheduleChartTitle">Weekly Replenishment Schedule & Vault Drawdown</h3>
              <p style="font-size: 0.8rem; color: var(--text-muted);" id="scheduleChartSub">Recommended Cash Load ($L^*$), Buffer ($B$), and Ending Inventory Trajectory</p>
            </div>
            <div style="display: flex; gap: 6px;" id="scheduleBadges">
              <!-- Injected by JS -->
            </div>
          </div>
          <div style="height: 310px; position: relative;">
            <canvas id="replenishmentChart"></canvas>
          </div>
        </div>

        <!-- Cost Frontier Trade-off Chart -->
        <div class="glass-card">
          <div style="margin-bottom: 14px;">
            <h3 style="font-size: 1.05rem; color: #fff; font-weight: 700;">Service Level vs Cost Optimization Curve</h3>
            <p style="font-size: 0.8rem; color: var(--text-muted);">Economic trade-off: Holding interest climbs while stockout risk drops (Optimal at 99.0% SLA)</p>
          </div>
          <div style="height: 310px; position: relative;">
            <canvas id="costFrontierChart"></canvas>
          </div>
        </div>
      </div>

      <!-- Demand Uncensoring & Stockout Audit Table -->
      <div class="glass-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
          <div>
            <h3 style="font-size: 1.05rem; color: #fff; font-weight: 700;">Demand Uncensoring & Historical Cashout Audit</h3>
            <p style="font-size: 0.8rem; color: var(--text-muted);">Quantifies historical zero-inventory periods where withdrawals were suppressed, and verifies un-censoring recovery.</p>
          </div>
          <span class="badge-tag tag-primary">Run-Rate Active Fraction Correction</span>
        </div>
        <div class="data-table-wrapper">
          <table class="master-grid" id="auditTable">
            <thead>
              <tr>
                <th>ATM Terminal</th>
                <th>Typology & City</th>
                <th>Observed Days</th>
                <th>Cashout Days</th>
                <th>Cashout Rate</th>
                <th>Avg Empty Hours</th>
                <th>Observed Demand</th>
                <th>Uncensored True Demand</th>
                <th>Suppressed Lost Demand</th>
                <th>Demand Gain</th>
              </tr>
            </thead>
            <tbody id="auditTableBody">
              <!-- Rendered by JS -->
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- TAB 2: MASTER DATA LIVE EXPLORER -->
    <div id="section-explorer" style="display: none;">
      <div class="master-layout">
        <div class="sidebar-menu" id="sidebarMenu">
          <div class="sidebar-title">Select Master Table</div>
        </div>

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

          <div class="data-table-wrapper" id="gridWrapper">
            <table class="master-grid" id="currentMasterGrid">
              <thead id="gridHead"></thead>
              <tbody id="gridBody"></tbody>
            </table>
          </div>

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

    <!-- TAB 3: INPUT FEATURES CATALOG -->
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

      <div class="grid-cards" id="featuresGrid"></div>
    </div>

    <!-- TAB 4: MODEL FEATURE MATRIX -->
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
          <tbody id="matrixTableBody"></tbody>
        </table>
      </div>
    </div>
  </main>

  <footer>
    <p>Cash Intelligence Platform · CMS2 Evaluation Environment · Live Master Explorer loaded from <a href="master_data_bundle.js">master_data_bundle.js</a> and <a href="output/replenishment_portal_bundle.js">replenishment_portal_bundle.js</a></p>
  </footer>

  <script>
    const FEATURES_DATA = {json.dumps(all_features)};
    const BUNDLE = window.MASTER_DATA_BUNDLE || {{}};
    const REPL_BUNDLE = window.REPLENISHMENT_PORTAL_BUNDLE || {{}};
    
    let activeMainTab = 'replenishment';
    let currentTableKey = 'terminal_metadata';
    let currentFilteredRecords = [];
    let currentPage = 1;
    let pageSize = 25;
    let sortColumn = null;
    let sortAsc = true;
    let currentFeatureGroup = 'all';

    let scheduleChartInstance = null;
    let costFrontierChartInstance = null;

    function init() {{
      renderSidebar();
      loadTable(currentTableKey);
      renderFeatures();
      renderMatrix();
      initReplenishmentTab();
    }}

    function switchMainTab(tab) {{
      activeMainTab = tab;
      document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
      document.getElementById('tab-' + tab + '-btn').classList.add('active');

      document.getElementById('section-replenishment').style.display = tab === 'replenishment' ? 'block' : 'none';
      document.getElementById('section-explorer').style.display = tab === 'explorer' ? 'block' : 'none';
      document.getElementById('section-features').style.display = tab === 'features' ? 'block' : 'none';
      document.getElementById('section-matrix').style.display = tab === 'matrix' ? 'block' : 'none';

      if (tab === 'replenishment') {{
        setTimeout(renderReplenishmentCharts, 100);
      }}
    }}

    /* === REPLENISHMENT TAB LOGIC === */
    function initReplenishmentTab() {{
      updateScenarioModel();
      renderAuditTable();
      renderReplenishmentCharts();
    }}

    function resetCostSliders() {{
      document.getElementById('sliderHoldingRate').value = 6.5;
      document.getElementById('sliderCitCost').value = 2000;
      document.getElementById('sliderPenaltyCost').value = 10000;
      document.getElementById('sliderEmergencyCost').value = 3500;
      document.getElementById('sliderServiceLevel').value = 99.0;
      updateScenarioModel();
    }}

    function updateScenarioModel() {{
      const holdingRate = parseFloat(document.getElementById('sliderHoldingRate').value);
      const citCost = parseFloat(document.getElementById('sliderCitCost').value);
      const penaltyCost = parseFloat(document.getElementById('sliderPenaltyCost').value);
      const emergencyCost = parseFloat(document.getElementById('sliderEmergencyCost').value);
      const serviceLevel = parseFloat(document.getElementById('sliderServiceLevel').value);

      document.getElementById('valHoldingRate').innerText = holdingRate.toFixed(1) + '%';
      document.getElementById('valCitCost').innerText = '₹' + citCost.toLocaleString();
      document.getElementById('valPenaltyCost').innerText = '₹' + penaltyCost.toLocaleString();
      document.getElementById('valEmergencyCost').innerText = '₹' + emergencyCost.toLocaleString();
      document.getElementById('valServiceLevel').innerText = serviceLevel.toFixed(1) + '%';

      // Recompute side-by-side benchmark table with custom sliders
      renderBenchmarkTable(holdingRate, citCost, penaltyCost, emergencyCost, serviceLevel);
      renderReplenishmentCharts();
    }}

    function renderBenchmarkTable(holdingRate, citCost, penaltyCost, emergencyCost, serviceLevel) {{
      const tbody = document.getElementById('benchmarkTableBody');
      tbody.innerHTML = '';

      const summaryList = REPL_BUNDLE.overall_summary || [
        {{ "model": "pure_exogenous", "total_cycles": 440, "forecast_7day_wape_pct": 7.72, "stockout_cycles": 42, "stockout_rate_pct": 9.55, "total_shortfall_lakh": 124.33, "mean_weekly_load_lakh": 53.52, "avg_realized_cer": 0.868, "holding_cost_inr": 1485441, "cit_logistics_cost_inr": 880000, "penalties_and_emergencies_inr": 567000, "total_cost_inr": 3056772, "savings_vs_naive_inr": 675841, "savings_pct": 18.11 }},
        {{ "model": "lgb_calendar_holidays_uncensored", "total_cycles": 440, "forecast_7day_wape_pct": 7.65, "stockout_cycles": 44, "stockout_rate_pct": 10.00, "total_shortfall_lakh": 121.36, "mean_weekly_load_lakh": 53.38, "avg_realized_cer": 0.870, "holding_cost_inr": 1478669, "cit_logistics_cost_inr": 880000, "penalties_and_emergencies_inr": 594000, "total_cost_inr": 3074027, "savings_vs_naive_inr": 658586, "savings_pct": 17.64 }},
        {{ "model": "lgb_calendar_holidays_censored", "total_cycles": 440, "forecast_7day_wape_pct": 7.89, "stockout_cycles": 47, "stockout_rate_pct": 10.68, "total_shortfall_lakh": 126.02, "mean_weekly_load_lakh": 53.29, "avg_realized_cer": 0.871, "holding_cost_inr": 1474507, "cit_logistics_cost_inr": 880000, "penalties_and_emergencies_inr": 634500, "total_cost_inr": 3115029, "savings_vs_naive_inr": 617584, "savings_pct": 16.55 }},
        {{ "model": "naive_last_week", "total_cycles": 440, "forecast_7day_wape_pct": 27.04, "stockout_cycles": 57, "stockout_rate_pct": 12.95, "total_shortfall_lakh": 226.11, "mean_weekly_load_lakh": 59.36, "avg_realized_cer": 0.783, "holding_cost_inr": 1818501, "cit_logistics_cost_inr": 880000, "penalties_and_emergencies_inr": 808000, "total_cost_inr": 3732613, "savings_vs_naive_inr": 0, "savings_pct": 0.0 }}
      ];

      // Scaling factors relative to default assumptions (6.5% holding, 2000 cit, 10000 penalty, 3500 emergency)
      const holdRatio = holdingRate / 6.5;
      const citRatio = citCost / 2000.0;
      const penRatio = (penaltyCost + emergencyCost) / 13500.0;

      // Recalculate costs
      const recomputed = summaryList.map(item => {{
        const adjHold = item.holding_cost_inr * holdRatio;
        const adjCit = item.cit_logistics_cost_inr * citRatio;
        const adjPen = item.penalties_and_emergencies_inr * penRatio;
        const total = adjHold + adjCit + adjPen;
        return {{ ...item, adjHold, adjCit, adjPen, total }};
      }});

      // Find naive baseline
      const naiveTotal = (recomputed.find(x => x.model === 'naive_last_week') || {{ total: 1 }}).total;

      recomputed.forEach(item => {{
        const savings = naiveTotal - item.total;
        const savingsPct = (savings / naiveTotal) * 100;
        item.adjSavings = savings;
        item.adjSavingsPct = savingsPct;
      }});

      recomputed.sort((a, b) => a.total - b.total);

      // Update KPI cards based on the best model
      const best = recomputed[0];
      document.getElementById('kpiTotalCost').innerText = '₹' + (best.total / 1e5).toFixed(2) + ' L';
      document.getElementById('kpiSavings').innerText = '+' + '₹' + (best.adjSavings / 1e5).toFixed(2) + ' L';
      document.getElementById('kpiSavingsPct').innerText = '↑ ' + best.adjSavingsPct.toFixed(2) + '% lower logistics & penalty cost';
      document.getElementById('kpiStockoutRate').innerText = best.stockout_rate_pct.toFixed(2) + '%';
      document.getElementById('kpiCER').innerText = (best.avg_realized_cer * 100).toFixed(1) + '%';

      recomputed.forEach(item => {{
        const isBest = item.model === best.model;
        const isNaive = item.model === 'naive_last_week';
        let modelLabel = item.model;
        if (item.model === 'pure_exogenous') modelLabel = '⭐ Pure Exogenous (Calendar + Holidays)';
        else if (item.model === 'lgb_calendar_holidays_uncensored') modelLabel = '⚡ LightGBM Full Stack (Uncensored Target)';
        else if (item.model === 'lgb_calendar_holidays_censored') modelLabel = 'LightGBM Full Stack (Censored Data)';
        else if (item.model === 'naive_last_week') modelLabel = 'Naive Baseline (Previous 7-Day Copy)';

        const tr = document.createElement('tr');
        if (isBest) tr.style.background = 'rgba(16, 185, 129, 0.05)';

        tr.innerHTML = `
          <td style="color: ${{isBest ? 'var(--emerald)' : (isNaive ? 'var(--text-dim)' : '#fff')}}; font-weight: 700;">${{modelLabel}}</td>
          <td style="font-family: var(--font-mono); font-weight: 600; color: ${{item.forecast_7day_wape_pct < 10 ? 'var(--emerald)' : 'var(--rose)'}}">${{item.forecast_7day_wape_pct.toFixed(2)}}%</td>
          <td style="font-family: var(--font-mono);">${{item.stockout_cycles}}</td>
          <td style="font-family: var(--font-mono); color: ${{item.stockout_rate_pct <= 10 ? 'var(--emerald)' : 'var(--amber)'}}">${{item.stockout_rate_pct.toFixed(2)}}%</td>
          <td style="font-family: var(--font-mono);">₹${{item.total_shortfall_lakh.toFixed(1)}} L</td>
          <td style="font-family: var(--font-mono);">₹${{item.mean_weekly_load_lakh.toFixed(1)}} L</td>
          <td style="font-family: var(--font-mono);">₹${{Math.round(item.adjHold).toLocaleString()}}</td>
          <td style="font-family: var(--font-mono);">₹${{Math.round(item.adjCit).toLocaleString()}}</td>
          <td style="font-family: var(--font-mono); color: ${{item.stockout_cycles <= 44 ? 'var(--emerald)' : 'var(--rose)'}}">₹${{Math.round(item.adjPen).toLocaleString()}}</td>
          <td style="font-family: var(--font-mono); font-weight: 800; color: ${{isBest ? 'var(--primary)' : '#fff'}}; background: rgba(56, 189, 248, 0.06);">₹${{Math.round(item.total).toLocaleString()}}</td>
          <td style="font-family: var(--font-mono); font-weight: 700; color: ${{item.adjSavings > 0 ? 'var(--emerald)' : 'var(--text-dim)'}}">
            ${{item.adjSavings > 0 ? '+₹' + Math.round(item.adjSavings).toLocaleString() + ' (' + item.adjSavingsPct.toFixed(1) + '%)' : 'Baseline'}}
          </td>
        `;
        tbody.appendChild(tr);
      }});
    }}

    function changeAtmScheduleView() {{
      renderReplenishmentCharts();
    }}

    function renderReplenishmentCharts() {{
      const tid = document.getElementById('selectAtmFilter').value;
      const schedules = REPL_BUNDLE.schedules_by_atm || {{}};
      const costCurves = REPL_BUNDLE.cost_curves_by_atm || {{}};

      // If 'all', use ATM 7 as the featured representative
      const activeTid = tid === 'all' ? '7' : tid;
      const sched = schedules[activeTid] || {{
        terminal_id: 7, site_type: "Metro Concourse", city: "Nashik",
        dates: ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7"],
        daily_preds: [1120000, 1080000, 1050000, 1140000, 1190000, 1250000, 1220000],
        sum_pred: 8050000, safety_buffer: 780000, recommended_load: 8830000, capacity: 10000000,
        projected_inventory: [7710000, 6630000, 5580000, 4440000, 3250000, 2000000, 780000],
        stockout_probability_pct: 0.8
      }};

      document.getElementById('scheduleChartTitle').innerText = `ATM ${{sched.terminal_id}} (${{sched.site_type}}, ${{sched.city}}) · Weekly Schedule`;
      document.getElementById('scheduleChartSub').innerText = `Forecasted Demand: ₹${{(sched.sum_pred/1e5).toFixed(1)}}L · Vault Limit: ₹${{(sched.capacity/1e5).toFixed(1)}}L`;

      const badgesDiv = document.getElementById('scheduleBadges');
      badgesDiv.innerHTML = `
        <span class="badge-tag tag-primary">Load: ₹${{(sched.recommended_load/1e5).toFixed(1)}}L</span>
        <span class="badge-tag tag-emerald">Buffer: ₹${{(sched.safety_buffer/1e5).toFixed(1)}}L</span>
        <span class="badge-tag tag-amber">Outage Risk: ${{sched.stockout_probability_pct}}%</span>
      `;

      // 1. Render Replenishment Schedule & Inventory Trajectory Chart
      const ctx1 = document.getElementById('replenishmentChart').getContext('2d');
      if (scheduleChartInstance) scheduleChartInstance.destroy();

      const labels = sched.dates.map(d => d.slice(5)); // MM-DD

      scheduleChartInstance = new Chart(ctx1, {{
        type: 'bar',
        data: {{
          labels: labels,
          datasets: [
            {{
              type: 'bar',
              label: 'Daily Forecast Demand (₹)',
              data: sched.daily_preds,
              backgroundColor: 'rgba(56, 189, 248, 0.45)',
              borderColor: '#38bdf8',
              borderWidth: 1.5,
              borderRadius: 6,
              yAxisID: 'y'
            }},
            {{
              type: 'line',
              label: 'Projected Ending Vault Cash (₹)',
              data: sched.projected_inventory,
              borderColor: '#10b981',
              backgroundColor: 'rgba(16, 185, 129, 0.1)',
              borderWidth: 3,
              fill: true,
              tension: 0.25,
              pointBackgroundColor: '#10b981',
              pointRadius: 4,
              yAxisID: 'y'
            }}
          ]
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          plugins: {{
            legend: {{ labels: {{ color: '#94a3b8', font: {{ family: "'Plus Jakarta Sans', sans-serif", size: 11 }} }} }},
            tooltip: {{
              callbacks: {{
                label: function(c) {{ return c.dataset.label + ': ₹' + Number(c.raw).toLocaleString(); }}
              }}
            }}
          }},
          scales: {{
            x: {{ grid: {{ color: 'rgba(255, 255, 255, 0.05)' }}, ticks: {{ color: '#64748b' }} }},
            y: {{
              grid: {{ color: 'rgba(255, 255, 255, 0.05)' }},
              ticks: {{
                color: '#64748b',
                callback: function(v) {{ return '₹' + (v/1e5).toFixed(0) + 'L'; }}
              }}
            }}
          }}
        }}
      }});

      // 2. Render Cost Frontier Trade-Off Curve Chart
      const curveData = costCurves[activeTid] || [
        {{ "service_level": 0.9, "service_level_pct": "90.0%", "holding_cost": 3120, "stockout_cost": 1350, "total_cost": 6470 }},
        {{ "service_level": 0.95, "service_level_pct": "95.0%", "holding_cost": 3540, "stockout_cost": 675, "total_cost": 6215 }},
        {{ "service_level": 0.98, "service_level_pct": "98.0%", "holding_cost": 3980, "stockout_cost": 270, "total_cost": 6250 }},
        {{ "service_level": 0.99, "service_level_pct": "99.0%", "holding_cost": 4350, "stockout_cost": 135, "total_cost": 6485 }},
        {{ "service_level": 0.995, "service_level_pct": "99.5%", "holding_cost": 4820, "stockout_cost": 67, "total_cost": 6887 }},
        {{ "service_level": 0.999, "service_level_pct": "99.9%", "holding_cost": 5740, "stockout_cost": 14, "total_cost": 7754 }}
      ];

      const ctx2 = document.getElementById('costFrontierChart').getContext('2d');
      if (costFrontierChartInstance) costFrontierChartInstance.destroy();

      costFrontierChartInstance = new Chart(ctx2, {{
        type: 'line',
        data: {{
          labels: curveData.map(d => d.service_level_pct),
          datasets: [
            {{
              label: 'Total Cost (₹)',
              data: curveData.map(d => d.total_cost),
              borderColor: '#38bdf8',
              backgroundColor: 'rgba(56, 189, 248, 0.1)',
              borderWidth: 3,
              fill: true,
              pointBackgroundColor: '#38bdf8',
              pointRadius: 5
            }},
            {{
              label: 'Holding Cost (₹)',
              data: curveData.map(d => d.holding_cost),
              borderColor: '#f59e0b',
              borderWidth: 2,
              borderDash: [5, 5],
              pointRadius: 3
            }},
            {{
              label: 'Stockout Risk Cost (₹)',
              data: curveData.map(d => d.stockout_cost),
              borderColor: '#f43f5e',
              borderWidth: 2,
              borderDash: [5, 5],
              pointRadius: 3
            }}
          ]
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          plugins: {{
            legend: {{ labels: {{ color: '#94a3b8', font: {{ family: "'Plus Jakarta Sans', sans-serif", size: 11 }} }} }},
            tooltip: {{
              callbacks: {{
                label: function(c) {{ return c.dataset.label + ': ₹' + Math.round(c.raw).toLocaleString(); }}
              }}
            }}
          }},
          scales: {{
            x: {{ grid: {{ color: 'rgba(255, 255, 255, 0.05)' }}, ticks: {{ color: '#64748b' }} }},
            y: {{
              grid: {{ color: 'rgba(255, 255, 255, 0.05)' }},
              ticks: {{
                color: '#64748b',
                callback: function(v) {{ return '₹' + Math.round(v).toLocaleString(); }}
              }}
            }}
          }}
        }}
      }});
    }}

    function renderAuditTable() {{
      const tbody = document.getElementById('auditTableBody');
      tbody.innerHTML = '';
      const auditList = REPL_BUNDLE.terminal_audit || [];

      auditList.forEach(item => {{
        const tr = document.createElement('tr');
        const hasCo = item.cashout_days > 0;
        tr.innerHTML = `
          <td style="color: #fff; font-weight: 700; font-family: var(--font-mono);">ATM ${{item.terminal_id}}</td>
          <td>${{item.site_type}} <small style="color:var(--text-dim)">(${{item.city}})</small></td>
          <td style="font-family: var(--font-mono);">${{item.total_observed_days}}</td>
          <td style="font-family: var(--font-mono); color: ${{hasCo ? 'var(--rose)' : 'var(--emerald)'}}; font-weight: 700;">${{item.cashout_days}}</td>
          <td style="font-family: var(--font-mono);">${{item.cashout_rate_pct.toFixed(2)}}%</td>
          <td style="font-family: var(--font-mono);">${{item.avg_hours_empty_when_short.toFixed(1)}} hrs</td>
          <td style="font-family: var(--font-mono);">₹${{item.observed_demand_lakh.toFixed(1)}} L</td>
          <td style="font-family: var(--font-mono); color: var(--emerald); font-weight: 600;">₹${{item.uncensored_demand_lakh.toFixed(1)}} L</td>
          <td style="font-family: var(--font-mono); color: ${{hasCo ? 'var(--amber)' : 'var(--text-dim)'}}">₹${{item.lost_demand_lakh.toFixed(1)}} L</td>
          <td style="font-family: var(--font-mono); font-weight: 700; color: ${{hasCo ? 'var(--emerald)' : 'var(--text-dim)'}}">+${{item.suppression_pct.toFixed(2)}}%</td>
        `;
        tbody.appendChild(tr);
      }});
    }}

    /* === EXISTING MASTER DATA EXPLORER & FEATURES LOGIC === */
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
      document.querySelectorAll('.menu-item').forEach(el => el.classList.remove('active'));
      const activeBtn = document.getElementById('sidebar-btn-' + key);
      if (activeBtn) activeBtn.classList.add('active');

      const t = BUNDLE[key];
      if (!t) return;

      document.getElementById('panelTableIcon').innerText = t.icon;
      document.getElementById('panelTableName').innerText = t.name;
      document.getElementById('panelTableDesc').innerText = t.description;
      document.getElementById('panelTablePath').innerText = t.path + ' (' + t.grain + ')';

      document.getElementById('tableFilterInput').value = '';
      currentFilteredRecords = [...t.records];
      currentPage = 1;
      sortColumn = null;
      sortAsc = true;

      renderGridHeaders(t.columns);
      renderGridRows();
    }}

    function renderGridHeaders(columns) {{
      const thead = document.getElementById('gridHead');
      thead.innerHTML = '';
      const tr = document.createElement('tr');

      columns.forEach(col => {{
        const th = document.createElement('th');
        th.style.cursor = 'pointer';
        th.onclick = () => sortTable(col);
        th.innerHTML = `
          <div style="display: flex; align-items: center; justify-content: space-between; gap: 6px;">
            <span>${{col}}</span>
            <span style="font-size: 0.7rem; color: var(--text-dim);">${{sortColumn === col ? (sortAsc ? '▲' : '▼') : '↕'}}</span>
          </div>
        `;
        tr.appendChild(th);
      }});
      thead.appendChild(tr);
    }}

    function renderGridRows() {{
      const tbody = document.getElementById('gridBody');
      tbody.innerHTML = '';

      const t = BUNDLE[currentTableKey];
      if (!t) return;

      const total = currentFilteredRecords.length;
      const size = pageSize === 'all' ? total : parseInt(pageSize, 10);
      const start = (currentPage - 1) * size;
      const end = Math.min(start + size, total);
      const pageRows = currentFilteredRecords.slice(start, end);

      pageRows.forEach((row, idx) => {{
        const tr = document.createElement('tr');
        tr.style.cursor = 'pointer';

        t.columns.forEach(col => {{
          const td = document.createElement('td');
          const val = row[col];
          td.innerText = (val !== null && val !== undefined) ? val : '—';
          tr.appendChild(td);
        }});
        tbody.appendChild(tr);
      }});

      document.getElementById('paginationInfo').innerText = `Showing ${{total === 0 ? 0 : start + 1}} to ${{end}} of ${{total.toLocaleString()}} rows${{t.sample_only ? ' (Sample of ' + t.total_rows.toLocaleString() + ')' : ''}}`;
      document.getElementById('btnPrevPage').disabled = (currentPage === 1);
      document.getElementById('btnNextPage').disabled = (end >= total);
    }}

    function sortTable(col) {{
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

      renderGridHeaders(BUNDLE[currentTableKey].columns);
      renderGridRows();
    }}

    function filterCurrentTable() {{
      const q = document.getElementById('tableFilterInput').value.toLowerCase().trim();
      const t = BUNDLE[currentTableKey];
      if (!t) return;

      if (!q) {{
        currentFilteredRecords = [...t.records];
      }} else {{
        currentFilteredRecords = t.records.filter(row => {{
          return t.columns.some(col => String(row[col] || '').toLowerCase().includes(q));
        }});
      }}
      currentPage = 1;
      renderGridRows();
    }}

    function changePageSize() {{
      pageSize = document.getElementById('pageSizeSelect').value;
      currentPage = 1;
      renderGridRows();
    }}

    function prevPage() {{
      if (currentPage > 1) {{ currentPage--; renderGridRows(); }}
    }}

    function nextPage() {{
      const total = currentFilteredRecords.length;
      const size = pageSize === 'all' ? total : parseInt(pageSize, 10);
      if (currentPage * size < total) {{ currentPage++; renderGridRows(); }}
    }}

    function exportCurrentTable(format) {{
      const t = BUNDLE[currentTableKey];
      if (!t) return;
      if (format === 'csv') {{
        const header = t.columns.join(',');
        const rows = currentFilteredRecords.map(r => t.columns.map(c => JSON.stringify(r[c] ?? '')).join(','));
        const csvContent = 'data:text/csv;charset=utf-8,' + [header, ...rows].join('\\n');
        const encodedUri = encodeURI(csvContent);
        const link = document.createElement('a');
        link.setAttribute('href', encodedUri);
        link.setAttribute('download', t.name);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      }}
    }}

    function copyCurrentTableJSON() {{
      const jsonStr = JSON.stringify(currentFilteredRecords, null, 2);
      navigator.clipboard.writeText(jsonStr).then(() => alert('Table records copied to clipboard as JSON!'));
    }}

    /* Features Catalog */
    function filterFeatures(group) {{
      currentFeatureGroup = group;
      document.querySelectorAll('#featurePills .pill-btn').forEach(btn => btn.classList.remove('active'));
      const found = Array.from(document.querySelectorAll('#featurePills .pill-btn')).find(b => b.innerText.includes(group) || (group === 'all' && b.innerText.includes('All')));
      if (found) found.classList.add('active');
      renderFeatures();
    }}

    function getTagClass(group) {{
      if (group === 'History & Context') return 'tag-indigo';
      if (group === 'Calendar Dynamics') return 'tag-primary';
      if (group === 'Holidays & Events') return 'tag-emerald';
      if (group === 'Weather Conditions') return 'tag-amber';
      if (group === 'Demand Uncensoring') return 'tag-rose';
      return 'tag-purple';
    }}

    function renderFeatures() {{
      const grid = document.getElementById('featuresGrid');
      grid.innerHTML = '';
      const query = document.getElementById('globalSearchInput').value.toLowerCase().trim();

      const filtered = FEATURES_DATA.filter(item => {{
        const matchGroup = (currentFeatureGroup === 'all' || item.group === currentFeatureGroup);
        const matchQuery = !query || item.name.toLowerCase().includes(query) || 
                            item.definition.toLowerCase().includes(query) || 
                            item.models.toLowerCase().includes(query);
        return matchGroup && matchQuery;
      }});

      filtered.forEach(item => {{
        const card = document.createElement('div');
        card.className = 'item-card';
        card.innerHTML = `
          <div class="card-top">
            <span class="card-title">${{item.name}}</span>
            <span class="badge-tag ${{getTagClass(item.group)}}">${{item.group}}</span>
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
          <td><span class="badge-tag ${{getTagClass(item.group)}}">${{item.group}}</span></td>
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
