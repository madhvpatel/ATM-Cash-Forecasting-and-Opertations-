"""Generate the interactive Web Page for Input Features and Data Masters.

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

def build_html():
    # 1. Load Feature Registry
    f_reg_path = DATA_DIR / "feature_registry.csv"
    df_reg = pd.read_csv(f_reg_path)
    
    # 2. Load Catalog
    catalog_path = DATA_DIR / "catalog.json"
    with open(catalog_path) as f:
        catalog = json.load(f)
        
    # 3. Model F inputs from run_experiment_4.py
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
    for f in history_feats + calendar_feats + holiday_feats + weather_feats + uncollected_feats:
        all_features.append({
            "name": f[0],
            "definition": f[1],
            "type": f[2],
            "group": f[3],
            "models": f[4],
            "availability": f[5],
        })
        
    # 4. Data Masters List
    data_masters = [
        {
            "name": "dim_calendar.csv",
            "path": "data/master/cash_files/dim_calendar.csv",
            "category": "Calendar & Governance",
            "rows": 754,
            "columns": 10,
            "grain": "Date (Calendar Day)",
            "description": "Standardized calendar dimension spanning 2024-09-01 to 2026-09-24 with weekend, payday, and month-cycle flags.",
            "source": "src/atm_cash_forecast/calendar.py",
            "status": "Production Master",
            "schema": ["calendar_date", "day_of_week", "month", "quarter", "is_weekend", "is_payday", "is_holiday", "days_to_month_end", "is_month_start", "is_month_end"]
        },
        {
            "name": "holiday_calendar.csv",
            "path": "data/master/cash_files/holiday_calendar.csv",
            "category": "Calendar & Governance",
            "rows": 1729,
            "columns": 9,
            "grain": "Date × Region × Holiday",
            "description": "Sourced Reserve Bank of India (RBI) Negotiable Instruments Act bank holidays across 29 regional offices with announcement provenance.",
            "source": "RBI Regional Holiday Matrices",
            "status": "Production Master",
            "schema": ["calendar_date", "region", "holiday_name", "holiday_type", "source_id", "source_url", "retrieved_at", "known_at", "evidence_mode"]
        },
        {
            "name": "holiday_coverage.csv",
            "path": "data/master/cash_files/holiday_coverage.csv",
            "category": "Calendar & Governance",
            "rows": 986,
            "columns": 6,
            "grain": "Region × Month",
            "description": "Audit matrix confirming verified RBI holiday disclosure coverage across all regions and observation periods.",
            "source": "RBI Source Verification",
            "status": "Audit Master",
            "schema": ["region", "period", "status", "source_id", "retrieved_at", "known_at"]
        },
        {
            "name": "terminal_metadata.csv",
            "path": "data/master/cash_files/terminal_metadata.csv",
            "category": "Terminal & Network",
            "rows": 100,
            "columns": 5,
            "grain": "Terminal ID",
            "description": "Catalog of 100 ATM terminals including unique IDs, terminal codes, host cities, and specific site descriptions.",
            "source": "Raw Cash Ingestion Pipeline",
            "status": "Production Master",
            "schema": ["terminal_id", "terminal_code", "city", "location", "site_type"]
        },
        {
            "name": "terminal_geography.csv",
            "path": "data/master/cash_files/terminal_geography.csv",
            "category": "Terminal & Network",
            "rows": 100,
            "columns": 9,
            "grain": "Terminal ID",
            "description": "Spatial and administrative mapping linking each terminal to RBI regions, state jurisdictions, metro tiers, and climate zones.",
            "source": "Curated Network Geography",
            "status": "Production Master",
            "schema": ["terminal_id", "city", "state", "rbi_region", "pin_code", "latitude", "longitude", "metro_tier", "climate_zone"]
        },
        {
            "name": "terminal_profile.csv",
            "path": "data/master/cash_files/terminal_profile.csv",
            "category": "Terminal & Network",
            "rows": 100,
            "columns": 6,
            "grain": "Terminal ID",
            "description": "Hardware and operational facts schema for cassette capacities, security, and access window hours.",
            "source": "Network Operations",
            "status": "Governance Schema",
            "schema": ["terminal_id", "cassette_capacity_notes", "max_amount_inr", "machine_make", "cctv_status", "access_hours"]
        },
        {
            "name": "payday_rules.csv & paydays.csv",
            "path": "data/master/cash_files/paydays.csv",
            "category": "Calendar & Governance",
            "rows": 74,
            "columns": 2,
            "grain": "Payday Date",
            "description": "74 discrete calendar dates flagged as salary credit dates based on the 1st, 15th, and month-end weekday protocol.",
            "source": "Treasury Calendar Protocol",
            "status": "Production Master",
            "schema": ["calendar_date", "holiday_name / payday_flag"]
        },
        {
            "name": "daily_cash_out.csv",
            "path": "data/processed/daily_cash_out.csv",
            "category": "Demand & Transactions",
            "rows": 75400,
            "columns": 8,
            "grain": "Terminal ID × Date",
            "description": "Primary transactional gold table with daily aggregated withdrawal cash-out demand for all 100 ATMs across 754 days.",
            "source": "Raw 15-Minute XLSX Archives (Parts 1-5)",
            "status": "Core Processed Table",
            "schema": ["terminal_id", "transaction_date", "denomination", "withdrawal_requested_amount", "amount_dispensed", "net_demand", "is_recycler", "target_semantics"]
        },
        {
            "name": "atm_daily.csv (Maharashtra Panel)",
            "path": "simulated/maharashtra/data/atm_daily.csv",
            "category": "Demand & Transactions",
            "rows": 6032,
            "columns": 26,
            "grain": "Terminal ID × Date (8 ATMs)",
            "description": "Fully joined research panel for the 8 Maharashtra benchmark ATMs enriched with ERA5 weather and RBI holiday calendars.",
            "source": "Maharashtra Study Pipeline",
            "status": "Benchmark Analytical Panel",
            "schema": ["terminal_id", "date", "city", "state", "rbi_region", "location", "loc_type", "demand", "dom", "month", "dow", "is_sunday", "is_2_4_sat", "is_early", "is_late", "is_payday", "is_regional_holiday", "holiday_name", "pre_holiday", "tmax_c", "tmean_c", "tmin_c", "rain_mm", "rain_ge_10", "rain_ge_25", "tmax_ge_35"]
        },
        {
            "name": "weather_city_daily.csv",
            "path": "simulated/maharashtra/data/weather_city_daily.csv",
            "category": "Exogenous Weather",
            "rows": 3016,
            "columns": 6,
            "grain": "City × Date",
            "description": "Public ERA5 atmospheric observations from Open-Meteo for Mumbai, Pune, Nagpur, and Nashik (rainfall, min/mean/max temperatures).",
            "source": "Open-Meteo ERA5 Archive API",
            "status": "Exogenous Master",
            "schema": ["city", "date", "tmax_c", "tmean_c", "tmin_c", "rain_mm"]
        },
        {
            "name": "demand_features.csv",
            "path": "data/processed/demand_features.csv",
            "category": "Feature Store",
            "rows": 75400,
            "columns": 19,
            "grain": "Terminal ID × Date",
            "description": "Feature store training matrix containing rolling windows, multi-week lags, and seasonal indicators across the full 100-ATM estate.",
            "source": "Demand Feature Engineering Pipeline",
            "status": "Feature Store Table",
            "schema": ["terminal_id", "transaction_date", "denomination", "net_demand", "day_of_week", "month", "quarter", "is_weekend", "is_payday", "is_holiday", "days_to_month_end", "is_month_start", "is_month_end", "lag_1", "lag_7", "lag_14", "rolling_mean_7", "rolling_mean_14", "rolling_std_7"]
        },
        {
            "name": "feature_registry.csv",
            "path": "data/feature_registry.csv",
            "category": "Governance & Quality",
            "rows": 30,
            "columns": 8,
            "grain": "Feature Name",
            "description": "Central catalog declaring feature grains, production status, lineage, missingness policies, and schema version constraints.",
            "source": "Feature Governance Registry",
            "status": "Governance Registry",
            "schema": ["feature", "definition", "grain", "status", "source", "availability", "missing_behavior", "schema_version"]
        }
    ]

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="description" content="Central Data Masters and Model Input Features Registry for ATM Cash Demand Forecasting and Intelligence.">
  <title>Cash Intelligence · Features Registry & Data Masters</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
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
      --amber-glow: rgba(245, 158, 11, 0.15);
      --indigo: #6366f1;
      --rose: #f43f5e;
      --purple: #a855f7;
      --font-sans: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }}

    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    body {{
      background-color: var(--bg-base);
      color: var(--text-main);
      font-family: var(--font-sans);
      line-height: 1.5;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      overflow-x: hidden;
    }}

    /* Glow backdrop effects */
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
      background: rgba(10, 13, 20, 0.85);
      backdrop-filter: blur(16px);
      border-bottom: 1px solid var(--border-subtle);
      padding: 16px 32px;
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

    .brand-icon svg {{
      width: 22px;
      height: 22px;
      color: #fff;
    }}

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
      gap: 20px;
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
      max-width: 1400px;
      margin: 0 auto;
      padding: 32px;
      position: relative;
      z-index: 10;
      flex: 1;
      width: 100%;
    }}

    .hero {{
      margin-bottom: 32px;
    }}

    .hero h2 {{
      font-size: 2.1rem;
      font-weight: 800;
      letter-spacing: -0.03em;
      margin-bottom: 8px;
      background: linear-gradient(to right, #fff, #cbd5e1);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}

    .hero p {{
      color: var(--text-muted);
      font-size: 1.05rem;
      max-width: 800px;
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
      gap: 8px;
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

    .tab-btn:hover {{
      color: #fff;
    }}

    .tab-btn.active {{
      color: var(--primary);
    }}

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

    /* Search & Filter Bar */
    .filter-bar {{
      display: flex;
      gap: 12px;
      margin-bottom: 24px;
      flex-wrap: wrap;
    }}

    .search-box {{
      position: relative;
      flex: 1;
      min-width: 280px;
    }}

    .search-box input {{
      width: 100%;
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 10px;
      padding: 10px 14px 10px 40px;
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
      left: 14px;
      top: 50%;
      transform: translateY(-50%);
      color: var(--text-dim);
      width: 16px;
      height: 16px;
      pointer-events: none;
    }}

    .filter-pills {{
      display: flex;
      gap: 8px;
      overflow-x: auto;
      padding-bottom: 4px;
    }}

    .pill-btn {{
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      color: var(--text-muted);
      border-radius: 8px;
      padding: 8px 14px;
      font-size: 0.82rem;
      font-weight: 600;
      cursor: pointer;
      white-space: nowrap;
      transition: all 0.2s;
    }}

    .pill-btn:hover {{
      background: var(--bg-surface-elevated);
      color: #fff;
    }}

    .pill-btn.active {{
      background: var(--primary);
      color: #000;
      border-color: var(--primary);
      font-weight: 700;
    }}

    /* Cards Grid */
    .grid-cards {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
      gap: 20px;
      margin-bottom: 32px;
    }}

    .item-card {{
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 14px;
      padding: 20px;
      backdrop-filter: blur(12px);
      transition: all 0.25s ease;
      display: flex;
      flex-direction: column;
      position: relative;
    }}

    .item-card:hover {{
      border-color: var(--border-hover);
      transform: translateY(-2px);
      box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4);
    }}

    .card-top {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      margin-bottom: 12px;
      gap: 12px;
    }}

    .card-title {{
      font-family: var(--font-mono);
      font-size: 1.05rem;
      font-weight: 600;
      color: #fff;
      word-break: break-all;
    }}

    .tag-badge {{
      font-size: 0.7rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      padding: 3px 9px;
      border-radius: 6px;
      white-space: nowrap;
    }}

    .tag-history {{ background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); }}
    .tag-calendar {{ background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }}
    .tag-holidays {{ background: rgba(244, 63, 94, 0.15); color: #f43f5e; border: 1px solid rgba(244, 63, 94, 0.3); }}
    .tag-weather {{ background: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); }}
    .tag-master {{ background: rgba(168, 85, 247, 0.15); color: #a855f7; border: 1px solid rgba(168, 85, 247, 0.3); }}
    .tag-uncollected {{ background: rgba(148, 163, 184, 0.15); color: #94a3b8; border: 1px solid rgba(148, 163, 184, 0.3); }}

    .card-desc {{
      color: var(--text-muted);
      font-size: 0.88rem;
      margin-bottom: 16px;
      flex-grow: 1;
    }}

    .card-meta-list {{
      display: flex;
      flex-direction: column;
      gap: 6px;
      border-top: 1px solid var(--border-subtle);
      padding-top: 12px;
      font-size: 0.8rem;
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

    /* Table View for Data Masters */
    .table-container {{
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 14px;
      overflow-x: auto;
      margin-bottom: 32px;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      text-align: left;
      font-size: 0.88rem;
    }}

    th {{
      background: var(--bg-surface-elevated);
      color: var(--text-muted);
      font-weight: 700;
      text-transform: uppercase;
      font-size: 0.75rem;
      letter-spacing: 0.05em;
      padding: 14px 18px;
      border-bottom: 1px solid var(--border-subtle);
    }}

    td {{
      padding: 14px 18px;
      border-bottom: 1px solid var(--border-subtle);
      color: var(--text-muted);
    }}

    tr:hover td {{
      background: rgba(255, 255, 255, 0.02);
      color: #fff;
    }}

    .mono-cell {{
      font-family: var(--font-mono);
      color: #fff;
      font-weight: 500;
    }}

    /* Model Matrix Grid */
    .model-matrix {{
      margin-top: 24px;
    }}

    .check-yes {{
      color: var(--emerald);
      font-weight: 700;
    }}

    .check-no {{
      color: var(--text-dim);
      font-weight: 400;
    }}

    /* Footer */
    footer {{
      border-top: 1px solid var(--border-subtle);
      padding: 24px 32px;
      text-align: center;
      color: var(--text-dim);
      font-size: 0.82rem;
    }}

    footer a {{
      color: var(--primary);
      text-decoration: none;
    }}

    /* Responsive */
    @media (max-width: 768px) {{
      header {{ padding: 14px 20px; }}
      .container {{ padding: 20px; }}
      .grid-cards {{ grid-template-columns: 1fr; }}
      .header-stats {{ display: none; }}
    }}
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
        <span>Data Masters & Feature Governance</span>
      </div>
    </a>

    <div class="header-stats">
      <div class="stat-pill">
        <div class="dot"></div>
        <span><strong>29</strong> Model Inputs</span>
      </div>
      <div class="stat-pill">
        <div class="dot"></div>
        <span><strong>12</strong> Core Masters</span>
      </div>
      <div class="stat-pill">
        <div class="dot"></div>
        <span><strong>75,400</strong> Observed ATM-Days</span>
      </div>
    </div>
  </header>

  <main class="container">
    <div class="hero">
      <h2>Data Masters & Input Features Registry</h2>
      <p>The authoritative catalog of operational data tables, temporal masters, and ML input feature definitions governing ATM cash replenishment forecasting.</p>
    </div>

    <div class="tabs-wrapper">
      <div class="tabs" role="tablist">
        <button id="tab-features-btn" class="tab-btn active" onclick="switchTab('features')">
          Input Features Catalog
          <span class="badge-count" id="badge-features">37</span>
        </button>
        <button id="tab-masters-btn" class="tab-btn" onclick="switchTab('masters')">
          Data Masters & Tables
          <span class="badge-count" id="badge-masters">12</span>
        </button>
        <button id="tab-matrix-btn" class="tab-btn" onclick="switchTab('matrix')">
          Model Feature Matrix
          <span class="badge-count">6 Models</span>
        </button>
      </div>

      <div class="search-box">
        <svg class="search-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="11" cy="11" r="8"></circle>
          <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
        </svg>
        <input type="text" id="searchInput" placeholder="Search features, definitions, masters, or columns..." oninput="handleSearch()">
      </div>
    </div>

    <!-- Section 1: Features Tab -->
    <div id="section-features">
      <div class="filter-bar">
        <div class="filter-pills" id="featurePills">
          <button class="pill-btn active" onclick="filterFeatures('all')">All Groups</button>
          <button class="pill-btn" onclick="filterFeatures('History & Context')">Withdrawal History (13)</button>
          <button class="pill-btn" onclick="filterFeatures('Calendar Dynamics')">Calendar Dynamics (6)</button>
          <button class="pill-btn" onclick="filterFeatures('Holidays & Events')">Holidays & Events (3)</button>
          <button class="pill-btn" onclick="filterFeatures('Weather Conditions')">Weather Signals (7)</button>
          <button class="pill-btn" onclick="filterFeatures('Uncollected Signals')">Uncollected / Governance (8)</button>
        </div>
      </div>

      <div class="grid-cards" id="featuresGrid">
        <!-- Rendered by JS -->
      </div>
    </div>

    <!-- Section 2: Masters Tab -->
    <div id="section-masters" style="display: none;">
      <div class="table-container">
        <table>
          <thead>
            <tr>
              <th>Table Name / Path</th>
              <th>Category</th>
              <th>Primary Grain</th>
              <th>Rows × Cols</th>
              <th>Description & Key Schema</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody id="mastersTableBody">
            <!-- Rendered by JS -->
          </tbody>
        </table>
      </div>
    </div>

    <!-- Section 3: Model Matrix Tab -->
    <div id="section-matrix" style="display: none;">
      <div class="table-container">
        <table>
          <thead>
            <tr>
              <th>Feature Name</th>
              <th>Information Tier</th>
              <th style="text-align: center;">Model A<br><small style="color:var(--text-dim)">Hist Only</small></th>
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

  <footer>
    <p>Cash Intelligence Platform · CMS2 Evaluation Environment · Governed by <a href="file://{f_reg_path}">feature_registry.csv</a> and <a href="file://{catalog_path}">catalog.json</a></p>
  </footer>

  <script>
    const FEATURES_DATA = {json.dumps(all_features)};
    const MASTERS_DATA = {json.dumps(data_masters)};

    let currentTab = 'features';
    let currentFeatureGroup = 'all';

    function switchTab(tab) {{
      currentTab = tab;
      document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
      document.getElementById('tab-' + tab + '-btn').classList.add('active');

      document.getElementById('section-features').style.display = tab === 'features' ? 'block' : 'none';
      document.getElementById('section-masters').style.display = tab === 'masters' ? 'block' : 'none';
      document.getElementById('section-matrix').style.display = tab === 'matrix' ? 'block' : 'none';

      handleSearch();
    }}

    function filterFeatures(group) {{
      currentFeatureGroup = group;
      document.querySelectorAll('#featurePills .pill-btn').forEach(btn => {{
        if (btn.innerText.toLowerCase().includes(group.toLowerCase()) || (group === 'all' && btn.innerText === 'All Groups')) {{
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
      const query = document.getElementById('searchInput').value.toLowerCase();
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
              <span>Data Grain / Type</span>
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

      document.getElementById('badge-features').innerText = filtered.length;
    }}

    function renderMasters() {{
      const query = document.getElementById('searchInput').value.toLowerCase();
      const tbody = document.getElementById('mastersTableBody');
      tbody.innerHTML = '';

      const filtered = MASTERS_DATA.filter(item => {{
        return !query || item.name.toLowerCase().includes(query) || 
               item.description.toLowerCase().includes(query) || 
               item.category.toLowerCase().includes(query) ||
               item.schema.join(' ').toLowerCase().includes(query);
      }});

      filtered.forEach(item => {{
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td>
            <div class="mono-cell" style="color: var(--primary); font-size: 0.95rem;">${{item.name}}</div>
            <small style="color: var(--text-dim); font-family: var(--font-mono);">${{item.path}}</small>
          </td>
          <td><span class="tag-badge tag-master">${{item.category}}</span></td>
          <td>${{item.grain}}</td>
          <td class="mono-cell">${{item.rows.toLocaleString()}} rows<br><small style="color:var(--text-dim);">${{item.columns}} cols</small></td>
          <td>
            <div style="color: #fff; margin-bottom: 6px;">${{item.description}}</div>
            <small style="font-family: var(--font-mono); color: var(--text-dim); word-break: break-all;">Schema: ${{item.schema.join(', ')}}</small>
          </td>
          <td><span class="tag-badge" style="background: rgba(16, 185, 129, 0.1); color: var(--emerald);">${{item.status}}</span></td>
        `;
        tbody.appendChild(tr);
      }});

      document.getElementById('badge-masters').innerText = filtered.length;
    }}

    function renderMatrix() {{
      const query = document.getElementById('searchInput').value.toLowerCase();
      const tbody = document.getElementById('matrixTableBody');
      tbody.innerHTML = '';

      const filtered = FEATURES_DATA.filter(item => {{
        return !query || item.name.toLowerCase().includes(query) || 
               item.group.toLowerCase().includes(query) || 
               item.models.toLowerCase().includes(query);
      }});

      filtered.forEach(item => {{
        const m = item.models;
        const inA = m.includes('A');
        const inB = m.includes('B');
        const inC = m.includes('C');
        const inD = m.includes('D');
        const inF = m.includes('F');
        const inExp3 = m.includes('Exp3');

        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td class="mono-cell">${{item.name}}</td>
          <td><span class="tag-badge ${{getTagClass(item.group)}}">${{item.group}}</span></td>
          <td style="text-align: center;"><span class="${{inA ? 'check-yes' : 'check-no'}}">${{inA ? '✓' : '—'}}</span></td>
          <td style="text-align: center;"><span class="${{inB ? 'check-yes' : 'check-no'}}">${{inB ? '✓' : '—'}}</span></td>
          <td style="text-align: center;"><span class="${{inC ? 'check-yes' : 'check-no'}}">${{inC ? '✓' : '—'}}</span></td>
          <td style="text-align: center;"><span class="${{inD ? 'check-yes' : 'check-no'}}">${{inD ? '✓' : '—'}}</span></td>
          <td style="text-align: center; background: rgba(56, 189, 248, 0.05);"><span class="${{inF ? 'check-yes' : 'check-no'}}" style="font-size: 1.1rem; color: var(--primary);">${{inF ? '✓' : '—'}}</span></td>
          <td style="text-align: center;"><span class="${{inExp3 ? 'check-yes' : 'check-no'}}">${{inExp3 ? '✓' : '—'}}</span></td>
        `;
        tbody.appendChild(tr);
      }});
    }}

    function handleSearch() {{
      if (currentTab === 'features') renderFeatures();
      else if (currentTab === 'masters') renderMasters();
      else if (currentTab === 'matrix') renderMatrix();
    }}

    // Initial load
    renderFeatures();
    renderMasters();
    renderMatrix();
  </script>
</body>
</html>
"""
    
    out_file1 = HERE / "features_and_masters.html"
    out_file2 = HERE / "index.html"
    out_file1.write_text(html_content, encoding="utf-8")
    out_file2.write_text(html_content, encoding="utf-8")
    print(f"Generated: {out_file1}")
    print(f"Generated: {out_file2}")

if __name__ == "__main__":
    build_html()
