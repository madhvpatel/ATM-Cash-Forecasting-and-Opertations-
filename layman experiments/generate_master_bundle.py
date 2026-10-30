"""Generate master_data_bundle.js and update features_and_masters.html to display live master data.

Outputs:
- layman experiments/master_data_bundle.js
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

def generate_bundle():
    tables_meta = [
        {
            "id": "terminal_metadata",
            "name": "terminal_metadata.csv",
            "title": "Terminal Metadata Master",
            "category": "Terminal & Network",
            "icon": "🏢",
            "path": "data/master/cash_files/terminal_metadata.csv",
            "full_path": str(DATA_DIR / "master" / "cash_files" / "terminal_metadata.csv"),
            "grain": "terminal_id (1 per ATM)",
            "description": "Primary register of all 100 ATM terminals in the cash estate, linking terminal codes to host cities and specific concourse/airport/mall locations.",
            "source": "Raw Ingestion Pipeline",
            "status": "Production Master",
        },
        {
            "id": "terminal_geography",
            "name": "terminal_geography.csv",
            "title": "Terminal Geography & Jurisdictions",
            "category": "Terminal & Network",
            "icon": "🗺️",
            "path": "data/master/cash_files/terminal_geography.csv",
            "full_path": str(DATA_DIR / "master" / "cash_files" / "terminal_geography.csv"),
            "grain": "terminal_id (1 per ATM)",
            "description": "Spatial, administrative, and RBI regional office mappings for all 100 ATMs, including pin codes, coordinates, metro tiers, and climate zones.",
            "source": "Network Geography Registry",
            "status": "Production Master",
        },
        {
            "id": "terminal_profile",
            "name": "terminal_profile.csv",
            "title": "Terminal Profile & Hardware Specs",
            "category": "Terminal & Network",
            "icon": "⚙️",
            "path": "data/master/cash_files/terminal_profile.csv",
            "full_path": str(DATA_DIR / "master" / "cash_files" / "terminal_profile.csv"),
            "grain": "terminal_id (1 per ATM)",
            "description": "Hardware specifications, maximum rupee capacity bounds, machine make/model, and operational accessibility hours for each machine.",
            "source": "ATM Operations",
            "status": "Governance Schema",
        },
        {
            "id": "dim_calendar",
            "name": "dim_calendar.csv",
            "title": "Calendar Dimension Master",
            "category": "Calendar & Time",
            "icon": "📅",
            "path": "data/master/cash_files/dim_calendar.csv",
            "full_path": str(DATA_DIR / "master" / "cash_files" / "dim_calendar.csv"),
            "grain": "calendar_date (1 per Day)",
            "description": "Standardized calendar dimension spanning 754 dates (2024-09-01 to 2026-09-24) with weekend flags, payday windows, and month boundaries.",
            "source": "src/atm_cash_forecast/calendar.py",
            "status": "Production Master",
        },
        {
            "id": "holiday_calendar",
            "name": "holiday_calendar.csv",
            "title": "RBI Regional Holiday Calendar",
            "category": "Calendar & Time",
            "icon": "🏛️",
            "path": "data/master/cash_files/holiday_calendar.csv",
            "full_path": str(DATA_DIR / "master" / "cash_files" / "holiday_calendar.csv"),
            "grain": "calendar_date × region × holiday",
            "description": "1,729 verified Reserve Bank of India (RBI) bank holidays across 29 regional offices under the Negotiable Instruments Act with official provenance.",
            "source": "RBI Official Gazette Disclosures",
            "status": "Production Master",
        },
        {
            "id": "holiday_coverage",
            "name": "holiday_coverage.csv",
            "title": "Holiday Coverage & Provenance Audit",
            "category": "Calendar & Time",
            "icon": "🔍",
            "path": "data/master/cash_files/holiday_coverage.csv",
            "full_path": str(DATA_DIR / "master" / "cash_files" / "holiday_coverage.csv"),
            "grain": "region × period (Month)",
            "description": "986 regional disclosure audit records verifying completeness, announcement dates, and source URLs for each RBI regional office.",
            "source": "RBI Source Verification",
            "status": "Audit Master",
        },
        {
            "id": "paydays",
            "name": "paydays.csv",
            "title": "Salary Credit Dates",
            "category": "Calendar & Time",
            "icon": "💰",
            "path": "data/master/cash_files/paydays.csv",
            "full_path": str(DATA_DIR / "master" / "cash_files" / "paydays.csv"),
            "grain": "calendar_date",
            "description": "74 discrete calendar dates flagged as salary disbursement days in the observed timeline matching the 1st, 15th, and month-end rule.",
            "source": "Treasury Calendar Protocol",
            "status": "Production Master",
        },
        {
            "id": "payday_rules",
            "name": "payday_rules.csv",
            "title": "Salary Window Policy Rules",
            "category": "Calendar & Time",
            "icon": "📜",
            "path": "data/master/cash_files/payday_rules.csv",
            "full_path": str(DATA_DIR / "master" / "cash_files" / "payday_rules.csv"),
            "grain": "rule_set",
            "description": "Formal governance rule definition establishing corporate and pension salary withdrawal assumptions.",
            "source": "Cash Management Protocol",
            "status": "Policy Master",
        },
        {
            "id": "holidays",
            "name": "holidays.csv",
            "title": "National Fixed Holidays",
            "category": "Calendar & Time",
            "icon": "🇮🇳",
            "path": "data/master/cash_files/holidays.csv",
            "full_path": str(DATA_DIR / "master" / "cash_files" / "holidays.csv"),
            "grain": "calendar_date",
            "description": "Six annual fixed national holidays observed nationwide across India (Republic Day, Independence Day, Gandhi Jayanti for 2024–2026).",
            "source": "National Gazette",
            "status": "Production Master",
        },
        {
            "id": "feature_registry",
            "name": "feature_registry.csv",
            "title": "Feature Governance Registry",
            "category": "Governance & Quality",
            "icon": "📋",
            "path": "data/feature_registry.csv",
            "full_path": str(DATA_DIR / "feature_registry.csv"),
            "grain": "feature name",
            "description": "30 governed feature declarations establishing grain, status (available, derived, uncollected), lineage, and missingness policies.",
            "source": "Feature Governance Registry",
            "status": "Governance Registry",
        },
        {
            "id": "weather_city_daily",
            "name": "weather_city_daily.csv",
            "title": "Open-Meteo ERA5 Daily Weather",
            "category": "Exogenous Weather",
            "icon": "🌦️",
            "path": "simulated/maharashtra/data/weather_city_daily.csv",
            "full_path": str(CMS_ROOT / "simulated" / "maharashtra" / "data" / "weather_city_daily.csv"),
            "grain": "city × date",
            "description": "3,016 daily atmospheric observations across Mumbai, Pune, Nagpur, and Nashik (rainfall in mm, min/mean/max temperatures in °C).",
            "source": "Open-Meteo ERA5 Archive API",
            "status": "Exogenous Master",
        },
        {
            "id": "atm_daily",
            "name": "atm_daily.csv (Maharashtra Panel)",
            "title": "Maharashtra Benchmark Analytical Panel",
            "category": "Demand & Transactions",
            "icon": "📊",
            "path": "simulated/maharashtra/data/atm_daily.csv",
            "full_path": str(CMS_ROOT / "simulated" / "maharashtra" / "data" / "atm_daily.csv"),
            "grain": "terminal_id × date (8 ATMs)",
            "description": "6,032 analytical rows joining daily cash demand with calendar flags, RBI regional holidays, and ERA5 weather across 8 benchmark ATMs.",
            "source": "Maharashtra Study Pipeline",
            "status": "Benchmark Analytical Panel",
        },
        {
            "id": "daily_cash_out",
            "name": "daily_cash_out.csv (Sample & Explorer)",
            "title": "Primary Daily Cash-Out Demand Table",
            "category": "Demand & Transactions",
            "icon": "💳",
            "path": "data/processed/daily_cash_out.csv",
            "full_path": str(DATA_DIR / "processed" / "daily_cash_out.csv"),
            "grain": "terminal_id × transaction_date (100 ATMs)",
            "description": "75,400 daily aggregated cash withdrawals across all 100 ATMs (754 consecutive days). Explorer presents the top 500 records with aggregate metrics.",
            "source": "Raw Ingestion Pipeline (Parts 1-5)",
            "status": "Core Processed Table",
        },
        {
            "id": "cashout_suppression_summary",
            "name": "cashout_suppression_summary.csv",
            "title": "ATM Cashout & Suppressed Demand Summary",
            "category": "Demand & Transactions",
            "icon": "⚠️",
            "path": "layman experiments/output/cashout_suppression_summary.csv",
            "full_path": str(HERE / "output" / "cashout_suppression_summary.csv"),
            "grain": "terminal_id (8 Maharashtra benchmark ATMs)",
            "description": "Quantified empty-hopper days, average hours empty, and lost rupee customer demand (₹1.36 Crore suppressed on ATM 7 alone).",
            "source": "uncensor_cashout_experiment.py",
            "status": "Analytical Master",
        },
        {
            "id": "uncensoring_model_comparison",
            "name": "uncensoring_model_comparison.csv",
            "title": "Censored vs. Uncensored Model Evaluation",
            "category": "Model Performance",
            "icon": "🎯",
            "path": "layman experiments/output/uncensoring_model_comparison.csv",
            "full_path": str(HERE / "output" / "uncensoring_model_comparison.csv"),
            "grain": "terminal_id (ATM 7 & ATM 1)",
            "description": "Comparative evaluation showing elimination of under-forecast peak bias (from -14.6% to -9.9%) and 80% reduction in stockouts.",
            "source": "uncensor_cashout_experiment.py",
            "status": "Model Evaluation",
        },
        {
            "id": "inventory_checkpoint_quality",
            "name": "inventory_checkpoint_quality.csv (Cashout Audit)",
            "title": "15-Min Inventory Snapshot Quality & Zero Balances",
            "category": "Demand & Transactions",
            "icon": "⏱️",
            "path": "data/processed/inventory_checkpoint_quality.csv",
            "full_path": str(DATA_DIR / "processed" / "inventory_checkpoint_quality.csv"),
            "grain": "terminal_id × calendar_date (100 ATMs, 75,400 days)",
            "description": "75,400 daily quality audit records with zero-balance snapshot counts and min/max reported cash. Explorer shows the top 500 cashout incidents.",
            "source": "Quality Ingestion Audit Pipeline",
            "status": "Core Processed Table",
        },
        {
            "id": "pseudo_masking_validation_summary",
            "name": "pseudo_masking_validation_summary.csv",
            "title": "Experiment 5: Pseudo-Masking Estimator Validation",
            "category": "Model Performance",
            "icon": "🔬",
            "path": "layman experiments/output/pseudo_masking_validation_summary.csv",
            "full_path": str(HERE / "output" / "pseudo_masking_validation_summary.csv"),
            "grain": "masking_scenario (5 validation regimes across 116 clean days)",
            "description": "Quantifies estimator accuracy under uniform random masking (MAPE 5%) versus diurnal peak missingness (-12.8% evening bias).",
            "source": "validate_reconstruction_masking.py",
            "status": "Validation Benchmark",
        },
        {
            "id": "experiment_5_replenishment_tradeoff",
            "name": "experiment_5_replenishment_tradeoff.csv",
            "title": "Experiment 5: Multi-Buffer Replenishment Frontier",
            "category": "Operational Policy",
            "icon": "⚖️",
            "path": "layman experiments/output/experiment_5_replenishment_tradeoff.csv",
            "full_path": str(HERE / "output" / "experiment_5_replenishment_tradeoff.csv"),
            "grain": "terminal_id × model × cer_buffer_target (55 cycles)",
            "description": "Multi-policy replenishment evaluation across 6 buffer levels (65% to 90% CER), balancing stockout counts against idle cash holding costs.",
            "source": "run_experiment_5.py",
            "status": "Policy Evaluation",
        }
    ]

    bundle = {}
    print("Exporting master tables to bundle...")
    for tm in tables_meta:
        p = Path(tm["full_path"])
        if not p.exists():
            print(f"Warning: {p} does not exist!")
            continue
        df = pd.read_csv(p)
        total_rows = len(df)
        
        # If very large table, sample 500 rows for smooth client-side rendering
        if tm["id"] == "inventory_checkpoint_quality":
            co_rows = df[df["zero_balance_snapshot_count"] > 0]
            records_to_store = co_rows.head(500) if len(co_rows) > 500 else co_rows
        else:
            records_to_store = df.head(500) if total_rows > 10000 else df
        records = records_to_store.fillna("").to_dict(orient="records")
        
        bundle[tm["id"]] = {
            "id": tm["id"],
            "name": tm["name"],
            "title": tm["title"],
            "category": tm["category"],
            "icon": tm["icon"],
            "path": tm["path"],
            "grain": tm["grain"],
            "description": tm["description"],
            "source": tm["source"],
            "status": tm["status"],
            "total_rows": total_rows,
            "columns": df.columns.tolist(),
            "sample_only": total_rows > 10000,
            "records": records
        }
        print(f"  Loaded {tm['name']}: {len(records)} records stored (total in CSV: {total_rows:,})")

    # Write JS file
    bundle_js_path = HERE / "master_data_bundle.js"
    with open(bundle_js_path, "w", encoding="utf-8") as f:
        f.write("// Master Data Bundle for Cash Intelligence Portal\n")
        f.write("window.MASTER_DATA_BUNDLE = ")
        json.dump(bundle, f)
        f.write(";\n")
    print(f"Wrote master_data_bundle.js ({bundle_js_path.stat().st_size / 1024:.1f} KB)")
    return tables_meta

if __name__ == "__main__":
    generate_bundle()
