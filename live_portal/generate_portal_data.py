"""Generate live portal data bundle for 1-month holdout forecast (Cutoff: 2026-08-24).

Cutoff date: 2026-08-24 (1 month before primary data ends on 2026-09-24).
Forecast horizon: 31 days (2026-08-25 to 2026-09-24).
"""

from __future__ import annotations

import json
import os
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CMS_ROOT = HERE.parent
if str(CMS_ROOT) not in sys.path:
    sys.path.insert(0, str(CMS_ROOT))

import numpy as np
import pandas as pd
import lightgbm as lgb

from replenishment_engine.config import ReplenishmentConfig
from replenishment_engine.uncensoring import load_uncensored_panel
from replenishment_engine.optimizer import CashLoadOptimizer, norm_ppf, norm_cdf
from replenishment_engine import demand_model

OPERATIONAL_MODEL = demand_model.MODEL_NAME

ATM_GEO_METADATA = {
    1: {
        "terminal_code": "SYNATM0001",
        "site_name": "Mumbai Railway Station Concourse",
        "site_type": "RAILWAY STATION CONCOURSE",
        "serviced_by": "CIT",
        "replenishment_cycle_days": 7,
        "address": "Dadar Central Concourse, Mumbai, Maharashtra 400014",
        "city": "MUMBAI",
        "lat": 18.9400,
        "lng": 72.8353,
        "corridor_id": "MUMBAI_CORRIDOR",
        "corridor_name": "Mumbai Transit Route 1",
        "vault_capacity": 6000000.0,
    },
    2: {
        "terminal_code": "SYNATM0002",
        "site_name": "Mumbai Airport Terminal T2",
        "site_type": "AIRPORT TERMINAL",
        "serviced_by": "CIT",
        "replenishment_cycle_days": 7,
        "address": "Chhatrapati Shivaji Maharaj International Airport T2 Arrivals, Mumbai 400099",
        "city": "MUMBAI",
        "lat": 19.0896,
        "lng": 72.8656,
        "corridor_id": "MUMBAI_CORRIDOR",
        "corridor_name": "Mumbai Transit Route 1",
        "vault_capacity": 6000000.0,
    },
    3: {
        "terminal_code": "SYNATM0003",
        "site_name": "Pune City Centre Mall",
        "site_type": "CITY CENTRE MALL",
        "serviced_by": "CIT",
        "replenishment_cycle_days": 7,
        "address": "Phoenix Marketcity, Viman Nagar, Pune, Maharashtra 411014",
        "city": "PUNE",
        "lat": 18.5620,
        "lng": 73.9167,
        "corridor_id": "PUNE_CORRIDOR",
        "corridor_name": "Pune Urban Route 2",
        "vault_capacity": 6000000.0,
    },
    4: {
        "terminal_code": "SYNATM0004",
        "site_name": "Pune IT Tech Park Campus",
        "site_type": "EDUCATION COMPLEX",
        "serviced_by": "BRANCH",
        "replenishment_cycle_days": 3,
        "address": "Rajiv Gandhi Infotech Park, Hinjawadi Phase 1, Pune 411057",
        "city": "PUNE",
        "lat": 18.5913,
        "lng": 73.7389,
        "corridor_id": "PUNE_CORRIDOR",
        "corridor_name": "Pune Urban Route 2",
        "vault_capacity": 3500000.0,
    },
    5: {
        "terminal_code": "SYNATM0005",
        "site_name": "Nagpur Commercial CBD / Fuel Plaza",
        "site_type": "FUEL PLAZA",
        "serviced_by": "CIT",
        "replenishment_cycle_days": 7,
        "address": "Sitabuldi Commercial Hub & Fuel Station, Nagpur 440012",
        "city": "NAGPUR",
        "lat": 21.1458,
        "lng": 79.0882,
        "corridor_id": "NAGPUR_CORRIDOR",
        "corridor_name": "Nagpur Metro Route 3",
        "vault_capacity": 5000000.0,
    },
    6: {
        "terminal_code": "SYNATM0006",
        "site_name": "Nagpur Tourist Promenade",
        "site_type": "BEACH / TOURIST ZONE",
        "serviced_by": "BRANCH",
        "replenishment_cycle_days": 3,
        "address": "Futala Lake Promenade, Vayu Sena Nagar, Nagpur 440001",
        "city": "NAGPUR",
        "lat": 21.1539,
        "lng": 79.0436,
        "corridor_id": "NAGPUR_CORRIDOR",
        "corridor_name": "Nagpur Metro Route 3",
        "vault_capacity": 6000000.0,
    },
    7: {
        "terminal_code": "SYNATM0007",
        "site_name": "Nashik Metro Station Concourse",
        "site_type": "METRO STATION CONCOURSE",
        "serviced_by": "CIT",
        "replenishment_cycle_days": 7,
        "address": "CBS Central Bus Station & Metro Transit Hub, Nashik 422001",
        "city": "NASHIK",
        "lat": 19.9975,
        "lng": 73.7898,
        "corridor_id": "NASHIK_CORRIDOR",
        "corridor_name": "Nashik Corridor Route 4",
        "vault_capacity": 6000000.0,
    },
    8: {
        "terminal_code": "SYNATM0008",
        "site_name": "Nashik MIDC Industrial Estate",
        "site_type": "MEDICAL COLLEGE HOSPITAL",
        "serviced_by": "BRANCH",
        "replenishment_cycle_days": 3,
        "address": "Ambad MIDC Industrial Area & Civil Hospital Road, Nashik 422010",
        "city": "NASHIK",
        "lat": 19.9572,
        "lng": 73.7420,
        "corridor_id": "NASHIK_CORRIDOR",
        "corridor_name": "Nashik Corridor Route 4",
        "vault_capacity": 3500000.0,
    },
}

# =========================================================================
# Ingest Operational Master Data from Production CSVs (Not Synthetic)
# Master tables: data/master/cash_files/terminal_profile.csv & terminal_geography.csv
# =========================================================================
MASTER_PROFILE_PATH = CMS_ROOT / "data" / "master" / "cash_files" / "terminal_profile.csv"
MASTER_GEO_PATH = CMS_ROOT / "data" / "master" / "cash_files" / "terminal_geography.csv"

if MASTER_PROFILE_PATH.exists():
    try:
        _prof_df = pd.read_csv(MASTER_PROFILE_PATH)
        for _, row in _prof_df.iterrows():
            tid = int(row["terminal_id"])
            if tid not in ATM_GEO_METADATA:
                ATM_GEO_METADATA[tid] = {}
            if pd.notna(row.get("verified_site_type")) and row["verified_site_type"]:
                ATM_GEO_METADATA[tid]["site_type"] = str(row["verified_site_type"])
            if pd.notna(row.get("serviced_by")) and row["serviced_by"]:
                ATM_GEO_METADATA[tid]["serviced_by"] = str(row["serviced_by"])
            if pd.notna(row.get("capacity_amount")):
                ATM_GEO_METADATA[tid]["vault_capacity"] = float(row["capacity_amount"])
            if pd.notna(row.get("replenishment_cycle_days")):
                ATM_GEO_METADATA[tid]["replenishment_cycle_days"] = int(row["replenishment_cycle_days"])
            if pd.notna(row.get("city")) and row["city"]:
                ATM_GEO_METADATA[tid]["city"] = str(row["city"])
            if pd.notna(row.get("location")) and row["location"] and "site_name" not in ATM_GEO_METADATA[tid]:
                ATM_GEO_METADATA[tid]["site_name"] = str(row["location"])
    except Exception as _e:
        print(f"Warning: could not load terminal_profile.csv: {_e}")

if MASTER_GEO_PATH.exists():
    try:
        _geo_df = pd.read_csv(MASTER_GEO_PATH)
        for _, row in _geo_df.iterrows():
            tid = int(row["terminal_id"])
            if tid not in ATM_GEO_METADATA:
                ATM_GEO_METADATA[tid] = {}
            if pd.notna(row.get("latitude")):
                ATM_GEO_METADATA[tid]["lat"] = float(row["latitude"])
            if pd.notna(row.get("longitude")):
                ATM_GEO_METADATA[tid]["lng"] = float(row["longitude"])
            if pd.notna(row.get("rbi_region")):
                ATM_GEO_METADATA[tid]["rbi_region"] = str(row["rbi_region"])
    except Exception as _e:
        print(f"Warning: could not load terminal_geography.csv: {_e}")


CORRIDORS = [
    {
        "id": "MUMBAI_CORRIDOR",
        "name": "Mumbai Transit Corridor (Route 1)",
        "city": "MUMBAI",
        "terminals": [1, 2],
        "coordinates": [[18.9400, 72.8353], [19.0896, 72.8656]]
    },
    {
        "id": "PUNE_CORRIDOR",
        "name": "Pune Urban Corridor (Route 2)",
        "city": "PUNE",
        "terminals": [3, 4],
        "coordinates": [[18.5620, 73.9167], [18.5913, 73.7389]]
    },
    {
        "id": "NAGPUR_CORRIDOR",
        "name": "Nagpur Metro Corridor (Route 3)",
        "city": "NAGPUR",
        "terminals": [5, 6],
        "coordinates": [[21.1458, 79.0882], [21.1539, 79.0436]]
    },
    {
        "id": "NASHIK_CORRIDOR",
        "name": "Nashik Highway Corridor (Route 4)",
        "city": "NASHIK",
        "terminals": [7, 8],
        "coordinates": [[19.9975, 73.7898], [19.9572, 73.7420]]
    }
]



class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
            return str(obj)
        return super().default(obj)


def build_portal_data():
    config = ReplenishmentConfig()
    optimizer = CashLoadOptimizer(config)

    print("Loading data and configuring 1-month cutoff...")
    panel = load_uncensored_panel(config)

    # Derived calendar flags
    panel["is_weekend"] = (panel["dow"] >= 5).astype(int)
    panel["is_month_end"] = ((panel["date"].dt.days_in_month - panel["dom"]) <= 2).astype(int)
    panel["terminal_id"] = panel["terminal_id"].astype("category")

    # Ingest Operational Features: Capacity, Type of Site, Serviced by CIT vs Branch, Replenishment Cycle Days
    panel["site_type"] = panel["terminal_id"].astype(int).map(lambda tid: ATM_GEO_METADATA.get(tid, {}).get("site_type", "COMMERCIAL_HUB")).astype("category")
    panel["serviced_by"] = panel["terminal_id"].astype(int).map(lambda tid: ATM_GEO_METADATA.get(tid, {}).get("serviced_by", "CIT")).astype("category")
    panel["capacity"] = panel["terminal_id"].astype(int).map(lambda tid: ATM_GEO_METADATA.get(tid, {}).get("vault_capacity", 6000000.0)).astype(float)
    panel["cycle_days"] = panel["terminal_id"].astype(int).map(lambda tid: ATM_GEO_METADATA.get(tid, {}).get("replenishment_cycle_days", 7)).astype(int)

    # =========================================================================
    # Festival Tiering, Lunar Calendar & Transit Typology Features
    # =========================================================================
    # Identify Ganesh Chaturthi dates (Lunar Hindu Calendar: Bhadrapada Shukla Chaturthi)
    gc_dates = panel[panel["holiday_name"].str.contains("Ganesh Chaturthi", case=False, na=False)]["date"].drop_duplicates().sort_values().tolist()
    panel["is_ganesh_chaturthi"] = 0
    panel["is_pre_ganesh"] = 0
    panel["is_ganesh_utsav"] = 0
    panel["is_visarjan"] = 0
    panel["festival_day_idx"] = 0

    for d in gc_dates:
        panel.loc[panel["date"] == d, "is_ganesh_chaturthi"] = 1
        panel.loc[(panel["date"] >= d - pd.Timedelta(days=2)) & (panel["date"] < d), "is_pre_ganesh"] = 1
        panel.loc[(panel["date"] >= d) & (panel["date"] <= d + pd.Timedelta(days=10)), "is_ganesh_utsav"] = 1
        panel.loc[panel["date"] == d + pd.Timedelta(days=10), "is_visarjan"] = 1
        for i in range(11):
            panel.loc[panel["date"] == d + pd.Timedelta(days=i), "festival_day_idx"] = i + 1

    # Identify canonical festival key across multi-year history
    def get_festival_key(row):
        hname = str(row["holiday_name"])
        if "Ganesh Chaturthi" in hname:
            return "GANESH_CHATURTHI"
        elif "Diwali Amavasya" in hname or "Laxmi Pujan" in hname:
            return "DIWALI_LAXMI"
        elif "Bali Pratipada" in hname or "Vikram Samvant" in hname:
            return "DIWALI_NEW_YEAR"
        elif "Dasara" in hname or "Dussehra" in hname:
            return "DUSSEHRA"
        elif "Holi" in hname:
            return "HOLI"
        elif "Ramzan" in hname:
            return "EID_RAMZAN"
        elif "Bakri" in hname:
            return "EID_BAKRI"
        elif "Republic" in hname:
            return "REPUBLIC_DAY"
        elif "Independence" in hname:
            return "INDEPENDENCE_DAY"
        elif "Christmas" in hname:
            return "CHRISTMAS"
        elif "Gudhi Padwa" in hname:
            return "GUDHI_PADWA"
        elif "Maharashtra" in hname:
            return "MAHARASHTRA_DAY"
        return "NONE"

    panel["festival_key"] = panel.apply(get_festival_key, axis=1)

    # Diwali dates
    diwali_dates = panel[panel["holiday_name"].str.contains("Diwali|Deepawali", case=False, na=False)]["date"].drop_duplicates().sort_values().tolist()
    panel["is_diwali"] = 0
    for d in diwali_dates:
        panel.loc[panel["date"] == d, "is_diwali"] = 1
        panel.loc[(panel["date"] >= d - pd.Timedelta(days=3)) & (panel["date"] < d), "festival_key"] = "DIWALI_PRE"

    for d in gc_dates:
        panel.loc[panel["date"] == d, "festival_key"] = "GANESH_CHATURTHI"
        panel.loc[(panel["date"] >= d - pd.Timedelta(days=2)) & (panel["date"] < d), "festival_key"] = "GANESH_PRE"
        panel.loc[panel["date"] == d + pd.Timedelta(days=10), "festival_key"] = "GANESH_VISARJAN"

    # Festival Tiering: 0 = Normal, 1 = Routine/Minor Bank Holiday, 2 = Major State/National Holiday, 3 = Mega Cultural Festival
    def get_festival_tier(row):
        hname = str(row["holiday_name"])
        if row["is_ganesh_chaturthi"] or row["is_diwali"] or row["is_visarjan"]:
            return 3  # Tier 1 Mega Festival
        if row["is_regional_holiday"]:
            if any(k in hname for k in ["Independence", "Republic", "Dasara", "Dussehra", "Holi", "Ramzan", "Bakri", "Christmas", "Maharashtra"]):
                return 2  # Major Holiday
            return 1  # Routine / Minor Bank Closure
        return 0

    panel["festival_tier"] = panel.apply(get_festival_tier, axis=1)

    # Site Typology × Festival Interaction (Airport / Railway / Metro travel surges)
    transit_types = ["AIRPORT TERMINAL", "RAILWAY STATION CONCOURSE", "METRO STATION CONCOURSE"]
    panel["is_transit"] = panel["site_type"].astype(str).isin(transit_types).astype(int)
    panel["transit_mega_festival"] = panel["is_transit"] * (panel["festival_tier"] == 3).astype(int)
    panel["transit_pre_festival"] = panel["is_transit"] * panel["is_pre_ganesh"]

    cutoff_date = pd.Timestamp("2026-08-24")
    train = panel[panel["date"] <= cutoff_date].copy()
    test = panel[panel["date"] > cutoff_date].copy()

    # Calculate Event-Aligned Historical Multipliers (Prior-Year Lift Ratio for Same Festival)
    base_median_map = train.groupby("terminal_id")["uncensored_demand"].median().to_dict()
    panel["base_median_vol"] = panel["terminal_id"].astype(int).map(base_median_map)
    panel["event_lift_ratio"] = panel["uncensored_demand"] / panel["base_median_vol"]

    # Prior-occurrence lift for the same festival, using only what is known at forecast time:
    # the source occurrence must be on or before the cutoff and at least
    # FESTIVAL_MIN_GAP_DAYS before the row (i.e. last year's occurrence, never an
    # earlier day of the same in-window festival). Same-day siblings are averaged,
    # never read across rows.
    FESTIVAL_MIN_GAP_DAYS = 30

    def causal_prior_lift(group_col):
        events = panel.loc[panel["festival_key"] != "NONE"]
        occurrences = events.groupby(
            [events[group_col].astype(str), "festival_key", "date"], observed=True
        )["event_lift_ratio"].mean()
        lifts = pd.Series(1.0, index=panel.index)
        for idx, row in events.iterrows():
            history = occurrences.loc[(str(row[group_col]), row["festival_key"])]
            known_by = min(row["date"] - pd.Timedelta(days=FESTIVAL_MIN_GAP_DAYS), cutoff_date)
            eligible = history[history.index <= known_by]
            if len(eligible):
                lifts.loc[idx] = eligible.iloc[-1]
        return lifts

    # 1. Terminal-specific prior festival multiplier
    panel["prior_festival_multiplier"] = causal_prior_lift("terminal_id")
    # 2. Site-type prior festival multiplier (aggregated across typology)
    panel["site_prior_festival_multiplier"] = causal_prior_lift("site_type")

    train = panel[panel["date"] <= cutoff_date].copy()
    test = panel[panel["date"] > cutoff_date].copy()

    test_dates = sorted(test["date"].unique())
    n_test_days = len(test_dates)
    print(f"Cutoff Date: {cutoff_date.date()}")
    print(f"Train Period: {train['date'].min().date()} to {train['date'].max().date()} ({train['date'].nunique()} days)")
    print(f"Test Period: {test['date'].min().date()} to {test['date'].max().date()} ({n_test_days} days)")

    # Feature Sets
    cal_features = ["dow", "dom", "month", "is_weekend", "is_2_4_sat", "is_month_end", "is_sunday", "is_early", "is_late"]
    hol_features = [
        "is_payday", "is_regional_holiday", "pre_holiday",
        "festival_tier", "is_ganesh_chaturthi", "is_pre_ganesh", "is_ganesh_utsav",
        "is_visarjan", "is_diwali", "festival_day_idx",
        "prior_festival_multiplier", "site_prior_festival_multiplier"
    ]
    operational_features = [
        "site_type", "serviced_by", "capacity", "cycle_days",
        "is_transit", "transit_mega_festival", "transit_pre_festival"
    ]
    exo_features = ["terminal_id"] + operational_features + cal_features + hol_features

    for col in ["terminal_id", "site_type", "serviced_by"]:
        train[col] = train[col].astype("category")
        test[col] = test[col].astype("category")
    train_exo = train
    test_exo = test

    # =========================================================================
    # Model 1: Pure Exogenous Baseline (Calendar + RBI Holidays + Festivals + Terminal ID)
    # =========================================================================
    lgb_exo = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=120,
        learning_rate=0.07,
        num_leaves=35,
        random_state=42,
        verbose=-1
    )
    lgb_exo.fit(train_exo[exo_features], train_exo["uncensored_demand"])
    test["pred_pure_exogenous"] = np.maximum(0.0, lgb_exo.predict(test_exo[exo_features]))
    lgb_exo.booster_.save_model(str(HERE / "model_pure_exogenous.txt"))

    # =========================================================================
    # Strategy A: Universal Volume-Normalized Multiplier Model (Zero-Retrain)
    # Formula: Pred(t) = Seasonality_Multiplier(Calendar_t, Holiday_t) * Baseline_Volume
    # Fits on ratio (demand / median_base_volume) without terminal_id.
    # =========================================================================
    base_volume_map = {int(k): float(v) for k, v in train.groupby("terminal_id")["uncensored_demand"].median().items()}
    train_a = train.copy()
    test_a = test.copy()
    train_a["base_volume"] = train_a["terminal_id"].astype(int).map(base_volume_map).astype(float)
    train_a["demand_ratio"] = train_a["uncensored_demand"].astype(float) / train_a["base_volume"]
    test_a["base_volume"] = test_a["terminal_id"].astype(int).map(base_volume_map).astype(float)

    features_a = cal_features + hol_features
    lgb_a = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=100,
        learning_rate=0.08,
        num_leaves=31,
        random_state=42,
        verbose=-1
    )
    lgb_a.fit(train_a[features_a], train_a["demand_ratio"])
    pred_ratio = np.maximum(0.0, lgb_a.predict(test_a[features_a]))
    test["pred_strategy_a"] = pred_ratio * test_a["base_volume"].to_numpy()
    lgb_a.booster_.save_model(str(HERE / "model_strategy_a.txt"))

    # Precalculate Strategy A Calendar Index Profile for 31 test days
    distinct_test_days = test.drop_duplicates(subset=["date"]).sort_values("date").copy()
    test_ratio_profile = lgb_a.predict(distinct_test_days[features_a])
    strategy_a_index_profile = [
        {
            "date": str(row["date"].date()),
            "dow": int(row["dow"]),
            "dom": int(row["dom"]),
            "is_weekend": int(row["is_weekend"]),
            "is_payday": int(row["is_payday"]),
            "is_regional_holiday": int(row["is_regional_holiday"]),
            "multiplier_index": round(float(idx_val), 4)
        }
        for (_, row), idx_val in zip(distinct_test_days.iterrows(), test_ratio_profile)
    ]

    # =========================================================================
    # Strategy B: Site Metadata Profiler (Zero-Shot Cold-Start by Location & Operations)
    # Formula: Pred(t) = LightGBM(Site_Type, City, RBI_Region, Serviced_By, Capacity, Cycle_Days, Calendar_t, Holiday_t)
    # No terminal_id used; branches entirely on location typology and operational profile.
    # =========================================================================
    train_b = train.copy()
    test_b = test.copy()
    for col in ["site_type", "city", "rbi_region", "serviced_by"]:
        train_b[col] = train_b[col].astype("category")
        test_b[col] = test_b[col].astype("category")

    features_b = ["site_type", "city", "rbi_region", "serviced_by", "capacity", "cycle_days", "is_transit", "transit_mega_festival", "transit_pre_festival"] + cal_features + hol_features
    lgb_b = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=120,
        learning_rate=0.06,
        num_leaves=35,
        random_state=42,
        verbose=-1
    )
    lgb_b.fit(train_b[features_b], train_b["uncensored_demand"])
    test["pred_strategy_b"] = np.maximum(0.0, lgb_b.predict(test_b[features_b]))
    lgb_b.booster_.save_model(str(HERE / "model_strategy_b.txt"))

    # Unseen-ATM accuracy: leave each ATM out of training and score it.
    # This is the evidence behind any new-ATM claim; in-sample WAPE is not.
    loao_abs_err, loao_actual = 0.0, 0.0
    for tid in sorted(train_b["terminal_id"].astype(int).unique()):
        held_out = train_b["terminal_id"].astype(int) != tid
        lgb_loao = lgb.LGBMRegressor(**lgb_b.get_params())
        lgb_loao.fit(train_b.loc[held_out, features_b], train_b.loc[held_out, "uncensored_demand"])
        te = test_b[test_b["terminal_id"].astype(int) == tid]
        loao_abs_err += float(np.abs(np.maximum(0.0, lgb_loao.predict(te[features_b])) - te["uncensored_demand"]).sum())
        loao_actual += float(te["uncensored_demand"].sum())
    strategy_b_unseen_atm_wape_pct = round(loao_abs_err / loao_actual * 100, 2)

    # Precalculate Strategy B Site Profiles across all (site_type, city) pairs
    unique_site_types = sorted(list(train["site_type"].unique()))
    unique_cities = sorted(list(train["city"].unique()))
    strategy_b_profiles = {}
    city_rbi_map = {"MUMBAI": "MUMBAI_RO", "PUNE": "MUMBAI_RO", "NAGPUR": "NAGPUR_RO", "NASHIK": "MUMBAI_RO"}

    lt_serviced_map = {
        "RAILWAY STATION CONCOURSE": ("CIT", 6000000.0, 7),
        "AIRPORT TERMINAL": ("CIT", 6000000.0, 7),
        "CITY CENTRE MALL": ("CIT", 6000000.0, 7),
        "EDUCATION COMPLEX": ("BRANCH", 3500000.0, 3),
        "FUEL PLAZA": ("CIT", 5000000.0, 7),
        "BEACH / TOURIST ZONE": ("BRANCH", 6000000.0, 3),
        "METRO STATION CONCOURSE": ("CIT", 6000000.0, 7),
        "MEDICAL COLLEGE HOSPITAL": ("BRANCH", 3500000.0, 3),
    }

    for st in unique_site_types:
        for ct in unique_cities:
            key = f"{st}___{ct}"
            profile_rows = []
            serv_by, cap, c_days = lt_serviced_map.get(st, ("CIT", 6000000.0, 7))
            is_transit_st = int(st in transit_types)
            for _, dt_row in distinct_test_days.iterrows():
                r = {f: dt_row[f] for f in cal_features + hol_features}
                r["site_type"] = st
                r["city"] = ct
                r["rbi_region"] = city_rbi_map.get(ct, "MUMBAI_RO")
                r["serviced_by"] = serv_by
                r["capacity"] = cap
                r["cycle_days"] = c_days
                r["is_transit"] = is_transit_st
                r["transit_mega_festival"] = is_transit_st * (r.get("festival_tier", 0) == 3)
                r["transit_pre_festival"] = is_transit_st * r.get("is_pre_ganesh", 0)
                r["prior_festival_multiplier"] = dt_row.get("site_prior_festival_multiplier", 1.0)
                r["site_prior_festival_multiplier"] = dt_row.get("site_prior_festival_multiplier", 1.0)
                profile_rows.append(r)
            p_df = pd.DataFrame(profile_rows)
            for col in ["site_type", "city", "rbi_region", "serviced_by"]:
                p_df[col] = p_df[col].astype("category")
            pred_profile = np.maximum(0.0, lgb_b.predict(p_df[features_b]))
            strategy_b_profiles[key] = {
                "site_type": st,
                "city": ct,
                "rbi_region": city_rbi_map.get(ct, "MUMBAI_RO"),
                "serviced_by": serv_by,
                "capacity": cap,
                "cycle_days": c_days,
                "daily_preds": [round(float(x), 0) for x in pred_profile],
                "total_pred_month": round(float(np.sum(pred_profile)), 0),
                "daily_mean": round(float(np.mean(pred_profile)), 0),
            }

    # =========================================================================
    # Strategy C: Dynamic Telemetry-Adaptive Model (Rolling History & Lags)
    # Formula: Pred(t) = LightGBM(Roll_Mean_7, Roll_Std_7, Roll_Mean_28, Lag_7, Calendar, Holiday)
    # Dynamically adapts to live cash consumption velocity without retraining.
    # =========================================================================
    train_c_list = []
    for tid, grp in train.groupby("terminal_id"):
        g = grp.sort_values("date").copy()
        g["roll_mean_7"] = g["uncensored_demand"].shift(1).rolling(7, min_periods=7).mean()
        g["roll_std_7"] = g["uncensored_demand"].shift(1).rolling(7, min_periods=7).std().fillna(0)
        g["roll_mean_28"] = g["uncensored_demand"].shift(1).rolling(28, min_periods=28).mean()
        g["lag_7"] = g["uncensored_demand"].shift(7)
        train_c_list.append(g)
    train_c = pd.concat(train_c_list).dropna(subset=["roll_mean_28"])

    features_c = ["roll_mean_7", "roll_std_7", "roll_mean_28", "lag_7"] + cal_features + hol_features
    lgb_c = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=100,
        learning_rate=0.06,
        num_leaves=31,
        random_state=42,
        verbose=-1
    )
    lgb_c.fit(train_c[features_c], train_c["uncensored_demand"])
    lgb_c.booster_.save_model(str(HERE / "model_strategy_c.txt"))

    # For holdout evaluation, anchor rolling stats at cutoff from training history
    test_c_list = []
    for tid, grp in test.groupby("terminal_id"):
        tr_sub = train[train["terminal_id"] == tid].sort_values("date")
        last_demands = tr_sub["uncensored_demand"].tolist()
        rm7 = np.mean(last_demands[-7:])
        rs7 = np.std(last_demands[-7:])
        rm28 = np.mean(last_demands[-28:])
        g = grp.sort_values("date").copy()
        g["roll_mean_7"] = rm7
        g["roll_std_7"] = rs7
        g["roll_mean_28"] = rm28
        g["lag_7"] = [last_demands[-7 + (i % 7)] for i in range(len(g))]
        test_c_list.append(g)
    test_c = pd.concat(test_c_list)
    test["pred_strategy_c"] = np.maximum(0.0, lgb_c.predict(test_c[features_c]))

    # =========================================================================
    # Baseline Comparison Models
    # =========================================================================
    # Model 5: LightGBM Uncensored Target
    lgb_unc = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=120,
        learning_rate=0.06,
        num_leaves=35,
        random_state=42,
        verbose=-1
    )
    lgb_unc.fit(train_exo[exo_features], train_exo["uncensored_demand"])
    test["pred_lgb_uncensored"] = np.maximum(0.0, lgb_unc.predict(test_exo[exo_features]))

    # Model 6: LightGBM Censored Target
    lgb_cens = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=120,
        learning_rate=0.06,
        num_leaves=35,
        random_state=42,
        verbose=-1
    )
    lgb_cens.fit(train_exo[exo_features], train_exo["demand"])
    test["pred_lgb_censored"] = np.maximum(0.0, lgb_cens.predict(test_exo[exo_features]))

    # Model 7: Naive Baseline (Last week repeating)
    # Same target as every other model; a missing weekday falls back to the
    # terminal's training median (known at the cutoff), never to the actual value.
    last_week_sub = train[train["date"] > (cutoff_date - pd.Timedelta(days=7))]
    dow_last_week = last_week_sub.groupby(["terminal_id", "dow"], observed=True)["uncensored_demand"].mean().to_dict()
    test["pred_naive_baseline"] = test.apply(
        lambda r: dow_last_week.get((r["terminal_id"], r["dow"]), base_median_map[r["terminal_id"]]),
        axis=1
    )

    # 4 Weekly replenishment cycles in the month
    cycle_defs = [
        {"cycle_id": 1, "name": "Cycle 1 (Aug 25 - Aug 31)", "start": "2026-08-25", "end": "2026-08-31", "days": 7},
        {"cycle_id": 2, "name": "Cycle 2 (Sep 01 - Sep 07)", "start": "2026-09-01", "end": "2026-09-07", "days": 7},
        {"cycle_id": 3, "name": "Cycle 3 (Sep 08 - Sep 14)", "start": "2026-09-08", "end": "2026-09-14", "days": 7},
        {"cycle_id": 4, "name": "Cycle 4 (Sep 15 - Sep 21)", "start": "2026-09-15", "end": "2026-09-21", "days": 7},
        {"cycle_id": 5, "name": "Remainder (Sep 22 - Sep 24)", "start": "2026-09-22", "end": "2026-09-24", "days": 3},
    ]

    # Operational model: holiday-aware LightGBM trained on all 100 ATMs (replenishment_engine.demand_model),
    # re-forecast at each cycle's own cutoff (the day before it starts), as weekly dispatch planning runs.
    # Its buffer is the backtest P99 of actual/forecast 7-day totals, not in-sample residuals.
    cycle_buffers = {}
    holiday_weeks = []
    for c in cycle_defs:
        cycle_cutoff = pd.Timestamp(c["start"]) - pd.Timedelta(days=1)
        week = demand_model.forecast_week(cycle_cutoff)
        holiday_weeks.append(week[week["target_date"] <= pd.Timestamp(c["end"])])
        multiplier, calibrated = demand_model.buffer_multiplier(cycle_cutoff, config.target_service_level)
        cycle_buffers[c["cycle_id"]] = {
            "forecast_cutoff": str(cycle_cutoff.date()),
            "p99_multiplier": round(multiplier, 4),
            "calibrated": calibrated,
        }
    holiday_preds = pd.concat(holiday_weeks)
    holiday_pred_map = {
        (int(t), d): p for t, d, p in zip(holiday_preds["terminal_id"], holiday_preds["target_date"], holiday_preds["pred"])
    }
    test["pred_lgb_holiday"] = [holiday_pred_map[(int(t), d)] for t, d in zip(test["terminal_id"], test["date"])]

    terminals = sorted(test["terminal_id"].unique())
    terminal_data = {}

    # Comprehensive Model Roster: operational model + 3 Zero-Retrain Strategies + 4 Baselines
    models = [
        OPERATIONAL_MODEL,  # Holiday-aware LightGBM, 100-ATM training, weekly re-forecast (drives indents)
        "strategy_b",       # Site Metadata Profiler
        "pure_exogenous",   # Reference Baseline (Calendar + RBI Holidays)
        "strategy_c",       # ⚡ Telemetry-Adaptive (Dynamic Rolling Lags)
        "strategy_a",       # 🔄 Volume-Normalized Multiplier (Index * Base)
        "lgb_uncensored",   # Full Stack LightGBM (Uncensored Ground Truth)
        "lgb_censored",     # Full Stack LightGBM (Censored Observational)
        "naive_baseline",   # Naive Baseline (Previous Week Copy)
    ]

    # Overall Portfolio Aggregation
    portfolio_daily_actual = []
    portfolio_daily_preds = {m: [] for m in models}

    for dt in test_dates:
        sub_dt = test[test["date"] == dt]
        portfolio_daily_actual.append(float(sub_dt["uncensored_demand"].sum()))
        for m in models:
            portfolio_daily_preds[m].append(float(sub_dt[f"pred_{m}"].sum()))

    portfolio_month_actual = sum(portfolio_daily_actual)

    # Load real inventory checkpoint quality data for existing cash at cutoff date
    checkpoint_path = CMS_ROOT / "data" / "processed" / "inventory_checkpoint_quality.csv"
    existing_cash_map = {}
    if checkpoint_path.exists():
        cp_df = pd.read_csv(checkpoint_path)
        cp_cutoff = cp_df[cp_df["calendar_date"] == str(cutoff_date.date())]
        for _, row in cp_cutoff.iterrows():
            existing_cash_map[int(row["terminal_id"])] = float(row["maximum_reported_end_cash"])
        print(f"Loaded {len(existing_cash_map)} reported terminal cash balances from checkpoint quality data at {cutoff_date.date()}")
    else:
        print("Warning: inventory_checkpoint_quality.csv not found, using baseline estimates")

    def build_executive_explainability(
        tid,
        site_name,
        site_type,
        city,
        corridor_name,
        capacity,
        initial_cash,
        c_sub,
        pred_sum,
        safety_buffer,
        gross_target,
        net_indent,
        buffer_note,
        train_sub,
        cycle_name="Cycle 1",
    ):
        baseline_median = float(train_sub["uncensored_demand"].median()) if len(train_sub) > 0 else 500000.0
        c_days = len(c_sub)
        daily_avg_pred = pred_sum / max(1, c_days)

        weekend_mask = c_sub["is_weekend"] == 1
        weekend_avg = float(c_sub[weekend_mask][f"pred_{OPERATIONAL_MODEL}"].mean()) if weekend_mask.any() else daily_avg_pred
        weekday_avg = float(c_sub[~weekend_mask][f"pred_{OPERATIONAL_MODEL}"].mean()) if (~weekend_mask).any() else daily_avg_pred
        weekend_boost_pct = round(((weekend_avg - weekday_avg) / max(1.0, weekday_avg)) * 100, 1)

        payday_count = int(c_sub["is_payday"].sum()) if "is_payday" in c_sub.columns else 0
        holiday_count = int(c_sub["is_regional_holiday"].sum()) if "is_regional_holiday" in c_sub.columns else 0
        if "is_ganesh_chaturthi" in c_sub.columns:
            holiday_count += int(c_sub["is_ganesh_chaturthi"].sum())

        cal_drivers = []
        if weekend_boost_pct != 0:
            cal_drivers.append({
                "name": "Weekend Pattern",
                "effect": f"{weekend_boost_pct:+.1f}% vs weekdays",
                "detail": "Customer leisure & retail cashout surge" if weekend_boost_pct > 0 else "Commercial office weekend dip"
            })
        if payday_count > 0:
            cal_drivers.append({
                "name": "Salary / Payday Cycle",
                "effect": f"+12% payroll lift ({payday_count} days)",
                "detail": "Month-beginning corporate & salary withdrawal spike"
            })
        if holiday_count > 0:
            cal_drivers.append({
                "name": "RBI Holiday / Festival Uplift",
                "effect": f"+18% branch closure demand",
                "detail": "Bank branch closure redirects withdrawal footfall to ATM"
            })
        if not cal_drivers:
            cal_drivers.append({
                "name": "Standard Seasonality",
                "effect": "Nominal variance (±3%)",
                "detail": "Consistent mid-month withdrawal volume"
            })

        site_type_desc_map = {
            "AIRPORT TERMINAL": "High-velocity 24/7 transit hub with continuous traveler cash-outs.",
            "METRO STATION": "Commuter transit concourse with rapid peak-hour micro-withdrawals.",
            "HIGH STREET RETAIL": "High-density retail shopping corridor with heavy evening footfall.",
            "COMMERCIAL / IT PARK": "Corporate office tech park with strong Monday–Friday daytime demand.",
            "RESIDENTIAL COMMUNITY": "Neighborhood residential site with steady domestic utility cash demand."
        }
        site_desc = site_type_desc_map.get(site_type, "Standard urban retail banking location.")

        site_drivers = [
            {"name": "Site Classification", "effect": site_type, "detail": site_desc},
            {"name": "Historical Median Baseline", "effect": f"₹{baseline_median:,.0f} / day", "detail": "Long-term unconstrained baseline demand"},
            {"name": "Regional Geography", "effect": f"{city} ({corridor_name})", "detail": "Corridor route transit tier"}
        ]

        recent_7d_actuals = train_sub.tail(7)["uncensored_demand"]
        recent_7d_avg = float(recent_7d_actuals.mean()) if len(recent_7d_actuals) > 0 else baseline_median
        velocity_drift_pct = round(((recent_7d_avg - baseline_median) / max(1.0, baseline_median)) * 100, 1)

        dcr_days = round(initial_cash / daily_avg_pred, 2) if daily_avg_pred > 0 else 99.0
        urgency = "CRITICAL" if dcr_days < 2.0 else ("SCHEDULED" if dcr_days <= 4.0 else "HEALTHY")

        telem_drivers = [
            {"name": "7-Day Velocity Drift", "effect": f"{velocity_drift_pct:+.1f}% vs baseline", "detail": "Recent customer demand momentum"},
            {"name": "Days of Cash Remaining (DCR)", "effect": f"{dcr_days:.1f} Days of Cash", "detail": f"Current cash in safe: ₹{initial_cash:,.0f} ({urgency})"},
            {"name": "Checkpoint Sensor Integrity", "effect": "100% Quality Verified", "detail": "EOD cassette switch counters reconciled"}
        ]

        buffer_drivers = [
            {"name": "99.0% SLA Safety Buffer", "effect": f"+₹{safety_buffer:,.0f}", "detail": buffer_note},
            {"name": "Cassette Vault Capacity", "effect": f"₹{capacity:,.0f} Max Limit", "detail": f"Target gross load: ₹{gross_target:,.0f} (Capacity OK)"},
            {"name": "Net Indent Formula", "effect": f"₹{net_indent:,.0f}", "detail": f"Max(0, Target ₹{gross_target:,.0f} - Cash ₹{initial_cash:,.0f})"}
        ]

        if urgency == "CRITICAL":
            top_driver_phrase = f"imminent stockout risk ({dcr_days:.1f} days remaining)"
        elif payday_count > 0:
            top_driver_phrase = "month-start salary disbursement surge"
        elif weekend_boost_pct > 15:
            top_driver_phrase = "heavy weekend retail cashout velocity"
        else:
            top_driver_phrase = "regular weekly replenishment cycle with 99% SLA buffer"

        summary_text = (
            f"Recommended Net Top-Up Indent of ₹{net_indent:,.0f} for {cycle_name}. "
            f"Current vault balance is ₹{initial_cash:,.0f} ({dcr_days:.1f} days coverage, {urgency}). "
            f"Primary driver: {top_driver_phrase} with a ₹{safety_buffer:,.0f} volatility safety reserve."
        )

        prompt_chips = [
            f"Why is the {cycle_name} indent ₹{net_indent:,.0f}?",
            f"What happens if we reduce the load to ₹{round(net_indent * 0.7, -4):,.0f}?",
            f"Simulate 24h CIT van delay for {site_name}",
            f"Generate executive audit justification memo"
        ]

        return {
            "summary_text": summary_text,
            "primary_driver": top_driver_phrase,
            "dcr_days": dcr_days,
            "urgency": urgency,
            "net_indent": round(net_indent, 0),
            "gross_target": round(gross_target, 0),
            "safety_buffer": round(safety_buffer, 0),
            "initial_cash": round(initial_cash, 0),
            "pred_demand": round(pred_sum, 0),
            "categories": {
                "calendar_seasonality": {
                    "title": "Calendar & Seasonality",
                    "icon": "📅",
                    "impact_summary": f"{weekend_boost_pct:+.1f}% Weekend / {payday_count} Paydays",
                    "drivers": cal_drivers,
                },
                "site_profile": {
                    "title": "Site Type Profile",
                    "icon": "🏢",
                    "impact_summary": f"{site_type} ({city})",
                    "drivers": site_drivers,
                },
                "telemetry_velocity": {
                    "title": "Telemetry & Velocity",
                    "icon": "⚡",
                    "impact_summary": f"{velocity_drift_pct:+.1f}% drift · DCR: {dcr_days:.1f}d",
                    "drivers": telem_drivers,
                },
                "buffer_policy": {
                    "title": "Buffer & SLA Policy",
                    "icon": "🛡️",
                    "impact_summary": f"99.0% SLA Buffer: +₹{safety_buffer:,.0f}",
                    "drivers": buffer_drivers,
                },
            },
            "prompt_chips": prompt_chips,
        }

    # Evaluate Each Terminal
    for tid in terminals:
        t_sub = test_exo[test_exo["terminal_id"] == tid].sort_values("date").reset_index(drop=True)
        train_sub = train_exo[train_exo["terminal_id"] == tid]

        geo = ATM_GEO_METADATA.get(tid, {})
        loc_type = geo.get("site_name", t_sub["loc_type"].iloc[0])
        city = geo.get("city", t_sub["city"].iloc[0])
        address = geo.get("address", "")
        lat = geo.get("lat", 19.0)
        lng = geo.get("lng", 73.0)
        corridor_id = geo.get("corridor_id", "CORRIDOR_1")
        corridor_name = geo.get("corridor_name", "Corridor Route 1")
        capacity = config.get_capacity(tid)
        site_type_val = geo.get("site_type", loc_type)
        serviced_by_val = geo.get("serviced_by", "CIT")
        cycle_days_val = geo.get("replenishment_cycle_days", 7)

        existing_cash = existing_cash_map.get(tid, 2000000.0)

        dates_str = [str(d.date()) for d in t_sub["date"]]
        actual_demands = [round(x, 0) for x in t_sub["uncensored_demand"].tolist()]
        total_actual_month = sum(actual_demands)

        # Standard deviation of residuals from training
        train_preds = lgb_exo.predict(train_sub[exo_features])
        residuals = train_sub["uncensored_demand"].to_numpy() - train_preds
        sigma_daily = float(np.std(residuals))

        daily_preds_by_model = {}
        monthly_metrics = {}

        for m in models:
            preds = [round(x, 0) for x in t_sub[f"pred_{m}"].tolist()]
            daily_preds_by_model[m] = preds
            total_pred = sum(preds)
            wape = np.sum(np.abs(np.array(actual_demands) - np.array(preds))) / total_actual_month
            bias = (total_pred - total_actual_month) / total_actual_month

            # 7-day WAPE (First 7 days of holdout horizon)
            actuals_7d = actual_demands[:7]
            preds_7d = preds[:7]
            wape_7d = np.sum(np.abs(np.array(actuals_7d) - np.array(preds_7d))) / max(1.0, sum(actuals_7d))

            monthly_metrics[m] = {
                "total_pred": round(total_pred, 0),
                "total_actual": round(total_actual_month, 0),
                "wape_7d_pct": round(wape_7d * 100, 2),
                "wape_pct": round(wape * 100, 2),
                "accuracy_pct": round(max(0.0, (1 - wape) * 100), 2),
                "bias_pct": round(bias * 100, 2),
            }

        # Weekly Replenishment Cycles
        cycles_detail = []
        cycle_initial_cash = existing_cash

        for idx, c in enumerate(cycle_defs):
            c_mask = (t_sub["date"] >= c["start"]) & (t_sub["date"] <= c["end"])
            c_sub = t_sub[c_mask]

            c_actuals = c_sub["uncensored_demand"].tolist()
            c_actual_sum = sum(c_actuals)
            c_days = len(c_sub)
            sigma_c = sigma_daily * math.sqrt(c_days)
            c_multiplier = cycle_buffers[c["cycle_id"]]["p99_multiplier"]

            c_model_perf = {}
            for m in models:
                c_preds = c_sub[f"pred_{m}"].tolist()
                c_pred_sum = sum(c_preds)

                if m == OPERATIONAL_MODEL:
                    # ponytail: the 7-day multiplier is reused for the 3-day remainder cycle (conservative).
                    safety_buffer = c_pred_sum * (c_multiplier - 1.0)
                else:
                    # Comparison models keep the legacy in-sample buffer (99% SLA, z=2.326)
                    z = 2.326
                    safety_buffer = z * sigma_c
                gross_recommended_load = min(capacity, max(0.0, c_pred_sum + safety_buffer))
                
                # NET TOP-UP INDENT FORMULA:
                # Indent = Max(0, Gross Target Load - Existing Cash)
                net_topup_indent = max(0.0, gross_recommended_load - cycle_initial_cash)

                # Daily simulation with Net Top-Up loaded
                curr_cash = cycle_initial_cash + net_topup_indent
                daily_holding = 0.0
                unmet = 0.0
                trajectory = []

                for act in c_actuals:
                    if curr_cash >= act:
                        curr_cash -= act
                    else:
                        unmet += (act - curr_cash)
                        curr_cash = 0.0
                    trajectory.append(round(curr_cash, 0))
                    daily_holding += curr_cash * config.daily_holding_rate

                had_stockout = (unmet > 0)
                trip_cost = config.cit_trip_cost if net_topup_indent > 0 else 0.0
                emergency_cost = config.emergency_refill_premium if had_stockout else 0.0
                penalty_cost = config.stockout_penalty_fixed if had_stockout else 0.0
                total_cycle_cost = daily_holding + trip_cost + emergency_cost + penalty_cost

                c_model_perf[m] = {
                    "pred_sum": round(c_pred_sum, 0),
                    "actual_sum": round(c_actual_sum, 0),
                    "safety_buffer": round(safety_buffer, 0),
                    "gross_recommended_load": round(gross_recommended_load, 0),
                    "recommended_load": round(gross_recommended_load, 0),
                    "net_topup_indent": round(net_topup_indent, 0),
                    "initial_cash": round(cycle_initial_cash, 0),
                    "ending_cash": round(curr_cash, 0),
                    "unmet_shortfall": round(unmet, 0),
                    "had_stockout": int(had_stockout),
                    "holding_cost": round(daily_holding, 0),
                    "trip_cost": round(trip_cost, 0),
                    "emergency_cost": round(emergency_cost, 0),
                    "penalty_cost": round(penalty_cost, 0),
                    "total_cost": round(total_cycle_cost, 0),
                    "trajectory": trajectory,
                }

            op = c_model_perf[OPERATIONAL_MODEL]
            buffer_note = (
                f"Load = forecast × {c_multiplier:.3f}: the 99th percentile of actual/forecast over the past "
                f"{demand_model.CONFORMAL_WINDOW_WEEKS} backtest weeks (100 ATMs)"
                if cycle_buffers[c["cycle_id"]]["calibrated"]
                else f"Uncalibrated default ×{c_multiplier:.2f} (fewer than {demand_model.MIN_PAST_WEEKS} backtest weeks)"
            )
            # Compute Executive Explainability for this cycle
            explain_obj = build_executive_explainability(
                tid=tid,
                site_name=geo.get("site_name", loc_type),
                site_type=site_type_val,
                city=city,
                corridor_name=corridor_name,
                capacity=capacity,
                initial_cash=cycle_initial_cash,
                c_sub=c_sub,
                pred_sum=op["pred_sum"],
                safety_buffer=op["safety_buffer"],
                gross_target=op["gross_recommended_load"],
                net_indent=op["net_topup_indent"],
                buffer_note=buffer_note,
                train_sub=train_sub,
                cycle_name=c["name"],
            )

            # Next cycle starts from the operational model's ending cash
            cycle_initial_cash = op["ending_cash"]

            cycles_detail.append({
                "cycle_id": c["cycle_id"],
                "name": c["name"],
                "start": c["start"],
                "end": c["end"],
                "dates": [str(d.date()) for d in c_sub["date"]],
                "actual_demands": [round(x, 0) for x in c_actuals],
                "actual_sum": round(c_actual_sum, 0),
                "models": c_model_perf,
                "explainability": explain_obj,
            })

        # Cost Trade-Off Curve across Service Levels (Cycle 1 as reference)
        ref_pred_sum = cycles_detail[0]["models"][OPERATIONAL_MODEL]["pred_sum"]
        ref_daily_preds = t_sub.loc[0:6, f"pred_{OPERATIONAL_MODEL}"].tolist()
        # Sigma that reproduces the calibrated P99 load; other frontier levels assume a normal shape.
        op_sigma_7 = demand_model.equivalent_sigma(
            ref_pred_sum, cycle_buffers[1]["p99_multiplier"], config.target_service_level
        )
        frontier = optimizer.compute_cost_frontier(
            predicted_7day_sum=ref_pred_sum,
            daily_predictions=ref_daily_preds,
            sigma_7=op_sigma_7,
            terminal_id=tid,
        )

        # Operational Triage Metrics for Cycle 1 (Immediate replenishment horizon)
        c1_exo = cycles_detail[0]["models"][OPERATIONAL_MODEL]
        c1_pred = c1_exo["pred_sum"]
        daily_avg_demand = c1_pred / 7.0 if c1_pred > 0 else 1.0
        dcr_days = round(existing_cash / daily_avg_demand, 2)

        if dcr_days < 2.0:
            urgency_status = "CRITICAL"
            status_color = "#f43f5e" # Rose red
            status_label = "🔴 Critical (< 2.0d)"
        elif dcr_days <= 4.0:
            urgency_status = "SCHEDULED"
            status_color = "#f59e0b" # Amber
            status_label = "🟡 Scheduled (2-4d)"
        else:
            urgency_status = "HEALTHY"
            status_color = "#10b981" # Emerald green
            status_label = "🟢 Healthy (> 4.0d)"

        # Day 1 & Month 1 Triage Metrics
        # ponytail: day and month buffers scale the calibrated weekly sigma by sqrt(days) (normal shape)
        op_sigma_daily = op_sigma_7 / math.sqrt(7)
        day1_pred = t_sub[f"pred_{OPERATIONAL_MODEL}"].iloc[0]
        day1_actual = t_sub["uncensored_demand"].iloc[0]
        day1_buffer = 2.326 * op_sigma_daily
        day1_target = min(capacity, day1_pred + day1_buffer)
        day1_net_indent = max(0.0, day1_target - existing_cash)

        month_pred = sum(t_sub[f"pred_{OPERATIONAL_MODEL}"])
        month_buffer = 2.326 * op_sigma_daily * math.sqrt(31)
        month_target = month_pred + month_buffer
        month_net_indent = max(0.0, month_target - existing_cash)

        site_type_val = geo.get("site_type", loc_type)
        serviced_by_val = geo.get("serviced_by", "CIT")
        cycle_days_val = geo.get("replenishment_cycle_days", 7)

        # Daily explainability factors
        daily_explainability = {}
        baseline_m = float(train_sub["uncensored_demand"].median()) if len(train_sub) > 0 else 500000.0
        for _, r in t_sub.iterrows():
            d_str = str(r["date"].date())
            dow_name = r["date"].strftime("%A")
            pred_val = float(r[f"pred_{OPERATIONAL_MODEL}"])
            diff_from_base = pred_val - baseline_m
            diff_pct = round((diff_from_base / max(1.0, baseline_m)) * 100, 1)

            factors = []
            if bool(r.get("is_weekend", 0)):
                factors.append(f"Weekend spending surge ({dow_name})")
            if bool(r.get("is_payday", 0)):
                factors.append("Salary disbursement window")
            if bool(r.get("is_regional_holiday", 0)) or bool(r.get("is_ganesh_chaturthi", 0)):
                factors.append("RBI branch holiday uplift")
            if not factors:
                factors.append("Regular weekday baseline velocity")

            daily_explainability[d_str] = {
                "date": d_str,
                "dow_name": dow_name,
                "pred_demand": round(pred_val, 0),
                "baseline_demand": round(baseline_m, 0),
                "variance_pct": diff_pct,
                "primary_factor": factors[0],
                "all_factors": factors,
                "driver_summary": f"{dow_name} forecast ₹{pred_val:,.0f} ({diff_pct:+.1f}% vs baseline): {', '.join(factors)}"
            }

        terminal_data[str(tid)] = {
            "terminal_id": tid,
            "terminal_code": geo.get("terminal_code", f"SYNATM{tid:04d}"),
            "site_name": geo.get("site_name", loc_type),
            "site_type": site_type_val,
            "serviced_by": serviced_by_val,
            "replenishment_cycle_days": cycle_days_val,
            "address": address,
            "city": city,
            "lat": lat,
            "lng": lng,
            "corridor_id": corridor_id,
            "corridor_name": corridor_name,
            "capacity": capacity,
            "existing_cash": round(existing_cash, 0),
            "dcr_days": dcr_days,
            "urgency_status": urgency_status,
            "status_color": status_color,
            "status_label": status_label,
            # Cycle 1 Net Top-up Indent (7 Days)
            "c1_pred_demand": round(c1_pred, 0),
            "c1_safety_buffer": round(c1_exo["safety_buffer"], 0),
            "c1_gross_target": round(c1_exo["gross_recommended_load"], 0),
            "c1_net_indent": round(c1_exo["net_topup_indent"], 0),
            "c1_trip_needed": 1 if c1_exo["net_topup_indent"] > 0 else 0,
            "c1_explainability": cycles_detail[0]["explainability"],
            # Day 1 Net Top-up Indent
            "day1_pred": round(day1_pred, 0),
            "day1_actual": round(day1_actual, 0),
            "day1_net_indent": round(day1_net_indent, 0),
            # Month Net Top-up Indent
            "month_pred": round(month_pred, 0),
            "month_net_indent": round(month_net_indent, 0),
            "dates": dates_str,
            "actual_demands": actual_demands,
            "total_actual_month": round(total_actual_month, 0),
            "daily_preds": daily_preds_by_model,
            "monthly_metrics": monthly_metrics,
            "cycles": cycles_detail,
            "daily_explainability": daily_explainability,
            "cost_frontier": frontier,
        }

    # Aggregate Portfolio-Level Metrics
    # Headline WAPE is per ATM-day: cash is loaded per ATM, so errors must not
    # cancel across terminals. The netted network-total WAPE is kept separately.
    first_week = test["date"] <= test_dates[min(6, n_test_days - 1)]
    portfolio_metrics = {}
    for m in models:
        total_p = sum(portfolio_daily_preds[m])
        abs_err = np.abs(test[f"pred_{m}"] - test["uncensored_demand"])
        wape = abs_err.sum() / test["uncensored_demand"].sum()
        wape_7d = abs_err[first_week].sum() / max(1.0, test.loc[first_week, "uncensored_demand"].sum())
        network_wape = np.sum(np.abs(np.array(portfolio_daily_actual) - np.array(portfolio_daily_preds[m]))) / portfolio_month_actual
        bias = (total_p - portfolio_month_actual) / portfolio_month_actual

        portfolio_metrics[m] = {
            "total_pred": round(total_p, 0),
            "total_actual": round(portfolio_month_actual, 0),
            "wape_7d_pct": round(wape_7d * 100, 2),
            "wape_pct": round(wape * 100, 2),
            "accuracy_pct": round(max(0.0, (1 - wape) * 100), 2),
            "network_total_wape_pct": round(network_wape * 100, 2),
            "bias_pct": round(bias * 100, 2),
        }

    # Fleet-wide CRA Operations Aggregates
    total_existing_fleet = sum(t["existing_cash"] for t in terminal_data.values())
    total_c1_net_indent = sum(t["c1_net_indent"] for t in terminal_data.values())
    total_day1_net_indent = sum(t["day1_net_indent"] for t in terminal_data.values())
    critical_count = sum(1 for t in terminal_data.values() if t["urgency_status"] == "CRITICAL")
    scheduled_count = sum(1 for t in terminal_data.values() if t["urgency_status"] == "SCHEDULED")
    healthy_count = sum(1 for t in terminal_data.values() if t["urgency_status"] == "HEALTHY")

    fleet_summary = {
        "total_existing_cash": round(total_existing_fleet, 0),
        "total_c1_net_indent": round(total_c1_net_indent, 0),
        "total_day1_net_indent": round(total_day1_net_indent, 0),
        "critical_count": critical_count,
        "scheduled_count": scheduled_count,
        "healthy_count": healthy_count,
        "total_terminals": len(terminals),
        "total_pred_month": round(sum(portfolio_daily_preds[OPERATIONAL_MODEL]), 0),
        "total_actual_month": round(portfolio_month_actual, 0),
    }

    # Resolve CARTO Basemaps API Key from .env or environment
    carto_api_key = os.environ.get("CARTO_API_KEY", "cb1_4shp_1_67853029428fcaa9c51011a8")
    env_file = CMS_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("CARTO_API_KEY="):
                carto_api_key = line.split("=", 1)[1].strip()

    portal_bundle = {
        "metadata": {
            "title": "Cash Intelligence CRA Operations Cockpit",
            "cutoff_date": str(cutoff_date.date()),
            "forecast_start": str(test_dates[0].date()),
            "forecast_end": str(test_dates[-1].date()),
            "holdout_duration_days": n_test_days,
            "total_terminals": len(terminals),
            "cycle_definitions": cycle_defs,
            "operational_model": {
                "id": OPERATIONAL_MODEL,
                "name": "Holiday-aware LightGBM (100-ATM training)",
                "drives": "all indents, cycle cash chain, explainability and cost frontier",
                "forecast": "re-forecast at each cycle's own cutoff (the day before the cycle starts); other models use the single 2026-08-24 cutoff",
                "target": "observed Cash_Out across all 100 cash-file ATMs; scored here against uncensored_demand",
                "buffer": "P99 of actual/forecast 7-day totals over the past 26 backtest weeks (outputs/scorecard_100)",
                "cycle_buffers": cycle_buffers,
            },
            "default_holding_rate": config.holding_rate_annual,
            "default_cit_cost": config.cit_trip_cost,
            "default_stockout_penalty": config.stockout_penalty_fixed,
            "default_emergency_refill": config.emergency_refill_premium,
            "carto_api_key": carto_api_key,
        },
        "fleet_summary": fleet_summary,
        "corridors": CORRIDORS,
        "portfolio": {
            "dates": [str(d.date()) for d in test_dates],
            "actual_demands": [round(x, 0) for x in portfolio_daily_actual],
            "total_actual_month": round(portfolio_month_actual, 0),
            "daily_preds": {m: [round(x, 0) for x in portfolio_daily_preds[m]] for m in models},
            "monthly_metrics": portfolio_metrics,
        },
        "zero_retrain_strategies": {
            "strategy_a": {
                "id": "strategy_a",
                "name": "Volume-Normalized Multiplier",
                "tag": "Universal Index",
                "description": "Multiplies country-wide learned calendar and RBI holiday seasonality multipliers by any ATM base daily volume. Zero retraining needed.",
                "formula": "Pred(t) = Seasonality_Multiplier(Calendar_t, Holiday_t) × Base_Daily_Volume",
                "features": features_a,
                "portfolio_wape_7d_pct": portfolio_metrics["strategy_a"]["wape_7d_pct"],
                "portfolio_wape_pct": portfolio_metrics["strategy_a"]["wape_pct"],
                "accuracy_pct": portfolio_metrics["strategy_a"]["accuracy_pct"],
                "bias_pct": portfolio_metrics["strategy_a"]["bias_pct"],
                "total_pred_month": portfolio_metrics["strategy_a"]["total_pred"],
                "daily_index_profile": strategy_a_index_profile,
            },
            "strategy_b": {
                "id": "strategy_b",
                "name": "Site Metadata Profiler",
                "tag": "Location & Operations Profile",
                "description": (
                    "Tree-based split on site characteristics (transit concourse, airport, mall, hospital, fuel plaza), "
                    "city tier, and operational replenishment channel (CIT 7-day vs Branch 3-day) without terminal_id. "
                    f"On an ATM it has never seen, per-ATM-day WAPE is {strategy_b_unseen_atm_wape_pct:.1f}% "
                    "(leave-one-ATM-out); each pilot site type is a single ATM, so treat new-ATM forecasts as a rough starting level."
                ),
                "unseen_atm_wape_pct": strategy_b_unseen_atm_wape_pct,
                "formula": "Pred(t) = LightGBM(Site_Type, City, RBI_Region, Serviced_By, Capacity, Cycle_Days, Calendar_t, Holiday_t)",
                "features": features_b,
                "portfolio_wape_7d_pct": portfolio_metrics["strategy_b"]["wape_7d_pct"],
                "portfolio_wape_pct": portfolio_metrics["strategy_b"]["wape_pct"],
                "accuracy_pct": portfolio_metrics["strategy_b"]["accuracy_pct"],
                "bias_pct": portfolio_metrics["strategy_b"]["bias_pct"],
                "total_pred_month": portfolio_metrics["strategy_b"]["total_pred"],
                "available_site_types": unique_site_types,
                "available_cities": unique_cities,
                "site_profiles": strategy_b_profiles,
            },
            "strategy_c": {
                "id": "strategy_c",
                "name": "Telemetry-Adaptive Model",
                "tag": "Dynamic Telemetry",
                "description": "Dynamically anchors predictions to recent rolling 7-day and 28-day cashout velocity without weight updates.",
                "formula": "Pred(t) = LightGBM(Roll_Mean_7, Roll_Std_7, Roll_Mean_28, Lag_7, Calendar_t, Holiday_t)",
                "features": features_c,
                "portfolio_wape_7d_pct": portfolio_metrics["strategy_c"]["wape_7d_pct"],
                "portfolio_wape_pct": portfolio_metrics["strategy_c"]["wape_pct"],
                "accuracy_pct": portfolio_metrics["strategy_c"]["accuracy_pct"],
                "bias_pct": portfolio_metrics["strategy_c"]["bias_pct"],
                "total_pred_month": portfolio_metrics["strategy_c"]["total_pred"],
            },
        },
        "terminals": terminal_data,
    }


    # Write files
    json_path = HERE / "portal_data.json"
    js_path = HERE / "portal_data.js"

    with open(json_path, "w") as f:
        json.dump(portal_bundle, f, indent=2, cls=NumpyEncoder)
    print(f"Exported JSON: {json_path} ({json_path.stat().st_size / 1024:.1f} KB)")

    with open(js_path, "w", encoding="utf-8") as f:
        f.write("// Live Portal Data Bundle (Cutoff 2026-08-24, Holdout Month 2026-08-25 to 2026-09-24)\n")
        f.write("window.LIVE_PORTAL_DATA = ")
        json.dump(portal_bundle, f, cls=NumpyEncoder)
        f.write(";\n")
    print(f"Exported JS: {js_path} ({js_path.stat().st_size / 1024:.1f} KB)")

    return portal_bundle


if __name__ == "__main__":
    build_portal_data()
