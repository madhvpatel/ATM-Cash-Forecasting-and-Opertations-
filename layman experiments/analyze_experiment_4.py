"""Experiment 4: Detailed Evaluation & Statistical Analysis

Computes:
1. Controlled comparison table (Model A, B, C, D, F, Leave-One-Out, and Exp3 Seasonal Reference)
2. Paired block-bootstrap 95% Confidence Intervals (resampling blocks of consecutive weekly cycles)
3. Relative WAPE improvements (%) on Daily and 7-Day horizons
4. Segmented slices:
   - Event days (Holidays, Paydays) vs Ordinary days
   - Rain days (>=10mm, >=25mm) vs Normal weather
   - Site types and Individual ATMs (all 8 Maharashtra ATMs)
   - Lead time horizon (Day 1 through Day 7)
5. Bias changes
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output"
PRED_PATH = OUT_DIR / "experiment_4_predictions.csv"

def wape(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.sum(np.abs(actual - pred)) / np.sum(actual))

def bias(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.sum(pred - actual) / np.sum(actual))

def block_bootstrap_ci(df: pd.DataFrame, base_col: str, target_col: str, n_boot: int = 1000, seed: int = 42):
    """Paired block bootstrap over consecutive forecast weeks keeping network together."""
    rng = np.random.default_rng(seed)
    unique_origins = df["origin_date"].unique()
    n_weeks = len(unique_origins)
    
    # Pre-group by origin to make resampling fast
    week_groups = {orig: df[df["origin_date"] == orig] for orig in unique_origins}
    
    daily_wape_diffs = []
    cycle_wape_diffs = []
    
    for _ in range(n_boot):
        sample_origins = rng.choice(unique_origins, size=n_weeks, replace=True)
        boot_df = pd.concat([week_groups[o] for o in sample_origins], ignore_index=True)
        
        # Daily WAPE diff (Base - Target, positive means improvement)
        act = boot_df["demand"].to_numpy(dtype=float)
        p_base = boot_df[base_col].to_numpy(dtype=float)
        p_tgt = boot_df[target_col].to_numpy(dtype=float)
        d_base = np.sum(np.abs(act - p_base)) / np.sum(act)
        d_tgt = np.sum(np.abs(act - p_tgt)) / np.sum(act)
        daily_wape_diffs.append((d_base - d_tgt) * 100)
        
        # 7-day Cycle WAPE diff
        c_agg = boot_df.groupby(["terminal_id", "origin_date"])[["demand", base_col, target_col]].sum()
        c_act = c_agg["demand"].to_numpy(dtype=float)
        c_base = c_agg[base_col].to_numpy(dtype=float)
        c_tgt = c_agg[target_col].to_numpy(dtype=float)
        cw_base = np.sum(np.abs(c_act - c_base)) / np.sum(c_act)
        cw_tgt = np.sum(np.abs(c_act - c_tgt)) / np.sum(c_act)
        cycle_wape_diffs.append((cw_base - cw_tgt) * 100)
        
    ci_daily = (np.percentile(daily_wape_diffs, 2.5), np.percentile(daily_wape_diffs, 97.5))
    ci_cycle = (np.percentile(cycle_wape_diffs, 2.5), np.percentile(cycle_wape_diffs, 97.5))
    return ci_daily, ci_cycle

def run_analysis():
    df = pd.read_csv(PRED_PATH, parse_dates=["origin_date", "target_date"])
    print(f"Loaded {len(df)} scored predictions across {df['origin_date'].nunique()} weekly origins.")
    
    # 7-day cycle aggregated dataframe
    cycle_df = df.groupby(["terminal_id", "loc_type", "city", "origin_date"], as_index=False).agg({
        "demand": "sum",
        "Model_A_History_Only": "sum",
        "Model_B_History_Calendar": "sum",
        "Model_C_History_Cal_Holidays": "sum",
        "Model_D_History_Cal_Weather": "sum",
        "Model_F_Full_Stack": "sum",
        "Model_F_minus_Weather": "sum",
        "Model_F_minus_Holidays": "sum",
        "Model_F_minus_Calendar": "sum",
        "Model_F_minus_History": "sum",
        "Seasonal_Exp3": "sum",
    })

    models = [
        "Model_A_History_Only",
        "Model_B_History_Calendar",
        "Model_C_History_Cal_Holidays",
        "Model_D_History_Cal_Weather",
        "Model_F_Full_Stack",
        "Model_F_minus_Weather",
        "Model_F_minus_Holidays",
        "Model_F_minus_Calendar",
        "Model_F_minus_History",
        "Seasonal_Exp3",
    ]

    # Baseline WAPEs
    base_m = "Model_B_History_Calendar"
    wape_d_base = wape(df["demand"], df[base_m])
    wape_c_base = wape(cycle_df["demand"], cycle_df[base_m])
    
    # 1. Overall Model Table
    overall_rows = []
    for m in models:
        dw = wape(df["demand"], df[m])
        cw = wape(cycle_df["demand"], cycle_df[m])
        b = bias(df["demand"], df[m])
        
        # Rel imp vs Model B (History + Calendar)
        rel_imp_d = (wape_d_base - dw) / wape_d_base * 100
        rel_imp_c = (wape_c_base - cw) / wape_c_base * 100
        
        overall_rows.append({
            "Model": m,
            "Daily_WAPE_%": round(dw * 100, 2),
            "Daily_Rel_Imp_%": round(rel_imp_d, 2),
            "7Day_WAPE_%": round(cw * 100, 2),
            "7Day_Rel_Imp_%": round(rel_imp_c, 2),
            "Bias_%": round(b * 100, 2),
        })
    df_overall = pd.DataFrame(overall_rows)
    print("\n=== OVERALL MODEL PERFORMANCE ===")
    print(df_overall.to_string(index=False))

    # 2. Key Pairwise Controlled Comparisons with Block Bootstrap CI
    comparisons = [
        ("Calendar vs History Only", "Model_A_History_Only", "Model_B_History_Calendar"),
        ("Events/Holidays vs History+Calendar", "Model_B_History_Calendar", "Model_C_History_Cal_Holidays"),
        ("Weather vs History+Calendar", "Model_B_History_Calendar", "Model_D_History_Cal_Weather"),
        ("All External (Full Stack) vs History+Calendar", "Model_B_History_Calendar", "Model_F_Full_Stack"),
        # Ablations
        ("Value of Weather in Full Stack (F vs F-minus-Weather)", "Model_F_minus_Weather", "Model_F_Full_Stack"),
        ("Value of Holidays in Full Stack (F vs F-minus-Holidays)", "Model_F_minus_Holidays", "Model_F_Full_Stack"),
        ("Value of Calendar in Full Stack (F vs F-minus-Calendar)", "Model_F_minus_Calendar", "Model_F_Full_Stack"),
        ("Value of History in Full Stack (F vs Pure Exogenous)", "Model_F_minus_History", "Model_F_Full_Stack"),
    ]

    comp_rows = []
    print("\nRunning paired block bootstrap (1000 resamples)...")
    for comp_name, base_col, target_col in comparisons:
        w_d_b = wape(df["demand"], df[base_col])
        w_d_t = wape(df["demand"], df[target_col])
        diff_d = (w_d_b - w_d_t) * 100  # pp improvement
        rel_d = (w_d_b - w_d_t) / w_d_b * 100
        
        w_c_b = wape(cycle_df["demand"], cycle_df[base_col])
        w_c_t = wape(cycle_df["demand"], cycle_df[target_col])
        diff_c = (w_c_b - w_c_t) * 100  # pp improvement
        rel_c = (w_c_b - w_c_t) / w_c_b * 100
        
        b_b = bias(df["demand"], df[base_col])
        b_t = bias(df["demand"], df[target_col])
        bias_change = (b_t - b_b) * 100
        
        ci_daily, ci_cycle = block_bootstrap_ci(df, base_col, target_col)
        
        comp_rows.append({
            "Comparison": comp_name,
            "Daily_WAPE_Change_pp": f"{diff_d:+.2f} pp ({rel_d:+.1f}%)",
            "7Day_WAPE_Change_pp": f"{diff_c:+.2f} pp ({rel_c:+.1f}%)",
            "Bias_Change_pp": f"{bias_change:+.2f} pp",
            "95%_CI_Daily_pp": f"[{ci_daily[0]:+.2f}, {ci_daily[1]:+.2f}]",
            "95%_CI_7Day_pp": f"[{ci_cycle[0]:+.2f}, {ci_cycle[1]:+.2f}]",
        })
    df_comp = pd.DataFrame(comp_rows)
    print("\n=== INCREMENTAL VALUE COMPARISONS & CONFIDENCE INTERVALS ===")
    print(df_comp.to_string(index=False))

    # 3. Segmented Slices: Event days vs Ordinary days
    # Regional holiday, payday, ordinary
    holiday_mask = df["is_regional_holiday"] == 1
    payday_mask = (df["is_payday"] == 1) & (~holiday_mask)
    ordinary_mask = (~holiday_mask) & (~payday_mask)
    
    slices_event = [
        ("All Days", np.ones(len(df), dtype=bool)),
        ("Regional Holiday Days", holiday_mask),
        ("Payday Days", payday_mask),
        ("Ordinary Days", ordinary_mask),
    ]
    event_rows = []
    for name, mask in slices_event:
        sub = df[mask]
        if len(sub) == 0:
            continue
        event_rows.append({
            "Slice": name,
            "N_ATM_Days": len(sub),
            "Demand_Share_%": round(sub["demand"].sum() / df["demand"].sum() * 100, 1),
            "Hist_Only (A)": round(wape(sub["demand"], sub["Model_A_History_Only"]) * 100, 2),
            "Hist+Cal (B)": round(wape(sub["demand"], sub["Model_B_History_Calendar"]) * 100, 2),
            "Hist+Cal+Holidays (C)": round(wape(sub["demand"], sub["Model_C_History_Cal_Holidays"]) * 100, 2),
            "Full Stack (F)": round(wape(sub["demand"], sub["Model_F_Full_Stack"]) * 100, 2),
            "Exp3 Seasonal": round(wape(sub["demand"], sub["Seasonal_Exp3"]) * 100, 2),
        })
    df_events = pd.DataFrame(event_rows)
    print("\n=== SLICE: EVENT DAYS VS ORDINARY DAYS ===")
    print(df_events.to_string(index=False))

    # 4. Segmented Slices: Weather conditions (Rain >= 10mm, >= 25mm, dry)
    heavy_rain = df["rain_ge_25"] == 1
    mod_rain = (df["rain_ge_10"] == 1) & (~heavy_rain)
    dry = df["rain_ge_10"] == 0
    weather_slices = [
        ("Dry (< 10mm rain)", dry),
        ("Moderate Rain (10-25mm)", mod_rain),
        ("Heavy Rain (>= 25mm)", heavy_rain),
    ]
    wx_rows = []
    for name, mask in weather_slices:
        sub = df[mask]
        if len(sub) == 0:
            continue
        wx_rows.append({
            "Weather Condition": name,
            "N_ATM_Days": len(sub),
            "Demand_Share_%": round(sub["demand"].sum() / df["demand"].sum() * 100, 1),
            "Hist+Cal (B)": round(wape(sub["demand"], sub["Model_B_History_Calendar"]) * 100, 2),
            "Hist+Cal+Weather (D)": round(wape(sub["demand"], sub["Model_D_History_Cal_Weather"]) * 100, 2),
            "Full Stack (F)": round(wape(sub["demand"], sub["Model_F_Full_Stack"]) * 100, 2),
            "Weather Lift (B - D)": f"{(wape(sub['demand'], sub['Model_B_History_Calendar']) - wape(sub['demand'], sub['Model_D_History_Cal_Weather']))*100:+.2f} pp",
        })
    df_wx = pd.DataFrame(wx_rows)
    print("\n=== SLICE: WEATHER CONDITIONS ===")
    print(df_wx.to_string(index=False))

    # 5. Segmented Slices: Site Types and Individual ATMs
    atm_rows = []
    for tid, grp in df.groupby("terminal_id"):
        c_grp = cycle_df[cycle_df["terminal_id"] == tid]
        loc = grp["loc_type"].iloc[0]
        city = grp["city"].iloc[0]
        atm_rows.append({
            "ATM": tid,
            "City": city,
            "Site Type": loc,
            "Daily_Hist (A)": round(wape(grp["demand"], grp["Model_A_History_Only"]) * 100, 1),
            "Daily_Cal (B)": round(wape(grp["demand"], grp["Model_B_History_Calendar"]) * 100, 1),
            "Daily_Full (F)": round(wape(grp["demand"], grp["Model_F_Full_Stack"]) * 100, 1),
            "Daily_Exp3": round(wape(grp["demand"], grp["Seasonal_Exp3"]) * 100, 1),
            "7Day_Hist (A)": round(wape(c_grp["demand"], c_grp["Model_A_History_Only"]) * 100, 1),
            "7Day_Cal (B)": round(wape(c_grp["demand"], c_grp["Model_B_History_Calendar"]) * 100, 1),
            "7Day_Full (F)": round(wape(c_grp["demand"], c_grp["Model_F_Full_Stack"]) * 100, 1),
            "7Day_Exp3": round(wape(c_grp["demand"], c_grp["Seasonal_Exp3"]) * 100, 1),
        })
    df_atm = pd.DataFrame(atm_rows)
    print("\n=== SLICE: INDIVIDUAL ATMS AND SITE TYPES ===")
    print(df_atm.to_string(index=False))

    # 6. Segmented Slices: Forecast Lead Time (Horizon 1 to 7)
    lead_rows = []
    for h in range(1, 8):
        sub = df[df["horizon"] == h]
        lead_rows.append({
            "Lead Time": f"Day {h} (h={h})",
            "Hist_Only (A)": round(wape(sub["demand"], sub["Model_A_History_Only"]) * 100, 2),
            "Hist+Cal (B)": round(wape(sub["demand"], sub["Model_B_History_Calendar"]) * 100, 2),
            "Hist+Cal+Holidays (C)": round(wape(sub["demand"], sub["Model_C_History_Cal_Holidays"]) * 100, 2),
            "Hist+Cal+Weather (D)": round(wape(sub["demand"], sub["Model_D_History_Cal_Weather"]) * 100, 2),
            "Full Stack (F)": round(wape(sub["demand"], sub["Model_F_Full_Stack"]) * 100, 2),
            "Exp3 Seasonal": round(wape(sub["demand"], sub["Seasonal_Exp3"]) * 100, 2),
        })
    df_lead = pd.DataFrame(lead_rows)
    print("\n=== SLICE: FORECAST LEAD TIME (HORIZON 1 TO 7) ===")
    print(df_lead.to_string(index=False))

    # Save summary tables to CSV
    df_overall.to_csv(OUT_DIR / "experiment_4_overall_models.csv", index=False)
    df_comp.to_csv(OUT_DIR / "experiment_4_controlled_comparisons.csv", index=False)
    df_events.to_csv(OUT_DIR / "experiment_4_slice_events.csv", index=False)
    df_wx.to_csv(OUT_DIR / "experiment_4_slice_weather.csv", index=False)
    df_atm.to_csv(OUT_DIR / "experiment_4_slice_atms.csv", index=False)
    df_lead.to_csv(OUT_DIR / "experiment_4_slice_horizons.csv", index=False)
    print(f"\nAll summary tables saved to {OUT_DIR}")

if __name__ == "__main__":
    run_analysis()
