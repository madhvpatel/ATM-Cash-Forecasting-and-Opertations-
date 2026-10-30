"""Experiment 5: Cashout-Aware Demand Reconstruction & Operational Trade-off Analysis

Rigorous investigation addressing:
1. Strict causal feature engineering (zero leakage of future cashout states).
2. Proper snapshot-count denominator with complete-empty fallback.
3. Multi-policy replenishment evaluation across 6 buffer targets (CER 65% to 90%).
4. Dual-metric evaluation: Stockout cycles & Rupee shortfall vs. Loaded cash & Ending idle cash (cost of capital).
5. Controlled evaluation on ATM 7 (severe cashouts) and ATM 1 (moderate cashouts).
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import shutil

HERE = Path(__file__).resolve().parent
CMS_ROOT = HERE.parent
OUT_DIR = HERE / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR = Path("/Users/madhavpatel/.gemini/antigravity-ide/brain/9cf32330-589c-44d5-b25f-e0c20f75e706")

MIN_TRAIN_DAYS = 365
HORIZON = 7
STEP = 7
SEED = 42
ANNUAL_HOLDING_RATE = 0.06  # 6% p.a. cost of idle working capital

LGB_PARAMS = {
    "objective": "regression_l1",
    "n_estimators": 100,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
    "force_col_wise": True,
}

HISTORY_FEATURES = [
    "horizon", "lag_1", "lag_7", "lag_14", "lag_28",
    "roll_mean_7", "roll_std_7", "roll_mean_14", "roll_std_14", "roll_mean_28", "roll_std_28"
]
CALENDAR_FEATURES = ["dow", "dom", "month", "is_weekend", "is_2_4_sat", "is_month_end"]
HOLIDAY_FEATURES = ["is_payday", "is_regional_holiday", "pre_holiday"]
WEATHER_FEATURES = ["tmax_c", "tmean_c", "tmin_c", "rain_mm", "rain_ge_10", "rain_ge_25", "tmax_ge_35"]

FEATURES = HISTORY_FEATURES + CALENDAR_FEATURES + HOLIDAY_FEATURES + WEATHER_FEATURES

def wape(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.sum(np.abs(actual - pred)) / np.sum(actual))

def bias(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.sum(pred - actual) / np.sum(actual))

def prepare_dataset():
    qual = pd.read_csv(CMS_ROOT / "data" / "processed" / "inventory_checkpoint_quality.csv", parse_dates=["calendar_date"])
    atm = pd.read_csv(CMS_ROOT / "simulated" / "maharashtra" / "data" / "atm_daily.csv", parse_dates=["date"])
    merged = pd.merge(atm, qual, left_on=["terminal_id", "date"], right_on=["terminal_id", "calendar_date"])

    # Methodological check: Use actual snapshot_count as denominator, guard against 0
    safe_snap_count = np.maximum(merged["snapshot_count"].fillna(96), 1)
    zero_snaps = merged["zero_balance_snapshot_count"].fillna(0)
    
    # Active operating fraction
    merged["active_fraction"] = np.clip(1.0 - (zero_snaps / safe_snap_count), 0.20, 1.0)
    
    # Reconstructed demand with complete empty fallback
    is_fully_empty = (zero_snaps >= safe_snap_count) & (merged["demand"] == 0)
    merged["reconstructed_demand"] = merged["demand"] / merged["active_fraction"]
    
    # For any fully empty days, fill with trailing 28-day median (none in this synthetic set, but guarded)
    if is_fully_empty.sum() > 0:
        merged.loc[is_fully_empty, "reconstructed_demand"] = np.nan
        merged["reconstructed_demand"] = merged.groupby("terminal_id")["reconstructed_demand"].transform(
            lambda s: s.fillna(s.rolling(28, min_periods=1).median())
        )

    merged["is_cashout_day"] = (zero_snaps > 0).astype(int)
    merged["hours_empty"] = zero_snaps * 0.25
    return merged

def run_backtest_atm(merged, tid):
    sub = merged[merged["terminal_id"] == tid].copy().sort_values("date").reset_index(drop=True)
    dates = np.sort(sub["date"].unique())
    n_dates = len(dates)

    demand_obs = sub["demand"].to_numpy(dtype=float)
    demand_recon = sub["reconstructed_demand"].to_numpy(dtype=float)
    dom = sub["dom"].to_numpy(dtype=int)
    dow = sub["dow"].to_numpy(dtype=int)
    month = sub["month"].to_numpy(dtype=int)
    days_in_month = sub["date"].dt.days_in_month.to_numpy(dtype=int)
    is_month_end = (days_in_month - dom <= 2).astype(int)
    is_weekend = (dow >= 5).astype(int)

    # Build rolling windows on Observed vs Reconstructed strictly ending on cutoff
    rolling_obs = {}
    rolling_recon = {}
    for w in (7, 14, 28):
        rolling_obs[f"roll_mean_{w}"] = pd.Series(demand_obs).rolling(w, min_periods=w).mean().to_numpy()
        rolling_obs[f"roll_std_{w}"] = pd.Series(demand_obs).rolling(w, min_periods=w).std(ddof=1).to_numpy()
        rolling_recon[f"roll_mean_{w}"] = pd.Series(demand_recon).rolling(w, min_periods=w).mean().to_numpy()
        rolling_recon[f"roll_std_{w}"] = pd.Series(demand_recon).rolling(w, min_periods=w).std(ddof=1).to_numpy()

    origin_parts = []
    horizon_parts = []
    for origin in range(27, n_dates - 1):
        max_h = min(HORIZON, n_dates - 1 - origin)
        origin_parts.append(np.full(max_h, origin, dtype=int))
        horizon_parts.append(np.arange(1, max_h + 1, dtype=int))
    origin = np.concatenate(origin_parts)
    horizon = np.concatenate(horizon_parts)
    target = origin + horizon

    df_exp = pd.DataFrame({
        "origin_idx": origin,
        "target_idx": target,
        "origin_date": pd.to_datetime(dates[origin]),
        "target_date": pd.to_datetime(dates[target]),
        "horizon": horizon,
        "demand_obs": demand_obs[target],
        "demand_recon": demand_recon[target],
        "is_cashout_day": sub["is_cashout_day"].to_numpy(dtype=int)[target],
        "hours_empty": sub["hours_empty"].to_numpy(dtype=float)[target],
        "dow": dow[target],
        "dom": dom[target],
        "month": month[target],
        "is_weekend": is_weekend[target],
        "is_2_4_sat": sub["is_2_4_sat"].to_numpy(dtype=int)[target],
        "is_month_end": is_month_end[target],
        "is_payday": sub["is_payday"].to_numpy(dtype=int)[target],
        "is_regional_holiday": sub["is_regional_holiday"].to_numpy(dtype=int)[target],
        "pre_holiday": sub["pre_holiday"].to_numpy(dtype=int)[target],
        "tmax_c": sub["tmax_c"].to_numpy(dtype=float)[target],
        "tmean_c": sub["tmean_c"].to_numpy(dtype=float)[target],
        "tmin_c": sub["tmin_c"].to_numpy(dtype=float)[target],
        "rain_mm": sub["rain_mm"].to_numpy(dtype=float)[target],
        "rain_ge_10": sub["rain_ge_10"].to_numpy(dtype=int)[target],
        "rain_ge_25": sub["rain_ge_25"].to_numpy(dtype=int)[target],
        "tmax_ge_35": sub["tmax_ge_35"].to_numpy(dtype=int)[target],
        # Lags ending strictly at origin (CAUSAL - NO FUTURE LEAKAGE)
        "lag_1_obs": demand_obs[origin],
        "lag_7_obs": demand_obs[target - 7],
        "lag_14_obs": demand_obs[target - 14],
        "lag_28_obs": demand_obs[target - 28],
        "roll_mean_7_obs": rolling_obs["roll_mean_7"][origin],
        "roll_std_7_obs": rolling_obs["roll_std_7"][origin],
        "roll_mean_14_obs": rolling_obs["roll_mean_14"][origin],
        "roll_std_14_obs": rolling_obs["roll_std_14"][origin],
        "roll_mean_28_obs": rolling_obs["roll_mean_28"][origin],
        "roll_std_28_obs": rolling_obs["roll_std_28"][origin],
        # Reconstructed Lags ending strictly at origin
        "lag_1_recon": demand_recon[origin],
        "lag_7_recon": demand_recon[target - 7],
        "lag_14_recon": demand_recon[target - 14],
        "lag_28_recon": demand_recon[target - 28],
        "roll_mean_7_recon": rolling_recon["roll_mean_7"][origin],
        "roll_std_7_recon": rolling_recon["roll_std_7"][origin],
        "roll_mean_14_recon": rolling_recon["roll_mean_14"][origin],
        "roll_std_14_recon": rolling_recon["roll_std_14"][origin],
        "roll_mean_28_recon": rolling_recon["roll_mean_28"][origin],
        "roll_std_28_recon": rolling_recon["roll_std_28"][origin],
    })

    feats_obs = [
        "horizon", "lag_1_obs", "lag_7_obs", "lag_14_obs", "lag_28_obs",
        "roll_mean_7_obs", "roll_std_7_obs", "roll_mean_14_obs", "roll_std_14_obs", "roll_mean_28_obs", "roll_std_28_obs",
        "dow", "dom", "month", "is_weekend", "is_2_4_sat", "is_month_end",
        "is_payday", "is_regional_holiday", "pre_holiday",
        "tmax_c", "tmean_c", "tmin_c", "rain_mm", "rain_ge_10", "rain_ge_25", "tmax_ge_35"
    ]

    feats_recon = [
        "horizon", "lag_1_recon", "lag_7_recon", "lag_14_recon", "lag_28_recon",
        "roll_mean_7_recon", "roll_std_7_recon", "roll_mean_14_recon", "roll_std_14_recon", "roll_mean_28_recon", "roll_std_28_recon",
        "dow", "dom", "month", "is_weekend", "is_2_4_sat", "is_month_end",
        "is_payday", "is_regional_holiday", "pre_holiday",
        "tmax_c", "tmean_c", "tmin_c", "rain_mm", "rain_ge_10", "rain_ge_25", "tmax_ge_35"
    ]

    first_cutoff = MIN_TRAIN_DAYS - 1
    cutoffs = list(range(first_cutoff, n_dates - HORIZON, STEP))

    pred_censored = []
    pred_reconstructed = []
    test_rows = []

    for step, c_idx in enumerate(cutoffs, 1):
        train = df_exp.loc[df_exp["target_idx"] <= c_idx].copy()
        test = df_exp.loc[df_exp["origin_idx"] == c_idx].sort_values("horizon").reset_index(drop=True)

        # Baseline: Trained on Censored Demand
        m_cens = lgb.LGBMRegressor(**LGB_PARAMS)
        m_cens.fit(train[feats_obs], train["demand_obs"])
        p_cens = np.clip(m_cens.predict(test[feats_obs]), 0, None)
        pred_censored.extend(p_cens)

        # Experiment 5: Trained on Reconstructed Demand
        m_recon = lgb.LGBMRegressor(**LGB_PARAMS)
        m_recon.fit(train[feats_recon], train["demand_recon"])
        p_recon = np.clip(m_recon.predict(test[feats_recon]), 0, None)
        pred_reconstructed.extend(p_recon)

        test_rows.append(test[[
            "origin_date", "target_date", "horizon", "demand_obs", "demand_recon",
            "is_cashout_day", "hours_empty", "dom", "is_payday", "is_regional_holiday"
        ]])

    out_df = pd.concat(test_rows, ignore_index=True)
    out_df["pred_censored"] = pred_censored
    out_df["pred_reconstructed"] = pred_reconstructed
    return out_df

def evaluate_replenishment_spectrum(df_res, tid, site_name):
    """Evaluate replenishment performance across multiple CER buffer levels (65% to 90%)."""
    cycles = df_res.groupby("origin_date", as_index=False).agg({
        "demand_obs": "sum",
        "demand_recon": "sum",
        "pred_censored": "sum",
        "pred_reconstructed": "sum",
        "is_cashout_day": "sum"
    })
    
    n_cycles = len(cycles)
    buffer_targets = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    eval_rows = []
    
    for buf in buffer_targets:
        for model_name, col_pred in [("Censored Baseline", "pred_censored"), ("Reconstructed Model", "pred_reconstructed")]:
            # Load = 7-day predicted demand / buffer_target
            loads = cycles[col_pred] / buf
            actual = cycles["demand_recon"]  # Evaluated against estimated customer demand
            
            stockouts = actual > loads
            shortfall = np.maximum(actual - loads, 0.0)
            ending_cash = np.maximum(loads - actual, 0.0)
            mean_inventory = (loads + ending_cash) / 2.0
            
            # Annualized holding cost for this ATM across the 55 weeks (55/52 years)
            annual_holding_cost = float(mean_inventory.mean() * ANNUAL_HOLDING_RATE)
            
            eval_rows.append({
                "terminal_id": tid,
                "site_name": site_name,
                "model": model_name,
                "cer_buffer_target_%": int(buf * 100),
                "total_cycles": n_cycles,
                "stockout_cycles": int(stockouts.sum()),
                "stockout_rate_%": round(stockouts.mean() * 100, 2),
                "total_shortfall_lakh": round(shortfall.sum() / 1e5, 2),
                "mean_shortfall_when_short_lakh": round((shortfall[stockouts].mean() if stockouts.sum() > 0 else 0) / 1e5, 2),
                "mean_loaded_per_cycle_lakh": round(loads.mean() / 1e5, 2),
                "mean_ending_idle_cash_lakh": round(ending_cash.mean() / 1e5, 2),
                "annual_idle_cash_holding_cost_inr": round(annual_holding_cost, 0),
            })
            
    df_eval = pd.DataFrame(eval_rows)
    return df_eval

def run_experiment_5():
    print("=" * 80)
    print("EXPERIMENT 5: CASHOUT-AWARE DEMAND RECONSTRUCTION & REPLENISHMENT TRADE-OFF")
    print("=" * 80)
    
    merged = prepare_dataset()
    
    # 1. Backtest ATM 7 (Nashik Metro Concourse)
    print("\nRunning Experiment 5 rolling-origin backtest for ATM 7 (Nashik Metro)...")
    res7 = run_backtest_atm(merged, 7)
    res7.to_csv(OUT_DIR / "experiment_5_atm7_predictions.csv", index=False)
    rep7 = evaluate_replenishment_spectrum(res7, 7, "Nashik Metro Concourse")
    
    # 2. Backtest ATM 1 (Mumbai Railway Concourse)
    print("Running Experiment 5 rolling-origin backtest for ATM 1 (Mumbai Railway)...")
    res1 = run_backtest_atm(merged, 1)
    res1.to_csv(OUT_DIR / "experiment_5_atm1_predictions.csv", index=False)
    rep1 = evaluate_replenishment_spectrum(res1, 1, "Mumbai Railway Concourse")
    
    df_all_rep = pd.concat([rep7, rep1], ignore_index=True)
    df_all_rep.to_csv(OUT_DIR / "experiment_5_replenishment_tradeoff.csv", index=False)
    
    print("\n=== REPLENISHMENT BUFFER SPECTRUM SUMMARY (ATM 7: NASHIK METRO) ===")
    print(rep7[["model", "cer_buffer_target_%", "stockout_cycles", "stockout_rate_%", "total_shortfall_lakh", "mean_loaded_per_cycle_lakh", "mean_ending_idle_cash_lakh"]].to_string(index=False))
    
    print("\n=== REPLENISHMENT BUFFER SPECTRUM SUMMARY (ATM 1: MUMBAI RAILWAY) ===")
    print(rep1[["model", "cer_buffer_target_%", "stockout_cycles", "stockout_rate_%", "total_shortfall_lakh", "mean_loaded_per_cycle_lakh", "mean_ending_idle_cash_lakh"]].to_string(index=False))
    
    return res7, res1, df_all_rep

if __name__ == "__main__":
    run_experiment_5()
