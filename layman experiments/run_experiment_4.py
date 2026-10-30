"""Experiment 4: Incremental Value of Exogenous Variables

Investigates: How much does information beyond past withdrawals improve forecasts on unseen dates?
Controlled LightGBM comparison on rolling-origin 7-day replenishment backtests across 8 Maharashtra ATMs.
All code and outputs stay within 'layman experiments/'.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

HERE = Path(__file__).resolve().parent
CMS_ROOT = HERE.parent
PANEL_PATH = CMS_ROOT / "simulated" / "maharashtra" / "data" / "atm_daily.csv"
OUT_DIR = HERE / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_TRAIN_DAYS = 365
HORIZON = 7
STEP = 7
SEED = 42

# LightGBM fixed hyperparameters for controlled comparison
LGB_PARAMS = {
    "objective": "regression_l1",  # MAE / L1 loss directly aligns with WAPE
    "n_estimators": 120,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 25,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
    "force_col_wise": True,
}

# Feature definitions
CATEGORICAL = ["terminal_id", "loc_type"]

HISTORY_FEATURES = [
    "horizon",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "roll_mean_7",
    "roll_std_7",
    "roll_mean_14",
    "roll_std_14",
    "roll_mean_28",
    "roll_std_28",
    "terminal_id",
    "loc_type",
]

CALENDAR_FEATURES = [
    "dow",
    "dom",
    "month",
    "is_weekend",
    "is_2_4_sat",
    "is_month_end",
]

HOLIDAY_FEATURES = [
    "is_payday",
    "is_regional_holiday",
    "pre_holiday",
]

WEATHER_FEATURES = [
    "tmax_c",
    "tmean_c",
    "tmin_c",
    "rain_mm",
    "rain_ge_10",
    "rain_ge_25",
    "tmax_ge_35",
]

MODEL_SPECS = {
    "Model_A_History_Only": HISTORY_FEATURES,
    "Model_B_History_Calendar": HISTORY_FEATURES + CALENDAR_FEATURES,
    "Model_C_History_Cal_Holidays": HISTORY_FEATURES + CALENDAR_FEATURES + HOLIDAY_FEATURES,
    "Model_D_History_Cal_Weather": HISTORY_FEATURES + CALENDAR_FEATURES + WEATHER_FEATURES,
    "Model_F_Full_Stack": HISTORY_FEATURES + CALENDAR_FEATURES + HOLIDAY_FEATURES + WEATHER_FEATURES,
    # Ablations (Leave-one-out from F)
    "Model_F_minus_Weather": HISTORY_FEATURES + CALENDAR_FEATURES + HOLIDAY_FEATURES,
    "Model_F_minus_Holidays": HISTORY_FEATURES + CALENDAR_FEATURES + WEATHER_FEATURES,
    "Model_F_minus_Calendar": HISTORY_FEATURES + HOLIDAY_FEATURES + WEATHER_FEATURES,
    "Model_F_minus_History": [c for c in (CALENDAR_FEATURES + HOLIDAY_FEATURES + WEATHER_FEATURES + ["terminal_id", "loc_type", "horizon"])],
}

def load_data():
    panel = pd.read_csv(PANEL_PATH, parse_dates=["date"])
    panel["terminal_id"] = panel["terminal_id"].astype(int)
    panel = panel.sort_values(["terminal_id", "date"]).reset_index(drop=True)
    dates = np.sort(panel["date"].unique())
    return panel, dates

def build_feature_frame(panel: pd.DataFrame, dates: np.ndarray) -> pd.DataFrame:
    chunks = []
    for terminal_id, group in panel.groupby("terminal_id", sort=True):
        group = group.sort_values("date").reset_index(drop=True)
        demand = group["demand"].to_numpy(dtype=float)
        dom = group["dom"].to_numpy(dtype=int)
        dow = group["dow"].to_numpy(dtype=int)
        month = group["month"].to_numpy(dtype=int)
        days_in_month = group["date"].dt.days_in_month.to_numpy(dtype=int)
        is_month_end = (days_in_month - dom <= 2).astype(int)
        is_weekend = (dow >= 5).astype(int)
        
        # Weather arrays
        tmax_c = group["tmax_c"].to_numpy(dtype=float)
        tmean_c = group["tmean_c"].to_numpy(dtype=float)
        tmin_c = group["tmin_c"].to_numpy(dtype=float)
        rain_mm = group["rain_mm"].to_numpy(dtype=float)
        rain_ge_10 = group["rain_ge_10"].to_numpy(dtype=int)
        rain_ge_25 = group["rain_ge_25"].to_numpy(dtype=int)
        tmax_ge_35 = group["tmax_ge_35"].to_numpy(dtype=int)

        # Rolling statistics strictly ending on origin
        series = pd.Series(demand)
        rolling = {}
        for window in (7, 14, 28):
            windowed = series.rolling(window, min_periods=window)
            rolling[f"roll_mean_{window}"] = windowed.mean().to_numpy()
            rolling[f"roll_std_{window}"] = windowed.std(ddof=1).to_numpy()

        n_dates = len(demand)
        origin_parts = []
        horizon_parts = []
        for origin in range(27, n_dates - 1):
            max_horizon = min(HORIZON, n_dates - 1 - origin)
            origin_parts.append(np.full(max_horizon, origin, dtype=int))
            horizon_parts.append(np.arange(1, max_horizon + 1, dtype=int))
        origin = np.concatenate(origin_parts)
        horizon = np.concatenate(horizon_parts)
        target = origin + horizon

        chunks.append(
            pd.DataFrame(
                {
                    "terminal_id": int(terminal_id),
                    "loc_type": group["loc_type"].iloc[0],
                    "city": group["city"].iloc[0],
                    "origin_idx": origin,
                    "target_idx": target,
                    "origin_date": pd.to_datetime(dates[origin]),
                    "target_date": pd.to_datetime(dates[target]),
                    "horizon": horizon,
                    "demand": demand[target],
                    "dow": dow[target],
                    "dom": dom[target],
                    "month": month[target],
                    "is_weekend": is_weekend[target],
                    "is_2_4_sat": group["is_2_4_sat"].to_numpy(dtype=int)[target],
                    "is_month_end": is_month_end[target],
                    "is_payday": group["is_payday"].to_numpy(dtype=int)[target],
                    "is_regional_holiday": group["is_regional_holiday"].to_numpy(dtype=int)[target],
                    "pre_holiday": group["pre_holiday"].to_numpy(dtype=int)[target],
                    # Weather features on target date (diagnostic / oracle test as specified)
                    "tmax_c": tmax_c[target],
                    "tmean_c": tmean_c[target],
                    "tmin_c": tmin_c[target],
                    "rain_mm": rain_mm[target],
                    "rain_ge_10": rain_ge_10[target],
                    "rain_ge_25": rain_ge_25[target],
                    "tmax_ge_35": tmax_ge_35[target],
                    # Lagged history available at origin cutoff
                    "lag_1": demand[origin],
                    "lag_7": demand[target - 7],
                    "lag_14": demand[target - 14],
                    "lag_28": demand[target - 28],
                    "roll_mean_7": rolling["roll_mean_7"][origin],
                    "roll_std_7": rolling["roll_std_7"][origin],
                    "roll_mean_14": rolling["roll_mean_14"][origin],
                    "roll_std_14": rolling["roll_std_14"][origin],
                    "roll_mean_28": rolling["roll_mean_28"][origin],
                    "roll_std_28": rolling["roll_std_28"][origin],
                }
            )
        )
    frame = pd.concat(chunks, ignore_index=True)
    return frame

def compute_seasonal_baseline(train_panel: pd.DataFrame, test_frame: pd.DataFrame) -> np.ndarray:
    """Experiment 3 frozen seasonal baseline: ATM mean * Day-of-Month index."""
    atm_means = train_panel.groupby("terminal_id")["demand"].mean()
    ratios = train_panel["demand"] / train_panel["terminal_id"].map(atm_means)
    dom_factors = ratios.groupby(train_panel["dom"]).mean()
    
    pred = test_frame["terminal_id"].map(atm_means).astype(float) * test_frame["dom"].map(dom_factors).astype(float)
    return pred.to_numpy()

def wape(actual, pred):
    return float(np.sum(np.abs(actual - pred)) / np.sum(actual))

def bias(actual, pred):
    return float(np.sum(pred - actual) / np.sum(actual))

def run_experiment():
    print("=" * 80)
    print("EXPERIMENT 4: INCREMENTAL VALUE OF EXOGENOUS VARIABLES")
    print("Question: How much does information beyond past withdrawals improve forecasts on unseen dates?")
    print("=" * 80)
    
    panel, dates = load_data()
    n_dates = len(dates)
    frame = build_feature_frame(panel, dates)
    
    # Calculate cutoff indices
    first_cutoff = MIN_TRAIN_DAYS - 1
    cutoffs = list(range(first_cutoff, n_dates - HORIZON, STEP))
    print(f"Panel spans {dates[0].astype('M8[D]')} to {dates[-1].astype('M8[D]')} ({n_dates} days).")
    print(f"Number of rolling weekly origins: {len(cutoffs)}")
    print(f"First cutoff: {dates[cutoffs[0]].astype('M8[D]')}, Last cutoff: {dates[cutoffs[-1]].astype('M8[D]')}")
    print(f"Models to evaluate: {list(MODEL_SPECS.keys())} + Seasonal_Baseline_Exp3")

    predictions_dict = {m: [] for m in MODEL_SPECS.keys()}
    predictions_dict["Seasonal_Exp3"] = []
    
    meta_test_rows = []
    
    start_time = time.time()
    
    for step, cutoff_idx in enumerate(cutoffs, start=1):
        cutoff = pd.Timestamp(dates[cutoff_idx])
        train_panel = panel.loc[panel["date"] <= cutoff]
        
        train = frame.loc[frame["target_idx"] <= cutoff_idx].copy()
        test = frame.loc[frame["origin_idx"] == cutoff_idx].sort_values(["terminal_id", "horizon"]).reset_index(drop=True)
        
        y_train = train["demand"].to_numpy(dtype=float)
        
        # 1. Experiment 3 Seasonal Baseline
        pred_seasonal = compute_seasonal_baseline(train_panel, test)
        predictions_dict["Seasonal_Exp3"].extend(pred_seasonal)
        
        # 2. LightGBM models
        for model_name, feature_list in MODEL_SPECS.items():
            cats = [c for c in CATEGORICAL if c in feature_list]
            
            x_train = train[feature_list].copy()
            x_test = test[feature_list].copy()
            for col in cats:
                cat_type = pd.CategoricalDtype(categories=sorted(x_train[col].unique()))
                x_train[col] = x_train[col].astype(cat_type)
                x_test[col] = x_test[col].astype(cat_type)
            
            model = lgb.LGBMRegressor(**LGB_PARAMS)
            model.fit(x_train, y_train, categorical_feature=cats)
            pred = np.clip(model.predict(x_test), 0, None)
            predictions_dict[model_name].extend(pred)
            
        meta_test_rows.append(test[[
            "terminal_id", "loc_type", "city", "origin_date", "target_date", "horizon",
            "demand", "dow", "dom", "month", "is_payday", "is_regional_holiday", "pre_holiday",
            "rain_mm", "rain_ge_10", "rain_ge_25", "tmax_ge_35"
        ]])
        
        if step % 10 == 0 or step == len(cutoffs):
            print(f"Processed cutoff {step}/{len(cutoffs)} ({cutoff.strftime('%Y-%m-%d')}) in {time.time() - start_time:.1f}s")
            
    # Assemble full scored test dataframe
    test_df = pd.concat(meta_test_rows, ignore_index=True)
    for m in predictions_dict:
        test_df[m] = predictions_dict[m]
        
    test_df.to_csv(OUT_DIR / "experiment_4_predictions.csv", index=False)
    print(f"\nSaved all predictions to {OUT_DIR / 'experiment_4_predictions.csv'}")
    
    return test_df

if __name__ == "__main__":
    run_experiment()
