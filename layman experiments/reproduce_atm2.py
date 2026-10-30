"""Test and reproduce Experiment results on ATM 2 (MUMBAI - AIRPORT TERMINAL).

Same city (Mumbai), same RBI region, same weather observations as ATM 1,
but different site typology (Airport Terminal vs Railway Station Concourse).

Evaluates:
1. Year-over-Year Pattern Reproducibility (Year 1 vs Year 2 stability, DOM, Month, DOW)
2. Controlled Exogenous Model Comparison (Models A, B, C, D, F, Exp 3) across 55 rolling weekly origins
3. Sliced performance on Holidays, Weather (Rain/Heat), and Lead-time Horizons
4. Head-to-Head Comparison: ATM 1 (Railway) vs ATM 2 (Airport)
"""

from __future__ import annotations

import json
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

MODEL_SPECS = {
    "Model_A_History_Only": HISTORY_FEATURES,
    "Model_B_History_Calendar": HISTORY_FEATURES + CALENDAR_FEATURES,
    "Model_C_History_Cal_Holidays": HISTORY_FEATURES + CALENDAR_FEATURES + HOLIDAY_FEATURES,
    "Model_D_History_Cal_Weather": HISTORY_FEATURES + CALENDAR_FEATURES + WEATHER_FEATURES,
    "Model_F_Full_Stack": HISTORY_FEATURES + CALENDAR_FEATURES + HOLIDAY_FEATURES + WEATHER_FEATURES,
}

def wape(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.sum(np.abs(actual - pred)) / np.sum(actual))

def bias(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.sum(pred - actual) / np.sum(actual))

def run_atm2_reproduction():
    panel = pd.read_csv(PANEL_PATH, parse_dates=["date"])
    atm2 = panel[panel["terminal_id"] == 2].copy().sort_values("date").reset_index(drop=True)
    dates = np.sort(atm2["date"].unique())
    n_dates = len(dates)
    
    print("=" * 80)
    print("REPRODUCING APPROACH ON ATM 2 (SYNATM0002: MUMBAI - AIRPORT TERMINAL)")
    print("City: Mumbai | State: Maharashtra | RBI Region: Mumbai")
    print(f"Total dates: {n_dates} ({dates[0].astype('M8[D]')} to {dates[-1].astype('M8[D]')})")
    print("=" * 80)

    # ----------------------------------------------------
    # Part 1: Macro & Seasonal Predictability (Year 1 vs Year 2)
    # ----------------------------------------------------
    y1 = atm2[(atm2['date'] >= '2024-09-01') & (atm2['date'] <= '2025-08-31')].copy().reset_index(drop=True)
    y2 = atm2[(atm2['date'] >= '2025-09-01') & (atm2['date'] <= '2026-08-31')].copy().reset_index(drop=True)

    y1_mean = y1['demand'].mean()
    y2_mean = y2['demand'].mean()
    y1_total = y1['demand'].sum()
    y2_total = y2['demand'].sum()
    y1_cv = y1['demand'].std() / y1_mean
    y2_cv = y2['demand'].std() / y2_mean
    drift = (y2_total - y1_total) / y1_total * 100

    dom1 = y1.groupby('dom')['demand'].mean() / y1_mean
    dom2 = y2.groupby('dom')['demand'].mean() / y2_mean
    corr_dom = dom1.corr(dom2)

    dow1 = y1.groupby('dow')['demand'].mean() / y1_mean
    dow2 = y2.groupby('dow')['demand'].mean() / y2_mean
    corr_dow = dow1.corr(dow2)

    m1 = y1.groupby('month')['demand'].mean() / y1_mean
    m2 = y2.groupby('month')['demand'].mean() / y2_mean
    corr_month = m1.corr(m2)

    # Structured Calendar Prediction of Year 2 from Year 1
    pred_y1_mean = np.full(len(y2), y1_mean)
    pred_y1_dom = y2['dom'].map(dom1).values * y1_mean
    pred_y1_dom_month = (y2['dom'].map(dom1).values * y2['month'].map(m1).values) * y1_mean

    wape_y2_mean = wape(y2['demand'], pred_y1_mean)
    wape_y2_dom = wape(y2['demand'], pred_y1_dom)
    wape_y2_dom_month = wape(y2['demand'], pred_y1_dom_month)

    # 7-day aggregated cycle on Year 2
    act_7d = y2['demand'].values[:52*7].reshape(52, 7).sum(axis=1)
    p_dom_7d = pred_y1_dom[:52*7].reshape(52, 7).sum(axis=1)
    p_dom_m_7d = pred_y1_dom_month[:52*7].reshape(52, 7).sum(axis=1)
    wape_7d_dom = wape(act_7d, p_dom_7d)
    wape_7d_dom_m = wape(act_7d, p_dom_m_7d)

    print("\n[PART 1: YEAR-OVER-YEAR STABILITY & SEASONALITY]")
    print(f"Year 1 Daily Mean: ₹{y1_mean:,.0f} (CV: {y1_cv:.3f}) | Total: ₹{y1_total/1e5:.1f} Lakh")
    print(f"Year 2 Daily Mean: ₹{y2_mean:,.0f} (CV: {y2_cv:.3f}) | Total: ₹{y2_total/1e5:.1f} Lakh")
    print(f"Annual Volume Drift: {drift:+.2f}%")
    print(f"Day-of-Month (DOM) Profile Correlation (Y1 vs Y2): {corr_dom:.4f}")
    print(f"Day-of-Week (DOW) Profile Correlation (Y1 vs Y2):   {corr_dow:.4f}")
    print(f"Month Seasonality Correlation (Y1 vs Y2):          {corr_month:.4f}")
    print(f"Year 2 Daily WAPE (Flat Y1 Mean):                  {wape_y2_mean*100:.2f}%")
    print(f"Year 2 Daily WAPE (Y1 DOM Profile):                {wape_y2_dom*100:.2f}%")
    print(f"Year 2 Daily WAPE (Y1 DOM * Month):                {wape_y2_dom_month*100:.2f}%")
    print(f"Year 2 7-Day Cycle WAPE (Y1 DOM Profile):          {wape_7d_dom*100:.2f}%")
    print(f"Year 2 7-Day Cycle WAPE (Y1 DOM * Month):          {wape_7d_dom_m*100:.2f}%")

    # ----------------------------------------------------
    # Part 2: Controlled Rolling-Origin Backtest on ATM 2
    # ----------------------------------------------------
    demand = atm2["demand"].to_numpy(dtype=float)
    dom = atm2["dom"].to_numpy(dtype=int)
    dow = atm2["dow"].to_numpy(dtype=int)
    month = atm2["month"].to_numpy(dtype=int)
    days_in_month = atm2["date"].dt.days_in_month.to_numpy(dtype=int)
    is_month_end = (days_in_month - dom <= 2).astype(int)
    is_weekend = (dow >= 5).astype(int)

    series = pd.Series(demand)
    rolling = {}
    for window in (7, 14, 28):
        windowed = series.rolling(window, min_periods=window)
        rolling[f"roll_mean_{window}"] = windowed.mean().to_numpy()
        rolling[f"roll_std_{window}"] = windowed.std(ddof=1).to_numpy()

    origin_parts = []
    horizon_parts = []
    for origin in range(27, n_dates - 1):
        max_h = min(HORIZON, n_dates - 1 - origin)
        origin_parts.append(np.full(max_h, origin, dtype=int))
        horizon_parts.append(np.arange(1, max_h + 1, dtype=int))
    origin = np.concatenate(origin_parts)
    horizon = np.concatenate(horizon_parts)
    target = origin + horizon

    frame2 = pd.DataFrame({
        "terminal_id": 2,
        "loc_type": "AIRPORT TERMINAL",
        "city": "MUMBAI",
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
        "is_2_4_sat": atm2["is_2_4_sat"].to_numpy(dtype=int)[target],
        "is_month_end": is_month_end[target],
        "is_payday": atm2["is_payday"].to_numpy(dtype=int)[target],
        "is_regional_holiday": atm2["is_regional_holiday"].to_numpy(dtype=int)[target],
        "pre_holiday": atm2["pre_holiday"].to_numpy(dtype=int)[target],
        "tmax_c": atm2["tmax_c"].to_numpy(dtype=float)[target],
        "tmean_c": atm2["tmean_c"].to_numpy(dtype=float)[target],
        "tmin_c": atm2["tmin_c"].to_numpy(dtype=float)[target],
        "rain_mm": atm2["rain_mm"].to_numpy(dtype=float)[target],
        "rain_ge_10": atm2["rain_ge_10"].to_numpy(dtype=int)[target],
        "rain_ge_25": atm2["rain_ge_25"].to_numpy(dtype=int)[target],
        "tmax_ge_35": atm2["tmax_ge_35"].to_numpy(dtype=int)[target],
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
    })

    first_cutoff = MIN_TRAIN_DAYS - 1
    cutoffs = list(range(first_cutoff, n_dates - HORIZON, STEP))

    preds = {m: [] for m in MODEL_SPECS}
    preds["Exp3_Seasonal"] = []
    test_meta = []
    last_model_f = None

    print(f"\n[PART 2: RUNNING 55 WEEKLY ORIGIN BACKTESTS FOR ATM 2]")
    for step, c_idx in enumerate(cutoffs, 1):
        cutoff = pd.Timestamp(dates[c_idx])
        train = frame2.loc[frame2["target_idx"] <= c_idx].copy()
        test = frame2.loc[frame2["origin_idx"] == c_idx].sort_values("horizon").reset_index(drop=True)
        y_tr = train["demand"].to_numpy(dtype=float)

        # Exp 3 Baseline
        m_level = y_tr.mean()
        r = y_tr / m_level
        d_factors = pd.Series(r).groupby(train["dom"]).mean().to_dict()
        preds["Exp3_Seasonal"].extend([d_factors.get(d, 1.0) * m_level for d in test["dom"]])

        # LightGBM models
        for m_name, feats in MODEL_SPECS.items():
            model = lgb.LGBMRegressor(**LGB_PARAMS)
            model.fit(train[feats], y_tr)
            p = np.clip(model.predict(test[feats]), 0, None)
            preds[m_name].extend(p)
            if m_name == "Model_F_Full_Stack":
                last_model_f = model

        test_meta.append(test[[
            "origin_date", "target_date", "horizon", "demand", "dow", "dom", "month",
            "is_payday", "is_regional_holiday", "pre_holiday", "rain_mm", "rain_ge_10", "rain_ge_25"
        ]])

    test_res = pd.concat(test_meta, ignore_index=True)
    for m in preds:
        test_res[m] = preds[m]

    # Save ATM 2 predictions
    test_res.to_csv(OUT_DIR / "atm2_predictions.csv", index=False)

    # Aggregate weekly replenishment cycles (55 cycles)
    cycle_res = test_res.groupby("origin_date", as_index=False).agg({
        "demand": "sum",
        "Model_A_History_Only": "sum",
        "Model_B_History_Calendar": "sum",
        "Model_C_History_Cal_Holidays": "sum",
        "Model_D_History_Cal_Weather": "sum",
        "Model_F_Full_Stack": "sum",
        "Exp3_Seasonal": "sum",
    })

    # Metrics table
    model_rows = []
    base_dw = wape(test_res["demand"], test_res["Model_A_History_Only"])
    base_cw = wape(cycle_res["demand"], cycle_res["Model_A_History_Only"])

    for m in ["Model_A_History_Only", "Model_B_History_Calendar", "Model_C_History_Cal_Holidays", "Model_D_History_Cal_Weather", "Model_F_Full_Stack", "Exp3_Seasonal"]:
        dw = wape(test_res["demand"], test_res[m])
        cw = wape(cycle_res["demand"], cycle_res[m])
        b = bias(test_res["demand"], test_res[m])
        rel_d = (base_dw - dw) / base_dw * 100
        rel_c = (base_cw - cw) / base_cw * 100
        model_rows.append({
            "Model": m,
            "Daily_WAPE_%": round(dw * 100, 2),
            "Daily_Rel_Gain_%": round(rel_d, 2),
            "7Day_WAPE_%": round(cw * 100, 2),
            "7Day_Rel_Gain_%": round(rel_c, 2),
            "Bias_%": round(b * 100, 2),
        })
    df_m = pd.DataFrame(model_rows)
    print("\n=== ATM 2 (AIRPORT) MODEL PERFORMANCE ===")
    print(df_m.to_string(index=False))

    # Event slices
    hol_mask = test_res["is_regional_holiday"] == 1
    ord_mask = test_res["is_regional_holiday"] == 0
    print("\n=== ATM 2 SLICE: REGIONAL HOLIDAYS VS ORDINARY ===")
    print(f"Holidays (N={hol_mask.sum()}): Hist_Only={wape(test_res.loc[hol_mask, 'demand'], test_res.loc[hol_mask, 'Model_A_History_Only'])*100:.2f}% | +Holidays={wape(test_res.loc[hol_mask, 'demand'], test_res.loc[hol_mask, 'Model_C_History_Cal_Holidays'])*100:.2f}% | Lift: {(wape(test_res.loc[hol_mask, 'demand'], test_res.loc[hol_mask, 'Model_A_History_Only']) - wape(test_res.loc[hol_mask, 'demand'], test_res.loc[hol_mask, 'Model_C_History_Cal_Holidays']))*100:+.2f} pp")
    print(f"Ordinary (N={ord_mask.sum()}): Hist_Only={wape(test_res.loc[ord_mask, 'demand'], test_res.loc[ord_mask, 'Model_A_History_Only'])*100:.2f}% | +Holidays={wape(test_res.loc[ord_mask, 'demand'], test_res.loc[ord_mask, 'Model_C_History_Cal_Holidays'])*100:.2f}%")

    # Weather slices
    heavy_rain = test_res["rain_ge_25"] == 1
    dry = test_res["rain_ge_10"] == 0
    print("\n=== ATM 2 SLICE: WEATHER CONDITIONS ===")
    print(f"Dry (<10mm, N={dry.sum()}): Hist+Cal={wape(test_res.loc[dry, 'demand'], test_res.loc[dry, 'Model_B_History_Calendar'])*100:.2f}% | +Weather={wape(test_res.loc[dry, 'demand'], test_res.loc[dry, 'Model_D_History_Cal_Weather'])*100:.2f}%")
    print(f"Heavy Rain (>=25mm, N={heavy_rain.sum()}): Hist+Cal={wape(test_res.loc[heavy_rain, 'demand'], test_res.loc[heavy_rain, 'Model_B_History_Calendar'])*100:.2f}% | +Weather={wape(test_res.loc[heavy_rain, 'demand'], test_res.loc[heavy_rain, 'Model_D_History_Cal_Weather'])*100:.2f}%")

    # Feature importances from Model F
    gain = last_model_f.booster_.feature_importance(importance_type="gain")
    feats = last_model_f.booster_.feature_name()
    df_imp = pd.DataFrame({"feature": feats, "gain": gain, "gain_pct": gain / gain.sum() * 100}).sort_values("gain", ascending=False).reset_index(drop=True)
    df_imp.to_csv(OUT_DIR / "atm2_feature_importance.csv", index=False)
    print("\n=== TOP 10 FEATURE IMPORTANCE (MODEL F ON ATM 2) ===")
    print(df_imp.head(10).to_string(index=False))

    df_m.to_csv(OUT_DIR / "atm2_model_performance.csv", index=False)
    print(f"\nAll ATM 2 results written to {OUT_DIR}")

if __name__ == "__main__":
    run_atm2_reproduction()
