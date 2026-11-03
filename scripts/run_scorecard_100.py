"""Rolling-origin scorecard for all 100 cash-file ATMs.

    python3 scripts/run_scorecard_100.py             # full run (55 weekly origins)
    python3 scripts/run_scorecard_100.py --limit 2   # smoke test on the last 2 origins

Protocol (rows come from simulated/maharashtra/run_rolling_origin.build_rows):
- The forecast is issued after the cutoff day's demand is known; horizons 1-7.
- Models train only on rows whose target date is on or before the cutoff.
- The headline score is each ATM's 7-day total (the cash-load decision), plus daily.
- Upper bounds are conformal: P95/P99 of actual/forecast 7-day ratios from past
  weeks only (weeks whose last day is on or before the cutoff).
- Cold start: ATMs held out in 5 groups. "day0" knows only location and calendar;
  "day28" also has the ATM's first 28 days of demand.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from replenishment_engine.demand_model import (  # noqa: E402
    CALENDAR, CONFORMAL_WINDOW_WEEKS, HOLIDAY_WEEK_LEAD_DAYS, MAJOR_FESTIVALS, MIN_PAST_WEEKS,
    add_features, conformal_multiplier, fit_predict, holiday_model_features, is_holiday_day, level_features,
    load_panel, load_rolling_origin, SEED,
)

PILOT_PANEL = ROOT / "simulated" / "maharashtra" / "data" / "atm_daily.csv"
OUT = ROOT / "outputs" / "scorecard_100"

RECENCY_HALF_LIFE_DAYS = 90
COLD_FOLDS = 5
COLD_ORIGIN_STEP = 8  # every 8th weekly origin for the cold-start test

RATIO_HISTORY = [
    "r_lag_1", "r_lag_7", "r_lag_14", "r_lag_28",
    "r_roll_mean_7", "r_roll_mean_14", "r_roll_std_7", "r_roll_std_14", "r_roll_std_28",
]
MODELS = ["seasonal_naive", "seasonal_mean_4w", "lgb_level", "lgb_holiday", "lgb_ratio", "blend"]


def pilot_flag_agreement(panel: pd.DataFrame) -> dict:
    """Share of ATM-days where our flags match the 8-ATM pilot panel."""
    pilot = pd.read_csv(PILOT_PANEL, parse_dates=["date"])
    merged = pilot.merge(panel, on=["terminal_id", "date"], suffixes=("_pilot", ""))
    flags = ["demand", "is_payday", "is_2_4_sat", "is_regional_holiday", "pre_holiday"]
    return {f: round(float((merged[f"{f}_pilot"] == merged[f]).mean()), 4) for f in flags}


def recency_weight(train: pd.DataFrame, cutoff: pd.Timestamp) -> np.ndarray:
    age = (cutoff - train["target_date"]).dt.days.to_numpy()
    return 0.5 ** (age / RECENCY_HALF_LIFE_DAYS)


def ratio_features(with_terminal: bool = True) -> list[str]:
    return CALENDAR + RATIO_HISTORY + ["log_level", "loc_type"] + (["terminal_id"] if with_terminal else [])


def forecast_origin(ro, lightgbm, frame, cutoff_idx, n_atm) -> pd.DataFrame:
    train = frame.loc[frame["target_idx"] <= cutoff_idx]
    test = frame.loc[frame["origin_idx"] == cutoff_idx].copy()
    if len(test) != n_atm * ro.HORIZON:
        raise AssertionError("test week is not one row per ATM per day")
    if train["target_idx"].max() > cutoff_idx or test["target_idx"].min() <= cutoff_idx:
        raise AssertionError("train and test dates overlap")
    cutoff = test["origin_date"].iloc[0]
    test["seasonal_naive"] = test["lag_7"]
    test["lgb_level"] = fit_predict(lightgbm, train, test, level_features(ro), train["demand"])
    test["lgb_holiday"] = fit_predict(lightgbm, train, test, holiday_model_features(ro), train["demand"])
    ratio = fit_predict(lightgbm, train, test, ratio_features(), train["demand"] / train["roll_mean_28"],
                        recency_weight(train, cutoff))
    test["lgb_ratio"] = ratio * test["roll_mean_28"]
    test["blend"] = test[["lgb_ratio", "lgb_level", "seasonal_mean_4w"]].mean(axis=1)
    return test


def weekly_totals(pred: pd.DataFrame) -> pd.DataFrame:
    keys = ["terminal_id", "origin_idx", "origin_date", "volume_tier", "loc_type"]
    totals = pred.groupby(keys, as_index=False)[["demand", *MODELS]].sum()
    holiday = pred.assign(holiday_week=is_holiday_day(pred["days_to_holiday"])).groupby(keys, as_index=False)["holiday_week"].any()
    return totals.merge(holiday, on=keys, validate="one_to_one")


def add_conformal_bounds(weekly: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Per model, P95/P99 multipliers from past weeks only (rolling 26-week window).

    Calibrated separately for holiday-touched and quiet weeks (pooled if a type has too few past weeks).
    """
    weekly = weekly.copy()
    origins = np.sort(weekly["origin_idx"].unique())
    for model in MODELS:
        ratio = weekly["demand"] / weekly[model].replace(0, np.nan)
        for q in (95, 99):
            weekly[f"{model}_q{q}"] = np.nan
        for origin in origins:
            past = (weekly["origin_idx"] + horizon <= origin) & (weekly["origin_idx"] > origin - CONFORMAL_WINDOW_WEEKS * 7)
            for week_type in (False, True):
                same_type = weekly["holiday_week"] == week_type
                now = (weekly["origin_idx"] == origin) & same_type
                for q in (95, 99):
                    mult = conformal_multiplier(ratio, past, weekly["origin_idx"], same_type, q / 100)
                    if mult is not None:
                        weekly.loc[now, f"{model}_q{q}"] = mult
    return weekly


def wape(actual, pred) -> float:
    return float(np.abs(np.asarray(pred) - np.asarray(actual)).sum() / np.asarray(actual).sum())


def bias(actual, pred) -> float:
    return float((np.asarray(pred) - np.asarray(actual)).sum() / np.asarray(actual).sum())


def metric_rows(daily: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    daily_slices = [("all", daily)] + [(f"h={h}", daily[daily["horizon"] == h]) for h in range(1, 8)]
    daily_slices += [
        ("holiday", daily[daily["is_regional_holiday"] == 1]),
        ("pre_holiday", daily[daily["pre_holiday"] == 1]),
        ("payday", daily[daily["is_payday"] == 1]),
        ("month_end", daily[daily["is_month_end"] == 1]),
        ("normal_day", daily[(daily[["is_regional_holiday", "pre_holiday", "is_payday", "is_month_end"]] == 0).all(axis=1)]),
        ("major_festival", daily[daily["holiday_family"].isin(MAJOR_FESTIVALS)]),
        ("minor_holiday", daily[(daily["is_regional_holiday"] == 1) & ~daily["holiday_family"].isin(MAJOR_FESTIVALS)]),
        ("3_days_before_major", daily[daily["days_to_holiday"].between(1, 3) & daily["next_holiday_family"].isin(MAJOR_FESTIVALS)]),
        ("long_closure", daily[(daily["is_bank_closed"] == 1) & (daily["closure_run"] >= 3)]),
    ]
    weekly_slices = [("all", weekly)] + [(f"tier:{t}", weekly[weekly["volume_tier"] == t]) for t in ["low", "mid", "high"]]
    weekly_slices += [("quiet_week", weekly[~weekly["holiday_week"]]), ("holiday_week", weekly[weekly["holiday_week"]])]
    for model in MODELS:
        for name, part in daily_slices:
            if part.empty:
                continue
            rows.append({"model": model, "level": "daily", "slice": name, "n": len(part),
                         "wape": wape(part["demand"], part[model]), "bias": bias(part["demand"], part[model])})
        for name, part in weekly_slices:
            row = {"model": model, "level": "weekly_7d_total", "slice": name, "n": len(part),
                   "wape": wape(part["demand"], part[model]), "bias": bias(part["demand"], part[model])}
            scored = part[part[f"{model}_q99"].notna()]
            if len(scored):
                for q in (95, 99):
                    upper = scored[model] * scored[f"{model}_q{q}"]
                    row[f"p{q}_coverage"] = float((scored["demand"] <= upper).mean())
                    row[f"p{q}_extra_cash_pct"] = float(upper.sum() / scored["demand"].sum() - 1)
                row["n_bounds_scored"] = len(scored)
            rows.append(row)
    return pd.DataFrame(rows).round(4)


def cold_start(lightgbm, frame, cutoffs, main_pred) -> pd.DataFrame:
    terminals = np.sort(frame["terminal_id"].unique())
    rng = np.random.default_rng(SEED)
    folds = np.array_split(rng.permutation(terminals), COLD_FOLDS)
    day0 = CALENDAR + ["loc_type", "city", "rbi_region"]
    parts = []
    for cutoff_idx in cutoffs:
        for held in folds:
            is_held = frame["terminal_id"].isin(held)
            train = frame.loc[(frame["target_idx"] <= cutoff_idx) & ~is_held]
            test = frame.loc[(frame["origin_idx"] == cutoff_idx) & is_held].copy()
            if train["terminal_id"].isin(held).any():
                raise AssertionError("a held-out ATM is in cold-start training")
            cutoff = test["origin_date"].iloc[0]
            test["cold_day0"] = fit_predict(lightgbm, train, test, day0, train["demand"])
            ratio = fit_predict(lightgbm, train, test, ratio_features(with_terminal=False),
                                train["demand"] / train["roll_mean_28"], recency_weight(train, cutoff))
            test["cold_day28"] = ratio * test["roll_mean_28"]
            parts.append(test[["terminal_id", "origin_idx", "horizon", "demand", "cold_day0", "cold_day28"]])
    cold = pd.concat(parts, ignore_index=True).merge(
        main_pred[["terminal_id", "origin_idx", "horizon", "lgb_ratio"]],
        on=["terminal_id", "origin_idx", "horizon"], how="left", validate="one_to_one",
    ).rename(columns={"lgb_ratio": "warm_lgb_ratio"})
    rows = []
    week = cold.groupby(["terminal_id", "origin_idx"])[["demand", "cold_day0", "cold_day28", "warm_lgb_ratio"]].sum()
    for model in ["cold_day0", "cold_day28", "warm_lgb_ratio"]:
        rows.append({"model": model, "daily_wape": wape(cold["demand"], cold[model]),
                     "weekly_wape": wape(week["demand"], week[model]), "weekly_bias": bias(week["demand"], week[model]),
                     "n_atm_weeks": len(week)})
    return pd.DataFrame(rows).round(4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="score only the last N origins")
    parser.add_argument("--skip-cold", action="store_true")
    args = parser.parse_args()

    ro = load_rolling_origin()
    lightgbm, _ = ro.load_lightgbm()
    panel, dates = load_panel()
    agreement = pilot_flag_agreement(panel)
    print("pilot flag agreement (ATMs 1-8):", agreement)
    frame = add_features(ro.build_rows(panel, dates), panel, dates)
    n_atm = panel["terminal_id"].nunique()
    cutoffs = ro.cutoff_indices(len(dates))
    if args.limit:
        cutoffs = cutoffs[-args.limit:]
    print(f"{n_atm} ATMs, {len(dates)} days, {len(cutoffs)} origins, {len(frame):,} candidate rows")

    parts = []
    for step, cutoff_idx in enumerate(cutoffs, start=1):
        started = time.perf_counter()
        parts.append(forecast_origin(ro, lightgbm, frame, cutoff_idx, n_atm))
        print(f"  origin {step}/{len(cutoffs)} {pd.Timestamp(dates[cutoff_idx]).date()} {time.perf_counter() - started:.1f}s", flush=True)
    daily = pd.concat(parts, ignore_index=True)
    weekly = add_conformal_bounds(weekly_totals(daily), ro.HORIZON)
    metrics = metric_rows(daily, weekly)

    per_atm = pd.DataFrame({m: weekly.groupby("terminal_id").apply(lambda g, m=m: wape(g["demand"], g[m])) for m in MODELS}).round(4)
    by_loc = pd.DataFrame({m: weekly.groupby("loc_type").apply(lambda g, m=m: wape(g["demand"], g[m])) for m in MODELS}).round(4)
    by_loc["n_atm"] = weekly.groupby("loc_type")["terminal_id"].nunique()

    OUT.mkdir(parents=True, exist_ok=True)
    cold = None
    if not args.skip_cold:
        cold_cutoffs = cutoffs[::COLD_ORIGIN_STEP] if not args.limit else cutoffs[-1:]
        print(f"cold start: {len(cold_cutoffs)} origins x {COLD_FOLDS} folds", flush=True)
        cold = cold_start(lightgbm, frame, cold_cutoffs, daily)
        cold.to_csv(OUT / "cold_start.csv", index=False)

    daily.drop(columns=[c for c in daily.columns if c.startswith("r_")]).to_csv(OUT / "predictions_daily.csv", index=False)
    weekly.to_csv(OUT / "predictions_weekly.csv", index=False)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    per_atm.to_csv(OUT / "per_atm_weekly_wape.csv")
    by_loc.to_csv(OUT / "loc_type_weekly_wape.csv")
    (OUT / "notes.json").write_text(json.dumps({
        "atms": int(n_atm), "days": len(dates), "origins": len(cutoffs),
        "first_cutoff": str(pd.Timestamp(dates[cutoffs[0]]).date()),
        "last_cutoff": str(pd.Timestamp(dates[cutoffs[-1]]).date()),
        "target": "observed net_demand (Cash_Out), not uncensored",
        "pilot_flag_agreement": agreement,
        "conformal": (
            f"P95/P99 of actual/forecast 7-day ratios, past {CONFORMAL_WINDOW_WEEKS} weeks, >= {MIN_PAST_WEEKS} weeks required; "
            f"separate for weeks touching a holiday (any day within {HOLIDAY_WEEK_LEAD_DAYS} days before one) and quiet weeks"
        ),
        "recency_half_life_days": RECENCY_HALF_LIFE_DAYS,
        "models": {
            "seasonal_naive": "same weekday last week",
            "seasonal_mean_4w": "mean of the same weekday over the last 4 weeks",
            "lgb_level": "global LightGBM on rupees (current approach, as in run_rolling_origin.py)",
            "lgb_holiday": "lgb_level plus holiday family, next holiday, bank-closure run and last year's lift for the same holiday",
            "lgb_ratio": "global LightGBM on demand / trailing 28-day mean, recency-weighted",
            "blend": "mean of lgb_ratio, lgb_level and seasonal_mean_4w",
        },
    }, indent=2))

    view = metrics[metrics["slice"] == "all"].pivot(index="model", columns="level", values="wape")
    weekly_all = metrics[(metrics["level"] == "weekly_7d_total") & (metrics["slice"] == "all")].set_index("model")
    view["weekly_bias"] = weekly_all["bias"]
    for col in ["p95_coverage", "p99_coverage", "p99_extra_cash_pct"]:
        if col in weekly_all:
            view[col] = weekly_all[col]
    print("\n" + view.loc[MODELS].to_string())
    if cold is not None:
        print("\n" + cold.to_string(index=False))
    print(f"\noutputs: {OUT}")


if __name__ == "__main__":
    main()
