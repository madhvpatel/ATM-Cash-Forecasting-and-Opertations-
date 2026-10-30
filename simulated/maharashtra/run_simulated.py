"""Standalone Maharashtra study. Nothing else in the repo imports this file.

Run it directly:
    python3 simulated/maharashtra/run_simulated.py

The baseline replenishment simulator under data/baselines/.../simulation.py
is a different job (CIT trips on the old 100-ATM feature set). This script
isolates the eight Maharashtra ATMs, joins public weather and RBI holidays,
and compares forecast specifications.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
HERE = Path(__file__).resolve().parent
PANEL_PATH = HERE / "data" / "atm_daily.csv"
WEATHER_PATH = HERE / "data" / "weather_city_daily.csv"
OUT = HERE / "output"
WEATHER_DROP = Path("/tmp/mh-wx")
CER_TARGET = 0.70
CYCLE_DAYS = 7

SPECS = [
    "atm_mean",
    "atm_dow",
    "atm_dom",
    "atm_dom_month",
    "dom_month_closure",
    "dom_month_holiday",
    "exogenous_stack",
]


def _load_weather() -> pd.DataFrame:
    frames = []
    provenance = []
    for path in sorted(WEATHER_DROP.glob("*.json")):
        payload = json.loads(path.read_text())
        daily = payload["daily"]
        city = path.stem.upper()
        frames.append(
            pd.DataFrame(
                {
                    "city": city,
                    "date": pd.to_datetime(daily["time"]),
                    "tmax_c": daily["temperature_2m_max"],
                    "tmean_c": daily["temperature_2m_mean"],
                    "tmin_c": daily["temperature_2m_min"],
                    "rain_mm": daily["precipitation_sum"],
                }
            )
        )
        provenance.append(
            {
                "city": city,
                "latitude": payload["latitude"],
                "longitude": payload["longitude"],
                "elevation_m": payload["elevation"],
                "timezone": payload["timezone"],
                "start": daily["time"][0],
                "end": daily["time"][-1],
                "rows": len(daily["time"]),
                "provider": "Open-Meteo archive API (ERA5)",
                "retrieved_for_this_run": "2026-10-21",
            }
        )
    if not frames:
        raise FileNotFoundError(f"No weather JSON in {WEATHER_DROP}")
    weather = pd.concat(frames, ignore_index=True)
    WEATHER_PATH.parent.mkdir(parents=True, exist_ok=True)
    weather.to_csv(WEATHER_PATH, index=False)
    (HERE / "data" / "weather_provenance.json").write_text(json.dumps(provenance, indent=2))
    return weather


def build_panel() -> pd.DataFrame:
    demand = pd.read_csv(DATA / "processed" / "daily_cash_out.csv", parse_dates=["transaction_date"])
    geo = pd.read_csv(DATA / "master" / "cash_files" / "terminal_geography.csv")
    meta = pd.read_csv(DATA / "master" / "cash_files" / "terminal_metadata.csv")
    calendar = pd.read_csv(DATA / "master" / "cash_files" / "dim_calendar.csv", parse_dates=["calendar_date"])
    holidays = pd.read_csv(DATA / "master" / "cash_files" / "holiday_calendar.csv", parse_dates=["calendar_date"])

    panel = demand.merge(geo, on="terminal_id").merge(meta[["terminal_id", "location"]], on="terminal_id")
    panel = panel[panel["state"] == "Maharashtra"].copy()
    panel = panel.rename(columns={"transaction_date": "date", "net_demand": "demand"})
    panel["loc_type"] = panel["location"].str.split(" - ", n=1).str[1]
    panel["dom"] = panel["date"].dt.day
    panel["month"] = panel["date"].dt.month
    panel["dow"] = panel["date"].dt.dayofweek
    panel["is_sunday"] = (panel["dow"] == 6).astype(int)
    sat_ord = (panel["date"].dt.day - 1) // 7 + 1
    panel["is_2_4_sat"] = ((panel["dow"] == 5) & sat_ord.isin([2, 4])).astype(int)
    panel["is_early"] = (panel["dom"] <= 7).astype(int)
    panel["is_late"] = (panel["dom"] >= 28).astype(int)

    checked = panel.merge(calendar, left_on="date", right_on="calendar_date", how="left")
    mismatched = int((checked["dow"] != checked["day_of_week"]).sum())
    if mismatched:
        raise AssertionError(f"Calendar day-of-week disagrees with pandas on {mismatched} rows")
    panel["is_payday"] = checked["is_payday"].astype(int).to_numpy()

    bank = holidays[holidays["holiday_type"] == "bank_holiday"][
        ["calendar_date", "region", "holiday_name"]
    ].drop_duplicates(["calendar_date", "region"])
    panel = panel.merge(
        bank,
        left_on=["date", "rbi_region"],
        right_on=["calendar_date", "region"],
        how="left",
    )
    panel["is_regional_holiday"] = panel["holiday_name"].notna().astype(int)
    nxt = bank.rename(columns={"calendar_date": "next_date", "holiday_name": "next_holiday"})
    nxt["date"] = nxt["next_date"] - pd.Timedelta(days=1)
    panel = panel.merge(
        nxt[["date", "region", "next_holiday"]],
        left_on=["date", "rbi_region"],
        right_on=["date", "region"],
        how="left",
        suffixes=("", "_next"),
    )
    panel["pre_holiday"] = (panel["next_holiday"].notna() & (panel["is_regional_holiday"] == 0)).astype(int)

    weather = _load_weather()
    panel = panel.merge(weather, on=["city", "date"], how="left")
    if panel["rain_mm"].isna().any():
        raise AssertionError("Weather does not cover every Maharashtra ATM-day")
    panel["rain_ge_10"] = (panel["rain_mm"] >= 10).astype(int)
    panel["rain_ge_25"] = (panel["rain_mm"] >= 25).astype(int)
    panel["tmax_ge_35"] = (panel["tmax_c"] >= 35).astype(int)

    keep = [
        "terminal_id", "date", "city", "state", "rbi_region", "location", "loc_type",
        "demand", "dom", "month", "dow", "is_sunday", "is_2_4_sat", "is_early", "is_late",
        "is_payday", "is_regional_holiday", "holiday_name", "pre_holiday",
        "tmax_c", "tmean_c", "tmin_c", "rain_mm", "rain_ge_10", "rain_ge_25", "tmax_ge_35",
    ]
    panel = panel[keep].sort_values(["terminal_id", "date"]).reset_index(drop=True)
    PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(PANEL_PATH, index=False)
    return panel


def _fit_factors(train: pd.DataFrame, columns: list[str]) -> dict:
    factors: dict = {"level": train.groupby("terminal_id")["demand"].mean(), "order": columns}
    ratio = train["demand"] / train["terminal_id"].map(factors["level"])
    for column in columns:
        index = ratio.groupby(train[column]).mean()
        factors[column] = index
        ratio = ratio / train[column].map(index)
    return factors


def _apply_factors(frame: pd.DataFrame, factors: dict) -> pd.Series:
    pred = frame["terminal_id"].map(factors["level"]).astype(float)
    for column in factors["order"]:
        pred = pred * frame[column].map(factors[column]).astype(float)
    return pred


def _multiplier(train: pd.DataFrame, base: pd.Series, column: str) -> float:
    mask = train[column] == 1
    if int(mask.sum()) < 15:
        return 1.0
    return float((train.loc[mask, "demand"] / base.loc[mask]).mean())


def _predict(train: pd.DataFrame, frame: pd.DataFrame, spec: str) -> pd.Series:
    if spec == "atm_mean":
        return _apply_factors(frame, _fit_factors(train, []))
    if spec == "atm_dow":
        return _apply_factors(frame, _fit_factors(train, ["dow"]))
    if spec == "atm_dom":
        return _apply_factors(frame, _fit_factors(train, ["dom"]))
    factors = _fit_factors(train, ["dom", "month"])
    base = _apply_factors(frame, factors)
    if spec == "atm_dom_month":
        return base
    base_train = _apply_factors(train, factors)
    sun = _multiplier(train, base_train, "is_sunday")
    sat = _multiplier(train, base_train, "is_2_4_sat")
    hol = _multiplier(train, base_train, "is_regional_holiday")
    pre = _multiplier(train, base_train, "pre_holiday")
    if spec == "dom_month_closure":
        return base * np.where(frame["is_sunday"].eq(1), sun, 1.0) * np.where(frame["is_2_4_sat"].eq(1), sat, 1.0)
    if spec == "dom_month_holiday":
        return base * np.where(frame["is_regional_holiday"].eq(1), hol, 1.0) * np.where(frame["pre_holiday"].eq(1), pre, 1.0)
    if spec == "exogenous_stack":
        return (
            base
            * np.where(frame["is_sunday"].eq(1), sun, 1.0)
            * np.where(frame["is_2_4_sat"].eq(1), sat, 1.0)
            * np.where(frame["is_regional_holiday"].eq(1), hol, 1.0)
            * np.where(frame["pre_holiday"].eq(1), pre, 1.0)
        )
    raise KeyError(spec)


def _wape(actual: pd.Series, pred: pd.Series) -> float:
    return float((actual - pred).abs().sum() / actual.sum())


def _bias(actual: pd.Series, pred: pd.Series) -> float:
    return float((pred - actual).sum() / actual.sum())


def _one_step_lag(ordered: pd.DataFrame, calendar: pd.Series, test_index: pd.Index) -> pd.Series:
    residual = ordered["demand"] - calendar
    lagged = residual.groupby(ordered["terminal_id"]).shift(1)
    pred = (calendar + lagged).loc[test_index].clip(lower=0)
    return pred


def _full_blocks(test: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    scored = test.copy()
    scored["block"] = ((scored["date"] - start).dt.days // CYCLE_DAYS).astype(int)
    sizes = scored.groupby(["terminal_id", "block"]).size()
    keep = sizes[sizes == CYCLE_DAYS].index
    grouped = (
        scored.groupby(["terminal_id", "block"], as_index=False)
        .agg(actual=("demand", "sum"), predicted=("pred", "sum"), first_date=("date", "min"))
    )
    key = pd.MultiIndex.from_frame(grouped[["terminal_id", "block"]])
    return grouped.loc[key.isin(keep)].copy()


def _indent_row(window: str, model: str, grouped: pd.DataFrame) -> dict:
    load = grouped["predicted"] / CER_TARGET
    realized = np.minimum(grouped["actual"], load) / load
    short = grouped["actual"] > load
    return {
        "window": window,
        "model": model,
        "cycles": int(len(grouped)),
        "wape_7d": round(_wape(grouped["actual"], grouped["predicted"]), 4),
        "mean_load_inr": round(float(load.mean()), 0),
        "mean_actual_inr": round(float(grouped["actual"].mean()), 0),
        "mean_realized_cer": round(float(realized.mean()), 3),
        "stockout_cycle_rate": round(float(short.mean()), 3),
        "mean_shortfall_when_short_inr": round(
            float((grouped.loc[short, "actual"] - load.loc[short]).mean()) if short.any() else 0.0,
            0,
        ),
    }


def evaluate(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    last = panel["date"].max()
    windows = [
        ("holdout_last_90d", last - pd.Timedelta(days=89), last),
        ("prior_90d", last - pd.Timedelta(days=179), last - pd.Timedelta(days=90)),
    ]
    forecast_rows = []
    indent_rows = []
    for window, start, end in windows:
        train = panel[panel["date"] < start]
        test = panel[(panel["date"] >= start) & (panel["date"] <= end)].copy()
        whole = pd.concat([train, test]).sort_values(["terminal_id", "date"])
        for spec in SPECS:
            pred_all = pd.Series(_predict(train, whole, spec).to_numpy(), index=whole.index)
            pred = pred_all.loc[test.index]
            forecast_rows.append(
                {
                    "window": window,
                    "model": spec,
                    "horizon": "1d",
                    "rows": int(len(test)),
                    "wape": round(_wape(test["demand"], pred), 4),
                    "bias_pct": round(100 * _bias(test["demand"], pred), 2),
                }
            )
            scored = test.assign(pred=pred.to_numpy())
            grouped = _full_blocks(scored, start)
            forecast_rows.append(
                {
                    "window": window,
                    "model": spec,
                    "horizon": "7d_sum",
                    "rows": int(len(grouped)),
                    "wape": round(_wape(grouped["actual"], grouped["predicted"]), 4),
                    "bias_pct": round(100 * _bias(grouped["actual"], grouped["predicted"]), 2),
                }
            )
            indent_rows.append(_indent_row(window, spec, grouped))
            if spec == "exogenous_stack":
                lag_pred = _one_step_lag(whole, pred_all, test.index)
                forecast_rows.append(
                    {
                        "window": window,
                        "model": "exogenous_stack_plus_lag1",
                        "horizon": "1d",
                        "rows": int(len(test)),
                        "wape": round(_wape(test["demand"], lag_pred), 4),
                        "bias_pct": round(100 * _bias(test["demand"], lag_pred), 2),
                    }
                )
        # Sum of the previous 7 observed days, known before each block starts.
        history = panel.sort_values(["terminal_id", "date"])
        scored = test.copy()
        scored["block"] = ((scored["date"] - start).dt.days // CYCLE_DAYS).astype(int)
        scored["pred"] = np.nan
        rows = []
        sizes = scored.groupby(["terminal_id", "block"]).size()
        for terminal_id, block in sizes[sizes == CYCLE_DAYS].index:
            chunk = scored[(scored.terminal_id == terminal_id) & (scored.block == block)]
            prev = history[(history.terminal_id == terminal_id) & (history.date < chunk["date"].min())].tail(CYCLE_DAYS)
            rows.append({"actual": float(chunk["demand"].sum()), "predicted": float(prev["demand"].sum())})
        naive = pd.DataFrame(rows)
        forecast_rows.append(
            {
                "window": window,
                "model": "trailing_7d",
                "horizon": "7d_sum",
                "rows": int(len(naive)),
                "wape": round(_wape(naive["actual"], naive["predicted"]), 4),
                "bias_pct": round(100 * _bias(naive["actual"], naive["predicted"]), 2),
            }
        )
        indent_rows.append(_indent_row(window, "trailing_7d", naive))
    return pd.DataFrame(forecast_rows), pd.DataFrame(indent_rows)


def correlations(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    last = panel["date"].max()
    start = last - pd.Timedelta(days=89)
    train = panel[panel["date"] < start]
    factors = _fit_factors(train, ["dom", "month"])
    residual_ratio = panel["demand"] / _apply_factors(panel, factors)
    flags = [
        "is_early", "is_late", "is_sunday", "is_2_4_sat", "is_payday",
        "is_regional_holiday", "pre_holiday", "rain_ge_10", "rain_ge_25", "tmax_ge_35",
    ]
    effect_rows = []
    for scope, mask in {
        "train_fit_scored_on_all_days": pd.Series(True, index=panel.index),
        "train_fit_scored_on_holdout": panel["date"] >= start,
    }.items():
        for flag in flags:
            on = mask & panel[flag].eq(1)
            effect_rows.append(
                {
                    "scope": scope,
                    "flag": flag,
                    "rows": int(on.sum()),
                    "days": int(panel.loc[on, "date"].nunique()) if on.any() else 0,
                    "median_ratio_to_dom_month": round(float(residual_ratio.loc[on].median()), 3) if on.any() else None,
                    "mean_ratio_to_dom_month": round(float(residual_ratio.loc[on].mean()), 3) if on.any() else None,
                }
            )
    corr_rows = []
    for column in ["rain_mm", "tmax_c", "tmean_c", "dom", "dow", *flags]:
        series = panel[column].astype(float)
        for scope, mask in [("train", panel["date"] < start), ("holdout", panel["date"] >= start)]:
            right = series.loc[mask]
            value = None if right.nunique(dropna=True) < 2 else round(float(residual_ratio.loc[mask].corr(right)), 3)
            corr_rows.append(
                {
                    "scope": scope,
                    "feature": column,
                    "pearson_with_dom_month_residual_ratio": value,
                    "rows": int(mask.sum()),
                }
            )
    city_rows = []
    train_idx = panel["date"] < start
    for city, frame in panel.loc[train_idx].groupby("city"):
        idx = frame.index
        wet = frame["rain_ge_10"].eq(1)
        city_rows.append(
            {
                "city": city,
                "rows": int(len(frame)),
                "pearson_rain_vs_residual": round(float(residual_ratio.loc[idx].corr(frame["rain_mm"])), 3),
                "pearson_tmax_vs_residual": round(float(residual_ratio.loc[idx].corr(frame["tmax_c"])), 3),
                "rain_ge_10_median_ratio": round(float(residual_ratio.loc[idx][wet].median()), 3) if wet.any() else None,
            }
        )
    return pd.DataFrame(effect_rows), pd.DataFrame(corr_rows), pd.DataFrame(city_rows)


def profiles(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for terminal_id, frame in panel.groupby("terminal_id"):
        rows.append(
            {
                "terminal_id": int(terminal_id),
                "city": frame["city"].iloc[0],
                "rbi_region": frame["rbi_region"].iloc[0],
                "loc_type": frame["loc_type"].iloc[0],
                "mean_demand_inr": round(float(frame["demand"].mean()), 0),
                "cv": round(float(frame["demand"].std() / frame["demand"].mean()), 3),
                "holiday_days": int(frame["is_regional_holiday"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("terminal_id")


def write_assumptions(panel: pd.DataFrame) -> None:
    payload = {
        "region": "Maharashtra",
        "why_this_region": "Named pilot on the engagement slide. Only state whose cities all received a public weather pull, and whose ATMs already map to RBI offices.",
        "atms": int(panel["terminal_id"].nunique()),
        "days": int(panel["date"].nunique()),
        "demand_source": "data/processed/daily_cash_out.csv, the daily sum of 15-minute Cash_Out. The series are synthetic.",
        "exogenous_included": [
            "Day of month, month, Sunday, 2nd and 4th Saturday, from the date.",
            "Payday flag from the existing 1st / 15th / last-weekday rule.",
            "RBI regional bank holidays for the Mumbai and Nagpur offices. known_at is blank, so this join is retrospective.",
            "City-centroid daily rain and temperature from the Open-Meteo archive (ERA5).",
            "Location type parsed from the location string. One ATM per type, so it is the site level, not a shared factor.",
        ],
        "exogenous_not_invented": [
            "Footfall, demographics, UPI, competing ATMs, card mix, festival intensity scores, cassette capacity, failed withdrawals, CIT routes.",
        ],
        "policy_assumptions_for_the_indent_sketch": {
            "cycle_days": CYCLE_DAYS,
            "cer_target": CER_TARGET,
            "load_rule": "predicted cycle demand / 0.70",
            "capacity_constraint": "none applied",
        },
        "holdout": "Last 90 days, factors fit only on earlier days. The previous 90 days are the stability check.",
    }
    (HERE / "assumptions.json").write_text(json.dumps(payload, indent=2))


def write_forward_indent(panel: pd.DataFrame) -> pd.DataFrame:
    """Freeze the day-of-month baseline and write the next seven-day indent."""
    factors = _fit_factors(panel, ["dom"])
    mu = factors["level"]
    index = factors["dom"]
    origin = panel["date"].max()
    sites = panel[["terminal_id", "city", "loc_type"]].drop_duplicates("terminal_id")
    sites["atm_id"] = sites["terminal_id"].map(lambda i: f"SYNATM{int(i):04d}")
    days = pd.date_range(origin + pd.Timedelta(days=1), periods=7, freq="D")
    rows = []
    for site in sites.itertuples(index=False):
        atm_level = float(mu.loc[site.terminal_id])
        for day in days:
            factor = float(index.loc[day.day])
            rows.append({
                "atm_id": site.atm_id,
                "terminal_id": int(site.terminal_id),
                "city": site.city,
                "loc_type": site.loc_type,
                "date": day.strftime("%Y-%m-%d"),
                "dom": int(day.day),
                "level_inr": round(atm_level, 2),
                "dom_index": round(factor, 4),
                "expected_demand_inr": round(atm_level * factor, 2),
            })
    daily = pd.DataFrame(rows)
    indent = (
        daily.groupby(["atm_id", "terminal_id", "city", "loc_type"], as_index=False)
        .agg(
            cycle_start=("date", "min"),
            cycle_end=("date", "max"),
            expected_demand_inr=("expected_demand_inr", "sum"),
        )
        .sort_values("terminal_id")
    )
    indent["cer_target"] = CER_TARGET
    indent["load_inr"] = (indent["expected_demand_inr"] / CER_TARGET).round(2)
    indent["expected_demand_inr"] = indent["expected_demand_inr"].round(2)
    OUT.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUT / "next_7d_forecast.csv", index=False)
    indent.to_csv(OUT / "next_indent.csv", index=False)
    (OUT / "frozen_baseline.json").write_text(json.dumps({
        "model": "atm_level_x_day_of_month",
        "fit_start": panel["date"].min().strftime("%Y-%m-%d"),
        "fit_end": origin.strftime("%Y-%m-%d"),
        "forecast_start": days.min().strftime("%Y-%m-%d"),
        "forecast_end": days.max().strftime("%Y-%m-%d"),
        "cer_target": CER_TARGET,
        "load_rule": "sum of seven daily forecasts / 0.70",
        "level_inr": {str(int(k)): round(float(v), 2) for k, v in mu.items()},
        "dom_index": {str(int(k)): round(float(v), 4) for k, v in index.items()},
    }, indent=2))
    return indent


def main() -> None:
    panel = build_panel()
    effects, corrs, cities = correlations(panel)
    forecasts, indents = evaluate(panel)
    OUT.mkdir(parents=True, exist_ok=True)
    profiles(panel).to_csv(OUT / "atm_profile.csv", index=False)
    effects.to_csv(OUT / "flag_effects.csv", index=False)
    corrs.to_csv(OUT / "correlations.csv", index=False)
    cities.to_csv(OUT / "weather_by_city.csv", index=False)
    forecasts.to_csv(OUT / "forecast_comparison.csv", index=False)
    indents.to_csv(OUT / "indent_simulation.csv", index=False)
    write_assumptions(panel)
    holdout = forecasts[forecasts.window == "holdout_last_90d"]
    print(f"panel rows={len(panel)} atms={panel.terminal_id.nunique()} days={panel.date.nunique()}")
    print(holdout.to_string(index=False))
    print(indents[indents.window == "holdout_last_90d"].to_string(index=False))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "indent":
        frozen = pd.read_csv(PANEL_PATH, parse_dates=["date"])
        print(write_forward_indent(frozen).to_string(index=False))
    else:
        main()
