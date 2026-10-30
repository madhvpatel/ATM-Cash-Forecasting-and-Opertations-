"""Rolling-origin backtest for the eight Maharashtra ATMs.

Nothing else imports this file. Run it directly:

    python3 simulated/maharashtra/run_rolling_origin.py

The seasonal benchmark is the shipped ATM-mean x day-of-month profile in
run_simulated.py, refit on each split. This script does not rebuild weather
and does not replace output/frozen_baseline.json.

Leakage rules that are easy to miss:
- The forecast is issued after the cutoff day's demand is known. The next
  seven days are the test week.
- lag_1 is that cutoff day's demand. For horizon 2-7, demand on the day
  before the target is inside the week being forecast, so it is not used.
- lag_7, lag_14 and lag_28 are demand on the target date minus that many
  days. With a 7-day horizon those dates fall on or before the cutoff.
- Rolling windows end on the cutoff, inclusive, and use a full window.
- Realized ERA5 weather is not a feature. It was not known as a forecast
  on the cutoff.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PANEL_PATH = HERE / "data" / "atm_daily.csv"
OUT = HERE / "output"
LIBOMP_CANDIDATES = (Path("/opt/homebrew/anaconda3/lib/libomp.dylib"),)

MIN_TRAIN_DAYS = 365
HORIZON = 7
STEP = 7
CER_TARGET = 0.70
IDLE_RATE = 0.08
SEED = 22
N_ESTIMATORS = 120
LEARNING_RATE = 0.08
NUM_LEAVES = 31
MIN_CHILD_SAMPLES = 30

FEATURES = [
    "dow",
    "dom",
    "month",
    "is_weekend",
    "is_2_4_sat",
    "is_month_end",
    "is_payday",
    "is_regional_holiday",
    "pre_holiday",
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
CATEGORICAL = ["terminal_id", "loc_type"]
HISTORY = [
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
]
WEATHER = {
    "tmax_c",
    "tmean_c",
    "tmin_c",
    "rain_mm",
    "rain_ge_10",
    "rain_ge_25",
    "tmax_ge_35",
}
MODELS = (
    ("seasonal_mean", "seasonal_mean", "seasonal_p80", "seasonal_p95"),
    ("seasonal_p50", "seasonal_p50", "seasonal_p80", "seasonal_p95"),
    ("lightgbm", "lightgbm_p50", "lightgbm_p80", "lightgbm_p95"),
)
PREDICTION_COLUMNS = [
    "terminal_id",
    "loc_type",
    "origin_date",
    "target_date",
    "horizon",
    "demand",
    "dow",
    "dom",
    "month",
    "is_weekend",
    "is_2_4_sat",
    "is_month_end",
    "is_payday",
    "is_regional_holiday",
    "pre_holiday",
    *HISTORY,
    "seasonal_mean",
    "seasonal_p50",
    "seasonal_p80",
    "seasonal_p95",
    "lightgbm_p50",
    "lightgbm_p80",
    "lightgbm_p95",
    "quantile_crossed",
]


def load_lightgbm():
    """Import LightGBM, preloading a local libomp when the wheel cannot find one."""
    libomp_used = None
    try:
        import lightgbm
    except OSError:
        for name in list(sys.modules):
            if name == "lightgbm" or name.startswith("lightgbm."):
                del sys.modules[name]
        for path in LIBOMP_CANDIDATES:
            if path.exists():
                ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
                libomp_used = str(path)
                break
        import lightgbm
    return lightgbm, libomp_used


def load_simulator():
    spec = importlib.util.spec_from_file_location("maharashtra_simulated", HERE / "run_simulated.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_panel() -> tuple[pd.DataFrame, np.ndarray]:
    panel = pd.read_csv(PANEL_PATH, parse_dates=["date"])
    panel["terminal_id"] = panel["terminal_id"].astype(int)
    panel = panel.sort_values(["terminal_id", "date"]).reset_index(drop=True)
    if panel.duplicated(["terminal_id", "date"]).any():
        raise AssertionError("duplicate ATM-day")
    dates = np.sort(panel["date"].unique())
    gaps = np.diff(dates.astype("datetime64[D]").astype(np.int64))
    if not np.all(gaps == 1):
        raise AssertionError("panel dates are not continuous")
    if len(panel) != panel["terminal_id"].nunique() * len(dates):
        raise AssertionError("panel is not balanced")
    return panel, dates


def cutoff_indices(n_dates: int) -> list[int]:
    # c + 7 is the last test index, so c <= n_dates - 8. range end is exclusive.
    first = MIN_TRAIN_DAYS - 1
    if first < 27:
        raise AssertionError("training window is shorter than the 28-day history")
    return list(range(first, n_dates - HORIZON, STEP))


def build_rows(panel: pd.DataFrame, dates: np.ndarray) -> pd.DataFrame:
    chunks = []
    for terminal_id, group in panel.groupby("terminal_id", sort=True):
        group = group.sort_values("date")
        if not np.array_equal(group["date"].to_numpy(), dates):
            raise AssertionError(f"ATM {terminal_id} is missing a day")
        if group["loc_type"].nunique() != 1:
            raise AssertionError(f"ATM {terminal_id} has more than one location type")
        demand = group["demand"].to_numpy(dtype=float)
        dom = group["dom"].to_numpy(dtype=int)
        dow = group["dow"].to_numpy(dtype=int)
        month = group["month"].to_numpy(dtype=int)
        days_in_month = group["date"].dt.days_in_month.to_numpy(dtype=int)
        is_month_end = (days_in_month - dom <= 2).astype(int)
        is_weekend = (dow >= 5).astype(int)
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
        if (target - 7 > origin).any() or (target - 28 < 0).any():
            raise AssertionError("a lag reaches past the cutoff or before the series")

        chunks.append(
            pd.DataFrame(
                {
                    "terminal_id": int(terminal_id),
                    "loc_type": group["loc_type"].iloc[0],
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
    if frame[HISTORY].isna().any().any():
        raise AssertionError("history feature is missing")
    return frame


def assert_history(panel: pd.DataFrame, frame: pd.DataFrame) -> None:
    if WEATHER & set(frame.columns) or WEATHER & set(FEATURES):
        raise AssertionError("realized weather entered the model frame")
    week = frame.loc[frame["horizon"] == 7, ["lag_1", "lag_7"]]
    if not np.allclose(week["lag_1"], week["lag_7"]):
        raise AssertionError("at horizon 7, lag_7 must be the cutoff day's demand")
    if not (frame["target_idx"] - frame["horizon"] == frame["origin_idx"]).all():
        raise AssertionError("horizon does not connect origin to target")

    sample = pd.concat(
        [
            frame.loc[frame["horizon"] == 1].sample(8, random_state=SEED),
            frame.loc[frame["horizon"] == 7].sample(8, random_state=SEED),
        ],
        ignore_index=True,
    )
    observed = {
        int(terminal_id): group.set_index("date")["demand"].sort_index()
        for terminal_id, group in panel.groupby("terminal_id")
    }
    for row in sample.itertuples(index=False):
        series = observed[int(row.terminal_id)]
        origin = pd.Timestamp(row.origin_date)
        target = pd.Timestamp(row.target_date)
        if target != origin + pd.Timedelta(days=int(row.horizon)):
            raise AssertionError("target date is not the origin plus the horizon")
        if series.loc[origin] != row.lag_1:
            raise AssertionError("lag_1 is not the cutoff day's demand")
        for lag, column in ((7, "lag_7"), (14, "lag_14"), (28, "lag_28")):
            if series.loc[target - pd.Timedelta(days=lag)] != getattr(row, column):
                raise AssertionError(column)
        if series.loc[target] != row.demand:
            raise AssertionError("target demand does not match the panel")
        for window in (7, 14, 28):
            history = series.loc[origin - pd.Timedelta(days=window - 1) : origin]
            if len(history) != window or target in history.index:
                raise AssertionError(f"{window}-day window is not closed on the cutoff")
            if not np.isclose(history.mean(), getattr(row, f"roll_mean_{window}")):
                raise AssertionError(f"roll_mean_{window}")
            if not np.isclose(history.std(ddof=1), getattr(row, f"roll_std_{window}")):
                raise AssertionError(f"roll_std_{window}")


def seasonal_forecast(simulator, panel: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """ATM mean x day-of-month, fit on dates through the cutoff only."""
    train = panel.loc[panel["date"] <= cutoff].copy()
    if train["date"].max() != cutoff:
        raise AssertionError("seasonal training does not end on the cutoff")
    factors = simulator._fit_factors(train, ["dom"])
    point = simulator._apply_factors(train, factors)
    if point.isna().any() or (point <= 0).any():
        raise AssertionError("seasonal profile is not positive on the training days")
    ratio = train["demand"].to_numpy(dtype=float) / point.to_numpy(dtype=float)
    ratios = pd.Series(ratio, index=train.index)
    scales = {
        0.5: ratios.groupby(train["terminal_id"]).quantile(0.5),
        0.8: ratios.groupby(train["terminal_id"]).quantile(0.8),
        0.95: ratios.groupby(train["terminal_id"]).quantile(0.95),
    }
    future = panel.loc[
        (panel["date"] > cutoff) & (panel["date"] <= cutoff + pd.Timedelta(days=HORIZON))
    ].copy()
    n_atm = panel["terminal_id"].nunique()
    if future["date"].nunique() != HORIZON or len(future) != n_atm * HORIZON:
        raise AssertionError("forecast week is incomplete")
    # Demand on these rows is the label. The profile uses terminal id and day of month.
    mean = simulator._apply_factors(future, factors).to_numpy(dtype=float)
    out = pd.DataFrame(
        {
            "terminal_id": future["terminal_id"].to_numpy(),
            "target_date": future["date"].to_numpy(),
            "seasonal_mean": mean,
            "seasonal_p50": mean * future["terminal_id"].map(scales[0.5]).to_numpy(dtype=float),
            "seasonal_p80": mean * future["terminal_id"].map(scales[0.8]).to_numpy(dtype=float),
            "seasonal_p95": mean * future["terminal_id"].map(scales[0.95]).to_numpy(dtype=float),
        }
    )
    if out.filter(like="seasonal_").isna().any().any():
        raise AssertionError("seasonal forecast is missing a scale")
    return out


def design_matrix(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_train = train[FEATURES].copy()
    x_test = test[FEATURES].copy()
    for column in CATEGORICAL:
        categories = pd.CategoricalDtype(categories=sorted(x_train[column].unique()))
        x_train[column] = x_train[column].astype(categories)
        x_test[column] = x_test[column].astype(categories)
    return x_train, x_test


def fit_quantiles(lightgbm, train: pd.DataFrame, test: pd.DataFrame):
    x_train, x_test = design_matrix(train, test)
    y_train = train["demand"].to_numpy(dtype=float)
    raw = []
    model_p50 = None
    for alpha in (0.5, 0.8, 0.95):
        model = lightgbm.LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=N_ESTIMATORS,
            learning_rate=LEARNING_RATE,
            num_leaves=NUM_LEAVES,
            min_child_samples=MIN_CHILD_SAMPLES,
            subsample=1.0,
            colsample_bytree=1.0,
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
            force_col_wise=True,
        )
        model.fit(x_train, y_train, categorical_feature=CATEGORICAL)
        raw.append(np.clip(model.predict(x_test), 0, None))
        if alpha == 0.5:
            model_p50 = model
    stacked = np.vstack(raw)
    crossed = (stacked[0] > stacked[1]) | (stacked[1] > stacked[2])
    ordered = np.maximum.accumulate(stacked, axis=0)
    if not np.isfinite(ordered).all():
        raise AssertionError("LightGBM returned a non-finite forecast")
    return ordered, crossed, model_p50


def _wape(actual, pred) -> float:
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.abs(pred - actual).sum() / actual.sum())


def _bias(actual, pred) -> float:
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float((pred - actual).sum() / actual.sum())


def _coverage(actual, pred) -> float:
    return float(np.mean(np.asarray(actual, dtype=float) <= np.asarray(pred, dtype=float)))


def metric_row(model: str, level: str, slice_name: str, actual, point, p80, p95) -> dict:
    return {
        "model": model,
        "level": level,
        "slice": slice_name,
        "n": int(len(actual)),
        "wape": round(_wape(actual, point), 6),
        "bias": round(_bias(actual, point), 6),
        "point_coverage": round(_coverage(actual, point), 6),
        "p80_coverage": round(_coverage(actual, p80), 6),
        "p95_coverage": round(_coverage(actual, p95), 6),
    }


def daily_slices(part: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    slices = [("all", part)]
    for horizon in range(1, HORIZON + 1):
        slices.append((f"h={horizon}", part.loc[part["horizon"] == horizon]))
    slices.append(("month_end", part.loc[part["is_month_end"] == 1]))
    slices.append(("not_month_end", part.loc[part["is_month_end"] == 0]))
    slices.append(("holiday", part.loc[part["is_regional_holiday"] == 1]))
    slices.append(("not_holiday", part.loc[part["is_regional_holiday"] == 0]))
    for terminal_id in sorted(part["terminal_id"].unique()):
        slices.append((f"atm:{int(terminal_id)}", part.loc[part["terminal_id"] == terminal_id]))
    return slices


def metrics_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, point, p80, p95 in MODELS:
        for name, part in daily_slices(predictions):
            if part.empty:
                continue
            rows.append(metric_row(model, "daily", name, part["demand"], part[point], part[p80], part[p95]))
        cycles = predictions.groupby(["terminal_id", "origin_date"], as_index=False).agg(
            actual=("demand", "sum"),
            point=(point, "sum"),
            p80=(p80, "sum"),
            p95=(p95, "sum"),
            n=("demand", "size"),
        )
        if not cycles["n"].eq(HORIZON).all():
            raise AssertionError("a cycle does not have 7 days")
        rows.append(metric_row(model, "cycle", "all", cycles["actual"], cycles["point"], cycles["p80"], cycles["p95"]))
        for terminal_id, atm_cycles in cycles.groupby("terminal_id"):
            rows.append(
                metric_row(
                    model,
                    "cycle",
                    f"atm:{int(terminal_id)}",
                    atm_cycles["actual"],
                    atm_cycles["point"],
                    atm_cycles["p80"],
                    atm_cycles["p95"],
                )
            )
    return pd.DataFrame(rows)


def fold_table(predictions: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for origin_date, part in predictions.groupby("origin_date", sort=True):
        row = {"origin_date": pd.Timestamp(origin_date).date().isoformat()}
        info = meta.loc[meta["origin_date"] == origin_date].iloc[0]
        row["train_rows"] = int(info["train_rows"])
        row["train_days"] = int(info["train_days"])
        row["cross_rate"] = float(info["cross_rate"])
        for model, point, p80, p95 in MODELS:
            row[f"{model}_wape"] = round(_wape(part["demand"], part[point]), 6)
            row[f"{model}_bias"] = round(_bias(part["demand"], part[point]), 6)
            row[f"{model}_p80_coverage"] = round(_coverage(part["demand"], part[p80]), 6)
            row[f"{model}_p95_coverage"] = round(_coverage(part["demand"], part[p95]), 6)
            totals = part.groupby("terminal_id")[["demand", point]].sum()
            row[f"{model}_cycle_wape"] = round(_wape(totals["demand"], totals[point]), 6)
        rows.append(row)
    return pd.DataFrame(rows)


def replenishment_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Same visit rule for both models. No cassette cap and no rupee visit tariff."""
    specs = (
        ("seasonal_mean", "seasonal_mean", "seasonal_p80", "seasonal_p95"),
        ("lightgbm", "lightgbm_p50", "lightgbm_p80", "lightgbm_p95"),
    )
    rows = []
    for model, point, p80, p95 in specs:
        cycles = predictions.groupby(["terminal_id", "origin_date"], as_index=False).agg(
            actual=("demand", "sum"),
            point=(point, "sum"),
            p80=(p80, "sum"),
            p95=(p95, "sum"),
        )
        scopes = [("all", cycles)] + [
            (f"atm:{int(terminal_id)}", cycles.loc[cycles["terminal_id"] == terminal_id])
            for terminal_id in sorted(cycles["terminal_id"].unique())
        ]
        rules = {
            "cer70": cycles["point"] / CER_TARGET,
            "sum_daily_p80": cycles["p80"],
            "sum_daily_p95": cycles["p95"],
        }
        for scope, part in scopes:
            for rule, load_all in rules.items():
                actual = part["actual"].to_numpy(dtype=float)
                load = load_all.loc[part.index].to_numpy(dtype=float)
                excess = np.maximum(load - actual, 0.0)
                shortfall = np.maximum(actual - load, 0.0)
                stockout = actual > load
                n_short = int(stockout.sum())
                dispensed = np.minimum(actual, load)
                rows.append(
                    {
                        "model": model,
                        "load_rule": rule,
                        "scope": scope,
                        "n_cycles": int(len(part)),
                        "stockout_rate": round(float(stockout.mean()), 6),
                        "n_stockouts": n_short,
                        "mean_excess_inr": round(float(excess.mean()), 2),
                        "mean_shortfall_inr": round(float(shortfall.mean()), 2),
                        "mean_shortfall_when_short_inr": None
                        if n_short == 0
                        else round(float(shortfall[stockout].mean()), 2),
                        "mean_cer": round(float((dispensed / load).mean()), 6),
                        "planned_visits": int(len(part)),
                        "emergency_visits": n_short,
                        "idle_interest_sketch_inr": round(float((excess * IDLE_RATE * HORIZON / 365).sum()), 2),
                        "total_actual_inr": round(float(actual.sum()), 2),
                        "total_load_inr": round(float(load.sum()), 2),
                        "total_excess_inr": round(float(excess.sum()), 2),
                        "total_shortfall_inr": round(float(shortfall.sum()), 2),
                    }
                )
    return pd.DataFrame(rows)


def importance_table(model) -> pd.DataFrame:
    booster = model.booster_
    gain = booster.feature_importance(importance_type="gain").astype(float)
    splits = booster.feature_importance(importance_type="split").astype(int)
    total = float(gain.sum()) or 1.0
    table = pd.DataFrame(
        {
            "feature": booster.feature_name(),
            "gain": gain,
            "gain_share": gain / total,
            "splits": splits,
        }
    )
    return table.sort_values("gain", ascending=False).reset_index(drop=True)


def run(limit: int, out_dir: Path) -> None:
    started = time.perf_counter()
    lightgbm, libomp_used = load_lightgbm()
    simulator = load_simulator()
    panel, dates = load_panel()
    frame = build_rows(panel, dates)
    assert_history(panel, frame)
    cutoffs = cutoff_indices(len(dates))
    if limit:
        cutoffs = cutoffs[:limit]

    scored = []
    meta_rows = []
    last_model = None
    n_atm = panel["terminal_id"].nunique()
    for step, cutoff_idx in enumerate(cutoffs, start=1):
        step_started = time.perf_counter()
        cutoff = pd.Timestamp(dates[cutoff_idx])
        train = frame.loc[frame["target_idx"] <= cutoff_idx]
        test = (
            frame.loc[frame["origin_idx"] == cutoff_idx]
            .sort_values(["terminal_id", "horizon"])
            .reset_index(drop=True)
        )
        if len(test) != n_atm * HORIZON:
            raise AssertionError("test week is not one row per ATM per day")
        if train["target_idx"].max() > cutoff_idx or test["target_idx"].min() <= cutoff_idx:
            raise AssertionError("train and test dates overlap")
        seasonal = seasonal_forecast(simulator, panel, cutoff)
        test = test.merge(seasonal, on=["terminal_id", "target_date"], how="left", validate="one_to_one")
        if test["seasonal_mean"].isna().any():
            raise AssertionError("seasonal forecast did not match a test row")
        ordered, crossed, last_model = fit_quantiles(lightgbm, train, test)
        test["lightgbm_p50"] = ordered[0]
        test["lightgbm_p80"] = ordered[1]
        test["lightgbm_p95"] = ordered[2]
        test["quantile_crossed"] = crossed.astype(int)
        meta_rows.append(
            {
                "origin_date": cutoff,
                "train_rows": int(len(train)),
                "train_days": int(cutoff_idx + 1),
                "cross_rate": float(crossed.mean()),
            }
        )
        scored.append(test[PREDICTION_COLUMNS])
        print(
            f"{step}/{len(cutoffs)} cutoff={cutoff.date()} train_rows={len(train)} "
            f"seasonal_wape={_wape(test['demand'], test['seasonal_mean']):.3f} "
            f"lightgbm_wape={_wape(test['demand'], test['lightgbm_p50']):.3f} "
            f"cross={crossed.mean():.3f} {time.perf_counter() - step_started:.1f}s",
            flush=True,
        )

    predictions = pd.concat(scored, ignore_index=True)
    counts = predictions.groupby(["terminal_id", "target_date"]).size()
    if not counts.eq(1).all():
        raise AssertionError("a forecast date was scored more than once")
    scored_dates = np.sort(predictions["target_date"].unique())
    gaps = np.diff(scored_dates.astype("datetime64[D]").astype(np.int64))
    if len(gaps) and not np.all(gaps == 1):
        raise AssertionError("scored dates have a gap")

    meta = pd.DataFrame(meta_rows)
    metrics = metrics_table(predictions)
    folds = fold_table(predictions, meta)
    replenishment = replenishment_table(predictions)
    importance = importance_table(last_model)

    out_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_dir / "rolling_origin_predictions.csv", index=False)
    metrics.to_csv(out_dir / "rolling_origin_metrics.csv", index=False)
    replenishment.to_csv(out_dir / "rolling_origin_replenishment.csv", index=False)
    folds.to_csv(out_dir / "rolling_origin_folds.csv", index=False)
    importance.to_csv(out_dir / "rolling_origin_importance.csv", index=False)

    loc_pairs = panel.groupby("terminal_id")["loc_type"].nunique()
    notes = {
        "protocol": "rolling-origin 7-day direct forecast",
        "prototype": (
            "754 days across eight simulated ATMs tests the pipeline. "
            "Treat the scores as a prototype until the same protocol is run on real ATM data."
        ),
        "panel": str(PANEL_PATH),
        "atms": int(n_atm),
        "panel_start": pd.Timestamp(dates[0]).date().isoformat(),
        "panel_end": pd.Timestamp(dates[-1]).date().isoformat(),
        "panel_days": int(len(dates)),
        "min_train_days": MIN_TRAIN_DAYS,
        "horizon_days": HORIZON,
        "origin_step_days": STEP,
        "n_origins": int(len(cutoffs)),
        "limit": int(limit),
        "first_cutoff": pd.Timestamp(dates[cutoffs[0]]).date().isoformat(),
        "last_cutoff": pd.Timestamp(dates[cutoffs[-1]]).date().isoformat(),
        "first_scored_date": pd.Timestamp(scored_dates[0]).date().isoformat(),
        "last_scored_date": pd.Timestamp(scored_dates[-1]).date().isoformat(),
        "unscored_tail_start": (
            None
            if pd.Timestamp(scored_dates[-1]) == pd.Timestamp(dates[-1])
            else (pd.Timestamp(scored_dates[-1]) + pd.Timedelta(days=1)).date().isoformat()
        ),
        "unscored_tail_end": pd.Timestamp(dates[-1]).date().isoformat(),
        "scored_rows": int(len(predictions)),
        "decision_time": "Cutoff day's demand is known. The model forecasts the next seven days.",
        "history": {
            "lag_1": "demand on the cutoff date",
            "lag_7_14_28": "demand on the target date minus 7, 14 and 28 days; these dates are on or before the cutoff",
            "rolling": "mean and sample standard deviation of 7, 14 and 28 days ending on the cutoff, full windows only",
        },
        "calendar": {
            "is_weekend": "Saturday or Sunday of the target date",
            "is_2_4_sat": "second or fourth Saturday of the target month",
            "is_month_end": "last three calendar days of the target month",
            "is_payday": "dim_calendar flag; rule legacy-payday-v1 is the 1st, the 15th, and the last weekday",
            "is_regional_holiday": "bank holiday for the ATM's RBI region",
            "pre_holiday": "day before a regional bank holiday, when that day is not itself a holiday",
            "horizon": "1 to 7, so one history vector is not asked to predict seven dates unlabeled",
        },
        "weather": (
            "Excluded. weather_city_daily.csv is the Open-Meteo ERA5 archive retrieved on "
            "2026-10-21, the weather that occurred. No forecast issued on each cutoff was stored."
        ),
        "atm_profile": (
            "terminal_id and loc_type are categorical. Each of the eight ATMs has its own "
            "location type, so loc_type is an ATM label. Do not read a location-type effect "
            "as a result for a larger network."
        ),
        "loc_type_per_atm_max": int(loc_pairs.max()),
        "seasonal_benchmark": (
            "mu_i x s_dom from run_simulated._fit_factors, refit on dates through the cutoff. "
            "P50/P80/P95 multiply that profile by the ATM's training quantiles of actual/profile. "
            "Each training day is used once. seasonal_mean is the profile itself and is the "
            "benchmark compared on WAPE and bias."
        ),
        "cycle_quantile_note": (
            "Cycle p80 and p95 coverage compare the week's demand with the sum of the seven "
            "daily quantile forecasts. Report that sum on its own; it is a different object "
            "from the quantile of the seven-day total."
        ),
        "horizon_weekday": (
            "Origins advance by 7 days from Sunday 2025-08-31, so each horizon is one weekday "
            "in every fold: horizon 1 is Monday and horizon 7 is Sunday. The pooled score "
            "still contains an equal number of each weekday."
        ),
        "quantile_monotone_repair": (
            "Separate LightGBM quantile fits can cross. After clipping at zero, P80 is raised "
            "to P50 and P95 to P80 when a fit crosses. P50 is unchanged. cross_rate is the "
            "share of test rows that crossed before that repair."
        ),
        "mean_cross_rate": round(float(meta["cross_rate"].mean()), 6),
        "model": {
            "library": "lightgbm",
            "version": lightgbm.__version__,
            "libomp": libomp_used,
            "objective": "quantile",
            "alphas": [0.5, 0.8, 0.95],
            "n_estimators": N_ESTIMATORS,
            "learning_rate": LEARNING_RATE,
            "num_leaves": NUM_LEAVES,
            "min_child_samples": MIN_CHILD_SAMPLES,
            "subsample": 1.0,
            "early_stopping": False,
            "random_state": SEED,
        },
        "replenishment": {
            "cycle_days": HORIZON,
            "cer70": "load = sum of the central daily forecast / 0.70. Central forecast is seasonal_mean or lightgbm P50.",
            "sum_daily_p80": "load = sum of the seven daily P80 forecasts",
            "sum_daily_p95": "load = sum of the seven daily P95 forecasts",
            "cer": "min(actual, load) / load. Dispensed is capped at the planned load. An emergency visit is counted and does not refill this CER.",
            "stockout": "actual week demand exceeds the planned load",
            "visits": "one planned visit per cycle; one extra visit when the cycle stocks out. No rupee price per visit.",
            "idle_interest_sketch": "sum of excess x 0.08 x 7/365. Holds the whole excess for the week. Not a float-day calculation.",
            "cassette_cap": None,
        },
        "bias_sign": "(forecast - actual) / actual. Negative means the forecast is low.",
        "frozen_baseline_unchanged": True,
        "switch_log_used": False,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "importance_cutoff": pd.Timestamp(dates[cutoffs[-1]]).date().isoformat(),
    }
    (out_dir / "rolling_origin_notes.json").write_text(json.dumps(notes, indent=2))
    print(f"DONE rolling-origin rows={len(predictions)} elapsed={notes['elapsed_seconds']}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rolling-origin Maharashtra cash-demand backtest")
    parser.add_argument("--limit", type=int, default=0, help="Score only the first N origins.")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.limit and args.out.resolve() == OUT.resolve():
        raise SystemExit("--limit refuses to write the official output directory")
    run(args.limit, args.out)


if __name__ == "__main__":
    main()
