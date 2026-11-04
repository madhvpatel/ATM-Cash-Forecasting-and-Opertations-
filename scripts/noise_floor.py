"""Estimate the irreducible forecast error (noise floor) of the 100-ATM data.

The generator is not in this repo, so the floor is measured from the data with a variogram.
On plain days (no holiday within 3 days, no payday, month end or 2nd/4th Saturday), take log
demand minus the ATM's weekday mean, then

    gamma(g) = 0.5 * mean((x[t+g] - x[t])^2)

Smooth structure (level, day-of-month cycle, season) adds ~nothing as g -> 0, so the intercept
of gamma at g = 1..3 extrapolated to 0 is the variance of pure day-to-day noise. No model,
however good, can predict that part. Done per ATM, because noise size differs by ATM.

A first version compared each day with the same weekday a week either side; that counted the
day-of-month cycle as noise and put the "floor" above the model, so it was dropped.

Run: python3 scripts/noise_floor.py   (needs outputs/scorecard_100/ from run_scorecard_100.py)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from replenishment_engine import demand_model  # noqa: E402

SCORECARD = ROOT / "outputs" / "scorecard_100"
MODEL = demand_model.MODEL_NAME
LEAD_IN_DAYS = 3
NUGGET_GAPS = (1, 2, 3)
VARIOGRAM_GAPS = (1, 2, 3, 7, 14, 21, 28, 91, 182, 364)
DRAWS = 200_000
SEED = 7


def wape(actual, pred) -> float:
    return float(np.abs(actual - pred).sum() / actual.sum())


def plain_mask(panel: pd.DataFrame) -> pd.Series:
    month_end = panel["date"].dt.days_in_month - panel["dom"] <= 2
    special = (
        (panel["days_to_holiday"] <= LEAD_IN_DAYS) | (panel["is_payday"] == 1)
        | month_end | (panel["is_2_4_sat"] == 1)
    )
    return ~special


def residual_matrix(panel: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Log demand minus ATM x weekday mean on plain days, NaN elsewhere. Rows = sorted terminals."""
    frame = panel.assign(log_d=np.log(panel["demand"].clip(lower=1)))
    plain = frame[frame["plain"]]
    weekday_mean = plain.groupby(["terminal_id", "dow"])["log_d"].mean().rename("weekday_mean")
    frame = frame.join(weekday_mean, on=["terminal_id", "dow"])
    frame["res"] = np.where(frame["plain"], frame["log_d"] - frame["weekday_mean"], np.nan)
    wide = frame.pivot(index="terminal_id", columns="date", values="res").sort_index()
    return wide.index.to_numpy(), wide.to_numpy()


def semivariance(matrix: np.ndarray, gap: int) -> np.ndarray:
    """Per-row semivariance at a gap."""
    diff = matrix[:, gap:] - matrix[:, :-gap]
    return 0.5 * np.nanmean(diff ** 2, axis=1)


def nugget(matrix: np.ndarray) -> np.ndarray:
    """Per-row noise variance: linear fit of gamma(1..3), evaluated at gap 0."""
    gaps = np.array(NUGGET_GAPS, dtype=float)
    gamma = np.column_stack([semivariance(matrix, g) for g in NUGGET_GAPS])
    slope, intercept = np.polyfit(gaps, gamma.T, 1)
    return np.clip(intercept, 0.0, None)


def noise_draws(sigma: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Multiplicative noise draws with mean 1 (lognormal)."""
    return np.exp(rng.normal(-0.5 * sigma ** 2, sigma, size=n))


def oracle_wape(actual_mean: np.ndarray, sigma: np.ndarray, rng: np.random.Generator, days: int) -> float:
    """WAPE of a forecast that knows the true mean of each item exactly, with the best constant scaling.

    actual_mean: expected demand per item (a day, or a week's daily means summed over `days`).
    """
    abs_err, total = 0.0, 0.0
    for mean, s in zip(actual_mean, sigma):
        noise = noise_draws(s, DRAWS * days, rng).reshape(DRAWS, days).mean(axis=1)
        point = np.median(noise)  # WAPE-optimal point forecast is the median
        abs_err += mean * np.abs(noise - point).mean()
        total += mean
    return abs_err / total


def main() -> None:
    rng = np.random.default_rng(SEED)
    panel, _ = demand_model.load_panel()
    panel = panel.sort_values(["terminal_id", "date"]).reset_index(drop=True)
    panel["plain"] = plain_mask(panel)
    terminals, matrix = residual_matrix(panel)
    noise_var = nugget(matrix)
    sigma = pd.Series(np.sqrt(noise_var), index=terminals)

    daily_preds = pd.read_csv(SCORECARD / "predictions_daily.csv", parse_dates=["origin_date", "target_date"])
    weekly = pd.read_csv(SCORECARD / "predictions_weekly.csv", parse_dates=["origin_date"])

    # Daily: model vs floor on the same plain days, weighted by each ATM's demand on those days.
    plain_keys = panel.loc[panel["plain"], ["terminal_id", "date"]].rename(columns={"date": "target_date"})
    scored = daily_preds.merge(plain_keys, on=["terminal_id", "target_date"])
    by_atm = scored.groupby("terminal_id")["demand"].sum()
    floor_daily = oracle_wape(by_atm.to_numpy(), sigma[by_atm.index].to_numpy(), rng, days=1)

    # Weekly: scorecard weeks with no holiday in or within 3 days of the week.
    near_holiday = panel.loc[panel["days_to_holiday"] <= LEAD_IN_DAYS, ["terminal_id", "date"]]
    weekly_days = daily_preds.merge(
        near_holiday.rename(columns={"date": "target_date"}).assign(near=1), on=["terminal_id", "target_date"], how="left"
    )
    calm_weeks = weekly_days.groupby(["terminal_id", "origin_date"])["near"].count().eq(0)
    calm_weeks = calm_weeks[calm_weeks].index
    calm = weekly.set_index(["terminal_id", "origin_date"]).loc[calm_weeks].reset_index()
    week_demand = calm.groupby("terminal_id")["demand"].sum()
    # ponytail: assumes the ATM's weekday shape is flat when summing noise over 7 days;
    # a peaky week averages out a little less, so the true weekly floor is slightly higher.
    floor_weekly = oracle_wape(week_demand.to_numpy(), sigma[week_demand.index].to_numpy(), rng, days=7)

    # Model daily errors on calm weeks: are they correlated within the week (a level error)?
    calm_daily = daily_preds.merge(calm[["terminal_id", "origin_date"]], on=["terminal_id", "origin_date"])
    calm_daily = calm_daily.assign(err=np.log(calm_daily[MODEL].clip(lower=1) / calm_daily["demand"].clip(lower=1)))
    week_mean_err = calm_daily.groupby(["terminal_id", "origin_date"])["err"].transform("mean")
    level_share = float(week_mean_err.var() / calm_daily["err"].var())

    # Buffer, on the scorecard's basis: sum(upper bound) / sum(actual) - 1.
    # Floor: a perfect forecaster still has to cover the P99 of each ATM's weekly noise.
    q99_noise = np.array([
        np.quantile(noise_draws(s, DRAWS * 7, rng).reshape(DRAWS, 7).mean(axis=1), 0.99)
        for s in sigma[week_demand.index]
    ])
    floor_extra = float((week_demand.to_numpy() * q99_noise).sum() / week_demand.sum() - 1)
    scored_calm = calm[calm[f"{MODEL}_q99"].notna()]
    model_extra_calm = float((scored_calm[MODEL] * scored_calm[f"{MODEL}_q99"]).sum() / scored_calm["demand"].sum() - 1)
    ratio = weekly["demand"] / weekly[MODEL]
    is_calm = weekly.set_index(["terminal_id", "origin_date"]).index.isin(calm_weeks)

    result = {
        "noise_sigma_log": {
            "median": float(sigma.median()),
            "p10": float(sigma.quantile(0.1)),
            "p90": float(sigma.quantile(0.9)),
        },
        "variogram_pooled": {g: float(np.nanmean(semivariance(matrix, g))) for g in VARIOGRAM_GAPS},
        "total_residual_var": float(np.nanvar(matrix)),
        "daily": {
            "plain_days_scored": int(len(scored)),
            "model_wape": wape(scored["demand"], scored[MODEL]),
            "floor_wape": floor_daily,
        },
        "weekly": {
            "calm_weeks": int(len(calm)),
            "model_wape": wape(calm["demand"], calm[MODEL]),
            "floor_wape": floor_weekly,
            "model_wape_all_weeks": wape(weekly["demand"], weekly[MODEL]),
        },
        # 1/7 = 0.143 if the model's daily errors were independent; above that is a shared level error.
        "model_error_level_share": level_share,
        "buffer_p99": {
            "model_extra_cash_calm_weeks": model_extra_calm,
            "floor_extra_cash_calm_weeks": floor_extra,
            # Whole-backtest (in-sample) P99 of actual/forecast, split by week type. Diagnostic only.
            "ratio_q99_calm_weeks": float(np.nanquantile(ratio[is_calm], 0.99)),
            "ratio_q99_holiday_weeks": float(np.nanquantile(ratio[~is_calm], 0.99)),
            "ratio_q99_all_weeks": float(np.nanquantile(ratio, 0.99)),
            "holiday_weeks": int((~is_calm).sum()),
        },
    }
    (SCORECARD / "noise_floor.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    assert result["daily"]["floor_wape"] <= result["daily"]["model_wape"], "floor above model: estimator broken"
    assert result["weekly"]["floor_wape"] <= result["weekly"]["model_wape"], "floor above model: estimator broken"


if __name__ == "__main__":
    main()
