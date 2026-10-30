from __future__ import annotations

import json
import math

import pandas as pd


def build_recommendation(
    forecast: pd.DataFrame,
    inventory_snapshots: pd.DataFrame,
    cassettes: pd.DataFrame,
    risk_quantile: str = "p80_amount",
    protection_horizon_days: int = 2,
    safety_buffer_pct: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if risk_quantile not in {"p50_amount", "p80_amount", "p95_amount"}:
        raise ValueError("risk_quantile must be p50_amount, p80_amount, or p95_amount")

    if protection_horizon_days < 1:
        raise ValueError("protection_horizon_days must be positive")
    if safety_buffer_pct < 0:
        raise ValueError("safety_buffer_pct cannot be negative")

    required_forecast_columns = {
        "forecast_date",
        "denomination",
        "p50_amount",
        "p80_amount",
        "p95_amount",
    }
    required_inventory_columns = {"snapshot_date", "denomination", "available_amount"}
    required_cassette_columns = {"denomination", "max_amount"}
    missing = required_forecast_columns - set(forecast.columns)
    if missing:
        raise ValueError(f"forecast is missing required columns: {sorted(missing)}")
    missing = required_inventory_columns - set(inventory_snapshots.columns)
    if missing:
        raise ValueError(f"inventory_snapshots is missing required columns: {sorted(missing)}")
    missing = required_cassette_columns - set(cassettes.columns)
    if missing:
        raise ValueError(f"cassettes is missing required columns: {sorted(missing)}")
    if forecast.empty or inventory_snapshots.empty or cassettes.empty:
        raise ValueError("forecast, inventory_snapshots, and cassettes must not be empty")

    def _validate_amounts(frame: pd.DataFrame, columns: list[str], name: str) -> None:
        for column in columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.isna().any() or not bool(values.map(math.isfinite).all()) or (values < 0).any():
                raise ValueError(f"{name}.{column} must contain finite non-negative amounts")

    _validate_amounts(forecast, ["p50_amount", "p80_amount", "p95_amount"], "forecast")
    _validate_amounts(inventory_snapshots, ["available_amount"], "inventory_snapshots")
    _validate_amounts(cassettes, ["max_amount"], "cassettes")
    for column in ["p50_amount", "p80_amount", "p95_amount"]:
        if (pd.to_numeric(forecast[column]) % 1 != 0).any():
            raise ValueError(f"forecast.{column} must contain whole currency amounts")
    if (pd.to_numeric(forecast["p80_amount"]) < pd.to_numeric(forecast["p50_amount"])).any():
        raise ValueError("forecast p80_amount must be at least p50_amount")
    if (pd.to_numeric(forecast["p95_amount"]) < pd.to_numeric(forecast["p80_amount"])).any():
        raise ValueError("forecast p95_amount must be at least p80_amount")

    try:
        forecast_dates = pd.to_datetime(forecast["forecast_date"]).dt.normalize()
    except (TypeError, ValueError) as exc:
        raise ValueError("forecast_date must contain valid dates") from exc
    if forecast_dates.isna().any():
        raise ValueError("forecast_date must contain valid dates")
    unique_dates = sorted(forecast_dates.dt.date.unique())
    if len(unique_dates) < protection_horizon_days:
        raise ValueError(
            "forecast does not cover the requested protection horizon "
            f"({protection_horizon_days} days; received {len(unique_dates)})"
        )
    expected_dates = pd.date_range(unique_dates[0], periods=len(unique_dates), freq="D").date
    if list(unique_dates) != list(expected_dates):
        raise ValueError("forecast protection dates must be contiguous daily dates")

    latest_date = inventory_snapshots["snapshot_date"].max()
    current_snapshot = inventory_snapshots.loc[
        inventory_snapshots["snapshot_date"] == latest_date
    ]
    if current_snapshot.empty:
        raise ValueError("inventory_snapshots has no rows at its latest snapshot_date")
    forecast_denoms = set(pd.to_numeric(forecast["denomination"], errors="coerce"))
    inventory_denoms = set(pd.to_numeric(current_snapshot["denomination"], errors="coerce"))
    cassette_denoms = set(pd.to_numeric(cassettes["denomination"], errors="coerce"))
    if any(not math.isfinite(float(value)) or float(value) <= 0 or float(value) % 1 for value in forecast_denoms):
        raise ValueError("forecast denominations must be positive whole amounts")
    if forecast_denoms != inventory_denoms or forecast_denoms != cassette_denoms:
        raise ValueError("forecast, inventory, and cassette denominations must match exactly")
    if current_snapshot["denomination"].duplicated().any() or cassettes["denomination"].duplicated().any():
        raise ValueError("latest inventory and cassette denominations must be unique")

    terminal_id = int(forecast["terminal_id"].iloc[0]) if "terminal_id" in forecast else 1001
    if "terminal_id" in forecast and forecast["terminal_id"].nunique(dropna=False) != 1:
        raise ValueError("forecast must contain exactly one terminal_id")
    for frame, name in ((inventory_snapshots, "inventory_snapshots"), (cassettes, "cassettes")):
        if "terminal_id" in frame and (
            frame["terminal_id"].nunique(dropna=False) != 1
            or int(frame["terminal_id"].iloc[0]) != terminal_id
        ):
            raise ValueError(f"{name} must belong to the same terminal as forecast")

    current = inventory_snapshots.loc[
        inventory_snapshots["snapshot_date"] == latest_date,
        ["denomination", "available_amount"],
    ]
    protected_dates = unique_dates[:protection_horizon_days]
    protected_forecast = forecast.loc[forecast_dates.dt.date.isin(protected_dates)].copy()
    demand = protected_forecast.groupby("denomination", as_index=False)[
        ["p50_amount", "p80_amount", "p95_amount"]
    ].sum()
    detail = demand.merge(current, on="denomination").merge(
        cassettes[["denomination", "max_amount"]], on="denomination"
    )
    detail.insert(0, "terminal_id", terminal_id)
    if (detail["available_amount"] > detail["max_amount"]).any():
        raise ValueError("inventory available_amount cannot exceed cassette max_amount")
    denominations = pd.to_numeric(detail["denomination"])
    if (denominations <= 0).any() or (detail["max_amount"] <= 0).any():
        raise ValueError("denominations and cassette max_amount must be positive")
    detail["base_stock_target"] = detail[risk_quantile] * (1 + safety_buffer_pct)
    detail["cash_required"] = (detail["base_stock_target"] - detail["available_amount"]).clip(lower=0)
    detail["available_capacity"] = detail["max_amount"] - detail["available_amount"]
    detail["recommended_amount"] = detail[["cash_required", "available_capacity"]].min(axis=1)
    detail["recommended_amount"] = detail.apply(
        lambda row: int(row["recommended_amount"] // row["denomination"] * row["denomination"]), axis=1
    )

    mix = {
        str(int(row.denomination)): int(row.recommended_amount)
        for row in detail.itertuples(index=False)
    }
    generated_at = pd.Timestamp.now(tz="Asia/Kolkata").isoformat()
    horizon_days = len(unique_dates)
    recommendation = pd.DataFrame(
        [
            {
                "recommendation_id": 1,
                "terminal_id": terminal_id,
                "recommend_date": str(forecast["forecast_date"].min()),
                "recommended_ts": generated_at,
                "forecast_horizon": f"{horizon_days}D",
                "protection_horizon_days": len(protected_dates),
                "risk_quantile": risk_quantile.removesuffix("_amount"),
                "safety_buffer_pct": safety_buffer_pct,
                "forecasted_demand": int(detail[risk_quantile].sum()),
                "current_inventory": int(detail["available_amount"].sum()),
                "recommended_amount": int(detail["recommended_amount"].sum()),
                "recommended_denomination_mix": json.dumps(mix, sort_keys=True),
                "recommended_by": "Capacity-aware quantile base-stock policy v1.0",
                # This policy score is not a calibrated probability.  Keep the
                # database field explicitly unavailable until one is measured.
                "confidence_score": None,
                "status": "Recommended",
            }
        ]
    )
    return recommendation, detail
