from __future__ import annotations

import json

import pandas as pd

from .config import DEFAULT_CONFIG, POCConfig


def run_simulation(
    backtest_predictions: pd.DataFrame,
    cassettes: pd.DataFrame,
    risk_quantile: str = "p80_amount",
    config: POCConfig = DEFAULT_CONFIG,
    protection_forecasts: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    """Run a daily order-up-to simulation over the end of the holdout window.

    Recommendations are calculated before demand is served each day. Lead time
    is represented in the protection horizon, so each visit loads enough for the
    review period plus lead time while still respecting cassette capacity.
    """
    if risk_quantile not in {"p50_amount", "p80_amount", "p95_amount"}:
        raise ValueError("risk_quantile must be p50_amount, p80_amount, or p95_amount")
    if protection_forecasts is None and config.protection_horizon_days > 1:
        raise ValueError("Multi-day simulation requires origin-specific protection_forecasts")

    predictions = backtest_predictions.copy()
    predictions["forecast_date"] = pd.to_datetime(predictions["forecast_date"])
    if protection_forecasts is not None:
        protection_forecasts = protection_forecasts.copy()
        protection_forecasts["forecast_origin"] = pd.to_datetime(protection_forecasts["forecast_origin"])
        protection_forecasts["forecast_date"] = pd.to_datetime(protection_forecasts["forecast_date"])
    simulation_dates = sorted(predictions["forecast_date"].unique())[-config.simulation_days :]
    simulation_start = simulation_dates[0]
    simulation_rows: list[dict[str, object]] = []
    replenishment_rows: list[dict[str, object]] = []

    cassette_lookup = cassettes.set_index("denomination").to_dict("index")
    balances = {
        int(denomination): int(row["max_amount"] * config.initial_inventory_ratio)
        // int(denomination)
        * int(denomination)
        for denomination, row in cassette_lookup.items()
    }

    for day in simulation_dates:
        day = pd.Timestamp(day)
        daily = predictions.loc[predictions["forecast_date"] == day]
        window_end = day + pd.Timedelta(days=config.protection_horizon_days - 1)
        protection = daily if protection_forecasts is None else protection_forecasts.loc[
            (protection_forecasts["forecast_origin"] == day)
            & protection_forecasts["forecast_date"].between(day, window_end)
        ]
        expected_dates = set(pd.date_range(day, window_end))
        for denomination in cassette_lookup:
            covered = protection.loc[protection["denomination"] == denomination, "forecast_date"]
            if len(covered) != len(expected_dates) or set(covered) != expected_dates:
                raise ValueError(f"Incomplete origin-specific protection for {day.date()}, {denomination}")
        day_rows: list[dict[str, object]] = []
        loaded_mix: dict[str, int] = {}

        for record in daily.sort_values("denomination").itertuples(index=False):
            denomination = int(record.denomination)
            capacity = int(cassette_lookup[denomination]["max_amount"])
            opening_inventory = balances[denomination]
            protected_demand = int(
                protection.loc[protection["denomination"] == denomination, risk_quantile].sum()
            )
            base_stock_target = min(
                capacity,
                int(protected_demand * (1 + config.safety_buffer_pct))
                // denomination
                * denomination,
            )
            recommended_load = max(0, base_stock_target - opening_inventory)
            recommended_load = min(recommended_load, capacity - opening_inventory)
            recommended_load = recommended_load // denomination * denomination
            inventory_after_load = opening_inventory + recommended_load
            actual_demand = max(0, int(record.actual_amount))
            dispensed_amount = min(actual_demand, inventory_after_load)
            shortage_amount = actual_demand - dispensed_amount
            closing_inventory = inventory_after_load - dispensed_amount
            balances[denomination] = closing_inventory

            if recommended_load:
                loaded_mix[str(denomination)] = recommended_load
            holding_cost = closing_inventory * config.holding_cost_rate_per_day
            shortage_cost = shortage_amount * config.shortage_penalty_rate
            cashout_cost = config.cashout_event_penalty if shortage_amount else 0.0
            day_rows.append(
                {
                    "simulation_date": day.date(),
                    "terminal_id": config.terminal_id,
                    "denomination": denomination,
                    "opening_inventory": opening_inventory,
                    "forecast_amount": int(getattr(record, risk_quantile)),
                    "protected_demand": protected_demand,
                    "base_stock_target": base_stock_target,
                    "recommended_load": recommended_load,
                    "inventory_after_load": inventory_after_load,
                    "actual_demand": actual_demand,
                    "dispensed_amount": dispensed_amount,
                    "shortage_amount": shortage_amount,
                    "closing_inventory": closing_inventory,
                    "holding_cost": round(holding_cost, 2),
                    "shortage_cost": round(shortage_cost, 2),
                    "cashout_event_cost": round(cashout_cost, 2),
                    "transport_cost": 0.0,
                }
            )

        if loaded_mix:
            day_rows[0]["transport_cost"] = config.transport_cost_per_trip
            replenishment_rows.append(
                {
                    "replenishment_id": len(replenishment_rows) + 1,
                    "terminal_id": config.terminal_id,
                    "visit_date": day.date(),
                    "visit_type": "Simulated Load",
                    "loaded_amount": sum(loaded_mix.values()),
                    "loaded_denomination_mix": json.dumps(loaded_mix, sort_keys=True),
                    "performed_by": "Policy Simulation",
                    "status": "Simulated",
                }
            )
        simulation_rows.extend(day_rows)

    detail = pd.DataFrame(simulation_rows)
    detail["total_cost"] = detail[
        ["holding_cost", "shortage_cost", "cashout_event_cost", "transport_cost"]
    ].sum(axis=1)
    replenishments = pd.DataFrame(replenishment_rows)
    actual_total = int(detail["actual_demand"].sum())
    shortage_total = int(detail["shortage_amount"].sum())
    metrics: dict[str, float | int] = {
        "simulation_days": len(simulation_dates),
        "simulation_start": str(pd.Timestamp(simulation_start).date()),
        "simulation_end": str(pd.Timestamp(simulation_dates[-1]).date()),
        "cashout_events": int((detail["shortage_amount"] > 0).sum()),
        "cashout_days": int(detail.loc[detail["shortage_amount"] > 0, "simulation_date"].nunique()),
        "cashout_rate": round(shortage_total / actual_total, 6) if actual_total else 0.0,
        "service_level": round(1 - (shortage_total / actual_total), 6) if actual_total else 1.0,
        "average_cash_held": round(
            float(detail.groupby("simulation_date")["closing_inventory"].sum().mean()), 2
        ),
        "total_actual_demand": actual_total,
        "total_shortage": shortage_total,
        "total_loaded": int(detail["recommended_load"].sum()),
        "cit_trips": len(replenishments),
        "holding_cost": round(float(detail["holding_cost"].sum()), 2),
        "transport_cost": round(float(detail["transport_cost"].sum()), 2),
        "shortage_cost": round(float(detail["shortage_cost"].sum()), 2),
        "cashout_event_cost": round(float(detail["cashout_event_cost"].sum()), 2),
        "total_cost": round(float(detail["total_cost"].sum()), 2),
    }
    return detail, replenishments, metrics
