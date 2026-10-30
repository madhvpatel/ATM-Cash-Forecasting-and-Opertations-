from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .config import DEFAULT_CONFIG, POCConfig
from .data import generate_demo_data
from .model import QuantileForecaster
from .recommendation import build_recommendation
from .regional import build_regional_operations
from .reporting import create_plots
from .simulation import run_simulation


def run_poc(
    output_dir: str | Path = "outputs",
    as_of: str | pd.Timestamp = "2026-08-13",
    horizon_days: int = 7,
    risk_quantile: str = "p80_amount",
    seed: int = 42,
    recycler_enabled: bool = True,
    config: POCConfig = DEFAULT_CONFIG,
    verbose: bool = True,
) -> dict[str, object]:
    """Run mock-data generation through reporting from one Python call."""
    if horizon_days not in {1, 2, 7, 14}:
        raise ValueError("horizon_days must be one of 1, 2, 7, or 14")
    if risk_quantile not in {"p50_amount", "p80_amount", "p95_amount"}:
        raise ValueError("risk_quantile must be p50_amount, p80_amount, or p95_amount")
    config.validate()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    as_of_date = pd.Timestamp(as_of)
    if verbose:
        print("[1/5] Generating two years of synthetic ATM data...")
    tables = generate_demo_data(
        as_of_date,
        seed=seed,
        recycler_enabled=recycler_enabled,
        config=config,
    )

    if verbose:
        print("[2/5] Training and evaluating quantile models...")
    forecaster = QuantileForecaster(random_state=seed).fit(
        tables["fact_transaction"], holdout_days=config.holdout_days
    )
    planning_forecast = forecaster.forecast(as_of_date, max(horizon_days, config.protection_horizon_days))
    forecast = planning_forecast.loc[
        pd.to_datetime(planning_forecast["forecast_date"]) < as_of_date + pd.Timedelta(days=horizon_days)
    ].copy()
    forecast.insert(0, "forecast_id", range(1, len(forecast) + 1))
    forecast["forecast_generated_ts"] = pd.Timestamp.now(tz="Asia/Kolkata").isoformat()
    forecast["forecast_horizon"] = f"{horizon_days}D"
    forecast["model_version"] = "v2.0"
    forecast["model_name"] = forecaster.model_name
    tables["fact_forecast"] = forecast
    tables["fact_forecast_evaluation"] = forecaster.evaluation_metrics

    if verbose:
        print("[3/5] Building the capacity-aware cash recommendation...")
    recommendation_parts: list[pd.DataFrame] = []
    recommendation_detail_parts: list[pd.DataFrame] = []
    terminal_ids = sorted(tables["dim_terminal"]["terminal_id"].astype(int).unique())
    for terminal_id in terminal_ids:
        recommendation_part, detail_part = build_recommendation(
            planning_forecast.loc[planning_forecast["terminal_id"] == terminal_id],
            tables["fact_cash_inventory_snapshot"].loc[
                tables["fact_cash_inventory_snapshot"]["terminal_id"] == terminal_id
            ],
            tables["dim_cassette_configuration"].loc[
                tables["dim_cassette_configuration"]["terminal_id"] == terminal_id
            ],
            risk_quantile,
            protection_horizon_days=config.protection_horizon_days,
            safety_buffer_pct=config.safety_buffer_pct,
        )
        recommendation_parts.append(recommendation_part)
        recommendation_detail_parts.append(detail_part)
    recommendation = pd.concat(recommendation_parts, ignore_index=True)
    recommendation["recommendation_id"] = range(1, len(recommendation) + 1)
    recommendation_detail = pd.concat(recommendation_detail_parts, ignore_index=True)
    tables["fact_cash_recommendation"] = recommendation
    build_regional_operations(tables, recommendation_detail, as_of_date)

    if verbose:
        print("[4/5] Simulating inventory, replenishments, cash-outs, and cost...")
    simulation_parts: list[pd.DataFrame] = []
    simulated_replenishment_parts: list[pd.DataFrame] = []
    simulation_by_terminal: dict[str, dict[str, float | int]] = {}
    origins = sorted(pd.to_datetime(forecaster.backtest_predictions["forecast_date"]).unique())[-config.simulation_days:]
    protection_forecasts = forecaster.forecast_at_origins(origins, config.protection_horizon_days)
    protection_forecasts.to_csv(output_path / "simulation_origin_forecasts.csv", index=False)
    for terminal_id in terminal_ids:
        simulation_part, replenishment_part, terminal_metrics = run_simulation(
            forecaster.backtest_predictions.loc[
                forecaster.backtest_predictions["terminal_id"] == terminal_id
            ],
            tables["dim_cassette_configuration"].loc[
                tables["dim_cassette_configuration"]["terminal_id"] == terminal_id
            ],
            risk_quantile,
            replace(config, terminal_id=terminal_id),
            protection_forecasts=protection_forecasts.loc[protection_forecasts["terminal_id"] == terminal_id],
        )
        simulation_parts.append(simulation_part)
        if not replenishment_part.empty:
            simulated_replenishment_parts.append(replenishment_part)
        simulation_by_terminal[str(terminal_id)] = terminal_metrics
    simulation_detail = pd.concat(simulation_parts, ignore_index=True)
    simulated_replenishments = (
        pd.concat(simulated_replenishment_parts, ignore_index=True)
        if simulated_replenishment_parts
        else pd.DataFrame(columns=tables["fact_replenishment"].columns)
    )
    if not simulated_replenishments.empty:
        simulated_replenishments["replenishment_id"] = range(
            1, len(simulated_replenishments) + 1
        )
    actual_total = int(simulation_detail["actual_demand"].sum())
    shortage_total = int(simulation_detail["shortage_amount"].sum())
    simulation_metrics: dict[str, float | int] = {
        "simulation_days": config.simulation_days,
        "simulation_start": str(simulation_detail["simulation_date"].min()),
        "simulation_end": str(simulation_detail["simulation_date"].max()),
        "cashout_events": int((simulation_detail["shortage_amount"] > 0).sum()),
        "cashout_days": int(
            simulation_detail.loc[
                simulation_detail["shortage_amount"] > 0, "simulation_date"
            ].nunique()
        ),
        "cashout_rate": round(shortage_total / actual_total, 6) if actual_total else 0.0,
        "service_level": round(1 - shortage_total / actual_total, 6) if actual_total else 1.0,
        "average_cash_held": round(
            float(
                simulation_detail.groupby(["simulation_date", "terminal_id"])[
                    "closing_inventory"
                ].sum().groupby("simulation_date").sum().mean()
            ),
            2,
        ),
        "total_actual_demand": actual_total,
        "total_shortage": shortage_total,
        "total_loaded": int(simulation_detail["recommended_load"].sum()),
        "cit_trips": len(simulated_replenishments),
        "holding_cost": round(float(simulation_detail["holding_cost"].sum()), 2),
        "transport_cost": round(float(simulation_detail["transport_cost"].sum()), 2),
        "shortage_cost": round(float(simulation_detail["shortage_cost"].sum()), 2),
        "cashout_event_cost": round(
            float(simulation_detail["cashout_event_cost"].sum()), 2
        ),
        "total_cost": round(float(simulation_detail["total_cost"].sum()), 2),
    }
    tables["fact_simulation_daily"] = simulation_detail
    if not simulated_replenishments.empty:
        simulated_replenishments["replenishment_id"] += len(tables["fact_replenishment"])
        tables["fact_replenishment"] = pd.concat(
            [tables["fact_replenishment"], simulated_replenishments], ignore_index=True
        )

    if verbose:
        print("[5/5] Writing tables, metrics, and plots...")
    database_path = output_path / "atm_forecast_poc.db"
    with sqlite3.connect(database_path) as connection:
        for name, frame in tables.items():
            frame.to_sql(name, connection, if_exists="replace", index=False)

    postgres_csv_path = output_path / "postgres_csv"
    postgres_csv_path.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(postgres_csv_path / f"{name}.csv", index=False)

    forecast.to_csv(output_path / "forecast_by_day_and_denomination.csv", index=False)
    recommendation.to_csv(output_path / "cash_recommendation.csv", index=False)
    recommendation_detail.to_csv(output_path / "cash_recommendation_detail.csv", index=False)
    forecaster.backtest_predictions.to_csv(output_path / "holdout_predictions.csv", index=False)
    forecaster.evaluation_metrics.to_csv(output_path / "forecast_evaluation.csv", index=False)
    simulation_detail.to_csv(output_path / "simulation_daily.csv", index=False)
    simulated_replenishments.to_csv(output_path / "simulation_replenishments.csv", index=False)
    tables["fact_vault_allocation"].to_csv(output_path / "vault_allocations.csv", index=False)
    tables["fact_route_plan"].to_csv(output_path / "route_plan.csv", index=False)
    tables["fact_route_stop"].to_csv(output_path / "route_stops.csv", index=False)
    for table_name, filename in {
        "fact_transaction": "network_daily_transactions.csv",
        "fact_crm_transaction_event": "crm_transaction_events.csv",
        "fact_cash_inventory_snapshot": "network_inventory_snapshots.csv",
        "fact_cash_evacuation": "crm_cash_evacuations.csv",
        "fact_crm_operational_event": "crm_operational_events.csv",
    }.items():
        if table_name in tables:
            tables[table_name].to_csv(output_path / filename, index=False)
    # Preserve CRM-specific extracts for consumers that only understand recycler data.
    tables["fact_transaction"].loc[
        tables["fact_transaction"]["terminal_id"] == config.terminal_id
    ].to_csv(output_path / "crm_daily_transactions.csv", index=False)
    tables["fact_cash_inventory_snapshot"].loc[
        tables["fact_cash_inventory_snapshot"]["terminal_id"] == config.terminal_id
    ].to_csv(output_path / "crm_inventory_snapshots.csv", index=False)
    plot_paths = create_plots(
        forecaster.backtest_predictions,
        simulation_detail,
        simulation_metrics,
        output_path,
    )

    overall = forecaster.evaluation_metrics.loc[
        (forecaster.evaluation_metrics["terminal_id"] == "ALL")
        & (forecaster.evaluation_metrics["denomination"] == "ALL")
    ].iloc[0]
    evaluation_summary = {
        "observations": int(overall["observations"]),
        "mae": float(overall["mae"]),
        "wape": float(overall["wape"]),
        "p50_coverage": float(overall["p50_coverage"]),
        "p80_coverage": float(overall["p80_coverage"]),
        "p95_coverage": float(overall["p95_coverage"]),
    }
    network_mix = (
        recommendation_detail.groupby("denomination")["recommended_amount"].sum().astype(int)
    )
    terminal_summaries = []
    for terminal_row in tables["dim_terminal"].sort_values("terminal_id").itertuples(index=False):
        rec = recommendation.loc[
            recommendation["terminal_id"] == int(terminal_row.terminal_id)
        ].iloc[0]
        terminal_summaries.append(
            {
                "terminal_id": int(terminal_row.terminal_id),
                "terminal_code": str(terminal_row.terminal_code),
                "terminal_type": str(terminal_row.terminal_type),
                "is_crm": bool(terminal_row.is_crm),
                "forecasted_demand": int(rec["forecasted_demand"]),
                "current_inventory": int(rec["current_inventory"]),
                "recommended_order": int(rec["recommended_amount"]),
            }
        )
    summary = {
        "forecast_contract": {
            "schema_version": 1,
            "provider": "local_quantile_gbm",
            "model_version": "v2.0",
            "target": "fulfilled withdrawals minus usable recycled deposits; withdrawal-only terminals use fulfilled withdrawals",
            "grain": "ATM × day × denomination",
            "currency": "INR",
            "forecast_origin": str(as_of_date.date()),
            "horizon_days": horizon_days,
            "planning_horizon_days": max(horizon_days, config.protection_horizon_days),
            "calibration_scope": "rolling one-day-ahead marginal denomination forecasts on synthetic data",
            "operationally_promoted": False,
        },
        "simulation_contract": {
            "forecast_origins": "decision morning; actuals strictly before origin",
            "protection_forecasts": "recursive from a single origin",
            "delivery_timing": "all loads precede daily withdrawals; not an intraday service model",
        },
        "terminal": str(tables["dim_terminal"].iloc[0]["terminal_code"]),
        "terminal_count": len(terminal_summaries),
        "terminals": terminal_summaries,
        "as_of": str(as_of_date.date()),
        "recycler_enabled": recycler_enabled,
        "horizon_days": horizon_days,
        "protection_horizon_days": config.protection_horizon_days,
        "risk_quantile": risk_quantile.removesuffix("_amount"),
        "model_name": forecaster.model_name,
        "forecasted_demand": int(recommendation["forecasted_demand"].sum()),
        "current_inventory": int(recommendation["current_inventory"].sum()),
        "recommended_order": int(recommendation["recommended_amount"].sum()),
        "denomination_mix": {
            str(int(denomination)): int(amount)
            for denomination, amount in network_mix.items()
        },
        "holdout_evaluation": evaluation_summary,
        "simulation": simulation_metrics,
        "simulation_by_terminal": simulation_by_terminal,
        "cra": {
            "cra_id": int(tables["dim_cra"].iloc[0]["cra_id"]),
            "cra_code": str(tables["dim_cra"].iloc[0]["cra_code"]),
            "cra_name": str(tables["dim_cra"].iloc[0]["cra_name"]),
            "region_name": str(tables["dim_region"].iloc[0]["region_name"]),
            "vault_code": str(tables["dim_vault"].iloc[0]["vault_code"]),
            "vehicle_code": str(tables["dim_vehicle"].iloc[0]["vehicle_code"]),
            "vault_available": int(tables["fact_vault_inventory"]["available_amount"].sum()),
            "vault_reserved": int(tables["fact_vault_inventory"]["reserved_amount"].sum()),
            "route_code": str(tables["fact_route_plan"].iloc[0]["route_code"]),
            "route_stops": int(tables["fact_route_plan"].iloc[0]["stop_count"]),
            "route_distance_km": float(
                tables["fact_route_plan"].iloc[0]["planned_distance_km"]
            ),
            "route_load": int(tables["fact_route_plan"].iloc[0]["planned_load_amount"]),
            "route_status": str(tables["fact_route_plan"].iloc[0]["status"]),
        },
        "database": str(database_path.resolve()),
        "plots": [str(path.resolve()) for path in plot_paths],
    }
    (output_path / "simulation_summary.json").write_text(
        json.dumps(simulation_metrics, indent=2) + "\n", encoding="utf-8"
    )
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
