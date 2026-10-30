from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd

from .calendar import calendar_context
from .demand_artifact import GlobalDemandArtifact, audit_demand_data
from .features import FEATURE_COLUMNS, future_feature_row


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PACKAGED_WEB = Path(__file__).resolve().parent / "web"
STATIC_DIR = _PACKAGED_WEB if _PACKAGED_WEB.is_dir() else PROJECT_ROOT / "web"


FEATURE_CATALOG = [
    {
        "name": "day_of_week",
        "family": "Calendar",
        "source": "dim_calendar.calendar_date",
        "mapping": "Monday=0 through Sunday=6",
        "model_use": "Tree splits learn denomination-specific weekday demand patterns.",
    },
    {
        "name": "month",
        "family": "Calendar",
        "source": "dim_calendar.calendar_date",
        "mapping": "Calendar month as integer 1–12",
        "model_use": "Captures broad within-year seasonality through numeric tree splits.",
    },
    {
        "name": "quarter",
        "family": "Calendar",
        "source": "dim_calendar.calendar_date",
        "mapping": "Derived quarter as integer 1–4",
        "model_use": "Provides a coarser seasonal grouping than month.",
    },
    {
        "name": "is_weekend",
        "family": "Calendar flag",
        "source": "dim_calendar.calendar_date",
        "mapping": "Saturday/Sunday=1; working day=0",
        "model_use": "Allows weekend demand to follow different tree branches.",
    },
    {
        "name": "is_payday",
        "family": "Calendar flag",
        "source": "dim_calendar.calendar_date",
        "mapping": "1st, 15th or last working day=1; otherwise=0",
        "model_use": "Identifies synthetic salary-window demand effects.",
    },
    {
        "name": "is_holiday",
        "family": "Calendar flag",
        "source": "dim_calendar.calendar_date",
        "mapping": "Configured UAT public holiday=1; otherwise=0",
        "model_use": "Allows holiday demand to follow separate tree branches.",
    },
    {
        "name": "days_to_month_end",
        "family": "Calendar",
        "source": "dim_calendar.calendar_date",
        "mapping": "Last calendar day minus forecast day",
        "model_use": "Represents proximity to month-end as a numeric signal.",
    },
    {
        "name": "is_month_start",
        "family": "Calendar flag",
        "source": "dim_calendar.calendar_date",
        "mapping": "First three calendar days=1; otherwise=0",
        "model_use": "Separates month-opening behaviour from regular days.",
    },
    {
        "name": "is_month_end",
        "family": "Calendar flag",
        "source": "dim_calendar.calendar_date",
        "mapping": "Final three calendar days=1; otherwise=0",
        "model_use": "Separates month-closing behaviour from regular days.",
    },
    {
        "name": "lag_1",
        "family": "Demand history",
        "source": "fact_transaction.net_demand",
        "mapping": "Same terminal and denomination, previous day",
        "model_use": "Supplies the most recent observed demand level.",
    },
    {
        "name": "lag_7",
        "family": "Demand history",
        "source": "fact_transaction.net_demand",
        "mapping": "Same terminal and denomination, seven days earlier",
        "model_use": "Supplies same-weekday weekly recurrence.",
    },
    {
        "name": "lag_14",
        "family": "Demand history",
        "source": "fact_transaction.net_demand",
        "mapping": "Same terminal and denomination, fourteen days earlier",
        "model_use": "Adds a second weekly reference less sensitive to one unusual day.",
    },
    {
        "name": "rolling_mean_7",
        "family": "Demand history",
        "source": "fact_transaction.net_demand",
        "mapping": "Mean of the previous seven days; forecast day excluded",
        "model_use": "Represents the recent local demand baseline without target leakage.",
    },
    {
        "name": "rolling_mean_14",
        "family": "Demand history",
        "source": "fact_transaction.net_demand",
        "mapping": "Mean of the previous fourteen days; forecast day excluded",
        "model_use": "Provides a smoother two-week baseline.",
    },
    {
        "name": "rolling_std_7",
        "family": "Demand history",
        "source": "fact_transaction.net_demand",
        "mapping": "Population standard deviation of the previous seven days",
        "model_use": "Represents recent volatility used by quantile tree splits.",
    },
]


NOT_IN_MODEL = [
    "Weather, rainfall and weather warnings",
    "Availability, downtime and cash-out minutes",
    "Days since refill and last refill amount",
    "Site type, city and urban/rural attributes",
    "Transaction counts and average ticket size",
]


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _build_dry_run(
    summary: dict[str, Any],
    terminal: dict[str, Any],
    forecast: pd.DataFrame,
    recommendation: pd.DataFrame,
    action: pd.DataFrame,
    transactions: pd.DataFrame,
    snapshots: pd.DataFrame,
    holdout: pd.DataFrame,
) -> dict[str, Any]:
    terminal_id = int(terminal["terminal_id"])
    forecast = forecast.loc[forecast["terminal_id"] == terminal_id]
    recommendation = recommendation.loc[recommendation["terminal_id"] == terminal_id]
    action = action.loc[action["terminal_id"] == terminal_id]
    transactions = transactions.loc[transactions["terminal_id"] == terminal_id]
    snapshots = snapshots.loc[snapshots["terminal_id"] == terminal_id]
    holdout = holdout.loc[holdout["terminal_id"] == terminal_id]
    latest_actual_date = str(transactions["transaction_date"].max())
    actual = transactions.loc[transactions["transaction_date"] == latest_actual_date]
    latest_snapshot_date = str(snapshots["snapshot_date"].max())
    latest_snapshot = snapshots.loc[snapshots["snapshot_date"] == latest_snapshot_date]
    first_forecast_date = str(forecast["forecast_date"].min())
    first_forecast = forecast.loc[forecast["forecast_date"] == first_forecast_date]
    latest_backtest_date = str(holdout["forecast_date"].max())
    latest_backtest = holdout.loc[holdout["forecast_date"] == latest_backtest_date]
    action_row = action.iloc[0]
    safety_buffer_pct = float(action_row["safety_buffer_pct"])

    opening_inventory = int(
        (latest_snapshot["opening_note_count"] * latest_snapshot["denomination"]).sum()
    )
    external_load = int(
        (
            latest_snapshot["externally_loaded_note_count"]
            * latest_snapshot["denomination"]
        ).sum()
    )
    evacuated_recycler = int(
        (
            latest_snapshot["evacuated_recycler_note_count"]
            * latest_snapshot["denomination"]
        ).sum()
    )
    closing_inventory = int(latest_snapshot["available_amount"].sum())
    recyclable_deposit = int(actual["deposit_recyclable_amount"].sum())
    dispensed = int(actual["amount_dispensed"].sum())

    denomination_rows: list[dict[str, Any]] = []
    for inventory_row in recommendation.sort_values("denomination").itertuples(index=False):
        denomination = int(inventory_row.denomination)
        prediction = first_forecast.loc[first_forecast["denomination"] == denomination].iloc[0]
        validation = latest_backtest.loc[latest_backtest["denomination"] == denomination].iloc[0]
        denomination_history = (
            transactions.loc[transactions["denomination"] == denomination]
            .sort_values("transaction_date")["net_demand"]
            .astype(float)
            .tolist()
        )
        feature_snapshot = future_feature_row(
            pd.Timestamp(first_forecast_date), denomination_history
        )
        selected_demand = int(prediction["p80_amount"])
        selected_load = int(inventory_row.recommended_amount)
        projected_closing = max(
            0, int(inventory_row.available_amount) + selected_load - selected_demand
        )
        denomination_rows.append(
            {
                "denomination": denomination,
                "current_inventory": int(inventory_row.available_amount),
                "capacity_amount": int(inventory_row.max_amount),
                "next_day": {
                    "p50": int(prediction["p50_amount"]),
                    "p80": selected_demand,
                    "p95": int(prediction["p95_amount"]),
                },
                "protected": {
                    "p50": int(inventory_row.p50_amount),
                    "p80": int(inventory_row.p80_amount),
                    "p95": int(inventory_row.p95_amount),
                },
                "policy_base_stock_target": int(inventory_row.base_stock_target),
                "policy_recommended_load": selected_load,
                "policy_projected_closing": projected_closing,
                "backtest_actual": int(validation["actual_amount"]),
                "backtest_p80": int(validation["p80_amount"]),
                "backtest_covered_by_p80": bool(
                    validation["actual_amount"] <= validation["p80_amount"]
                ),
                "feature_snapshot": {
                    feature: round(float(feature_snapshot[feature]), 2)
                    for feature in FEATURE_COLUMNS
                },
            }
        )

    next_day_totals = {
        quantile: int(first_forecast[f"{quantile}_amount"].sum())
        for quantile in ("p50", "p80", "p95")
    }
    protected_totals = {
        quantile: int(recommendation[f"{quantile}_amount"].sum())
        for quantile in ("p50", "p80", "p95")
    }
    backtest_actual = int(latest_backtest["actual_amount"].sum())
    backtest_p80 = int(latest_backtest["p80_amount"].sum())
    forecast_context = calendar_context(pd.Timestamp(first_forecast_date))

    return {
        "terminal_id": terminal_id,
        "terminal_code": str(terminal["terminal_code"]),
        "terminal_type": str(terminal["terminal_type"]),
        "is_crm": bool(terminal["is_crm"]),
        "history_window": {
            "start": str(transactions["transaction_date"].min()),
            "end": latest_actual_date,
            "days": int(transactions["transaction_date"].nunique()),
        },
        "actual_day": {
            "date": latest_actual_date,
            "operating_case": str(actual["operating_case"].iloc[0]),
            "opening_inventory": opening_inventory,
            "external_load": external_load,
            "deposit_presented": int(actual["deposit_presented_amount"].sum()),
            "deposit_accepted": int(actual["deposit_amount"].sum()),
            "deposit_recyclable": recyclable_deposit,
            "deposit_collection": int(actual["deposit_collection_amount"].sum()),
            "deposit_rejected": int(actual["deposit_rejected_amount"].sum()),
            "dispensed": dispensed,
            "same_day_reused": int(actual["same_day_recycled_dispensed_amount"].sum()),
            "evacuated_recycler": evacuated_recycler,
            "net_demand": int(actual["net_demand"].sum()),
            "closing_inventory": closing_inventory,
            "balance_reconciles": bool(
                opening_inventory
                + external_load
                + recyclable_deposit
                - dispensed
                - evacuated_recycler
                == closing_inventory
            ),
        },
        "forecast_day": {
            "date": first_forecast_date,
            "weekday": forecast_context["weekday"],
            "calendar_signals": forecast_context["signals"],
            **next_day_totals,
        },
        "policy": {
            "engine_name": "Capacity-aware quantile base-stock policy",
            "engine_version": "v1.0",
            "engine_type": "Deterministic policy engine",
            "selected_quantile": str(action_row["risk_quantile"]),
            "protection_horizon_days": int(action_row["protection_horizon_days"]),
            "safety_buffer_pct": safety_buffer_pct,
            "protected": protected_totals,
            "current_inventory": int(action_row["current_inventory"]),
            "recommended_load": int(action_row["recommended_amount"]),
            "policy_base_stock_target": int(
                round(protected_totals[str(action_row["risk_quantile"])] * (1 + safety_buffer_pct))
            ),
        },
        "backtest": {
            "engine_name": "Rolling hold-out evaluation",
            "engine_type": "Historical validation, not a forecast model",
            "date": latest_backtest_date,
            "actual": backtest_actual,
            "p80": backtest_p80,
            "covered_by_p80": backtest_actual <= backtest_p80,
        },
        "denominations": denomination_rows,
        "model": {
            "name": str(summary["model_name"]),
            "version": str(first_forecast["model_version"].iloc[0]),
            "engine_type": "Machine-learning quantile regression",
            "model_instances": 12,
            "forecast_generated_at": str(first_forecast["forecast_generated_ts"].iloc[0]),
            "features": FEATURE_COLUMNS,
            "feature_catalog": FEATURE_CATALOG,
            "preprocessing": (
                "Raw numeric and 0/1 values are passed to the tree models without scaling. "
                "History is isolated by terminal and denomination, and only prior dates are used."
            ),
            "target_mapping": {
                "grain": "terminal_id × transaction_date × denomination",
                "source": "fact_transaction",
                "raw_fields": (
                    ["amount_dispensed", "deposit_recyclable_amount"]
                    if terminal["is_crm"]
                    else ["amount_dispensed"]
                ),
                "formula": (
                    "net_demand = amount_dispensed - deposit_recyclable_amount"
                    if terminal["is_crm"]
                    else "withdrawal_demand = amount_dispensed"
                ),
                "use": "Training label and the historical series used to derive lag/rolling features.",
            },
            "not_in_model": NOT_IN_MODEL,
            "target": (
                "Net external cash demand by denomination"
                if terminal["is_crm"]
                else "Withdrawal cash demand by denomination"
            ),
            "definition": (
                "Successful withdrawals minus accepted deposits routed to recycle cassettes"
                if terminal["is_crm"]
                else "Successful withdrawals; this terminal has no deposit offset"
            ),
            "data_source": "Terminal facts mirrored in PostgreSQL schema crm_cash",
        },
        "projection": {
            "engine_name": "Cassette cash-balance projection",
            "engine_version": "v1.0",
            "engine_type": "Deterministic scenario simulation",
            "formula": "current cash + recommended load - selected demand",
        },
    }


def _explainability(daily: pd.DataFrame) -> dict[str, Any]:
    daily = daily.copy().sort_values("forecast_date")
    daily_dates = pd.to_datetime(daily["forecast_date"])
    typical_p80 = float(daily["p80_amount"].median())
    explanation_rows: list[dict[str, Any]] = []
    for row, forecast_date in zip(daily.itertuples(index=False), daily_dates, strict=True):
        context = calendar_context(forecast_date)
        relative_pct = round((float(row.p80_amount) / typical_p80 - 1) * 100) if typical_p80 else 0
        if relative_pct >= 15:
            demand_pattern = "Above the seven-day typical demand"
        elif relative_pct <= -15:
            demand_pattern = "Below the seven-day typical demand"
        else:
            demand_pattern = "Near the seven-day typical demand"
        explanation_rows.append(
            {
                "forecast_date": forecast_date.strftime("%Y-%m-%d"),
                "weekday": context["weekday"],
                "date_label": context["date_label"],
                "p80_amount": int(row.p80_amount),
                "relative_to_typical_pct": relative_pct,
                "demand_pattern": demand_pattern,
                "days_to_month_end": context["days_to_month_end"],
                "calendar_signals": context["signals"],
            }
        )
    return {
        "typical_daily_p80": int(typical_p80),
        "days": explanation_rows,
        "input_families": [
            "Day of week and weekend",
            "Payday window",
            "Holiday",
            "Month and month boundary",
            "Recent demand lags and rolling behaviour",
        ],
        "disclaimer": (
            "Calendar signals are model inputs and operational context. The comparison shows how "
            "each forecast differs from this seven-day window; it is not a causal contribution score."
        ),
    }


def _probable_demand_reasons(
    daily_forecast: pd.DataFrame, historical: pd.DataFrame
) -> dict[str, Any]:
    """Create auditable, plain-language forecast context for the demand demo.

    These are deliberately evidence statements about inputs available at forecast
    time, not post-hoc claims that a feature *caused* a prediction.  That keeps
    the operator explanation useful without overstating what a tree model can
    prove from this synthetic data.
    """
    history = historical.sort_values("date").copy()
    demand = history["requested_amount"].astype(float)
    recent = demand.tail(7)
    reference = demand.tail(90)
    recent_mean = float(recent.mean()) if not recent.empty else 0.0
    typical_mean = float(reference.mean()) if not reference.empty else 0.0
    recent_cv = float(recent.std(ddof=0) / recent_mean) if recent_mean else 0.0
    change_pct = round((recent_mean / typical_mean - 1) * 100) if typical_mean else 0

    history_reason: dict[str, str]
    if change_pct >= 10:
        history_reason = {
            "kind": "recent-demand",
            "title": "Recent demand is elevated",
            "evidence": (
                f"The latest 7 days averaged ₹{recent_mean:,.0f}, "
                f"{change_pct}% above the 90-day daily average."
            ),
        }
    elif change_pct <= -10:
        history_reason = {
            "kind": "recent-demand",
            "title": "Recent demand is lower than usual",
            "evidence": (
                f"The latest 7 days averaged ₹{recent_mean:,.0f}, "
                f"{abs(change_pct)}% below the 90-day daily average."
            ),
        }
    else:
        history_reason = {
            "kind": "recent-demand",
            "title": "Recent demand is close to its usual level",
            "evidence": (
                f"The latest 7 days averaged ₹{recent_mean:,.0f}, "
                f"within {abs(change_pct)}% of the 90-day daily average."
            ),
        }

    days: list[dict[str, Any]] = []
    for row in daily_forecast.itertuples(index=False):
        forecast_date = pd.Timestamp(row.forecast_date)
        context = calendar_context(forecast_date)
        reasons = [history_reason.copy()]
        for signal in context["signals"]:
            if signal["type"] != "regular":
                reasons.append(
                    {
                        "kind": signal["type"],
                        "title": signal["label"],
                        "evidence": signal["meaning"],
                    }
                )
        planning_spread = int(row.p80_amount) - int(row.p50_amount)
        if recent_cv >= 0.25 or planning_spread >= max(int(row.p50_amount) * 0.20, 1):
            reasons.append(
                {
                    "kind": "uncertainty",
                    "title": "A wider planning allowance is prudent",
                    "evidence": (
                        f"P80 is ₹{planning_spread:,.0f} above P50"
                        + (" and recent demand has been variable." if recent_cv >= 0.25 else ".")
                    ),
                }
            )
        days.append(
            {
                "forecast_date": forecast_date.strftime("%Y-%m-%d"),
                "date_label": context["date_label"],
                "p80_amount": int(row.p80_amount),
                "reasons": reasons,
            }
        )
    return {
        "history_reference": {
            "recent_7_day_average": int(round(recent_mean)),
            "trailing_90_day_average": int(round(typical_mean)),
            "recent_vs_trailing_pct": change_pct,
        },
        "days": days,
        "disclaimer": (
            "Probable reasons describe observed demand history and forecast-time calendar inputs. "
            "They are not causal proof or a feature-attribution score."
        ),
    }


def build_dashboard_payload(output_dir: str | Path = "outputs") -> dict[str, Any]:
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    required = {
        "summary": output_path / "summary.json",
        "forecast": output_path / "forecast_by_day_and_denomination.csv",
        "recommendation": output_path / "cash_recommendation_detail.csv",
        "action": output_path / "cash_recommendation.csv",
        "evaluation": output_path / "forecast_evaluation.csv",
        "transactions": output_path / "network_daily_transactions.csv",
        "snapshots": output_path / "network_inventory_snapshots.csv",
        "holdout": output_path / "holdout_predictions.csv",
        "cra": output_path / "postgres_csv" / "dim_cra.csv",
        "region": output_path / "postgres_csv" / "dim_region.csv",
        "locations": output_path / "postgres_csv" / "dim_location.csv",
        "vault": output_path / "postgres_csv" / "dim_vault.csv",
        "vehicle": output_path / "postgres_csv" / "dim_vehicle.csv",
        "terminal_master": output_path / "postgres_csv" / "dim_terminal.csv",
        "vault_inventory": output_path / "postgres_csv" / "fact_vault_inventory.csv",
        "vault_allocations": output_path / "postgres_csv" / "fact_vault_allocation.csv",
        "route_plan": output_path / "postgres_csv" / "fact_route_plan.csv",
        "route_stops": output_path / "postgres_csv" / "fact_route_stop.csv",
    }
    missing = [path.name for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing dashboard inputs: {', '.join(missing)}. Run `.venv/bin/python main.py` first."
        )

    summary = json.loads(required["summary"].read_text(encoding="utf-8"))
    forecast = pd.read_csv(required["forecast"])
    recommendation = pd.read_csv(required["recommendation"])
    action = pd.read_csv(required["action"])
    evaluation = pd.read_csv(
        required["evaluation"], dtype={"terminal_id": str, "denomination": str}
    )
    transactions = pd.read_csv(required["transactions"])
    snapshots = pd.read_csv(required["snapshots"])
    holdout = pd.read_csv(required["holdout"])
    cra_master = pd.read_csv(required["cra"])
    region_master = pd.read_csv(required["region"])
    locations = pd.read_csv(required["locations"])
    vault_master = pd.read_csv(required["vault"])
    vehicle_master = pd.read_csv(required["vehicle"])
    terminal_master = pd.read_csv(required["terminal_master"])
    vault_inventory = pd.read_csv(required["vault_inventory"])
    vault_allocations = pd.read_csv(required["vault_allocations"])
    route_plan = pd.read_csv(required["route_plan"])
    route_stops = pd.read_csv(required["route_stops"])

    cra_row = cra_master.iloc[0]
    region_row = region_master.iloc[0]
    vault_row = vault_master.iloc[0]
    vehicle_row = vehicle_master.iloc[0]
    route_row = route_plan.iloc[0]
    vault_location = locations.loc[locations["location_id"] == int(vault_row["location_id"])].iloc[0]
    route_stop_payloads: list[dict[str, Any]] = []
    for stop in route_stops.sort_values("stop_sequence").itertuples(index=False):
        master = terminal_master.loc[terminal_master["terminal_id"] == int(stop.terminal_id)].iloc[0]
        location = locations.loc[locations["location_id"] == int(master["location_id"])].iloc[0]
        route_stop_payloads.append(
            {
                "id": int(stop.route_stop_id),
                "sequence": int(stop.stop_sequence),
                "terminal_id": int(stop.terminal_id),
                "terminal_code": str(master["terminal_code"]),
                "location_name": str(location["location_name"]),
                "site_type": str(master["site_type"]),
                "latitude": float(location["latitude"]),
                "longitude": float(location["longitude"]),
                "arrival": str(stop.planned_arrival_ts),
                "departure": str(stop.planned_departure_ts),
                "travel_distance_km": float(stop.travel_distance_km),
                "service_minutes": int(stop.service_minutes),
                "planned_load_amount": int(stop.planned_load_amount),
                "denomination_mix": json.loads(str(stop.planned_denomination_mix)),
                "status": str(stop.status),
            }
        )
    route_terminal_ids = {stop["terminal_id"] for stop in route_stop_payloads}

    network_daily = (
        forecast.groupby("forecast_date", as_index=False)[["p50_amount", "p80_amount", "p95_amount"]]
        .sum()
        .sort_values("forecast_date")
    )
    network_daily["forecast_date"] = pd.to_datetime(network_daily["forecast_date"]).dt.strftime(
        "%Y-%m-%d"
    )
    current_inventory = int(summary["current_inventory"])
    capacity_total = int(recommendation["max_amount"].sum())
    seven_day_p80 = int(network_daily["p80_amount"].sum())
    average_daily_p80 = seven_day_p80 / max(len(network_daily), 1)
    cash_cover_days = round(current_inventory / average_daily_p80, 1) if average_daily_p80 else None
    overall = evaluation.loc[
        (evaluation["terminal_id"] == "ALL") & (evaluation["denomination"] == "ALL")
    ].iloc[0]
    recommended_order = int(summary["recommended_order"])
    terminal_payloads: list[dict[str, Any]] = []
    dry_runs: list[dict[str, Any]] = []

    for terminal_meta in summary["terminals"]:
        terminal_id = int(terminal_meta["terminal_id"])
        terminal_master_row = terminal_master.loc[
            terminal_master["terminal_id"] == terminal_id
        ].iloc[0]
        location_row = locations.loc[
            locations["location_id"] == int(terminal_master_row["location_id"])
        ].iloc[0]
        terminal_forecast = forecast.loc[forecast["terminal_id"] == terminal_id]
        terminal_daily = (
            terminal_forecast.groupby("forecast_date", as_index=False)[
                ["p50_amount", "p80_amount", "p95_amount"]
            ]
            .sum()
            .sort_values("forecast_date")
        )
        terminal_daily["forecast_date"] = pd.to_datetime(
            terminal_daily["forecast_date"]
        ).dt.strftime("%Y-%m-%d")
        terminal_recommendation = recommendation.loc[
            recommendation["terminal_id"] == terminal_id
        ].sort_values("denomination")
        terminal_action = action.loc[action["terminal_id"] == terminal_id].iloc[0]
        inventory_rows: list[dict[str, Any]] = []
        for row in terminal_recommendation.itertuples(index=False):
            available = int(row.available_amount)
            capacity = int(row.max_amount)
            inventory_rows.append(
                {
                    "terminal_id": terminal_id,
                    "denomination": int(row.denomination),
                    "available_amount": available,
                    "capacity_amount": capacity,
                    "utilisation": round(available / capacity, 4) if capacity else 0,
                    "protected_p80": int(row.p80_amount),
                    "recommended_amount": int(row.recommended_amount),
                }
            )
        terminal_capacity = int(terminal_recommendation["max_amount"].sum())
        terminal_inventory = int(terminal_action["current_inventory"])
        terminal_seven_day_p80 = int(terminal_daily["p80_amount"].sum())
        terminal_average_p80 = terminal_seven_day_p80 / max(len(terminal_daily), 1)
        terminal_status = (
            "Action required" if int(terminal_action["recommended_amount"]) else "Healthy"
        )
        terminal_evaluation = evaluation.loc[
            (evaluation["terminal_id"] == str(terminal_id))
            & (evaluation["denomination"] == "ALL")
        ].iloc[0]
        terminal_payload = {
            "id": terminal_id,
            "code": str(terminal_meta["terminal_code"]),
            "type": "Cash recycler" if terminal_meta["is_crm"] else "Withdrawal-only ATM",
            "is_crm": bool(terminal_meta["is_crm"]),
            "location": str(location_row["location_name"]),
            "address": str(location_row["address_line"]),
            "latitude": float(location_row["latitude"]),
            "longitude": float(location_row["longitude"]),
            "site_type": str(terminal_master_row["site_type"]),
            "service_window": (
                f"{terminal_master_row['service_window_start']}–"
                f"{terminal_master_row['service_window_end']}"
            ),
            "operating_model": "Onsite · Bank owned",
            "status": terminal_status,
            "on_route": terminal_id in route_terminal_ids,
            "current_inventory": terminal_inventory,
            "capacity_total": terminal_capacity,
            "protected_demand": int(terminal_action["forecasted_demand"]),
            "recommended_order": int(terminal_action["recommended_amount"]),
            "risk_quantile": str(terminal_action["risk_quantile"]).upper(),
            "protection_horizon_days": int(terminal_action["protection_horizon_days"]),
            "cash_cover_days": (
                round(terminal_inventory / terminal_average_p80, 1)
                if terminal_average_p80
                else None
            ),
            "model_name": summary["model_name"],
            "seven_day_p80": terminal_seven_day_p80,
            "forecast": _records(terminal_daily),
            "inventory": inventory_rows,
            "explainability": _explainability(terminal_daily),
            "evaluation": {
                "wape": float(terminal_evaluation["wape"]),
                "mae": float(terminal_evaluation["mae"]),
                "p50_coverage": float(terminal_evaluation["p50_coverage"]),
                "p80_coverage": float(terminal_evaluation["p80_coverage"]),
                "p95_coverage": float(terminal_evaluation["p95_coverage"]),
                "observations": int(terminal_evaluation["observations"]),
            },
        }
        terminal_payloads.append(terminal_payload)
        dry_runs.append(
            _build_dry_run(
                summary,
                terminal_meta,
                forecast,
                recommendation,
                action,
                transactions,
                snapshots,
                holdout,
            )
        )

    healthy_count = sum(terminal["status"] == "Healthy" for terminal in terminal_payloads)

    return {
        "generated_at": forecast["forecast_generated_ts"].iloc[0],
        "forecast_contract": summary.get("forecast_contract", {"evidence_status": "legacy_unverified"}),
        "simulation_contract": summary.get("simulation_contract", {"evidence_status": "legacy_unverified"}),
        "as_of": summary["as_of"],
        "network": {
            "terminal_count": len(terminal_payloads),
            "healthy_count": healthy_count,
            "attention_count": len(terminal_payloads) - healthy_count,
            "current_inventory": current_inventory,
            "capacity_total": capacity_total,
            "inventory_utilisation": round(current_inventory / capacity_total, 4),
            "seven_day_p80": seven_day_p80,
            "recommended_order": recommended_order,
            "cash_cover_days": cash_cover_days,
        },
        "cra": {
            "id": int(cra_row["cra_id"]),
            "code": str(cra_row["cra_code"]),
            "name": str(cra_row["cra_name"]),
            "operator": str(cra_row["operator_name"]),
            "status": str(cra_row["status"]),
            "region": {
                "id": int(region_row["region_id"]),
                "code": str(region_row["region_code"]),
                "name": str(region_row["region_name"]),
                "service_radius_km": float(region_row["service_radius_km"]),
            },
            "vault": {
                "id": int(vault_row["vault_id"]),
                "code": str(vault_row["vault_code"]),
                "name": str(vault_row["vault_name"]),
                "location_name": str(vault_location["location_name"]),
                "latitude": float(vault_location["latitude"]),
                "longitude": float(vault_location["longitude"]),
                "cash_capacity": int(vault_row["cash_capacity"]),
                "opening_amount": int(vault_inventory["opening_amount"].sum()),
                "reserved_amount": int(vault_inventory["reserved_amount"].sum()),
                "available_amount": int(vault_inventory["available_amount"].sum()),
                "dispatch_cutoff": str(vault_row["dispatch_cutoff"]),
                "inventory": _records(vault_inventory.sort_values("denomination")),
            },
            "vehicle": {
                "id": int(vehicle_row["vehicle_id"]),
                "code": str(vehicle_row["vehicle_code"]),
                "type": str(vehicle_row["vehicle_type"]),
                "cash_capacity": int(vehicle_row["cash_capacity"]),
                "max_stops": int(vehicle_row["max_stops"]),
                "shift": f"{vehicle_row['shift_start']}–{vehicle_row['shift_end']}",
                "status": str(vehicle_row["status"]),
            },
            "route": {
                "id": int(route_row["route_id"]),
                "code": str(route_row["route_code"]),
                "date": str(route_row["route_date"]),
                "start": str(route_row["planned_start_ts"]),
                "end": str(route_row["planned_end_ts"]),
                "distance_km": float(route_row["planned_distance_km"]),
                "load_amount": int(route_row["planned_load_amount"]),
                "stop_count": int(route_row["stop_count"]),
                "vehicle_capacity_amount": int(route_row["vehicle_capacity_amount"]),
                "status": str(route_row["status"]),
                "stops": route_stop_payloads,
            },
            "allocations": _records(vault_allocations),
        },
        "terminal": terminal_payloads[0],
        "terminals": terminal_payloads,
        "forecast": _records(network_daily),
        "explainability": terminal_payloads[0]["explainability"],
        "inventory": terminal_payloads[0]["inventory"],
        "evaluation": {
            "wape": float(overall["wape"]),
            "mae": float(overall["mae"]),
            "p50_coverage": float(overall["p50_coverage"]),
            "p80_coverage": float(overall["p80_coverage"]),
            "p95_coverage": float(overall["p95_coverage"]),
            "observations": int(overall["observations"]),
        },
        "dry_run": dry_runs[0],
        "dry_runs": dry_runs,
    }


def build_demand_demo_payload(
    output_dir: str | Path = "outputs",
    terminal_id: int | None = None,
    horizon: int = 7,
) -> dict[str, Any]:
    """Build the demand-only demo from the promoted, persisted model artifact."""
    if horizon < 1 or horizon > 14:
        raise ValueError("horizon must be between 1 and 14 days")
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    demand_root = output_path / "demand_model"
    required = {
        "training_summary": demand_root / "training_summary.json",
        "transactions": output_path / "network_daily_transactions.csv",
        "terminal_master": output_path / "postgres_csv" / "dim_terminal.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing demand demo inputs: " + ", ".join(missing)
        )

    training_summary = json.loads(
        required["training_summary"].read_text(encoding="utf-8")
    )
    model_version = str(training_summary["model_version"])
    version_dir = demand_root / "models" / model_version
    artifact = GlobalDemandArtifact.load(version_dir)
    manifest = json.loads((version_dir / "manifest.json").read_text(encoding="utf-8"))

    known_terminal_ids = sorted(artifact.terminal_profiles)
    selected_terminal_id = known_terminal_ids[0] if terminal_id is None else terminal_id
    if selected_terminal_id not in artifact.terminal_profiles:
        raise ValueError(f"Unknown terminal_id: {selected_terminal_id}")

    terminal_master = pd.read_csv(required["terminal_master"])
    terminal_rows: list[dict[str, Any]] = []
    for known_id in known_terminal_ids:
        matches = terminal_master.loc[terminal_master["terminal_id"] == known_id]
        if matches.empty:
            profile = artifact.terminal_profiles[known_id]
            terminal_rows.append(
                {
                    "terminal_id": known_id,
                    "terminal_code": str(known_id),
                    "terminal_type": "ATM",
                    "site_type": str(profile["site_type"]),
                }
            )
            continue
        row = matches.iloc[0]
        terminal_rows.append(
            {
                "terminal_id": known_id,
                "terminal_code": str(row["terminal_code"]),
                "terminal_type": str(row["terminal_type"]),
                "site_type": str(row["site_type"]),
            }
        )
    selected_terminal = next(
        row for row in terminal_rows if row["terminal_id"] == selected_terminal_id
    )

    start_date = pd.Timestamp(artifact.training_end) + pd.Timedelta(days=1)
    forecast = artifact.forecast(
        start_date, horizon, terminal_ids=[selected_terminal_id]
    )
    daily_forecast = (
        forecast.groupby("forecast_date", as_index=False)[
            ["p50_amount", "p80_amount", "p95_amount"]
        ]
        .sum()
        .sort_values("forecast_date")
    )
    denomination_forecast = forecast.sort_values(
        ["forecast_date", "denomination"]
    )
    denomination_totals = (
        forecast.groupby("denomination", as_index=False)[
            ["p50_amount", "p80_amount", "p95_amount"]
        ]
        .sum()
        .sort_values("denomination")
    )

    transactions = pd.read_csv(
        required["transactions"], parse_dates=["transaction_date"]
    )
    data_quality = audit_demand_data(transactions)
    terminal_transactions = transactions.loc[
        transactions["terminal_id"] == selected_terminal_id
    ].copy()
    historical = (
        terminal_transactions.groupby("transaction_date", as_index=False)[
            [
                "withdrawal_requested_amount",
                "amount_dispensed",
                "withdrawal_short_amount",
            ]
        ]
        .sum()
        .sort_values("transaction_date")
        .tail(90)
        .rename(
            columns={
                "transaction_date": "date",
                "withdrawal_requested_amount": "requested_amount",
                "amount_dispensed": "dispensed_amount",
                "withdrawal_short_amount": "short_amount",
            }
        )
    )
    probable_reasons = _probable_demand_reasons(daily_forecast, historical)

    candidate = artifact.metrics
    baseline = artifact.promotion.get("baseline_metrics", {})
    baseline_wape = float(baseline["wape"]) if "wape" in baseline else None
    wape_improvement = ((baseline_wape - float(candidate["wape"])) / baseline_wape
                        if baseline_wape else None)

    feature_families = [
        {
            "name": "Calendar",
            "features": artifact.feature_columns[:9],
            "description": "Weekday, month, holidays, payday and month boundaries.",
        },
        {
            "name": "Demand history",
            "features": artifact.feature_columns[9:15],
            "description": "Prior-day and prior-week lags plus recent level and volatility.",
        },
        {
            "name": "ATM context",
            "features": artifact.feature_columns[15:],
            "description": "Recycler status, terminal, denomination and site type.",
        },
    ]
    forecast_totals = {
        quantile: int(daily_forecast[f"{quantile}_amount"].sum())
        for quantile in ("p50", "p80", "p95")
    }
    return {
        "demo_scope": "cash_demand_only",
        "forecast_contract": {
            "schema_version": 1,
            "provider": "global_quantile_gbm",
            "model_version": artifact.model_version,
            "target": "requested gross withdrawal amount",
            "grain": "ATM × day × denomination",
            "currency": "INR",
            "forecast_origin": str(start_date.date()),
            "horizon_days": horizon,
            "calibration_scope": "rolling one-day-ahead marginal denomination forecasts; sums are not calibrated quantiles",
            "evidence_status": artifact.promotion["status"],
            "operationally_promoted": False,
        },
        "data_provenance": "Synthetic POC daily aggregates",
        "terminals": terminal_rows,
        "selected_terminal": selected_terminal,
        "horizon_days": horizon,
        "history": _records(historical),
        "forecast": _records(daily_forecast),
        "denomination_forecast": _records(denomination_forecast),
        "denomination_totals": _records(denomination_totals),
        "forecast_totals": forecast_totals,
        "probable_reasons": probable_reasons,
        "model": {
            "version": artifact.model_version,
            "created_at": artifact.created_at,
            "training_start": artifact.training_start,
            "training_end": artifact.training_end,
            "target": "Requested withdrawal amount",
            "promotion_status": artifact.promotion["status"],
            "artifact_checksum": manifest["artifact_sha256"],
            "feature_count": len(artifact.feature_columns),
            "runtime_configurable_feature_count": 0,
            "feature_columns": artifact.feature_columns,
            "feature_families": feature_families,
        },
        "evaluation": {
            "observations": int(candidate.get("observations", 0)),
            "wape": float(candidate["wape"]),
            "bias_pct": float(candidate["bias_pct"]),
            "p80_coverage": float(candidate["p80_coverage"]),
            "p95_coverage": float(candidate["p95_coverage"]),
            "baseline_wape": baseline_wape,
            "wape_improvement_vs_baseline": round(wape_improvement, 4) if wape_improvement is not None else None,
            "evidence_status": artifact.promotion["status"],
            "cadence": "rolling one-day-ahead",
            "folds": artifact.evaluation_evidence.get("folds", []),
            "grain": "ATM × day × denomination",
            "aggregate_bands_calibrated": False,
        },
        "data_quality": {
            **data_quality,
            "selected_terminal_censored_rows": int(
                (terminal_transactions["withdrawal_short_amount"] > 0).sum()
            ),
        },
    }


class DashboardRequestHandler(SimpleHTTPRequestHandler):
    output_dir = "outputs"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self) -> None:  # noqa: N802 - inherited HTTP method name
        parsed = urlparse(self.path)
        route = parsed.path
        if route == "/api/dashboard":
            self._send_dashboard_data()
            return
        if route == "/api/demand-demo":
            query = parse_qs(parsed.query)
            self._send_demand_demo_data(query)
            return
        if route.startswith("/api/"):
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown API endpoint")
            return
        if route == "/operations-mock":
            self.path = "/operations-mock.html"
        elif route == "/demand-demo":
            self.path = "/demand-demo.html"
        elif route in {"/client-portal", "/portal"}:
            self.path = "/client-portal.html"
        elif route in {"/", "/network", "/atm", "/dry-run", "/routes", "/performance"}:
            self.path = "/index.html"
        super().do_GET()

    def _send_dashboard_data(self) -> None:
        try:
            payload = build_dashboard_payload(self.output_dir)
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            status = HTTPStatus.OK
        except (FileNotFoundError, KeyError, ValueError) as error:
            body = json.dumps({"error": str(error)}).encode("utf-8")
            status = HTTPStatus.INTERNAL_SERVER_ERROR
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_demand_demo_data(self, query: dict[str, list[str]]) -> None:
        try:
            terminal_value = query.get("terminal_id", [None])[0]
            terminal_id = int(terminal_value) if terminal_value is not None else None
            horizon = int(query.get("horizon", ["7"])[0])
            payload = build_demand_demo_payload(
                self.output_dir, terminal_id=terminal_id, horizon=horizon
            )
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            status = HTTPStatus.OK
        except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
            body = json.dumps({"error": str(error)}).encode("utf-8")
            status = HTTPStatus.BAD_REQUEST
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        super().end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[dashboard] {self.address_string()} - {format % args}")


def serve_dashboard(host: str = "127.0.0.1", port: int = 8000, output_dir: str = "outputs") -> None:
    if not STATIC_DIR.exists():
        raise FileNotFoundError(f"Dashboard assets were not found at {STATIC_DIR}")
    mimetypes.add_type("text/javascript", ".js")
    DashboardRequestHandler.output_dir = output_dir
    server = ThreadingHTTPServer((host, port), DashboardRequestHandler)
    print(f"ATM Cash Intelligence dashboard: http://{host}:{port}")
    print("Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the ATM Cash Intelligence dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    serve_dashboard(args.host, args.port, args.output_dir)


if __name__ == "__main__":
    main()
