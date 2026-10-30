from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .calendar import calendar_values, make_calendar
from .config import DEFAULT_CONFIG, POCConfig
from .regional import (
    REGIONAL_TERMINALS,
    apply_latest_cash_scenarios,
    enrich_terminal_dimension,
    regional_dimensions,
)


TERMINAL_ID = 1001
WITHDRAWAL_ONLY_TERMINAL_ID = 1002


def terminal_dimension(
    terminal_id: int = TERMINAL_ID,
    terminal_code: str = "ATM-MUM-001",
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "terminal_id": terminal_id,
                "terminal_code": terminal_code,
                "terminal_type": "ATM",
                "is_crm": False,
                "operating_model": "Onsite",
                "ownership": "Bank",
                "status": "Active",
                "installation_date": "2023-01-15",
            }
        ]
    )


def cassette_dimension(
    config: POCConfig = DEFAULT_CONFIG,
    terminal_id: int = TERMINAL_ID,
    config_id_offset: int = 0,
) -> pd.DataFrame:
    rows = [
        (index, index, denomination, max_notes)
        for index, (denomination, max_notes) in enumerate(
            zip(config.denominations, config.cassette_max_notes, strict=True), start=1
        )
    ]
    return pd.DataFrame(
        [
            {
                "cassette_config_id": config_id + config_id_offset,
                "terminal_id": terminal_id,
                "cassette_id": cassette_id,
                "denomination": denomination,
                "cassette_type": "Dispense",
                "max_note_count": max_notes,
                "max_amount": denomination * max_notes,
            }
            for config_id, cassette_id, denomination, max_notes in rows
        ]
    )


def generate_transactions(
    start: pd.Timestamp,
    end: pd.Timestamp,
    seed: int = 42,
    recycler_enabled: bool = False,
    terminal_id: int = TERMINAL_ID,
    demand_scale: float = 1.0,
) -> pd.DataFrame:
    """Generate daily withdrawal/deposit aggregates for one ATM.

    When ``recycler_enabled`` is true, deposits reduce the cash demand that must
    be served from loaded inventory. The withdrawal column remains available so
    dispense-only and recycler scenarios can use the same table.
    """
    rng = np.random.default_rng(seed)
    denominations = np.array([100, 200, 500, 2_000])
    mix = np.array([0.08, 0.16, 0.56, 0.20])
    rows: list[dict[str, object]] = []
    transaction_id = 1

    for index, day in enumerate(pd.date_range(start, end, freq="D")):
        cal = calendar_values(day)
        weekday_factor = [0.96, 0.93, 0.98, 1.04, 1.22, 1.14, 0.78][day.dayofweek]
        payday_factor = 1.42 if cal["is_payday"] else 1.0
        holiday_factor = 1.30 if cal["is_holiday"] else 1.0
        month_factor = 1 + 0.07 * np.sin(2 * np.pi * (day.month - 1) / 12)
        trend_factor = 1 + 0.00018 * index
        noise = max(0.72, rng.normal(1, 0.11))
        total_demand = (
            155_000
            * demand_scale
            * weekday_factor
            * payday_factor
            * holiday_factor
            * month_factor
            * trend_factor
            * noise
        )

        daily_mix = rng.dirichlet(mix * 120)
        for denomination, share in zip(denominations, daily_mix, strict=True):
            withdrawal_amount = int(round((total_demand * share) / denomination) * denomination)
            withdrawal_amount = max(int(denomination), withdrawal_amount)
            deposit_ratio = {
                100: 0.18,
                200: 0.15,
                500: 0.12,
                2_000: 0.05,
            }[int(denomination)]
            deposit_noise = max(0.25, rng.normal(1.0, 0.20))
            deposit_amount = (
                int(round(withdrawal_amount * deposit_ratio * deposit_noise / denomination) * denomination)
                if recycler_enabled
                else 0
            )
            rows.append(
                {
                    "transaction_id": transaction_id,
                    "terminal_id": terminal_id,
                    "transaction_date": day.date(),
                    "transaction_type": "DailyAggregate",
                    "transaction_status": "Success",
                    "denomination": int(denomination),
                    "operating_case": "Withdrawal-only ATM",
                    "withdrawal_transaction_count": max(1, withdrawal_amount // 3_000),
                    "deposit_transaction_count": 0,
                    "withdrawal_requested_amount": withdrawal_amount,
                    "amount_withdrawn": withdrawal_amount,
                    "amount_dispensed": withdrawal_amount,
                    "withdrawal_short_amount": 0,
                    "deposit_presented_amount": deposit_amount,
                    "deposit_amount": deposit_amount,
                    "deposit_recyclable_amount": deposit_amount,
                    "deposit_collection_amount": 0,
                    "deposit_rejected_amount": 0,
                    "deposit_returned_amount": 0,
                    "same_day_recycled_dispensed_amount": 0,
                    "net_demand": withdrawal_amount - deposit_amount,
                    "is_recycler": recycler_enabled,
                }
            )
            transaction_id += 1

    return pd.DataFrame(rows)


def simulate_inventory(
    transactions: pd.DataFrame,
    cassettes: pd.DataFrame,
    config: POCConfig = DEFAULT_CONFIG,
    terminal_id: int = TERMINAL_ID,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate daily replenishments and closing cassette balances."""
    cassette_lookup = cassettes.set_index("denomination").to_dict("index")
    balances = {
        denomination: int(row["max_amount"] * config.historical_target_fill_ratio)
        for denomination, row in cassette_lookup.items()
    }
    snapshot_rows: list[dict[str, object]] = []
    replenishment_rows: list[dict[str, object]] = []
    snapshot_id = 1
    replenishment_id = 1

    for day, daily in transactions.groupby("transaction_date", sort=True):
        loaded_mix: dict[str, int] = {}
        for record in daily.itertuples(index=False):
            denomination = int(record.denomination)
            cassette = cassette_lookup[denomination]
            capacity = int(cassette["max_amount"])
            demand = int(record.net_demand)
            target = int(capacity * config.historical_target_fill_ratio) // denomination * denomination
            threshold = int(capacity * config.historical_reorder_ratio)
            opening_balance = balances[denomination]
            load = 0
            dispensed = 0

            if balances[denomination] - demand < threshold:
                load = max(0, target - balances[denomination])
                load = min(load, capacity - balances[denomination])
                load = load // denomination * denomination
                balances[denomination] += load
                if load:
                    loaded_mix[str(denomination)] = load

            if demand >= 0:
                dispensed = min(demand, balances[denomination])
                balances[denomination] -= dispensed
            else:
                balances[denomination] = min(capacity, balances[denomination] - demand)
            snapshot_rows.append(
                {
                    "snapshot_id": snapshot_id,
                    "terminal_id": terminal_id,
                    "cassette_config_id": int(cassette["cassette_config_id"]),
                    "snapshot_date": day,
                    "denomination": denomination,
                    "opening_note_count": opening_balance // denomination,
                    "externally_loaded_note_count": load // denomination,
                    "deposit_recyclable_note_count": 0,
                    "dispensed_note_count": dispensed // denomination,
                    "evacuated_recycler_note_count": 0,
                    "available_note_count": balances[denomination] // denomination,
                    "available_amount": balances[denomination],
                    "recycle_capacity_notes": capacity // denomination,
                    "recycle_utilisation": round(balances[denomination] / capacity, 4),
                    "collection_note_count": 0,
                    "collection_amount": 0,
                    "collection_total_note_count": 0,
                    "collection_utilisation": 0.0,
                    "reject_bin_note_count": 0,
                    "logical_balance": balances[denomination],
                    "physical_balance": balances[denomination],
                    "source_system": "Withdrawal-only ATM Mock",
                }
            )
            snapshot_id += 1

        if loaded_mix:
            replenishment_rows.append(
                {
                    "replenishment_id": replenishment_id,
                    "terminal_id": terminal_id,
                    "visit_date": day,
                    "visit_type": "Load",
                    "loaded_amount": sum(loaded_mix.values()),
                    "loaded_denomination_mix": json.dumps(loaded_mix, sort_keys=True),
                    "performed_by": "POC Route Team",
                    "status": "Completed",
                }
            )
            replenishment_id += 1

    return pd.DataFrame(snapshot_rows), pd.DataFrame(replenishment_rows)


def generate_demo_data(
    as_of: pd.Timestamp,
    history_days: int | None = None,
    seed: int = 42,
    recycler_enabled: bool = True,
    config: POCConfig = DEFAULT_CONFIG,
) -> dict[str, pd.DataFrame]:
    config.validate()
    history_days = history_days or config.history_days
    end = as_of.normalize() - pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=history_days - 1)
    if recycler_enabled:
        # Local import avoids coupling the smaller dispense-only generator to
        # the richer stateful cash-recycler ledger.
        from .crm_data import generate_crm_mock_data

        crm_tables = generate_crm_mock_data(start, end, seed=seed, config=config)
        atm_terminals: list[pd.DataFrame] = []
        atm_cassettes: list[pd.DataFrame] = []
        atm_transactions: list[pd.DataFrame] = []
        atm_inventories: list[pd.DataFrame] = []
        atm_replenishments: list[pd.DataFrame] = []
        for spec in REGIONAL_TERMINALS[1:]:
            terminal_id = int(spec["terminal_id"])
            terminal = terminal_dimension(
                terminal_id=terminal_id,
                terminal_code=str(spec["terminal_code"]),
            )
            cassettes = cassette_dimension(
                config,
                terminal_id=terminal_id,
                config_id_offset=(terminal_id - 1001) * 100,
            )
            transactions = generate_transactions(
                start,
                end,
                seed=seed + int(spec["seed_offset"]),
                recycler_enabled=False,
                terminal_id=terminal_id,
                demand_scale=float(spec["demand_scale"]),
            )
            inventory, replenishments = simulate_inventory(
                transactions,
                cassettes,
                config,
                terminal_id=terminal_id,
            )
            atm_terminals.append(terminal)
            atm_cassettes.append(cassettes)
            atm_transactions.append(transactions)
            atm_inventories.append(inventory)
            atm_replenishments.append(replenishments)
        crm_tables["dim_terminal"] = pd.concat(
            [crm_tables["dim_terminal"], *atm_terminals], ignore_index=True
        )
        crm_tables["dim_cassette_configuration"] = pd.concat(
            [crm_tables["dim_cassette_configuration"], *atm_cassettes], ignore_index=True
        )
        crm_tables["fact_transaction"] = pd.concat(
            [crm_tables["fact_transaction"], *atm_transactions], ignore_index=True
        ).sort_values(["transaction_date", "terminal_id", "denomination"])
        crm_tables["fact_transaction"]["transaction_id"] = range(
            1, len(crm_tables["fact_transaction"]) + 1
        )
        crm_tables["fact_cash_inventory_snapshot"] = pd.concat(
            [crm_tables["fact_cash_inventory_snapshot"], *atm_inventories], ignore_index=True
        ).sort_values(["snapshot_date", "terminal_id", "denomination"])
        crm_tables["fact_cash_inventory_snapshot"]["snapshot_id"] = range(
            1, len(crm_tables["fact_cash_inventory_snapshot"]) + 1
        )
        crm_tables["fact_replenishment"] = pd.concat(
            [crm_tables["fact_replenishment"], *atm_replenishments], ignore_index=True
        ).sort_values(["visit_date", "terminal_id"])
        crm_tables["fact_replenishment"]["replenishment_id"] = range(
            1, len(crm_tables["fact_replenishment"]) + 1
        )
        crm_tables["dim_terminal"] = enrich_terminal_dimension(crm_tables["dim_terminal"])
        crm_tables.update(regional_dimensions(as_of))
        apply_latest_cash_scenarios(crm_tables)
        return crm_tables
    terminal = terminal_dimension()
    cassettes = cassette_dimension(config)
    transactions = generate_transactions(start, end, seed, recycler_enabled)
    inventory, replenishments = simulate_inventory(transactions, cassettes, config)
    calendar = make_calendar(start, as_of + pd.Timedelta(days=30))
    return {
        "dim_terminal": terminal,
        "dim_cassette_configuration": cassettes,
        "dim_calendar": calendar,
        "fact_transaction": transactions,
        "fact_cash_inventory_snapshot": inventory,
        "fact_replenishment": replenishments,
    }
