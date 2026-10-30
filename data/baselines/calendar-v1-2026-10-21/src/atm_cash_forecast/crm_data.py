from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pandas as pd

from .calendar import calendar_values, make_calendar
from .config import DEFAULT_CONFIG, POCConfig


def _terminal_dimension(config: POCConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "terminal_id": config.terminal_id,
                "terminal_code": config.terminal_code.replace("ATM", "CRM"),
                "terminal_type": "CRM",
                "is_crm": True,
                "operating_model": "Onsite",
                "ownership": "Bank",
                "status": "Active",
                "installation_date": "2023-01-15",
            }
        ]
    )


def _cassette_dimension(config: POCConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "cassette_config_id": index,
                "terminal_id": config.terminal_id,
                "cassette_id": index,
                "denomination": denomination,
                "cassette_type": "Recycle",
                "max_note_count": max_notes,
                "max_amount": denomination * max_notes,
            }
            for index, (denomination, max_notes) in enumerate(
                zip(config.denominations, config.cassette_max_notes, strict=True), start=1
            )
        ]
    )


def _module_dimension(config: POCConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "module_id": 1,
                "terminal_id": config.terminal_id,
                "module_type": "Escrow",
                "purpose": "Temporarily holds presented notes until customer confirmation",
                "can_dispense": False,
                "max_note_count": 200,
            },
            {
                "module_id": 2,
                "terminal_id": config.terminal_id,
                "module_type": "All-in collection",
                "purpose": "Stores unfit, unsupported, and overflow accepted notes",
                "can_dispense": False,
                "max_note_count": config.crm_collection_max_notes,
            },
            {
                "module_id": 3,
                "terminal_id": config.terminal_id,
                "module_type": "Reject/retract bin",
                "purpose": "Stores suspect, rejected, or retracted notes for reconciliation",
                "can_dispense": False,
                "max_note_count": config.crm_reject_bin_max_notes,
            },
        ]
    )


def _operating_case(day: pd.Timestamp, index: int) -> str:
    if index and index % 89 == 0:
        return "Denomination mismatch"
    if index and index % 73 == 0:
        return "Deposit surge"
    if index and index % 61 == 0:
        return "Withdrawal surge"
    values = calendar_values(day)
    if values["is_payday"]:
        return "Payday demand"
    if values["is_holiday"]:
        return "Holiday demand"
    if values["is_weekend"]:
        return "Weekend pattern"
    return "Normal"


def _daily_amounts(
    day: pd.Timestamp,
    index: int,
    rng: np.random.Generator,
    config: POCConfig,
) -> tuple[dict[int, int], dict[int, int], str]:
    values = calendar_values(day)
    weekday_factor = [0.96, 0.93, 0.98, 1.04, 1.22, 1.14, 0.78][day.dayofweek]
    payday_factor = 1.42 if values["is_payday"] else 1.0
    holiday_factor = 1.30 if values["is_holiday"] else 1.0
    month_factor = 1 + 0.07 * np.sin(2 * np.pi * (day.month - 1) / 12)
    trend_factor = 1 + 0.00018 * index
    total_withdrawal = (
        155_000
        * weekday_factor
        * payday_factor
        * holiday_factor
        * month_factor
        * trend_factor
        * max(0.72, rng.normal(1, 0.11))
    )
    case = _operating_case(day, index)
    if case == "Withdrawal surge":
        total_withdrawal *= 2.0

    denominations = np.asarray(config.denominations)
    withdrawal_mix = rng.dirichlet(np.array([0.08, 0.16, 0.56, 0.20]) * 120)
    withdrawal_notes: dict[int, int] = {}
    deposit_notes: dict[int, int] = {}
    deposit_ratios = {100: 0.58, 200: 0.50, 500: 0.42, 2_000: 0.22}

    for denomination, share in zip(denominations, withdrawal_mix, strict=True):
        denomination = int(denomination)
        amount = total_withdrawal * float(share)
        if case == "Denomination mismatch" and denomination == 500:
            amount *= 2.4
        withdrawal_notes[denomination] = max(1, int(round(amount / denomination)))

        deposit_amount = total_withdrawal * float(share) * deposit_ratios[denomination]
        if case == "Deposit surge":
            deposit_amount *= 3.2
        if case == "Denomination mismatch" and denomination == 500:
            deposit_amount *= 0.20
        deposit_amount *= max(0.35, rng.normal(1.0, 0.24))
        deposit_notes[denomination] = max(0, int(round(deposit_amount / denomination)))

    return withdrawal_notes, deposit_notes, case


def generate_crm_mock_data(
    start: pd.Timestamp,
    end: pd.Timestamp,
    seed: int = 42,
    config: POCConfig = DEFAULT_CONFIG,
) -> dict[str, pd.DataFrame]:
    """Generate a stateful cash-recycler ledger and forecast-ready daily facts.

    Only notes routed into denomination-specific recycle cassettes can fund a
    later withdrawal. Accepted notes in the collection module are credited to
    the customer but never treated as available dispense inventory.
    """
    config.validate()
    rng = np.random.default_rng(seed)
    cassettes = _cassette_dimension(config)
    cassette_lookup = cassettes.set_index("denomination").to_dict("index")
    capacities = {
        int(denomination): int(max_notes)
        for denomination, max_notes in zip(
            config.denominations, config.cassette_max_notes, strict=True
        )
    }
    recycler_notes = {
        denomination: int(capacity * config.crm_target_fill_ratio)
        for denomination, capacity in capacities.items()
    }
    collection_notes: dict[int, int] = {denomination: 0 for denomination in capacities}
    reject_notes = 0

    event_rows: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []
    snapshot_rows: list[dict[str, object]] = []
    replenishment_rows: list[dict[str, object]] = []
    evacuation_rows: list[dict[str, object]] = []
    operational_rows: list[dict[str, object]] = []
    transaction_id = snapshot_id = replenishment_id = evacuation_id = operational_id = 1
    block_probabilities = np.array([0.10, 0.16, 0.22, 0.21, 0.18, 0.13])
    block_hours = (8, 10, 12, 14, 16, 18)

    for day_index, day in enumerate(pd.date_range(start, end, freq="D")):
        opening_notes = recycler_notes.copy()
        loaded_notes = {denomination: 0 for denomination in capacities}
        evacuated_recycler = {denomination: 0 for denomination in capacities}
        evacuated_collection = {denomination: 0 for denomination in capacities}

        if (
            sum(collection_notes.values())
            >= int(config.crm_collection_max_notes * config.crm_collection_evacuation_ratio)
            or reject_notes >= int(config.crm_reject_bin_max_notes * 0.80)
            or any(
                recycler_notes[denomination] >= int(capacity * config.crm_high_fill_ratio)
                for denomination, capacity in capacities.items()
            )
        ):
            for denomination, capacity in capacities.items():
                evacuated_collection[denomination] = collection_notes[denomination]
                collection_notes[denomination] = 0
                target = int(capacity * config.crm_target_fill_ratio)
                if recycler_notes[denomination] > target:
                    evacuated_recycler[denomination] = recycler_notes[denomination] - target
                    recycler_notes[denomination] = target
            rejected_removed = reject_notes
            reject_notes = 0
            evacuated_mix = {
                str(denomination): (
                    evacuated_collection[denomination] + evacuated_recycler[denomination]
                )
                * denomination
                for denomination in capacities
                if evacuated_collection[denomination] or evacuated_recycler[denomination]
            }
            evacuation_rows.append(
                {
                    "evacuation_id": evacuation_id,
                    "terminal_id": config.terminal_id,
                    "visit_date": day.date(),
                    "visit_type": "Collection and excess recycle unload",
                    "evacuated_amount": sum(evacuated_mix.values()),
                    "evacuated_denomination_mix": json.dumps(evacuated_mix, sort_keys=True),
                    "reject_notes_removed": rejected_removed,
                    "performed_by": "Mock CIT Team",
                    "status": "Completed",
                }
            )
            operational_rows.append(
                {
                    "operational_event_id": operational_id,
                    "terminal_id": config.terminal_id,
                    "event_ts": day.replace(hour=6).isoformat(),
                    "event_type": "CashEvacuation",
                    "severity": "Info",
                    "denomination": None,
                    "amount": sum(evacuated_mix.values()),
                    "details": "Collection/reject modules cleared; high recycle cassettes balanced",
                }
            )
            operational_id += 1
            evacuation_id += 1

        load_mix: dict[str, int] = {}
        for denomination, capacity in capacities.items():
            if recycler_notes[denomination] < int(capacity * config.crm_reorder_ratio):
                target = int(capacity * config.crm_target_fill_ratio)
                loaded_notes[denomination] = target - recycler_notes[denomination]
                recycler_notes[denomination] = target
                load_mix[str(denomination)] = loaded_notes[denomination] * denomination
        if load_mix:
            replenishment_rows.append(
                {
                    "replenishment_id": replenishment_id,
                    "terminal_id": config.terminal_id,
                    "visit_date": day.date(),
                    "visit_type": "Load recycle cassettes",
                    "loaded_amount": sum(load_mix.values()),
                    "loaded_denomination_mix": json.dumps(load_mix, sort_keys=True),
                    "performed_by": "Mock CIT Team",
                    "status": "Completed",
                }
            )
            operational_rows.append(
                {
                    "operational_event_id": operational_id,
                    "terminal_id": config.terminal_id,
                    "event_ts": day.replace(hour=6, minute=30).isoformat(),
                    "event_type": "ExternalLoad",
                    "severity": "Info",
                    "denomination": None,
                    "amount": sum(load_mix.values()),
                    "details": "Low recycle cassettes restored to target fill",
                }
            )
            operational_id += 1
            replenishment_id += 1

        withdrawal_notes, deposit_notes, operating_case = _daily_amounts(
            day, day_index, rng, config
        )
        aggregates: dict[int, defaultdict[str, int]] = {
            denomination: defaultdict(int) for denomination in capacities
        }
        same_day_deposit_pool = {denomination: 0 for denomination in capacities}

        scheduled: list[tuple[pd.Timestamp, str, int, int]] = []
        for denomination in capacities:
            withdrawal_blocks = rng.multinomial(
                withdrawal_notes[denomination], block_probabilities
            )
            deposit_blocks = rng.multinomial(deposit_notes[denomination], block_probabilities)
            for block, hour in enumerate(block_hours):
                for transaction_type, note_count in (
                    ("Deposit", int(deposit_blocks[block])),
                    ("Withdrawal", int(withdrawal_blocks[block])),
                ):
                    if note_count:
                        minute = int(rng.integers(0, 110))
                        scheduled.append(
                            (
                                day.replace(hour=hour) + pd.Timedelta(minutes=minute),
                                transaction_type,
                                denomination,
                                note_count,
                            )
                        )

        for event_ts, transaction_type, denomination, requested_notes in sorted(scheduled):
            record = {
                "transaction_id": transaction_id,
                "terminal_id": config.terminal_id,
                "transaction_ts": event_ts.isoformat(),
                "transaction_date": day.date(),
                "transaction_type": transaction_type,
                "channel": rng.choice(["Card", "UPI"], p=[0.72, 0.28]),
                "denomination": denomination,
                "operating_case": operating_case,
                "notes_presented": 0,
                "notes_accepted": 0,
                "notes_recyclable": 0,
                "notes_to_collection": 0,
                "notes_rejected": 0,
                "notes_returned": 0,
                "notes_requested": 0,
                "notes_dispensed": 0,
                "notes_short": 0,
                "same_day_recycled_notes_dispensed": 0,
            }
            aggregate = aggregates[denomination]

            if transaction_type == "Deposit":
                record["notes_presented"] = requested_notes
                aggregate["deposit_events"] += 1
                aggregate["notes_presented"] += requested_notes
                if rng.random() < 0.008:
                    returned = requested_notes
                    accepted = recyclable = to_collection = rejected = 0
                    status = "Cancelled"
                else:
                    suspect = int(rng.binomial(requested_notes, 0.004))
                    reject_room = max(0, config.crm_reject_bin_max_notes - reject_notes)
                    rejected = min(suspect, reject_room)
                    reject_notes += rejected
                    candidates = requested_notes - suspect
                    unfit = int(rng.binomial(candidates, 0.045))
                    recyclable_candidates = candidates - unfit
                    recycle_room = capacities[denomination] - recycler_notes[denomination]
                    recyclable = min(recyclable_candidates, recycle_room)
                    recycler_notes[denomination] += recyclable
                    same_day_deposit_pool[denomination] += recyclable

                    collection_candidates = unfit + (recyclable_candidates - recyclable)
                    collection_room = max(
                        0, config.crm_collection_max_notes - sum(collection_notes.values())
                    )
                    to_collection = min(collection_candidates, collection_room)
                    collection_notes[denomination] += to_collection
                    returned = requested_notes - rejected - recyclable - to_collection
                    accepted = recyclable + to_collection
                    status = "Success" if returned == 0 else ("Partial" if accepted else "Declined")

                record.update(
                    {
                        "transaction_status": status,
                        "notes_accepted": accepted,
                        "notes_recyclable": recyclable,
                        "notes_to_collection": to_collection,
                        "notes_rejected": rejected,
                        "notes_returned": returned,
                    }
                )
                for key, value in (
                    ("notes_accepted", accepted),
                    ("notes_recyclable", recyclable),
                    ("notes_to_collection", to_collection),
                    ("notes_rejected", rejected),
                    ("notes_returned", returned),
                ):
                    aggregate[key] += value
            else:
                aggregate["withdrawal_events"] += 1
                aggregate["notes_requested"] += requested_notes
                dispensed = min(requested_notes, recycler_notes[denomination])
                short = requested_notes - dispensed
                same_day_reused = min(dispensed, same_day_deposit_pool[denomination])
                same_day_deposit_pool[denomination] -= same_day_reused
                recycler_notes[denomination] -= dispensed
                status = "Success" if short == 0 else ("Partial" if dispensed else "Declined")
                record.update(
                    {
                        "transaction_status": status,
                        "notes_requested": requested_notes,
                        "notes_dispensed": dispensed,
                        "notes_short": short,
                        "same_day_recycled_notes_dispensed": same_day_reused,
                    }
                )
                aggregate["notes_dispensed"] += dispensed
                aggregate["notes_short"] += short
                aggregate["same_day_recycled_notes_dispensed"] += same_day_reused

            record.update(
                {
                    "amount_requested": requested_notes * denomination,
                    "amount_accepted": int(record["notes_accepted"]) * denomination,
                    "amount_dispensed": int(record["notes_dispensed"]) * denomination,
                    "recyclable_amount": int(record["notes_recyclable"]) * denomination,
                    "collection_amount": int(record["notes_to_collection"]) * denomination,
                    "rejected_amount": int(record["notes_rejected"]) * denomination,
                    "returned_amount": int(record["notes_returned"]) * denomination,
                    "same_day_recycled_dispensed_amount": int(
                        record["same_day_recycled_notes_dispensed"]
                    )
                    * denomination,
                }
            )
            event_rows.append(record)
            transaction_id += 1

        for denomination, capacity in capacities.items():
            aggregate = aggregates[denomination]
            dispensed_amount = aggregate["notes_dispensed"] * denomination
            recyclable_amount = aggregate["notes_recyclable"] * denomination
            daily_rows.append(
                {
                    "transaction_id": len(daily_rows) + 1,
                    "terminal_id": config.terminal_id,
                    "transaction_date": day.date(),
                    "transaction_type": "DailyAggregate",
                    "transaction_status": "Success",
                    "denomination": denomination,
                    "operating_case": operating_case,
                    "withdrawal_transaction_count": aggregate["withdrawal_events"],
                    "deposit_transaction_count": aggregate["deposit_events"],
                    "withdrawal_requested_amount": aggregate["notes_requested"] * denomination,
                    "amount_withdrawn": dispensed_amount,
                    "amount_dispensed": dispensed_amount,
                    "withdrawal_short_amount": aggregate["notes_short"] * denomination,
                    "deposit_presented_amount": aggregate["notes_presented"] * denomination,
                    "deposit_amount": aggregate["notes_accepted"] * denomination,
                    "deposit_recyclable_amount": recyclable_amount,
                    "deposit_collection_amount": aggregate["notes_to_collection"] * denomination,
                    "deposit_rejected_amount": aggregate["notes_rejected"] * denomination,
                    "deposit_returned_amount": aggregate["notes_returned"] * denomination,
                    "same_day_recycled_dispensed_amount": aggregate[
                        "same_day_recycled_notes_dispensed"
                    ]
                    * denomination,
                    "net_demand": dispensed_amount - recyclable_amount,
                    "is_recycler": True,
                }
            )
            snapshot_rows.append(
                {
                    "snapshot_id": snapshot_id,
                    "terminal_id": config.terminal_id,
                    "cassette_config_id": int(
                        cassette_lookup[denomination]["cassette_config_id"]
                    ),
                    "snapshot_date": day.date(),
                    "denomination": denomination,
                    "opening_note_count": opening_notes[denomination],
                    "externally_loaded_note_count": loaded_notes[denomination],
                    "deposit_recyclable_note_count": aggregate["notes_recyclable"],
                    "dispensed_note_count": aggregate["notes_dispensed"],
                    "evacuated_recycler_note_count": evacuated_recycler[denomination],
                    "available_note_count": recycler_notes[denomination],
                    "available_amount": recycler_notes[denomination] * denomination,
                    "recycle_capacity_notes": capacity,
                    "recycle_utilisation": round(recycler_notes[denomination] / capacity, 4),
                    "collection_note_count": collection_notes[denomination],
                    "collection_amount": collection_notes[denomination] * denomination,
                    "collection_total_note_count": sum(collection_notes.values()),
                    "collection_utilisation": round(
                        sum(collection_notes.values()) / config.crm_collection_max_notes, 4
                    ),
                    "reject_bin_note_count": reject_notes,
                    "logical_balance": recycler_notes[denomination] * denomination,
                    "physical_balance": recycler_notes[denomination] * denomination,
                    "source_system": "Stateful CRM Mock",
                }
            )
            snapshot_id += 1

            if aggregate["notes_short"]:
                operational_rows.append(
                    {
                        "operational_event_id": operational_id,
                        "terminal_id": config.terminal_id,
                        "event_ts": day.replace(hour=20).isoformat(),
                        "event_type": "PartialDispenseOrCashOut",
                        "severity": "Critical",
                        "denomination": denomination,
                        "amount": aggregate["notes_short"] * denomination,
                        "details": "Requested notes exceeded recyclable denomination inventory",
                    }
                )
                operational_id += 1
            elif recycler_notes[denomination] < int(capacity * config.crm_reorder_ratio):
                operational_rows.append(
                    {
                        "operational_event_id": operational_id,
                        "terminal_id": config.terminal_id,
                        "event_ts": day.replace(hour=20).isoformat(),
                        "event_type": "DenominationLow",
                        "severity": "Warning",
                        "denomination": denomination,
                        "amount": recycler_notes[denomination] * denomination,
                        "details": "Total cash may be adequate, but this denomination needs loading",
                    }
                )
                operational_id += 1

        total_reused = sum(
            aggregate["same_day_recycled_notes_dispensed"] * denomination
            for denomination, aggregate in aggregates.items()
        )
        total_rejected = sum(
            aggregate["notes_rejected"] for aggregate in aggregates.values()
        )
        if total_reused:
            operational_rows.append(
                {
                    "operational_event_id": operational_id,
                    "terminal_id": config.terminal_id,
                    "event_ts": day.replace(hour=21).isoformat(),
                    "event_type": "AutomaticCashReuse",
                    "severity": "Info",
                    "denomination": None,
                    "amount": total_reused,
                    "details": "Customer-deposited notes were dispensed later the same day",
                }
            )
            operational_id += 1
        if total_rejected:
            operational_rows.append(
                {
                    "operational_event_id": operational_id,
                    "terminal_id": config.terminal_id,
                    "event_ts": day.replace(hour=21, minute=5).isoformat(),
                    "event_type": "NotesRejected",
                    "severity": "Warning",
                    "denomination": None,
                    "amount": total_rejected,
                    "details": "Count of suspect notes diverted to the reject/retract module",
                }
            )
            operational_id += 1

    operational_events = pd.DataFrame(operational_rows)
    operational_events["denomination"] = operational_events["denomination"].astype("Int64")

    return {
        "dim_terminal": _terminal_dimension(config),
        "dim_cassette_configuration": cassettes,
        "dim_crm_module": _module_dimension(config),
        "dim_calendar": make_calendar(start, end + pd.Timedelta(days=31)),
        "fact_transaction": pd.DataFrame(daily_rows),
        "fact_crm_transaction_event": pd.DataFrame(event_rows),
        "fact_cash_inventory_snapshot": pd.DataFrame(snapshot_rows),
        "fact_replenishment": pd.DataFrame(replenishment_rows),
        "fact_cash_evacuation": pd.DataFrame(evacuation_rows, columns=[
            "evacuation_id", "terminal_id", "visit_date", "visit_type",
            "evacuated_amount", "evacuated_denomination_mix", "reject_notes_removed",
            "performed_by", "status",
        ]),
        "fact_crm_operational_event": operational_events,
    }
