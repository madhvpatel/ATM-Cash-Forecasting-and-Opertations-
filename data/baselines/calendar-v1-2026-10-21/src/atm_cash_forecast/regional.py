from __future__ import annotations

import json
import math
from datetime import datetime, time, timedelta

import pandas as pd


CRA_ID = 1
REGION_ID = 11
VAULT_ID = 201
VAULT_LOCATION_ID = 5001
VEHICLE_ID = 301


REGIONAL_TERMINALS = [
    {
        "terminal_id": 1001,
        "terminal_code": "CRM-MUM-001",
        "location_name": "Dadar TT Circle",
        "site_type": "Transit hub",
        "latitude": 19.0183,
        "longitude": 72.8444,
        "address_line": "Dadar East, Mumbai",
        "pincode": "400014",
        "demand_scale": 1.00,
        "seed_offset": 0,
        "low_cash_ratio": None,
        "service_window_start": "08:00",
        "service_window_end": "11:00",
    },
    {
        "terminal_id": 1002,
        "terminal_code": "ATM-MUM-002",
        "location_name": "Matunga Central",
        "site_type": "Residential high street",
        "latitude": 19.0274,
        "longitude": 72.8553,
        "address_line": "Lakhamsi Nappu Road, Matunga",
        "pincode": "400019",
        "demand_scale": 1.12,
        "seed_offset": 101,
        "low_cash_ratio": None,
        "service_window_start": "08:00",
        "service_window_end": "12:00",
    },
    {
        "terminal_id": 1003,
        "terminal_code": "ATM-PAR-003",
        "location_name": "Parel Station",
        "site_type": "Transit hub",
        "latitude": 18.9977,
        "longitude": 72.8376,
        "address_line": "Dr Babasaheb Ambedkar Road, Parel",
        "pincode": "400012",
        "demand_scale": 1.34,
        "seed_offset": 203,
        "low_cash_ratio": 0.05,
        "service_window_start": "08:00",
        "service_window_end": "10:30",
    },
    {
        "terminal_id": 1004,
        "terminal_code": "ATM-WOR-004",
        "location_name": "Worli Naka",
        "site_type": "Commercial district",
        "latitude": 19.0114,
        "longitude": 72.8174,
        "address_line": "Dr Annie Besant Road, Worli",
        "pincode": "400018",
        "demand_scale": 0.93,
        "seed_offset": 307,
        "low_cash_ratio": 0.14,
        "service_window_start": "09:00",
        "service_window_end": "13:00",
    },
    {
        "terminal_id": 1005,
        "terminal_code": "ATM-MAH-005",
        "location_name": "Mahim Junction",
        "site_type": "Mixed retail",
        "latitude": 19.0402,
        "longitude": 72.8427,
        "address_line": "L J Road, Mahim",
        "pincode": "400016",
        "demand_scale": 1.05,
        "seed_offset": 409,
        "low_cash_ratio": None,
        "service_window_start": "08:30",
        "service_window_end": "12:30",
    },
    {
        "terminal_id": 1006,
        "terminal_code": "ATM-PRB-006",
        "location_name": "Prabhadevi Business Centre",
        "site_type": "Office district",
        "latitude": 19.0148,
        "longitude": 72.8295,
        "address_line": "Sayani Road, Prabhadevi",
        "pincode": "400025",
        "demand_scale": 1.21,
        "seed_offset": 503,
        "low_cash_ratio": 0.08,
        "service_window_start": "08:00",
        "service_window_end": "11:30",
    },
]


def terminal_spec(terminal_id: int) -> dict[str, object]:
    return next(item for item in REGIONAL_TERMINALS if item["terminal_id"] == terminal_id)


def enrich_terminal_dimension(terminals: pd.DataFrame) -> pd.DataFrame:
    metadata = pd.DataFrame(
        [
            {
                "terminal_id": spec["terminal_id"],
                "cra_id": CRA_ID,
                "region_id": REGION_ID,
                "location_id": spec["terminal_id"],
                "site_type": spec["site_type"],
                "service_window_start": spec["service_window_start"],
                "service_window_end": spec["service_window_end"],
            }
            for spec in REGIONAL_TERMINALS
        ]
    )
    enriched = terminals.merge(metadata, on="terminal_id", how="left")
    return enriched[
        [
            "terminal_id",
            "terminal_code",
            "terminal_type",
            "is_crm",
            "cra_id",
            "region_id",
            "location_id",
            "site_type",
            "operating_model",
            "ownership",
            "status",
            "installation_date",
            "service_window_start",
            "service_window_end",
        ]
    ]


def regional_dimensions(as_of: pd.Timestamp) -> dict[str, pd.DataFrame]:
    locations = [
        {
            "location_id": VAULT_LOCATION_ID,
            "region_id": REGION_ID,
            "location_name": "Dadar Regional Cash Centre",
            "location_type": "Vault",
            "latitude": 19.0212,
            "longitude": 72.8426,
            "address_line": "Naigaon Cross Road, Dadar East",
            "pincode": "400014",
        }
    ]
    locations.extend(
        {
            "location_id": spec["terminal_id"],
            "region_id": REGION_ID,
            "location_name": spec["location_name"],
            "location_type": "Terminal",
            "latitude": spec["latitude"],
            "longitude": spec["longitude"],
            "address_line": spec["address_line"],
            "pincode": spec["pincode"],
        }
        for spec in REGIONAL_TERMINALS
    )
    vault_stock = {100: 3_000_000, 200: 6_000_000, 500: 18_000_000, 2_000: 24_000_000}
    return {
        "dim_cra": pd.DataFrame(
            [
                {
                    "cra_id": CRA_ID,
                    "cra_code": "CRA-MUM-CENTRAL-01",
                    "cra_name": "Mumbai Central Cash Replenishment Agency",
                    "operator_name": "Demo Secure Cash Services",
                    "status": "Active",
                }
            ]
        ),
        "dim_region": pd.DataFrame(
            [
                {
                    "region_id": REGION_ID,
                    "cra_id": CRA_ID,
                    "region_code": "MUM-CENTRAL",
                    "region_name": "Mumbai Central Micro-region",
                    "service_radius_km": 10.0,
                    "timezone": "Asia/Kolkata",
                }
            ]
        ),
        "dim_location": pd.DataFrame(locations),
        "dim_vault": pd.DataFrame(
            [
                {
                    "vault_id": VAULT_ID,
                    "cra_id": CRA_ID,
                    "region_id": REGION_ID,
                    "location_id": VAULT_LOCATION_ID,
                    "vault_code": "VLT-DADAR-01",
                    "vault_name": "Dadar Regional Cash Centre",
                    "cash_capacity": sum(vault_stock.values()),
                    "dispatch_cutoff": "07:30",
                    "status": "Active",
                }
            ]
        ),
        "dim_vehicle": pd.DataFrame(
            [
                {
                    "vehicle_id": VEHICLE_ID,
                    "cra_id": CRA_ID,
                    "vehicle_code": "CIT-MUM-01",
                    "vehicle_type": "Armoured van",
                    "cash_capacity": 12_000_000,
                    "max_stops": 6,
                    "shift_start": "08:00",
                    "shift_end": "15:00",
                    "status": "Available",
                }
            ]
        ),
        "fact_vault_inventory": pd.DataFrame(
            [
                {
                    "vault_inventory_id": index,
                    "vault_id": VAULT_ID,
                    "snapshot_date": as_of.date(),
                    "denomination": denomination,
                    "opening_amount": amount,
                    "reserved_amount": 0,
                    "available_amount": amount,
                }
                for index, (denomination, amount) in enumerate(vault_stock.items(), start=1)
            ]
        ),
    }


def apply_latest_cash_scenarios(tables: dict[str, pd.DataFrame]) -> None:
    snapshots = tables["fact_cash_inventory_snapshot"]
    transactions = tables["fact_transaction"]
    latest_date = snapshots["snapshot_date"].max()
    for spec in REGIONAL_TERMINALS:
        ratio = spec["low_cash_ratio"]
        if ratio is None:
            continue
        terminal_id = int(spec["terminal_id"])
        terminal_mask = (snapshots["terminal_id"] == terminal_id) & (
            snapshots["snapshot_date"] == latest_date
        )
        for index in snapshots.index[terminal_mask]:
            denomination = int(snapshots.at[index, "denomination"])
            capacity_notes = int(snapshots.at[index, "recycle_capacity_notes"])
            available_notes = max(1, int(capacity_notes * float(ratio)))
            actual = transactions.loc[
                (transactions["terminal_id"] == terminal_id)
                & (transactions["transaction_date"] == latest_date)
                & (transactions["denomination"] == denomination)
            ].iloc[0]
            dispensed_notes = int(actual["amount_dispensed"]) // denomination
            snapshots.at[index, "opening_note_count"] = available_notes + dispensed_notes
            snapshots.at[index, "externally_loaded_note_count"] = 0
            snapshots.at[index, "deposit_recyclable_note_count"] = 0
            snapshots.at[index, "dispensed_note_count"] = dispensed_notes
            snapshots.at[index, "available_note_count"] = available_notes
            snapshots.at[index, "available_amount"] = available_notes * denomination
            snapshots.at[index, "recycle_utilisation"] = round(available_notes / capacity_notes, 4)
            snapshots.at[index, "logical_balance"] = available_notes * denomination
            snapshots.at[index, "physical_balance"] = available_notes * denomination
            snapshots.at[index, "source_system"] = "Regional UAT low-cash scenario"
    selected_ids = {
        int(spec["terminal_id"])
        for spec in REGIONAL_TERMINALS
        if spec["low_cash_ratio"] is not None
    }
    replenishments = tables["fact_replenishment"]
    tables["fact_replenishment"] = replenishments.loc[
        ~(
            replenishments["terminal_id"].isin(selected_ids)
            & (replenishments["visit_date"] == latest_date)
        )
    ].reset_index(drop=True)
    tables["fact_replenishment"]["replenishment_id"] = range(
        1, len(tables["fact_replenishment"]) + 1
    )


def _distance_km(first: dict[str, float], second: dict[str, float]) -> float:
    radius = 6371.0
    lat1, lat2 = math.radians(first["latitude"]), math.radians(second["latitude"])
    delta_lat = lat2 - lat1
    delta_lon = math.radians(second["longitude"] - first["longitude"])
    value = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    )
    return radius * 2 * math.atan2(math.sqrt(value), math.sqrt(1 - value))


def _clock(value: object) -> time:
    """Parse the small set of time representations used by generated tables."""
    if isinstance(value, time):
        return value
    parsed = pd.Timestamp(str(value))
    return parsed.time()


def _route_unserved_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=[
            "route_id",
            "terminal_id",
            "requested_amount",
            "reason_code",
            "reason_detail",
        ],
    )


def build_regional_operations(
    tables: dict[str, pd.DataFrame],
    recommendation_detail: pd.DataFrame,
    as_of: pd.Timestamp,
) -> None:
    vault_inventory = tables["fact_vault_inventory"].copy()
    if (vault_inventory["opening_amount"] != vault_inventory["reserved_amount"] + vault_inventory["available_amount"]).any():
        raise ValueError("vault inventory reservation ledger is inconsistent")
    # Rebuilding a route replaces the previous plan. Release reservations for
    # cancelled/unserved stops before committing the new feasible allocation.
    vault_inventory["reserved_amount"] = 0
    vault_inventory["available_amount"] = vault_inventory["opening_amount"]
    vehicle = tables["dim_vehicle"].iloc[0]
    vehicle_capacity = int(vehicle["cash_capacity"])
    max_stops = int(vehicle["max_stops"])
    shift_start = _clock(vehicle["shift_start"])
    shift_end = _clock(vehicle["shift_end"])
    if vehicle_capacity <= 0 or max_stops <= 0 or shift_end <= shift_start:
        raise ValueError("vehicle capacity, max_stops, and shift window must be valid")

    required_detail = {"terminal_id", "denomination", "recommended_amount"}
    missing = required_detail - set(recommendation_detail.columns)
    if missing:
        raise ValueError(f"recommendation_detail is missing columns: {sorted(missing)}")
    detail = recommendation_detail.copy()
    if detail.empty:
        pass
    else:
        if detail[["terminal_id", "denomination", "recommended_amount"]].isna().any().any():
            raise ValueError("route recommendations cannot contain null identifiers or amounts")
        if detail.duplicated(["terminal_id", "denomination"]).any():
            raise ValueError("route recommendations must contain one row per terminal and denomination")
        for row in detail.itertuples(index=False):
            denomination = int(row.denomination)
            requested = int(row.recommended_amount)
            if denomination <= 0 or requested < 0 or requested % denomination:
                raise ValueError("route loads must be non-negative denomination multiples")
        known_terminals = set(tables["dim_terminal"]["terminal_id"].astype(int))
        unknown_terminals = set(detail["terminal_id"].astype(int)) - known_terminals
        if unknown_terminals:
            raise ValueError(f"route recommendations reference unknown terminals: {sorted(unknown_terminals)}")
        known_locations = set(tables["dim_location"]["location_id"].astype(int))
        missing_locations = known_terminals - known_locations
        if missing_locations:
            raise ValueError(f"terminal dimension is missing locations: {sorted(missing_locations)}")

    # A terminal is only a candidate when its complete denomination mix can be
    # funded. Partial reservations are not dispatched: they make the planned
    # mix and the vault ledger disagree.
    vault_available = {
        int(row.denomination): int(row.available_amount)
        for row in vault_inventory.itertuples(index=False)
    }
    candidates: list[dict[str, object]] = []
    unserved: list[dict[str, object]] = []
    for terminal_id, group in detail.groupby("terminal_id"):
        terminal_id = int(terminal_id)
        terminal_total = int(group["recommended_amount"].sum())
        if terminal_total <= 0:
            continue
        candidates.append({
            "terminal_id": terminal_id,
            "requested_amount": terminal_total,
            "group": group,
        })

    locations = tables["dim_location"].set_index("location_id")
    vault_location = locations.loc[VAULT_LOCATION_ID].to_dict()
    unvisited = {int(candidate["terminal_id"]) for candidate in candidates}
    ordered: list[int] = []
    current = vault_location
    while unvisited:
        terminal_id = min(
            unvisited,
            key=lambda item: _distance_km(current, locations.loc[item].to_dict()),
        )
        next_location = locations.loc[terminal_id].to_dict()
        leg_distance = _distance_km(current, next_location)
        ordered.append(terminal_id)
        current = next_location
        unvisited.remove(terminal_id)

    start = pd.Timestamp(datetime.combine(as_of.date(), shift_start))
    current_time = start
    route_stops: list[dict[str, object]] = []
    total_distance = 0.0
    accepted_candidates: list[dict[str, object]] = []
    last_location = vault_location
    for terminal_id in ordered:
        # A candidate may be rejected (window, capacity, or shift). Distances
        # and arrival times must therefore be measured from the last accepted
        # stop, never from a rejected candidate in the precomputed ordering.
        terminal_location = locations.loc[terminal_id].to_dict()
        distance = _distance_km(last_location, terminal_location)
        candidate = next(item for item in candidates if item["terminal_id"] == terminal_id)
        insufficient = [
            int(row.denomination)
            for row in candidate["group"].itertuples(index=False)
            if int(row.recommended_amount) > vault_available.get(int(row.denomination), 0)
        ]
        if insufficient:
            unserved.append({"route_id": 7001, "terminal_id": terminal_id,
                             "requested_amount": int(candidate["requested_amount"]),
                             "reason_code": "vault_insufficient_funding",
                             "reason_detail": f"Insufficient denominations: {sorted(set(insufficient))}"})
            continue
        travel_minutes = max(8, int(round(distance / 22 * 60)))
        arrival = current_time + timedelta(minutes=travel_minutes)
        terminal_rows = tables["dim_terminal"].loc[
            tables["dim_terminal"]["terminal_id"].astype(int) == terminal_id
        ]
        if terminal_rows.empty:
            unserved.append({"route_id": 7001, "terminal_id": terminal_id,
                             "requested_amount": int(candidate["requested_amount"]),
                             "reason_code": "unknown_terminal", "reason_detail": "No terminal dimension exists"})
            continue
        terminal_row = terminal_rows.iloc[0]
        window_start = pd.Timestamp(datetime.combine(as_of.date(), _clock(terminal_row["service_window_start"])))
        window_end = pd.Timestamp(datetime.combine(as_of.date(), _clock(terminal_row["service_window_end"])))
        service_start = max(arrival, window_start)
        if service_start > window_end:
            unserved.append({"route_id": 7001, "terminal_id": terminal_id,
                             "requested_amount": int(candidate["requested_amount"]),
                             "reason_code": "service_window_missed",
                             "reason_detail": f"Arrival/service start {service_start.isoformat()} is after {window_end.isoformat()}"})
            continue
        departure = service_start + timedelta(minutes=25)
        if departure > window_end:
            unserved.append({"route_id": 7001, "terminal_id": terminal_id,
                             "requested_amount": int(candidate["requested_amount"]),
                             "reason_code": "service_window_missed",
                             "reason_detail": f"Completion {departure.isoformat()} is after {window_end.isoformat()}"})
            continue
        if len(accepted_candidates) >= max_stops:
            unserved.append({"route_id": 7001, "terminal_id": terminal_id,
                             "requested_amount": int(candidate["requested_amount"]),
                             "reason_code": "vehicle_max_stops",
                             "reason_detail": f"Vehicle allows at most {max_stops} stops"})
            continue
        projected_load = sum(int(item["requested_amount"]) for item in accepted_candidates) + int(candidate["requested_amount"])
        if projected_load > vehicle_capacity:
            unserved.append({"route_id": 7001, "terminal_id": terminal_id,
                             "requested_amount": int(candidate["requested_amount"]),
                             "reason_code": "vehicle_capacity_exceeded",
                             "reason_detail": f"Projected load {projected_load} exceeds {vehicle_capacity}"})
            continue
        # The return leg is part of feasibility. If this candidate would push
        # the shift over its end, leave its reservation uncommitted.
        return_distance = _distance_km(locations.loc[terminal_id].to_dict(), vault_location)
        projected_end = departure + timedelta(minutes=max(8, int(round(return_distance / 22 * 60))))
        shift_end_ts = pd.Timestamp(datetime.combine(as_of.date(), shift_end))
        if projected_end > shift_end_ts:
            unserved.append({"route_id": 7001, "terminal_id": terminal_id,
                             "requested_amount": int(candidate["requested_amount"]),
                             "reason_code": "shift_end_exceeded",
                             "reason_detail": f"Return would finish at {projected_end.isoformat()} after {shift_end}"})
            continue
        terminal_allocations = candidate["group"]
        mix = {
            str(int(row.denomination)): int(row.recommended_amount)
            for row in terminal_allocations.itertuples(index=False)
            if int(row.recommended_amount) > 0
        }
        sequence = len(accepted_candidates) + 1
        route_stops.append(
            {
                "route_stop_id": 7100 + sequence,
                "route_id": 7001,
                "stop_sequence": sequence,
                "terminal_id": terminal_id,
                "planned_arrival_ts": service_start.isoformat(),
                "planned_departure_ts": departure.isoformat(),
                "travel_distance_km": round(distance, 2),
                "service_minutes": 25,
                "planned_load_amount": int(candidate["requested_amount"]),
                "planned_denomination_mix": json.dumps(mix, sort_keys=True),
                "status": "Planned",
            }
        )
        total_distance += distance
        current_time = departure
        accepted_candidates.append(candidate)
        for row in terminal_allocations.itertuples(index=False):
            denomination = int(row.denomination)
            vault_available[denomination] -= int(row.recommended_amount)
        last_location = terminal_location
    if accepted_candidates:
        final_location = last_location
        return_distance = _distance_km(final_location, vault_location)
        total_distance += return_distance
        current_time += timedelta(minutes=max(8, int(round(return_distance / 22 * 60))))

    planned_load = sum(int(item["requested_amount"]) for item in accepted_candidates)
    selected_terminals = {int(item["terminal_id"]) for item in accepted_candidates}
    allocations: list[dict[str, object]] = []
    allocation_id = 1
    for row in detail.sort_values(["terminal_id", "denomination"]).itertuples(index=False):
        requested = int(row.recommended_amount)
        if requested <= 0:
            continue
        allocated = requested if int(row.terminal_id) in selected_terminals else 0
        stock_matches = vault_inventory.index[vault_inventory["denomination"] == int(row.denomination)]
        if allocated and len(stock_matches) != 1:
            raise ValueError(f"vault has no unique stock row for denomination {row.denomination}")
        if allocated:
            stock_index = stock_matches[0]
            vault_inventory.at[stock_index, "reserved_amount"] += allocated
            vault_inventory.at[stock_index, "available_amount"] -= allocated
        matching_reason = next((item for item in unserved if int(item["terminal_id"]) == int(row.terminal_id)), None)
        allocations.append({
            "allocation_id": allocation_id,
            "vault_id": VAULT_ID,
            "terminal_id": int(row.terminal_id),
            "allocation_date": as_of.date(),
            "denomination": int(row.denomination),
            "requested_amount": requested,
            "allocated_amount": allocated,
            "status": "Allocated" if allocated == requested else f"Unserved: {matching_reason['reason_code'] if matching_reason else 'not_selected'}",
        })
        allocation_id += 1
    tables["fact_vault_inventory"] = vault_inventory
    tables["fact_vault_allocation"] = pd.DataFrame(allocations, columns=[
        "allocation_id", "vault_id", "terminal_id", "allocation_date", "denomination",
        "requested_amount", "allocated_amount", "status",
    ])
    tables["fact_route_unserved"] = _route_unserved_frame(unserved)
    tables["fact_route_plan"] = pd.DataFrame(
        [
            {
                "route_id": 7001,
                "cra_id": CRA_ID,
                "vault_id": VAULT_ID,
                "vehicle_id": VEHICLE_ID,
                "route_date": as_of.date(),
                "route_code": "RTE-MUM-CENTRAL-001",
                "planned_start_ts": start.isoformat(),
                "planned_end_ts": current_time.isoformat(),
                "planned_distance_km": round(total_distance, 2),
                "planned_load_amount": planned_load,
                "stop_count": len(route_stops),
                "vehicle_capacity_amount": int(tables["dim_vehicle"].iloc[0]["cash_capacity"]),
                "max_stops": max_stops,
                "status": (
                    "Ready for dispatch"
                    if route_stops and not unserved
                    else "Partial - unserved stops"
                    if route_stops and unserved
                    else "Blocked - no feasible stops"
                    if unserved
                    else "No service required"
                ),
            }
        ]
    )
    tables["fact_route_stop"] = pd.DataFrame(
        route_stops,
        columns=[
            "route_stop_id",
            "route_id",
            "stop_sequence",
            "terminal_id",
            "planned_arrival_ts",
            "planned_departure_ts",
            "travel_distance_km",
            "service_minutes",
            "planned_load_amount",
            "planned_denomination_mix",
            "status",
            
        ],
    )
