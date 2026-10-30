"""Build portal display masters from the tables this project actually uses.

Unknown operational fields stay blank. The legacy six-terminal demo is listed
as its own population and is not joined onto the 100 synthetic ATMs.

Run from the repository root:

    python3 scripts/prepare_portal_masters.py
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MASTER = DATA / "master"
CASH = MASTER / "cash_files"
LEGACY = MASTER / "legacy_demo"
OUT = MASTER / "portal"
SIM = ROOT / "simulated" / "maharashtra" / "data"

PORTAL_GROUP = {
    "day_of_week": "calendar",
    "month": "calendar",
    "quarter": "calendar",
    "is_weekend": "calendar",
    "is_payday": "payday",
    "is_holiday": "holiday",
    "days_to_month_end": "calendar",
    "is_month_start": "calendar",
    "is_month_end": "calendar",
    "lag_1": "demand_history",
    "lag_7": "demand_history",
    "lag_14": "demand_history",
    "rolling_mean_7": "demand_history",
    "rolling_mean_14": "demand_history",
    "rolling_std_7": "demand_history",
    "is_bank_closed_day": "holiday",
    "is_pre_holiday_3d": "holiday",
    "is_post_holiday_3d": "holiday",
    "local_event": "unavailable",
    "verified_site_type": "unavailable",
    "device_type": "unavailable",
    "capacity_amount": "unavailable",
    "uptime_pct": "unavailable",
    "failed_amount": "unavailable",
    "weather": "unavailable",
    "footfall": "unavailable",
    "terminal_id": "terminal_context",
    "denomination": "terminal_context",
    "site_type": "terminal_context",
    "is_recycler": "terminal_context",
}

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

README = """# Portal masters

Display tables for a later frontend. `index.json` is the entry point. It lists
every file written here, with size and SHA-256. `dataset.csv` lists those
files plus the canonical tables they were built from.

Two ATM populations stay separate. `cash_files` is terminal ids 1–100.
`legacy_demo` is terminal ids 1001–1006. Do not join them.

Blank cells are unknown. Device type, cassette capacity, verified site type,
coordinates, uptime, failed withdrawals, footfall, and local events were not
collected for the 100 synthetic ATMs. `unavailable.csv` is the list to show
instead of filling those fields.

`location_label` is the text after ` - ` in the location string. It is not a
verified site classification. State and RBI office are a curated proxy.
Regional holidays come from the RBI office matrix with a blank `known_at`, so
they are retrospective. Payday is the assumed 1st, 15th, and last-weekday rule.

`cash_summary.json` rolls up `net_demand` from `data/processed/daily_cash_out.csv`.
The denomination value 100 is a rounding unit.

Weather coordinates cover the four Maharashtra city centroids used by
`simulated/maharashtra/`. That series is ERA5 reanalysis. The governed feature
registry still marks weather unavailable. The switch field list applies only
to the SYNATM0001 mock extract.

Cassette rows live only in `data/master/legacy_demo/dim_cassette_configuration.csv`.

Regenerate with `python3 scripts/prepare_portal_masters.py`.
"""


def _money(value: Decimal, quant: str) -> str:
    return str(value.quantize(Decimal(quant), rounding=ROUND_HALF_UP))


def _crore(rupees: int) -> str:
    return _money(Decimal(int(rupees)) / Decimal(10_000_000), "0.01")


def _lakh(rupees: float) -> str:
    return _money(Decimal(str(rupees)) / Decimal(100_000), "0.01")


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, na_rep="")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _csv_rows(path: Path) -> int:
    with path.open(newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def _load() -> dict[str, pd.DataFrame]:
    return {
        "meta": pd.read_csv(CASH / "terminal_metadata.csv"),
        "geo": pd.read_csv(CASH / "terminal_geography.csv"),
        "profile": pd.read_csv(CASH / "terminal_profile.csv"),
        "calendar": pd.read_csv(CASH / "dim_calendar.csv", parse_dates=["calendar_date"]),
        "holidays": pd.read_csv(CASH / "holiday_calendar.csv"),
        "coverage": pd.read_csv(CASH / "holiday_coverage.csv"),
        "fixed_holidays": pd.read_csv(CASH / "holidays.csv", parse_dates=["calendar_date"]),
        "paydays": pd.read_csv(CASH / "paydays.csv", parse_dates=["calendar_date"]),
        "features": pd.read_csv(DATA / "feature_registry.csv"),
        "demand": pd.read_csv(DATA / "processed" / "daily_cash_out.csv", parse_dates=["transaction_date"]),
        "legacy_terminal": pd.read_csv(LEGACY / "dim_terminal.csv"),
        "legacy_cassette": pd.read_csv(LEGACY / "dim_cassette_configuration.csv"),
        "weather_place": pd.read_json(SIM / "weather_provenance.json"),
        "switch_fields": pd.read_csv(SIM / "switch_log_SYNATM0001_fields.csv"),
    }


def _check_inputs(tables: dict[str, pd.DataFrame]) -> None:
    meta = tables["meta"]
    geo = tables["geo"]
    profile = tables["profile"]
    demand = tables["demand"]
    if not meta.terminal_id.is_unique or len(meta) != 100:
        raise AssertionError("terminal metadata is not 100 unique ATMs")
    if set(meta.terminal_id) != set(geo.terminal_id) or set(meta.terminal_id) != set(profile.terminal_id):
        raise AssertionError("terminal, geography, and profile ids differ")
    if set(meta.terminal_id) & set(tables["legacy_terminal"].terminal_id):
        raise AssertionError("primary and legacy demo terminal ids overlap")
    paired = meta.merge(geo, on="terminal_id", suffixes=("_meta", "_geo"))
    if len(paired) != 100 or not (paired.city_meta == paired.city_geo).all():
        raise AssertionError("terminal city disagrees with geography")
    if meta.location.str.contains(" - ", regex=False).sum() != len(meta):
        raise AssertionError("a location string has no ' - ' separator")
    if len(demand) != 100 * len(tables["calendar"]) or demand.terminal_id.nunique() != 100:
        raise AssertionError("daily demand is not one row per ATM per calendar date")
    if set(demand.denomination.unique()) != {100}:
        raise AssertionError("denomination is no longer the constant rounding unit 100")
    missing_groups = set(tables["features"].feature) - set(PORTAL_GROUP)
    if missing_groups:
        raise AssertionError(f"feature registry has ungrouped features: {sorted(missing_groups)}")
    pilot_cities = set(paired.loc[paired.state == "Maharashtra", "city_meta"])
    if pilot_cities != set(tables["weather_place"].city):
        raise AssertionError("weather places do not match the Maharashtra cities")


def _atm_terminal(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frame = tables["meta"].merge(tables["geo"], on=["terminal_id", "city"], how="inner").merge(
        tables["profile"][
            ["terminal_id", "verified_site_type", "device_type", "capacity_amount", "verification_status"]
        ],
        on="terminal_id",
        how="inner",
    )
    frame["location_label"] = frame.location.str.split(" - ", n=1).str[1]
    frame["population"] = "cash_files"
    columns = [
        "terminal_id",
        "terminal_code",
        "city",
        "state",
        "rbi_region",
        "timezone",
        "location",
        "location_label",
        "site_type",
        "mapping_status",
        "source",
        "effective_from",
        "effective_to",
        "verified_site_type",
        "device_type",
        "capacity_amount",
        "verification_status",
        "population",
    ]
    return frame[columns].sort_values("terminal_id")


def _city(terminals: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    places = weather.drop(columns=["timezone"]).rename(
        columns={
            "latitude": "weather_latitude",
            "longitude": "weather_longitude",
            "elevation_m": "weather_elevation_m",
            "provider": "weather_provider",
            "retrieved_for_this_run": "weather_retrieved_on",
            "start": "weather_start",
            "end": "weather_end",
            "rows": "weather_days",
        }
    )
    frame = terminals.groupby(["city", "state", "rbi_region", "timezone"], as_index=False).agg(
        atm_count=("terminal_id", "size")
    )
    if frame.city.duplicated().any():
        raise AssertionError("a city maps to more than one state or RBI office")
    frame = frame.merge(places, on="city", how="left")
    frame["weather_series"] = frame.weather_provider.notna().map({True: "maharashtra_experiment", False: ""})
    frame["weather_provider"] = frame.weather_provider.fillna("")
    frame["weather_days"] = frame.weather_days.astype("Int64")
    keep = [
        "city",
        "state",
        "rbi_region",
        "timezone",
        "atm_count",
        "weather_series",
        "weather_latitude",
        "weather_longitude",
        "weather_elevation_m",
        "weather_provider",
        "weather_retrieved_on",
        "weather_start",
        "weather_end",
        "weather_days",
    ]
    return frame[keep].sort_values("city")


def _state(terminals: pd.DataFrame) -> pd.DataFrame:
    return (
        terminals.groupby("state", as_index=False)
        .agg(
            atm_count=("terminal_id", "size"),
            city_count=("city", "nunique"),
            rbi_regions=("rbi_region", lambda values: " | ".join(sorted(set(values)))),
        )
        .sort_values(["atm_count", "state"], ascending=[False, True])
    )


def _offices(tables: dict[str, pd.DataFrame], terminals: pd.DataFrame) -> pd.DataFrame:
    assigned = terminals.groupby("rbi_region", as_index=False).agg(
        atm_count=("terminal_id", "size"),
        states=("state", lambda values: " | ".join(sorted(set(values)))),
        cities=("city", lambda values: " | ".join(sorted(set(values)))),
    )
    holidays = tables["holidays"].groupby("region").size().rename("holiday_rows")
    coverage = tables["coverage"].groupby("region").size().rename("coverage_rows")
    complete = (
        tables["coverage"].loc[tables["coverage"].status == "complete"].groupby("region").size().rename("coverage_complete_rows")
    )
    offices = pd.Index(sorted(set(assigned.rbi_region) | set(holidays.index) | set(coverage.index)), name="rbi_region")
    frame = pd.DataFrame({"rbi_region": offices}).merge(assigned, on="rbi_region", how="left")
    frame = frame.merge(holidays, left_on="rbi_region", right_index=True, how="left")
    frame = frame.merge(coverage, left_on="rbi_region", right_index=True, how="left")
    frame = frame.merge(complete, left_on="rbi_region", right_index=True, how="left")
    for column in ("atm_count", "holiday_rows", "coverage_rows", "coverage_complete_rows"):
        frame[column] = frame[column].fillna(0).astype(int)
    frame["in_terminal_geography"] = frame.atm_count.gt(0).map({True: "yes", False: "no"})
    frame["states"] = frame.states.fillna("")
    frame["cities"] = frame.cities.fillna("")
    return frame.sort_values("rbi_region")


def _location_label(terminals: pd.DataFrame) -> pd.DataFrame:
    frame = terminals.groupby("location_label", as_index=False).agg(
        atm_count=("terminal_id", "size"),
        city_count=("city", "nunique"),
    )
    frame.insert(1, "verification_status", "parsed_from_location_string")
    return frame.sort_values(["atm_count", "location_label"], ascending=[False, True])


def _features(features: pd.DataFrame) -> pd.DataFrame:
    frame = features.copy()
    frame.insert(1, "portal_group", frame.feature.map(PORTAL_GROUP))
    return frame


def _denomination() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "denomination_code": 100,
                "currency": "INR",
                "meaning": "rounding_unit",
                "observed_note_mix": "no",
                "population": "cash_files",
                "source": "data/processed/daily_cash_out.csv",
                "note": "Every primary demand row carries denomination 100. That value is a rounding unit, not a count of 100-rupee notes.",
            }
        ]
    )


def _population(terminals: pd.DataFrame, legacy: pd.DataFrame) -> pd.DataFrame:
    pilot = terminals.loc[terminals.state == "Maharashtra", "terminal_id"]
    return pd.DataFrame(
        [
            {
                "population_id": "cash_files",
                "what": "100 synthetic cash-file ATMs",
                "terminal_ids": "1-100",
                "terminal_count": int(len(terminals)),
                "join_key": "terminal_id",
                "do_not_join_with": "legacy_demo",
            },
            {
                "population_id": "maharashtra_pilot",
                "what": "Maharashtra subset of cash_files used by the simulated study",
                "terminal_ids": ",".join(str(value) for value in pilot.tolist()),
                "terminal_count": int(len(pilot)),
                "join_key": "terminal_id",
                "do_not_join_with": "legacy_demo",
            },
            {
                "population_id": "legacy_demo",
                "what": "Six-terminal operations demo",
                "terminal_ids": ",".join(str(value) for value in legacy.terminal_id.tolist()),
                "terminal_count": int(len(legacy)),
                "join_key": "terminal_id",
                "do_not_join_with": "cash_files",
            },
        ]
    )


def _pilot(terminals: pd.DataFrame) -> pd.DataFrame:
    frame = terminals.loc[terminals.state == "Maharashtra"].copy()
    frame["weather_series"] = "maharashtra_experiment"
    keep = [
        "terminal_id",
        "terminal_code",
        "city",
        "state",
        "rbi_region",
        "location",
        "location_label",
        "weather_series",
        "population",
    ]
    return frame[keep].sort_values("terminal_id")


def _switch_fields(fields: pd.DataFrame) -> pd.DataFrame:
    frame = fields.copy()
    frame.insert(0, "applies_to", "SYNATM0001 mock extract only")
    frame["fleet_master"] = "no"
    return frame


def _unavailable() -> pd.DataFrame:
    rows = [
        ("device_type", "terminal", "blank on all 100 cash-file ATMs", "Do not show a dispenser or recycler type."),
        ("capacity_amount", "terminal", "blank on all 100 cash-file ATMs", "Do not show a rupee cassette capacity."),
        ("verified_site_type", "terminal", "blank on all 100 cash-file ATMs", "Show location_label as a parsed string, not a verified site class."),
        ("cassette_configuration", "terminal", "present only on legacy_demo terminals 1001-1006", "Do not copy demo cassettes onto SYNATM terminals."),
        ("local_event", "city/date", "schema only, zero rows", "An empty table is not a confirmed quiet day."),
        ("uptime_pct", "ATM/date", "availability_daily.csv has a header and zero rows", "Do not show uptime as observed."),
        ("failed_amount", "ATM/date", "not in the cash-file snapshots", "Cash_Out is the demand target. It is not failed-withdrawal demand."),
        ("footfall", "site/date", "not collected", "Airport passengers and events are not ATM footfall."),
        ("coordinates", "terminal", "not on the terminal master", "Weather coordinates are city centroids for four Maharashtra cities."),
        ("bank_applicability", "holiday", "RBI office matrix, known_at blank", "Regional holidays are retrospective. They are not a verified bank calendar."),
        ("salary_calendar", "date", "one assumed rule", "Payday is 1st, 15th, and last weekday. It is not a verified salary file."),
    ]
    return pd.DataFrame(rows, columns=["field", "grain", "status", "portal_rule"])


def _cash_summary(tables: dict[str, pd.DataFrame]) -> dict:
    demand = tables["demand"]
    calendar = tables["calendar"][["calendar_date", "day_of_week", "is_payday"]]
    joined = demand.merge(calendar, left_on="transaction_date", right_on="calendar_date", how="left")
    if joined.calendar_date.isna().any():
        raise AssertionError("demand dates are missing from the calendar master")
    total = int(joined.net_demand.sum())
    atm_days = int(len(joined))
    days = int(joined.transaction_date.nunique())
    payday = joined.loc[joined.is_payday == 1, "net_demand"]
    other = joined.loc[joined.is_payday == 0, "net_demand"]
    if int((tables["calendar"].is_payday == 1).sum()) != int(tables["paydays"].calendar_date.nunique()):
        raise AssertionError("calendar payday flags do not match paydays.csv")
    monthly = []
    for period, group in joined.groupby(joined.transaction_date.dt.to_period("M"), sort=True):
        amount = int(group.net_demand.sum())
        monthly.append(
            {
                "year_month": str(period),
                "calendar_days": int(group.transaction_date.nunique()),
                "atm_days": int(len(group)),
                "total_inr": amount,
                "total_crore": _crore(amount),
                "mean_atm_day_inr": round(float(group.net_demand.mean()), 2),
                "mean_atm_day_lakh": _lakh(float(group.net_demand.mean())),
            }
        )
    by_day = []
    for day_of_week, group in joined.groupby("day_of_week", sort=True):
        amount = int(group.net_demand.sum())
        by_day.append(
            {
                "day_of_week": int(day_of_week),
                "day_name": DAY_NAMES[int(day_of_week)],
                "atm_days": int(len(group)),
                "total_inr": amount,
                "total_crore": _crore(amount),
                "share_of_total": round(amount / total, 4),
                "mean_atm_day_inr": round(float(group.net_demand.mean()), 2),
                "mean_atm_day_lakh": _lakh(float(group.net_demand.mean())),
            }
        )
    payday_mean = float(payday.mean())
    other_mean = float(other.mean())
    return {
        "population": "cash_files",
        "source": "data/processed/daily_cash_out.csv",
        "measure": "net_demand",
        "target_semantics": "cash_out_equals_withdrawal_demand",
        "currency": "INR",
        "terminals": 100,
        "calendar_days": days,
        "atm_days": atm_days,
        "start": str(joined.transaction_date.min().date()),
        "end": str(joined.transaction_date.max().date()),
        "total_inr": total,
        "total_crore": _crore(total),
        "mean_atm_day_inr": round(total / atm_days, 2),
        "mean_atm_day_lakh": _lakh(total / atm_days),
        "mean_fleet_day_inr": round(total / days, 2),
        "mean_fleet_day_crore": _crore(int(round(total / days))),
        "monthly": monthly,
        "day_of_week": by_day,
        "payday": {
            "rule_id": "legacy-payday-v1",
            "rule": "1st, 15th, and last weekday; weekend month-end rolls backward; holidays are not removed",
            "verification_status": "assumption",
            "payday_dates": int(tables["paydays"].calendar_date.nunique()),
            "atm_days": int(payday.shape[0]),
            "total_inr": int(payday.sum()),
            "total_crore": _crore(int(payday.sum())),
            "mean_atm_day_inr": round(payday_mean, 2),
            "mean_atm_day_lakh": _lakh(payday_mean),
            "other_atm_days": int(other.shape[0]),
            "other_total_inr": int(other.sum()),
            "other_mean_atm_day_inr": round(other_mean, 2),
            "other_mean_atm_day_lakh": _lakh(other_mean),
            "mean_lift_vs_other": round(payday_mean / other_mean - 1, 4),
        },
    }


def _dataset_rows(written: dict[str, int]) -> pd.DataFrame:
    raw_archives = list(DATA.glob("raw/ATM_Raw_15Min_XLSX_CashFiles_PART*/*.tar.gz"))
    snapshot = DATA / "raw" / "mumbai_public" / "2026-10-21"
    mumbai_files = [path for path in snapshot.iterdir() if path.is_file() and path.name != "sources.json"]
    rows = [
        ("atm_terminal", "master", "cash_files", "data/master/portal/atm_terminal.csv", written["atm_terminal"], "terminal", "portal display", "prepared"),
        ("city", "master", "cash_files", "data/master/portal/city.csv", written["city"], "city", "portal display", "prepared"),
        ("state", "master", "cash_files", "data/master/portal/state.csv", written["state"], "state", "portal display", "prepared"),
        ("rbi_region", "master", "cash_files", "data/master/portal/rbi_region.csv", written["rbi_region"], "RBI office", "portal display", "prepared"),
        ("location_label", "master", "cash_files", "data/master/portal/location_label.csv", written["location_label"], "parsed location label", "portal display", "parsed, not verified"),
        ("denomination", "master", "cash_files", "data/master/portal/denomination.csv", written["denomination"], "demand column", "portal display", "rounding unit"),
        ("feature", "master", "cash_files", "data/master/portal/feature.csv", written["feature"], "feature", "model registry", "prepared"),
        ("population", "master", "all", "data/master/portal/population.csv", written["population"], "population", "portal display", "prepared"),
        ("maharashtra_pilot_atm", "master", "maharashtra_pilot", "data/master/portal/maharashtra_pilot_atm.csv", written["maharashtra_pilot_atm"], "terminal", "Maharashtra study", "prepared"),
        ("weather_place", "master", "maharashtra_pilot", "data/master/portal/weather_place.csv", written["weather_place"], "city centroid", "Maharashtra study", "ERA5 reanalysis"),
        ("switch_field", "master", "mock_switch", "data/master/portal/switch_field.csv", written["switch_field"], "field", "SYNATM0001 mock extract", "mock"),
        ("unavailable", "master", "cash_files", "data/master/portal/unavailable.csv", written["unavailable"], "field", "portal display", "explicit gap"),
        ("terminal_metadata", "master", "cash_files", "data/master/cash_files/terminal_metadata.csv", _csv_rows(CASH / "terminal_metadata.csv"), "terminal", "ingestion", "canonical"),
        ("terminal_geography", "master", "cash_files", "data/master/cash_files/terminal_geography.csv", _csv_rows(CASH / "terminal_geography.csv"), "terminal", "holiday join", "curated region proxy"),
        ("terminal_profile", "master", "cash_files", "data/master/cash_files/terminal_profile.csv", _csv_rows(CASH / "terminal_profile.csv"), "terminal", "not used as a model feature", "unknowns blank"),
        ("dim_calendar", "master", "cash_files", "data/master/cash_files/dim_calendar.csv", _csv_rows(CASH / "dim_calendar.csv"), "date", "calendar features and payday flag", "canonical"),
        ("holiday_calendar", "master", "cash_files", "data/master/cash_files/holiday_calendar.csv", _csv_rows(CASH / "holiday_calendar.csv"), "date/region", "regional holiday flag", "retrospective"),
        ("holiday_coverage", "master", "cash_files", "data/master/cash_files/holiday_coverage.csv", _csv_rows(CASH / "holiday_coverage.csv"), "region/month", "coverage check", "canonical"),
        ("holidays", "master", "cash_files", "data/master/cash_files/holidays.csv", _csv_rows(CASH / "holidays.csv"), "date", "legacy three-holiday rule", "canonical"),
        ("paydays", "master", "cash_files", "data/master/cash_files/paydays.csv", _csv_rows(CASH / "paydays.csv"), "date", "payday flag", "assumption"),
        ("payday_rules", "master", "cash_files", "data/master/cash_files/payday_rules.csv", _csv_rows(CASH / "payday_rules.csv"), "rule", "payday flag", "assumption"),
        ("local_events", "master", "cash_files", "data/master/cash_files/local_events.csv", _csv_rows(CASH / "local_events.csv"), "event", "not used", "unavailable"),
        ("daily_cash_out", "fact", "cash_files", "data/processed/daily_cash_out.csv", _csv_rows(DATA / "processed" / "daily_cash_out.csv"), "ATM/date", "demand target", "synthetic"),
        ("demand_features", "fact", "cash_files", "data/processed/demand_features.csv", _csv_rows(DATA / "processed" / "demand_features.csv"), "ATM/date", "legacy training matrix", "derived"),
        ("inventory_checkpoint_quality", "fact", "cash_files", "data/processed/inventory_checkpoint_quality.csv", _csv_rows(DATA / "processed" / "inventory_checkpoint_quality.csv"), "ATM/date", "snapshot quality", "not downtime"),
        ("availability_daily", "fact", "cash_files", "data/processed/availability_daily.csv", _csv_rows(DATA / "processed" / "availability_daily.csv"), "ATM/date", "not used", "unavailable"),
        ("raw_cash_archives", "raw", "cash_files", "data/raw/ATM_Raw_15Min_XLSX_CashFiles_PART*", len(raw_archives), "monthly archive", "daily demand source", "retained"),
        ("mumbai_public_snapshot", "raw", "mumbai_public", "data/raw/mumbai_public/2026-10-21/", len(mumbai_files), "source file", "not extracted in this workspace", "retained"),
        ("weather_city_daily", "fact", "maharashtra_pilot", "simulated/maharashtra/data/weather_city_daily.csv", _csv_rows(SIM / "weather_city_daily.csv"), "city/date", "Maharashtra study", "ERA5 reanalysis"),
        ("atm_daily", "fact", "maharashtra_pilot", "simulated/maharashtra/data/atm_daily.csv", _csv_rows(SIM / "atm_daily.csv"), "ATM/date", "Maharashtra study and layman experiment 4", "derived"),
        ("switch_log_SYNATM0001", "fact", "mock_switch", "simulated/maharashtra/data/switch_log_SYNATM0001.csv", _csv_rows(SIM / "switch_log_SYNATM0001.csv"), "transaction", "mock extract for one ATM", "mock"),
    ]
    for path in sorted(LEGACY.glob("dim_*.csv")):
        rows.append(
            (
                f"legacy_{path.stem}",
                "master",
                "legacy_demo",
                f"data/master/legacy_demo/{path.name}",
                _csv_rows(path),
                "legacy demo",
                "legacy dashboard only",
                "separate population",
            )
        )
    return pd.DataFrame(
        rows,
        columns=["dataset_id", "kind", "population", "path", "rows", "grain", "used_by", "status"],
    )


def _index(summary: dict, tables: dict[str, pd.DataFrame]) -> dict:
    cassette = tables["legacy_cassette"]
    files = []
    for path in sorted(OUT.iterdir()):
        if path.name == "index.json" or not path.is_file():
            continue
        files.append(
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return {
        "prepared_on": date.today().isoformat(),
        "entry_point": "data/master/portal/index.json",
        "readme": "data/master/portal/README.md",
        "dataset_catalog": "data/master/portal/dataset.csv",
        "populations_do_not_join": ["cash_files", "legacy_demo"],
        "cash_summary": "data/master/portal/cash_summary.json",
        "cash_total_inr": summary["total_inr"],
        "cash_total_crore": summary["total_crore"],
        "legacy_fixed_holidays": sorted(tables["fixed_holidays"].calendar_date.dt.strftime("%Y-%m-%d").tolist()),
        "legacy_demo_cassette": {
            "path": "data/master/legacy_demo/dim_cassette_configuration.csv",
            "rows": int(len(cassette)),
            "terminals": sorted(int(value) for value in cassette.terminal_id.unique()),
            "denominations_inr": sorted(int(value) for value in cassette.denomination.unique()),
            "applies_to": "legacy_demo only",
        },
        "files": files,
    }


def main() -> None:
    tables = _load()
    _check_inputs(tables)
    terminals = _atm_terminal(tables)
    if len(terminals) != 100:
        raise AssertionError("terminal join did not keep one row per ATM")
    OUT.mkdir(parents=True, exist_ok=True)
    written = {
        "atm_terminal": terminals,
        "city": _city(terminals, tables["weather_place"]),
        "state": _state(terminals),
        "rbi_region": _offices(tables, terminals),
        "location_label": _location_label(terminals),
        "denomination": _denomination(),
        "feature": _features(tables["features"]),
        "population": _population(terminals, tables["legacy_terminal"]),
        "maharashtra_pilot_atm": _pilot(terminals),
        "weather_place": tables["weather_place"].sort_values("city"),
        "switch_field": _switch_fields(tables["switch_fields"]),
        "unavailable": _unavailable(),
    }
    counts = {}
    for name, frame in written.items():
        _write_csv(frame, OUT / f"{name}.csv")
        counts[name] = int(len(frame))
    if counts["maharashtra_pilot_atm"] != 8 or counts["weather_place"] != 4:
        raise AssertionError("pilot or weather master has an unexpected row count")
    _write_csv(_dataset_rows(counts), OUT / "dataset.csv")
    summary = _cash_summary(tables)
    if summary["atm_days"] != 75400 or summary["payday"]["atm_days"] != 7400:
        raise AssertionError("cash summary row counts drifted from the source files")
    (OUT / "cash_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "README.md").write_text(README)
    (OUT / "index.json").write_text(json.dumps(_index(summary, tables), indent=2) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"terminals {counts['atm_terminal']} cities {counts['city']} states {counts['state']} offices {counts['rbi_region']}")
    print(f"demand {summary['total_inr']} INR ({summary['total_crore']} Cr) over {summary['atm_days']} ATM-days")
    print(f"payday lift vs other days {summary['payday']['mean_lift_vs_other']}")


if __name__ == "__main__":
    main()
