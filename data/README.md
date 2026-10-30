# Project data

| Location | Contents | Origin |
| --- | --- | --- |
| `raw/ATM_Raw_15Min_XLSX_CashFiles_PART*/` | 25 monthly archives, original part READMEs | Supplied synthetic 15-minute XLSX snapshots; unchanged |
| `master/cash_files/terminal_metadata.csv` | 100 terminal identifiers, codes, cities and locations | Existing raw-cash ingestion output |
| `master/cash_files/dim_calendar.csv` | 754 dates and model calendar flags | `src/atm_cash_forecast/calendar.py` |
| `master/cash_files/holidays.csv` | Six dates in the observed period | The model's three fixed annual holidays |
| `master/cash_files/paydays.csv` | 74 dates in the observed period | Model rule: 1st, 15th, last weekday of month |
| `master/legacy_demo/dim_*.csv` | All nine demo master tables | Snapshots of `outputs/postgres_csv/dim_*.csv` |
| `processed/daily_cash_out.csv` | 75,400 ATM-day rows | Daily sum of snapshot `Cash_Out` |
| `catalog.json` | Relative paths, sizes, SHA-256, CSV columns and row counts | `scripts/prepare_data.py` |
| `raw/reference/rbi/` | 29 retained monthly RBI holiday matrices and source metadata | Public RBI regional holiday matrix |
| `master/cash_files/holiday_calendar.csv` | Sourced regional holiday occurrences | Validated retained RBI responses |
| `master/cash_files/holiday_coverage.csv` | Explicit region/month completeness | Validated retained RBI responses |
| `master/cash_files/terminal_geography.csv` | City/state and regional office proxies for all 100 ATMs | Curated mapping, bank applicability unverified |
| `master/cash_files/payday_rules.csv` | Named salary-window assumptions | Original model rule |
| `master/cash_files/local_events.csv`, `terminal_profile.csv` | Schemas with unknown/uncollected fields | No fabricated observations |
| `references/` | Immutable calendar parity and regional feature bundles | `scripts/prepare_calendar_reference.py` |
| `feature_registry.csv`, `reference_coverage.json` | Feature definitions and reference coverage/gaps | Reference preparation |
| `processed/inventory_checkpoint_quality.csv` | Per-ATM/day observed counts and min/max end cash, once exported | Raw snapshot quality extraction |
| `processed/demand_features.csv`, `demand_features.manifest.json` | Reviewable training matrix and source/reference identity, once exported | Shared feature builder |
| `baselines/calendar-v1-2026-10-21/` | Frozen original source, masters and reported metrics | Pre-change baseline |
| `master/portal/` | Display masters, dataset catalog, and cash rollup for a later frontend | `scripts/prepare_portal_masters.py` |

The primary dataset covers 2024-09-01 through 2026-09-24. These are synthetic
cash snapshots, not customer transaction events. The daily target treats
`Cash_Out` as withdrawal demand. The value `100` in its denomination column
is a rounding unit, not a measured denomination split.

Keep the primary 100-ATM masters separate from the six-terminal demo masters.
In the current files the ids do not overlap: cash-file terminals are 1–100 and
the demo terminals are 1001–1006. They still identify different populations.
`master/portal/index.json` is the display catalog. Unknown device, capacity,
cassette, and event fields stay blank there. The demo
tables include calendar, cassette configuration, CRA, CRM module, location,
region, terminal, vault, and vehicle. They are reference snapshots; the legacy
dashboard and PostgreSQL loader still read the generated exports in `outputs/`.

The original `dim_calendar.csv`, `holidays.csv` and `paydays.csv` remain legacy
rule exports. The richer `holiday_calendar.csv` and immutable `references/`
bundles are the new governed-input path. The new evaluation workflow consumes
these bundles in both training and forecasting. Named legacy commands retain
their defaults unless a reference is supplied. No primary ATM capacity or
device-type master has been invented from the legacy demo.

RBI source coverage is distinct from bank-specific applicability. Historical
regional results are labeled retrospective because original announcement dates
are unknown. See [the implementation guide](../docs/CALENDAR_FEATURE_IMPLEMENTATION.md)
for source provenance, assumptions, preparation, evaluation and serving commands.

Run `.venv/bin/python scripts/prepare_data.py` to refresh calendar/reference
tables, validate and prepare sourced reference bundles, copy current demo masters,
and rebuild the catalog. Add `--rebuild-raw`
to re-read all raw workbooks and reconstruct daily demand and terminal metadata
without training forecasting models. This also writes `raw_audit.json`.
The full benchmark also refreshes the primary data tables; run `prepare_data.py`
afterward to refresh the catalog.

Reports, evaluation results, saved models, operational facts and databases belong
in `outputs/`. Raw archives and processed CSVs are ignored by Git; retain them
separately when transferring the project. Small masters and the catalog remain
versionable. No original data was deleted during organization.
