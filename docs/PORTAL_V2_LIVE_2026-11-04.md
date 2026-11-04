# v2 is the operational portal (4 Nov 2026)

`live_portal_v2` is now the portal to use. `live_portal` (v1) is a demo: its approvals, dispatches and audit entries live only in the browser, and its route, van and site details are made up. A banner at the top of v1 now says so.

## What was wrong in v2

Five inputs to the indent amount were typed by hand or came from an older model:

| Input | Before | Now |
|---|---|---|
| Forecast | `strategy_c` | `lgb_holiday`, the 100-ATM holiday-aware model |
| Safety buffer | Assumed demand varies ±22% a day, then a normal z-score | Backtest P99 multiplier on totals of the same length as the refill cycle, chosen by holiday or quiet window (`demand_model.buffer_multiplier(..., days)`) |
| Refill cycle | Typed: 7 days (CIT), 3 days (branch) | Measured from the raw data: 1 day for 93 ATMs, 2 days for 7 |
| Capacity | ₹28.6L for every ATM (invented cassette mix) | Highest balance each ATM held in the 28 days to the cutoff (₹17.8–28.8L for ATMs 1–8) |
| Opening cash | Typed `last_feed` values (ATM 1: ₹15.0L) | `End_Cash` in the raw 24 Aug 23:45 snapshot (ATM 1: ₹23.2L) |
| Demand events | Four invented events (salary ×1.28, Ganesh ×1.45, …) | None by default. `lgb_holiday` already models paydays and holidays, so these counted them twice. Operators can still add events. |
| Refill date | Typed `suggested_date` (26 Aug), so a 7-day cycle only covered 6 days | The cycle start (25 Aug), matching the week the buffer was calibrated on |

v1 also had an error: it used the day's **highest** cash reading as opening cash (ATM 1: ₹26.0L), not the end-of-day balance. Both portals now read the 23:45 snapshot.

Also changed:
- The last-feed figure and cassette note counts now add up to the real balance. The per-slot split is still the typed one, rescaled, because the raw files have no denomination breakdown.
- The copilot's "how was this indent worked out" answer and the formula card now describe the calibrated buffer.
- The map key is no longer in the code or the data bundle. Public CARTO basemaps need no key. **Rotate the old key**, because it was committed.
- `generate_portal_data.py` now writes the bundle to both portals, so they can't drift apart.

## Measured refill cycles

`python3 scripts/measure_refill_cadence.py` (~15 min) reads every 15-minute snapshot from 28 Jul to 24 Aug 2026 and writes `data/processed/refill_cadence.csv`. A refill is a rise in `End_Cash` of more than ₹1L between two snapshots, since withdrawals can only lower it.

| ATM | Refills a week | Cycle | Highest balance |
|---|---|---|---|
| 1 | 5.3 | 1 day | ₹26.4L |
| 2 | 6.3 | 1 day | ₹19.8L |
| 3 | 6.0 | 1 day | ₹23.1L |
| 4 | 4.0 | 2 days | ₹17.8L |
| 5 | 4.0 | 2 days | ₹28.1L |
| 6 | 8.3 | 1 day | ₹20.3L |
| 7 | 5.8 | 1 day | ₹28.8L |
| 8 | 5.3 | 1 day | ₹20.5L |

The cycle is 7 divided by refills per week, rounded. Across all 100 ATMs, 93 come out at 1 day and 7 at 2 days.

The buffer gets wider as the cycle gets shorter, because one day is noisier than a week:

| Cycle | Quiet multiplier | Near a holiday |
|---|---|---|
| 1 day | ×1.55 | ×1.70 |
| 2 days | ×1.40 | ×1.48 |
| 7 days | ×1.24 | ×1.32 |

**What it means for the 25 Aug plan:** seven of the eight ATMs already hold enough for one day plus the buffer, so they need ₹0. ATM 2 needs ₹56,000. Before this change, v2 planned large weekly loads that would not fit in the cassette.

## Still assumed (no data source)

- **Capacity.** The highest balance seen is a lower bound on the real cassette limit. The engine's ₹60L default is also an assumption, which is why its "123 stockout cycles" figure is not a physical floor.
- **CRA and risk flags.** The typed `cra_tracking`, `human_errors` and `reject_bin` values feed the urgency status and the "emergency hot dispatch" recommendation. For example, ATM 3 shows CRITICAL with a ₹0 indent. Treat urgency as illustrative until those values come from real logs.
- **Surplus sweeps.** "Surplus" means more than 1.5 cycles of cash. On a 1-day cycle, most ATMs holding over ₹20L qualify.
- **v1 and the replenishment engine** still plan weekly cycles.
- **Display-only details** typed in `TERMINAL_OPERATIONAL_METADATA`: van, vault distance, CRA status, threat level, telemetry, ticket size, addresses, reject bins. They don't change the indent, but they aren't real.
- **The copilot's feature list** ("30 ML model variables", monsoon index and so on) describes an older model, not `lgb_holiday`.
- Sign-in identifies the officer. Roles come from `users.json`.

## How to check it

```bash
python3 scripts/measure_refill_cadence.py         # refill cadence + highest balance (~15 min)
python3 live_portal/generate_portal_data.py       # rebuilds both portal bundles (~2 min)
python3 -m unittest tests.test_audit_remediation  # 23 tests
python3 tests/test_buffer_week_type.py
python3 tests/test_forecaster_leakage.py
```

In v2, ATM 1 should show a usable balance of ₹23,19,500 and a last feed of "2026-08-24 23:45 IST (raw snapshot)".

Test changes:
- **A5** used to expect ATM 4 to need ₹0, because its typed cash was ₹26.6L. It now checks that ATM 4 is sized on its measured cycle only, with no phantom second cycle.
- **A1** now checks that capacity equals the highest measured balance and that no indent loads above it.
- **B8** approves ATM 2, the CIT ATM that needs cash on a daily cycle. ATM 1 now needs ₹0, so a manifest with only ATM 1 approved has nothing to lock.
