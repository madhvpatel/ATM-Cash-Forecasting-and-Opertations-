# v2 is the operational portal (4 Nov 2026)

`live_portal_v2` is now the portal to use. `live_portal` (v1) is a demo: its approvals, dispatches and audit entries live only in the browser, and its route, van and site details are made up. A banner at the top of v1 now says so.

## What was wrong in v2

Five inputs to the indent amount were typed by hand or came from an older model:

| Input | Before | Now |
|---|---|---|
| Forecast | `strategy_c` | `lgb_holiday`, the 100-ATM holiday-aware model |
| Safety buffer | Assumed demand varies ±22% a day, then a normal z-score | Backtest P99 multiplier, chosen by holiday or quiet week (`demand_model.cycle_multiplier`); scaled by √(7/days) for 3-day cycles |
| Opening cash | Typed `last_feed` values (ATM 1: ₹15.0L) | `End_Cash` in the raw 24 Aug 23:45 snapshot (ATM 1: ₹23.2L) |
| Demand events | Four invented events (salary ×1.28, Ganesh ×1.45, …) | None by default. `lgb_holiday` already models paydays and holidays, so these counted them twice. Operators can still add events. |
| Refill date | Typed `suggested_date` (26 Aug), so a 7-day cycle only covered 6 days | The cycle start (25 Aug), matching the week the buffer was calibrated on |

v1 also had an error: it used the day's **highest** cash reading as opening cash (ATM 1: ₹26.0L), not the end-of-day balance. Both portals now read the 23:45 snapshot.

Also changed:
- The last-feed figure and cassette note counts now add up to the real balance. The per-slot split is still the typed one, rescaled, because the raw files have no denomination breakdown.
- The copilot's "how was this indent worked out" answer and the formula card now describe the calibrated buffer.
- The map key is no longer in the code or the data bundle. Public CARTO basemaps need no key. **Rotate the old key**, because it was committed.
- `generate_portal_data.py` now writes the bundle to both portals, so they can't drift apart.

## Still assumed (no data source)

- **Refill frequency.** v2 assumes 7-day CIT and 3-day branch cycles. The raw data shows ATMs 1–8 refilled **3 to 8 times a week** (18–24 Aug: a jump of more than ₹1L in `End_Cash` between snapshots). With about ₹79L of weekly demand and a ₹28.6L cassette, ATM 1 cannot last a week on one visit. Treat v2's weekly plans as an assumption until this is settled.
- **Capacity.** ₹28.6L (4 cassettes × 2,200 notes) is an assumption. The highest balance ever reported for these ATMs is ₹29.1L (ATM 7), which slightly exceeds it. The engine's ₹60L default is also an assumption, which is why the engine's "123 stockout cycles" figure is not a physical floor.
- **Display-only details** typed in `TERMINAL_OPERATIONAL_METADATA`: van, vault distance, CRA status, threat level, telemetry, ticket size, addresses, reject bins. They don't change the indent, but they aren't real.
- **The copilot's feature list** ("30 ML model variables", monsoon index and so on) describes an older model, not `lgb_holiday`.
- Sign-in identifies the officer. Roles come from `users.json`.

## How to check it

```bash
python3 live_portal/generate_portal_data.py       # rebuilds both portal bundles (~2 min)
python3 -m unittest tests.test_audit_remediation  # 23 tests
python3 tests/test_buffer_week_type.py
python3 tests/test_forecaster_leakage.py
```

In v2, ATM 1 should show a usable balance of ₹23,19,500 and a last feed of "2026-08-24 23:45 IST (raw snapshot)".

The A5 test used to expect ATM 4 to need ₹0, because its typed cash was ₹26.6L. Its real cash is ₹11.7L, so it now needs a top-up. The test now checks what it was meant to: ATM 4 is sized on its 3-day horizon only, with no phantom second cycle.
