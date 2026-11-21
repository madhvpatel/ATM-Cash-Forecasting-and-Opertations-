# Review of 29-10-2026: what to change, and what to ask

**Written:** 2026-11-04, against `live_portal_v2/` on branch `feat/holiday-demand-model`.

## The short version

Most items in the review already show up in the v2 portal. The Daily Cash Action Report even uses the reviewer's column list. But most of those values are **typed by hand** in `TERMINAL_OPERATIONAL_METADATA` (`masters_manager.py`, about lines 70–500), and only for **8 ATMs**. The forecasting engine and the raw 15-minute files cover **100 ATMs**.

So most of the work is not "build the feature". It is "replace the hard-coded number with one worked out from the data", or "get the data from the bank".

**Bug found while checking:** the Daily Cash Action Report (CSV and Excel download) crashes today. Past cycles store `actual_demand = "N/A (Unmeasured)"`, and the report then divides that text by 100000 (`masters_manager.py:2129`). The result is `TypeError: unsupported operand type(s) for /: 'str' and 'int'`.

## What the raw data actually has

Each 15-minute file has these columns: `ATM_ID, Term_City, Term_Loc, Cash_Out, End_Cash`, and a cash amount plus note denomination for each of 4 hoppers, plus `TIMESTAMP`.

It does **not** have:
- transaction counts
- reject bin counts
- rated cassette capacity
- CRA visit or SLA logs
- device fault or status codes
- deposits (needed for CRM)
- how much was authorised versus how much was actually loaded

## Every item, sorted into four groups

**A: Already real.** Just show the reviewer where it is.
| Item | Where |
|---|---|
| Plausible P50 / P80 / P95 | The `quantile_forecast` block. The 100-ATM model calibrates the buffer from backtests. |
| Maker/checker on indents | `approve_terminal_indent`, `apply_checker_override`, tier-1 controls |
| Indent manifest download | `/api/manifest`, `/api/manifest/lock` |

**B: The screen exists but the data is hand-typed.** Keep the UI and feed it computed values.
| Item | What it shows today |
|---|---|
| Location category | Typed per ATM. It could be mapped from `Term_Loc`. |
| Avg ticket size (day/week/month) | Typed constants |
| Reject bin | Typed `current_notes` |
| CRA missed visit / SLA breach | Typed `cra_status` |
| Threat monitoring | Typed |
| Human errors | Typed `human_errors` |
| Last 3 predictions and how well they did | "Predicted" is `demand × 0.95` and "actual" is "N/A". This is the crash above. |
| Downtime risk score | Rules applied to the typed fields |
| Last feed end cash | Typed, frozen at 2026-08-24 |

**C: Can be worked out from the 15-minute files.** This is real work, but it needs no new data.
| Item | How |
|---|---|
| Last feed end cash | The last `End_Cash` row for each ATM |
| Denomination demand | The drop in each hopper's cash, grouped by note value |
| Replenishment history | Find the jumps in `End_Cash` |
| Missed or late CRA visit | Compare actual refill dates with the configured cycle |
| Cashout events and hours | `End_Cash` is 0, or the hopper for a note value is empty |
| Downtime (rule-based) | Long runs of `Cash_Out = 0` during hours that normally have demand while cash > 0, or missing intervals |
| Cassette capacity (lower bound) | The highest load ever seen in each hopper |
| Prediction vs actual for past cycles | The backtest the scorecard already runs (`scripts/run_scorecard_100.py`) |
| Anomaly report | Large forecast misses that the calendar does not explain |
| Surplus sweep between 2 ATMs | Logic exists (`execute_inter_atm_sweep`). It needs real balances and the forecast. |
| Critical-first network view | Sort by hours to cashout. The UI change only. |
| Forecast Excel, historical predictions | Exports exist. Point them at the backtest instead of the made-up rows. |

**D: Needs data from the bank.** Without it, the item stays labelled "sample" or is hidden.
- Transaction counts. **Ticket size cannot be computed without them.**
- Reject bin counters (for both CRM and non-CRM ATMs)
- Rated cassette capacity for each machine model
- CRA visit logs and SLA penalty terms
- Device fault or status feed (needed for true downtime, not just "nothing dispensed")
- Deposit and recycle data for CRM machines
- Authorised versus loaded amounts (needed for "human error")

## Suggested order

1. Fix the report crash. This is a one-line fix.
2. Run v2 on the 100 ATMs and the real model, instead of the 8 seeded ATMs. Most of group C depends on this.
3. Group C items, one small PR each: last feed, refill history and missed visits, downtime rules, denomination demand, past-cycle efficacy, anomalies.
4. Critical-first network view, then the download set.
5. Group D items once the bank's data arrives. Until then, every hand-typed field gets a visible "sample data" label.

## Questions to answer before building

1. **Bank data:** Which group D feeds can the bank give us, and when? Until they arrive, should hand-typed fields be **hidden**, or **shown with a "sample" label**?
2. **Downtime, model or rules:** I recommend rules. The synthetic data has no outages, so a model would have nothing to learn from or be tested against. Is that acceptable?
3. **P50/P80/P95:** Are these for **display only**, or should the planner be able to **pick which one sets the indent**? Today the P99 buffer sets it. What did the reviewer mean by "Plausible"?
4. **"Critical" ATM:** What is the rule? For example, cashout within X hours, a missed CRA visit, or a full reject bin. Is it the same for metro and non-metro?
5. **Location taxonomy:** Who owns the list metro / non-metro / Residential / Corporate / Hood? What is "Hood"? Is metro/non-metro a separate field from the category?
6. **Exceptional events:** Who enters them (ops or treasury)? Do they change the forecast automatically, or only flag it for review?
7. **Replanning after a disruption:** What triggers a replan (missed visit, downtime, a big miss)? Does the new plan go back through maker/checker?
8. **Human error:** Which errors count? For example, loading short, a cassette in the wrong slot, or a reject bin left full. Where would they be recorded?
9. **Surplus sweep:** Is moving cash between two ATMs allowed operationally (CIT rules, insurance)? What is the minimum amount that makes it worth a trip?
10. **Downloads:** Excel, CSV or PDF? What goes in "Single Report – Params"? Is it one ATM with every parameter?
11. **Ticket size:** Ticket size needs transaction counts. Is "cash out per day" an acceptable stand-in until those counts arrive?

## The crash: fixed on 2026-11-04

**What broke:** past cycles that have never been measured store `actual_demand` as the text "N/A (Unmeasured)". Two places did maths on that text:
- The Daily Cash Action Report (`masters_manager.py`) raised a `TypeError`. That broke both the CSV and the Excel download.
- The agent's "explain ATM performance" answer (`agent.py`) raised a `ValueError`.

**The fix:** both places now print "N/A" when the actual isn't a number. The data is unchanged. The Excel past-cycles sheet still shows "N/A (Unmeasured)".

**To check:**
```bash
python3 -m unittest tests.test_audit_remediation tests.test_tier1_controls
```
All 40 tests should pass. `test_unmeasured_cycle_actuals_render_without_crashing` is the test for this bug.

## Feature-importance panel: made-up numbers replaced (2026-11-04)

**What was wrong:** the AI copilot's "Model Variables" answer (`agent.py`) and its bar panel (`index.html`) showed hand-written weights for a "30-variable Strategy C" model. They listed inputs the model doesn't use, such as `lag_1` and weather, and percentages nobody measured. For example, `lag_7` was shown at 22.6% and `is_payday` at 16.8%.

**What changed:**
- `demand_model.feature_importance(cutoff)` now reads the split-gain shares from the actual trained model.
- `generate_portal_data.py` saves them as `metadata.operational_model.feature_importance`, and `/api/data` passes them through.
- The agent answer and the panel both read from there.

The real top inputs are: 28-day average demand 35.9%, demand 4 weeks before 18.8%, which ATM 15.4%, day of month 8.3%, and same weekday last week 6.6%. Payday is under 1%.

**Limits:**
- One model serves all 100 ATMs, so the weights are fleet-wide, not per ATM.
- Gain shows how much the model relies on an input. It does not show whether the input pushes demand up or down.

**To check:** ask the copilot `/variables 1`. If the data file has no importances, the answer says so instead of showing numbers.
