# 100-ATM Forecast Scorecard

**Run:** 3 November 2026 · `python3 scripts/run_scorecard_100.py` (~15 min) · outputs in `outputs/scorecard_100/`
**Data:** all 100 synthetic cash-file ATMs, 754 days. 55 weekly forecast origins (31 Aug 2025 → 13 Sep 2026), each forecasting the next 7 days.
**Rules:** models train only on days known at the cutoff. Every number below is out-of-sample. The script asserts this at every origin, reusing the row builder from `run_rolling_origin.py`. The 100-ATM calendar flags reproduce the 8-ATM pilot's exactly (100% agreement).

## What "accuracy" means here

- **Weekly WAPE:** error on each ATM's 7-day total, which is the cash-load decision. This is the headline figure.
- **Daily WAPE:** error on each ATM-day.
- Both are per ATM, so errors are never netted across ATMs.

## Results

| Model | Weekly WAPE | Daily WAPE | Weekly bias | P99 bound coverage | Extra cash for P99 |
| --- | --- | --- | --- | --- | --- |
| Same weekday last week | 18.5% | 30.7% | +0.5% | 99.3% | +69% |
| Same weekday, 4-week mean | 14.7% | 24.3% | +0.2% | 99.6% | +57% |
| **LightGBM on rupees (`lgb_level`)** | **8.8%** | **18.2%** | −1.6% | 99.2% | **+34%** |
| LightGBM on ratio to 28-day mean | 9.7% | 18.8% | −1.8% | 99.4% | +40% |
| Blend of the three above | 9.7% | 19.0% | −1.1% | 99.5% | +39% |

**What this says**

1. **The ML model is clearly worth having on 100 ATMs.** It cuts weekly error by 53% against "same weekday last week", and wins on all 100 ATMs. Per-ATM weekly error ranges from 5.7% to 10.9%.
2. **The money is in the buffer, not the WAPE.** To be 99% sure an ATM doesn't run dry, the naive forecast needs 69% extra cash on top of actual demand. LightGBM needs 34%. That roughly halves idle safety stock.
3. **The new intervals are honest.** The P99 bounds come from past-week errors only (conformal), and they cover 99.2–99.6% of weeks. The old in-sample σ×√7 method covered 95.7%.
4. **The ratio model and the blend did *not* win here.** A 2-origin smoke test suggested they would, but over 55 origins `lgb_level` beats the blend on 89 of 100 ATMs. The ratio model exists to survive level shifts, and this synthetic data has none, so it pays its cost without collecting the benefit. Keep it as the fallback when the drift alarm fires (see the review doc, §4), not as the default.
5. **Accuracy improves with history.** Weekly WAPE was 12.1% in Q3 2025 (one year of training data), then 10.0%, 8.7%, 7.0% and 7.7% in later quarters.

## Where it is weak

| Slice (daily, `lgb_level`) | WAPE | Bias |
| --- | --- | --- |
| Normal days | 17.3% | −1.7% |
| Month end | 19.1% | +1.9% |
| Payday | 19.8% | +0.8% |
| Day before a bank holiday | 21.0% | **−4.6%** |
| Bank holiday | **24.0%** | **−5.3%** |

**Holidays are the main weakness.** The model under-forecasts holidays and the day before them, which is exactly when running out of cash is most costly. The baselines are worse (−14% to −18%), but this is the next thing to fix.

## Holiday fix (`lgb_holiday`)

**What was wrong:** the model had a single `is_regional_holiday` flag, so every bank holiday looked the same. In the data, Holi lifts demand by about 65% over a normal week, Diwali by 27%, and Gandhi Jayanti by 61% (it shares a day with Dussehra); minor holidays barely move it. The model averaged these, so it under-forecast big festivals by ~13% and over-forecast minor holidays by ~4%.

**What was added** (all known from the RBI calendar before the forecast is made):

| Feature | Meaning |
| --- | --- |
| `holiday_family` | ~28 families (diwali, holi, dussehra, eid_fitr, …). RBI names are combined strings that change yearly, so they are mapped by pattern |
| `next_holiday_family`, `days_to_holiday` | Which holiday is coming within 7 days, and how far away |
| `is_bank_closed`, `closure_run`, `days_to_closure` | Sundays, 2nd/4th Saturdays and holidays as closed days, and the length of the closed stretch |
| `prior_holiday_lift` | This ATM's lift on the same holiday family last year (≥300 days back, so always before the cutoff) |

**Result** (55 origins, 100 ATMs, same protocol):

| | `lgb_level` (before) | `lgb_holiday` (after) |
| --- | --- | --- |
| **Weekly WAPE (headline)** | 8.8% | **8.2%** |
| Extra cash for a 99% buffer | +34.3% | **+30.5%** (coverage 99.4%) |
| ATMs improved | — | 94 / 100 |
| Weeks containing a major festival: WAPE / bias | 12.5% / −9.2% | 11.1% / −6.6% |
| Major-festival days: WAPE / bias | 25.2% / −12.6% | 24.0% / **−4.3%** |
| Minor holidays: bias | +4.4% | −1.4% |
| 3 days before a major festival: bias | −9.8% | −5.1% |
| Long closures (3+ days): bias | −8.4% | −4.8% |
| Normal days: WAPE | 17.3% | 17.3% (unchanged) |

**Still open:**
- Festival weeks are still under-forecast by ~7%.
- Diwali flipped from −16% to +15% bias. The prior-year lift rests on a single previous Diwali, and Diwali moves between October and November. With only two years of history, these estimates are noisy. They will settle as history grows, or can be damped (e.g. average the ATM's lift with the family-wide lift).
- Holiday-day error is still ~23%. Most of it is day-to-day noise, but the remaining festival-week bias matters because those weeks carry the stockout risk.

`lgb_holiday` is now the best model on the scorecard and should replace `lgb_level` as the default.

## Wired into the portal and engine (3 Nov 2026)

**One model, one place:** `replenishment_engine/demand_model.py` holds the holiday-aware model and the buffer calibration. The scorecard, the portal generator and the replenishment engine all import it.

**Buffer:**
- The load is the forecast × the P99 of actual/forecast 7-day totals over the past 26 backtest weeks, read from `outputs/scorecard_100/predictions_weekly.csv`.
- With fewer than 8 past weeks it falls back to ×1.6 and marks the buffer uncalibrated.
- The optimizer and cost frontier still take a σ. `equivalent_sigma` converts the multiplier, so the 99% load matches exactly. Other frontier levels still assume a normal shape.

**Portal (`live_portal/generate_portal_data.py`):**
- `lgb_holiday` drives every indent, the cycle cash chain, the explainability text and the cost frontier (previously `pure_exogenous` did).
- Each weekly cycle is re-forecast at its own cutoff, the day before the cycle starts. The other models in the comparison table still use the single 24 Aug cutoff, so the comparison favours `lgb_holiday` slightly: it has fresher data.
- Cycle buffer multipliers: ×1.24–1.29, all calibrated.
- Portal holdout, per ATM-day: 17.6% WAPE (vs 18.5% pure exogenous), 7-day 16.4%, bias −2.4%.

**Engine (`python3 -m replenishment_engine.run_simulation`, 440 weekly cycles, 8 pilot ATMs):**

| Model | 7-day WAPE | Stockout cycles | Total cost | vs naive |
| --- | --- | --- | --- | --- |
| **`lgb_holiday`** | **8.4%** | **28.2%** | **₹49.3L** | **−1.9%** |
| Pure exogenous | 9.0% | 31.1% | ₹50.1L | −0.2% |
| Same weekday last week | 18.9% | 29.3% | ₹50.2L | — |
| LightGBM calendar (old default) | 9.2% | 32.5% | ₹51.1L | +1.6% |

**Why stockouts stay near 28%:** in 123 of 448 ATM-weeks (27%), actual demand exceeds the cassette capacity in `config.py`. That happens at ATM 7 in 89% of weeks and at ATM 1 in 57%. No forecast can prevent those stockouts with one visit a week, so `lgb_holiday` is at that floor. The next saving comes from a second weekly visit (or a larger cassette) at ATMs 1, 2 and 7, not from the model.

**Not synced:** `live_portal_v2/` and `UI_2/` still hold the old bundle copies.

## New ATMs (cold start)

ATMs were held out in 5 groups of 20 across 7 origins (700 ATM-weeks). The model never saw them in training.

| Scenario | Weekly WAPE | Bias |
| --- | --- | --- |
| Day 0: only location, city and region known | 24.9% | −7.7% |
| Day 28: four weeks of the ATM's own demand known | 11.3% | +0.3% |
| Same ATMs, fully trained (reference) | 10.6% | −0.4% |

**Policy that follows:** for a new ATM, use the location-based forecast with a wide buffer for the first four weeks. After 28 days, the ratio model is almost as good as one trained on the ATM itself.

## Caveats

- Synthetic data. These numbers measure the simulator, not real customers.
- The target is observed `Cash_Out`. Stockout days are not corrected, because the correction heuristic is unreliable (see `ML_REVIEW_2026-11-03.md` §6).
- Capacity and visit frequency are unknown for 92 of the ATMs, so this scores forecasts and buffer size, not rupee cost.
- The comparison with "industry" is approximate. Published ATM competitions (e.g. NN5) report daily per-ATM error, which compares to the 18.2% daily figure here, not the 8.8% weekly one.

## Noise floor: how much accuracy is left? (4 Nov 2026)

`python3 scripts/noise_floor.py` (~3 s) → `outputs/scorecard_100/noise_floor.json`

**Question:** some day-to-day noise is pure luck that no model can predict. How close is `lgb_holiday` to that limit?

**Method:** the data generator is not in this repo, so the floor is measured from the data.
- **Plain days only:** no holiday within 3 days, no payday, no month end, no 2nd/4th Saturday.
- **Remove the weekday pattern:** subtract each ATM's average for that weekday from its log demand.
- **Compare days by gap:** measure how different two days are as the gap between them grows. Level drift, the month cycle and the seasons barely change from one day to the next. So the difference extrapolated to a gap of zero is the pure noise. This is done per ATM.

**Result:**

| | `lgb_holiday` | Floor | Room left |
| --- | --- | --- | --- |
| Plain days, daily WAPE | 17.0% | 16.4% | 0.7 pts |
| Holiday-free weeks, weekly WAPE (3,082 weeks) | 7.3% | 6.3% | 1.0 pt |
| All weeks, weekly WAPE | 8.2% | not measured | — |

- Day-to-day noise is about ±21% per day (log σ 0.19–0.23 across ATMs). That is the simulator's own randomness.
- **On ordinary weeks the model is within about 1 point of perfect.** The floor assumes a forecaster that knows each ATM's true level exactly, so the reachable gap is smaller still.
- **The model has almost no week-wide level error.** 18% of the variance in its daily errors is shared across the week. Independent errors would already give 1/7 = 14%, so the true level error is only about 4 points.
- The difference between 8.2% (all weeks) and 7.3% (holiday-free weeks) is the holiday weeks. That is where forecast room is left.

**Buffer**, on the scorecard's basis (sum of P99 loads ÷ sum of actual demand − 1):

| | Value |
| --- | --- |
| Holiday-free weeks, today's buffer | +30.8% extra cash |
| Holiday-free weeks, perfect forecast | +19.8% (floor) |
| P99 of actual/forecast on holiday-free weeks | ×1.24 |
| P99 of actual/forecast on weeks with a holiday in them or within 3 days (2,418 weeks, 44%) | ×1.36 |
| P99 pooled over all weeks, which every week gets today | ×1.31 |

The pooled multiplier is set by holiday weeks and then applied to quiet weeks too. Noise size barely differs between ATMs (log σ 0.19–0.23), so a separate buffer per ATM would gain little. **A buffer per week type** (holiday-touched vs. not) would cut quiet-week extra cash from about 31% towards 24%. That figure uses ratios over the whole backtest, not a rolling window, so it needs confirming in a proper backtest.

**What went wrong in the first attempt:** I first compared each day with the same weekday one week either side. That "floor" came out at 20%, above the model's 17%, which is impossible. The cause: the data has a monthly cycle (days 14 apart differ most, days 28 apart differ least), so that comparison counted the cycle as noise. The script now asserts that the floor stays below the model, so this mistake would be caught.

**Caveats:**
- If demand spills across midnight, neighbouring days would be negatively correlated, and the floor would then be slightly lower than shown.
- The weekly floor treats each week's days as equal in size, which slightly understates it.
- Holiday-day noise is not measured.

**So:** better day-to-day features or a bigger model will not pay off on this data. Both remaining gains come from holiday weeks: their forecast error (step 1 below) and the buffer they inflate for every other week (step 2).

## Next steps

Re-ordered after the noise-floor result above.

1. **Damp the prior-year holiday lift towards the family-wide lift.** Holiday weeks are where the remaining forecast error is.
2. **Calibrate the buffer by week type.** In `buffer_multiplier`, compute the rolling P99 separately for weeks that touch a holiday and weeks that don't, and use the multiplier for this week's type. Expected: holiday-free weeks go from +31% towards +24% extra cash. The floor is +20%.
3. **Add a drift alarm** that switches an ATM to the ratio model when its 14-day bias exceeds ±10%. This is protection for real data, not an accuracy gain here.
4. Done:
   - ~~Holiday features~~
   - ~~Wire `lgb_holiday` and the conformal bounds into the portal and engine~~. Re-run the scorecard before the portal whenever data changes, because the buffer reads its output.

## How to re-run

```bash
python3 scripts/run_scorecard_100.py            # full, ~15 min
python3 scripts/run_scorecard_100.py --limit 2  # smoke test, ~2 min
```
