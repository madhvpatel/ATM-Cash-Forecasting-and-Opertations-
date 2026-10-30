# CMS2 Project Review

**Review date:** 27 October 2026  
**Project:** Cash Intelligence — ATM cash-demand forecasting and replenishment decision support  
**Basis:** Local source code, data documentation, saved models, experiment outputs, and limited read-only execution checks.

## 1. Overall assessment

CMS2 is a substantial analytical and operational prototype for ATM cash management. It addresses a practical question: how much cash should be replenished at each ATM, given expected withdrawals, existing cash, uncertainty, capacity, and delivery arrangements?

The project progresses from raw cash snapshots to daily demand analysis, forecasting experiments, cashout correction, replenishment simulation, and an interactive treasury portal. Its most valuable achievement is connecting forecasting to an operational decision: a proposed cash indent that an operator can inspect, adjust, approve, and include in a dispatch manifest.

The implementation demonstrates both research depth and attention to the operator's workflow. Calendar ablations, rolling-origin evaluations, reconstruction experiments, scenario analysis, and an AI assistant all have concrete implementations or saved outputs. However, the evidence supports describing the project as a working prototype on synthetic data. Real-world forecasting performance, financial savings, service guarantees, and secure dispatch authorization remain unproven.

## 2. Scope and architecture

The workspace contains several stages of development rather than a single packaged application.

| Component | Purpose and contribution |
| --- | --- |
| `data/` | Raw archives, processed demand tables, reference calendars, master data, catalogs, and a frozen baseline |
| `data/baselines/calendar-v1-2026-10-21/` | Earlier forecasting package and retained benchmark evidence |
| `simulated/maharashtra/` | Eight-ATM pilot, seasonal analysis, weather exploration, quantile forecasts, and rolling-origin evaluation |
| `layman experiments/` | Feature comparisons, demand reconstruction, validation experiments, charts, and initial portal generation |
| `replenishment_engine/` | Reusable forecasting, cashout correction, buffer sizing, cost calculation, and backtesting modules |
| `live_portal/` | First operational portal, model and bundle generation, new-ATM forecasting, and explanation/scenario endpoints |
| `live_portal_v2/` | Latest treasury interface, editable masters, recomputed indents, checker workflow, manifests, and a tool-using AI assistant |

The main flow is:

**15-minute snapshots → daily demand and inventory-quality tables → calendar/reference features → demand forecasts → cash-buffer and indent calculations → operator review → local dispatch-manifest generation.**

Python supports ingestion, analysis, modeling, and serving. Pandas, NumPy, and LightGBM appear in the analytical code; the latest assistant uses LangGraph and a Gemini integration. The frontend uses HTML, CSS, JavaScript, and Chart.js. JSON bundles connect offline model outputs to the portals.

This structure makes the progression understandable, but duplicated calculations and independently copied bundles create a risk that experiments, portal displays, and operational recommendations use different assumptions.

## 3. Methodology

### Data preparation and provenance

The primary synthetic dataset covers **100 ATMs across 754 days**, from **1 September 2024 to 24 September 2026**. The retained baseline audit reports **25 archives, 72,384 snapshot files, and 7,238,400 snapshot rows**, with no missing or duplicate snapshot files in that audit. These are retained audit results, not a fresh re-ingestion performed for this review.

Daily demand is formed by summing `Cash_Out`. This is a practical modeling target, but observed dispensed cash is not necessarily total customer demand when a machine is empty. The project recognizes this distinction in later experiments.

Data documentation also makes useful distinctions: the primary 100-ATM population is separate from the six-terminal legacy demo; unknown capacity and device attributes remain blank in governed primary masters; and the denomination value of ₹100 is a rounding unit rather than an observed note mix. File hashes, catalogs, reference coverage, and frozen artifacts provide a foundation for reproducibility.

The eight-ATM Maharashtra work is a smaller experimental population across Mumbai, Pune, Nagpur, and Nashik. It supports pipeline testing and comparative analysis, but cannot establish performance across the full network or isolate general site-type effects when each site type identifies a single ATM.

### Forecast development and comparison

The approach appropriately starts with interpretable benchmarks before adding complexity. Implemented comparisons include ATM averages, day-of-month factors, previous-week demand, history-based models, calendar features, regional holidays, payday rules, weather, and combinations of these inputs.

The stronger evaluation workflows use **rolling origins**: train on available history, forecast the next seven days, move the origin forward, and repeat. The Maharashtra rolling-origin record contains **55 origins, 440 ATM-week cycles, and 3,080 daily predictions**, after a 365-day initial training period.

WAPE measures absolute forecast error relative to total demand; bias identifies systematic over- or underprediction. Daily and seven-day scores are reported separately, which is appropriate because replenishment depends on cumulative demand. Quantile forecasts at P50, P80, and P95 explore uncertainty and service trade-offs. The notes correctly distinguish a sum of daily quantiles from a quantile of weekly total demand.

Controlled feature ablations are a particular strength. They investigate whether an input earns its place rather than assuming that more features improve forecasts. Weather findings require care: historical reanalysis describes what occurred, rather than information necessarily available at a forecast origin. The governed rolling-origin workflow explicitly excludes weather for that reason.

### Cashout correction

The replenishment engine estimates an active fraction from zero-balance snapshots and scales observed demand by the inverse of that fraction. A floor of 20% limits extreme inflation. It produces cashout indicators, estimated empty hours, reconstructed demand, and inferred lost demand.

The project also tests reconstruction through pseudo-masking. This is valuable because true unmet withdrawals are unavailable. Results show that reconstruction depends on when missing intervals occur: uniform masking is less problematic than losing peak or quiet periods. Reconstructed demand should therefore be described as an estimate, not verified ground truth.

### Replenishment and economics

Two main policy approaches are implemented: a forecast divided by a target Cash Exhaustion Rate, and a forecast plus a service-level buffer. The optimizer estimates inventory trajectories, holding costs, delivery costs, stockout risk, and expected penalties. A cost frontier compares several service levels.

These calculations connect forecast quality to operating consequences. However, the current optimizer primarily sizes cash for a selected policy and evaluates its cost. It does not jointly optimize delivery dates, routes, vehicle constraints, denomination allocation, and cash amounts. Capacity, cost, and penalty values in the prototype should be treated as configurable assumptions; comments describing them as industry standards do not provide independent evidence.

## 4. Results and things achieved

Results come from different populations, horizons, targets, and evaluation protocols. They should remain separate rather than being combined into one accuracy claim.

| Evidence set | Saved result | Interpretation |
| --- | --- | --- |
| Frozen 100-ATM benchmark | Local quantile GBM WAPE **17.94%**, versus **29.16%** for the same-weekday baseline | Reported relative WAPE improvement of about **38.5%**, under three 30-day rolling one-day-ahead folds |
| Maharashtra rolling-origin evaluation | LightGBM daily WAPE **18.59%** and weekly WAPE **9.02%**, versus seasonal-mean **21.09%** and **10.35%** | Demonstrates improvement over an interpretable seasonal benchmark on the synthetic pilot |
| Experiment 4 feature comparison | History-only daily/weekly WAPE **21.75% / 12.13%**; history plus calendar **18.74% / 9.37%** | Calendar inputs deliver the clearest incremental gain |
| Experiment 4 broader models | Full stack **18.50% / 9.08%**; model without demand-history features **17.98% / 8.77%** | More inputs do not consistently improve performance; “without history” still includes ATM identity and weather |
| Capacity-constrained replenishment-engine simulation | Pure-exogenous total cost **₹30,56,772**, versus naive **₹37,32,613**, across 440 cycles | Saved modeled saving of **₹6,75,841**, or **18.11%**; requires correction and rerun before use as a reliable benefit claim |

The last result has material evaluation issues described below. It also records a **9.55% stockout-cycle rate**, despite a nominal 99% service-level setting. A configured target is not an achieved service guarantee.

An earlier Maharashtra simulation has no cassette cap and reports much lower stockouts under the 70% CER policy. This difference illustrates how strongly policy assumptions and physical constraints affect results; the two simulations are not directly comparable.

The most defensible achievements are:

1. A documented transformation from synthetic snapshots into a daily forecasting dataset.
2. Preserved baseline evidence and repeatable experimental scripts.
3. Forecast comparisons using temporal holdouts, rolling origins, and feature ablations.
4. Evidence that calendar structure matters more consistently than weather in the examined data.
5. A tested prototype for estimating cashout-suppressed demand, including evidence of its limitations.
6. Translation of forecasts into capacity-aware cash recommendations and modeled cost trade-offs.
7. An operations interface that brings recommendations, explanations, manual adjustments, and local approvals into one workflow.

Forecast quality should be communicated using WAPE and bias. Describing `100 − WAPE` as generic “accuracy” can obscure the metric's meaning and should be qualified.

## 5. Features and functions

| Feature | Implemented functions and current scope |
| --- | --- |
| Fleet overview | Fleet cash and indent totals, ATM roster, critical/scheduled/healthy status, approval counts, and network charts |
| ATM inspection | Demand charts, cash trajectories, existing cash, capacity, cash runway, buffers, and recommendation details |
| Explainability | Facts, observations, assumptions, active signals, and a formula ledger showing the cash calculation |
| Forecast alternatives | Pure exogenous models, volume-normalized Strategy A, metadata-based Strategy B, and telemetry/history-based Strategy C |
| New-ATM exploration | First-portal endpoint and profiles for estimating demand from baseline volume, site metadata, or supplied history; unseen-site performance is not established |
| Master configuration | Editable financial assumptions, capacity, servicing mode, cycle length, site attributes, and event multipliers, with save/reset support |
| Scenario analysis | Custom-indent and dispatch-delay calculations, estimated runway, end cash, and holding cost |
| Maker/checker workflow | System-proposed indents, manual checker adjustments, reasons and timestamps, and individual or fleet approval |
| Dispatch preparation | Route-grouped local manifest, chest-pull total, checker details, override records, and a stored lock flag |
| AI copilot | Nine tools for network status, ATM explanation, simulation, masters retrieval/update, approvals, indent adjustment, manifest retrieval, and manifest locking |
| Assistant fallback | Slash commands and deterministic request handling when an API key is unavailable or model calls fail |
| Data governance | Catalogs, file identities, reference coverage, unavailable-field records, and preserved baseline artifacts |

The assistant is integrated with application functions rather than being an isolated chat window. That is a meaningful product achievement. Its model-backed path was not exercised in this review, and fallback support should not be interpreted as covering every possible conversational request.

The latest portal recomputes recommendations from a saved **25 August–24 September 2026** forecast bundle. “Live” currently refers mainly to parameter editing and recalculation. The reviewed path does not establish a continuously refreshed connection to ATM telemetry or a bank switch.

## 6. Important limitations and improvements

### Correct evaluation before promoting savings claims

In `replenishment_engine/forecaster.py`, training rows are selected by `cutoff_date < forecast_origin`. Because each training origin has targets up to seven days ahead, this can include targets occurring after the forecast origin. Training eligibility must also constrain the target date to information available at decision time. Experiment 4 already uses a target-index cutoff, demonstrating the stronger approach elsewhere in the project.

The engine's `naive_last_week` path repeats a single origin-level `lag_7` value across the seven forecast horizons. It does not reproduce the corresponding seven historical weekdays. Correct this benchmark and rerun the engine comparison before relying on the reported savings or relative forecast advantage.

### Align operational recalculation with dated evidence

The latest masters manager applies all city-matching event multipliers without checking their start and end dates. A read-only check confirmed that moving event dates to 2099 leaves calculated demand unchanged. Apply events only to overlapping forecast days and distinguish assumed scenario multipliers from learned effects.

The portal also uses a fixed 22% daily-volatility heuristic for buffer sizing, while other paths use fitted residuals or quantile models. Consolidate the policy calculation and calibrate uncertainty on out-of-sample residuals, including dependence between forecast days.

### Strengthen approval and manifest integrity

The current maker/checker interface demonstrates the workflow, but the reviewed server does not enforce authenticated, separate roles. A manifest can be generated from all terminals without requiring approval, and the lock flag does not prevent master edits. A read-only in-memory check confirmed an update succeeds while the flag is set. Checker overrides also lack explicit bounds against negative cash or available capacity.

The displayed signature token is a truncated hash of manifest metadata. It is not a private-key digital signature. The manifest is regenerated from current data rather than preserved as a signed, immutable payload. Implement server-side validation, approval invalidation after changes, immutable approved snapshots, and enforceable locking before operational use. Ordinary approvals are held in memory unless subsequently persisted through another save path.

Assistant text claiming transmission to an armored fleet exceeds the reviewed implementation: the lock function writes local state and returns a manifest; it does not integrate with a carrier dispatch system.

### Make the application reproducible and consistent

The available root `.venv` lacks Pandas, and an attempted analytical import in an alternate Anaconda installation failed because of a NumPy binary compatibility problem. The active application needs its own dependency specification and documented setup; a lockfile inside the frozen baseline does not establish a reproducible current portal environment.

Some data documentation references missing preparation scripts and an absent implementation guide. Consolidate instructions, identify the canonical portal version, and document data/model regeneration. Reconcile site labels and servicing assumptions across pilot data, generated bundles, and V2 defaults; some terminals have different site descriptions across these stages.

The Python HTTP server, broad CORS configuration, local JSON state, and lack of visible authentication are suitable prototype choices but need replacement or hardening for a shared operational service. No dedicated automated test suite was found in the inspected project files. Prioritize tests for temporal splits, cash conservation, capacity limits, event applicability, approval persistence, and lock enforcement.

## 7. Recommended next phase

First, correct the engine's temporal split and previous-week baseline, then publish refreshed metrics with explicit dataset, target, horizon, and policy identities. Second, use one shared replenishment calculation in experiments and serving, with dated event application and calibrated weekly uncertainty. Third, make approvals and manifests enforceable, bounded, persistent, and attributable to authenticated users. Fourth, package the current application and repair its regeneration documentation.

After those changes, run a controlled pilot with real withdrawals, inventory, failed transactions, replenishment events, and verified capacity. Measure forecast error, actual cash availability, idle cash, emergency visits, and realized operating cost separately. Denomination planning and route optimization can then extend the system using verified operational inputs.

## 8. Review conclusion and verification scope

CMS2 has achieved a connected research-to-operations prototype with meaningful forecasting experiments and a detailed operator workflow. Its strongest contribution is the link between demand estimates, cash-loading decisions, human review, and explainable calculations. The next milestone is to make that chain consistent and auditable, then demonstrate value on real operational data.

This review inspected source and retained outputs and validated Python syntax for **32 active scripts/modules**. Read-only execution verified eight-terminal fleet recalculation and the event-date and lock-edit behaviors above. No application configuration was saved or changed. The full model-training pipeline, raw-archive audit, browser workflow, external AI calls, and real dispatch integrations were not rerun or validated. Reported experimental metrics are attributed to retained project artifacts.

## Source guide

- Data scope and governance: `data/README.md`, `data/master/portal/README.md`.
- Retained full-network benchmark: `data/baselines/calendar-v1-2026-10-21/outputs/raw_cash_experiment/summary.json`.
- Pilot methodology: `simulated/maharashtra/README.md`, `FINDINGS.md`, and `output/rolling_origin_notes.json`.
- Pilot metrics: `simulated/maharashtra/output/rolling_origin_metrics.csv` and `rolling_origin_replenishment.csv`.
- Feature comparisons: `layman experiments/run_experiment_4.py` and `output/experiment_4_overall_models.csv`.
- Reconstruction evidence: `layman experiments/output/pseudo_masking_validation_summary.csv` and `uncensoring_model_comparison.csv`.
- Replenishment implementation and results: `replenishment_engine/forecaster.py`, `optimizer.py`, `uncensoring.py`, `simulator.py`, and `output/replenishment_simulation_summary.csv`.
- Portal model generation: `live_portal/generate_portal_data.py` and `serve.py`.
- Current operational workflow: `live_portal_v2/index.html`, `masters_manager.py`, `agent.py`, `serve.py`, and `portal_data.json`.
