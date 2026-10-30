"""Execution Runner for ATM Cash Optimization & Replenishment Engine.

Runs rolling-origin backtests across the 8 Maharashtra benchmark ATMs,
evaluates Rupee cost trade-offs, computes side-by-side model benchmarks,
and exports structured artifacts for the web portal.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

from replenishment_engine.config import ReplenishmentConfig
from replenishment_engine.uncensoring import load_uncensored_panel, get_cashout_audit_summary
from replenishment_engine.simulator import ReplenishmentSimulator
from replenishment_engine.forecaster import ReplenishmentForecaster
from replenishment_engine.optimizer import CashLoadOptimizer


class NumpyEncoder(json.JSONEncoder):
    """Handles NumPy scalars and arrays for JSON export."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
            return str(obj)
        return super().default(obj)


def run_full_simulation():
    start_time = time.time()
    config = ReplenishmentConfig()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    layman_out = config.cms_root / "layman experiments" / "output"
    layman_out.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ATM CASH OPTIMIZATION & REPLENISHMENT ENGINE - SIMULATION RUN")
    print("=" * 80)

    # 1. Load data with uncensored demand
    print("\n[1/5] Loading panel data and applying demand un-censoring...")
    panel = load_uncensored_panel(config)
    audit_df = get_cashout_audit_summary(panel)
    audit_df.to_csv(output_dir / "terminal_cashout_audit.csv", index=False)
    audit_df.to_csv(layman_out / "terminal_cashout_audit.csv", index=False)
    print("  -> Terminal cashout audit saved.")
    print(audit_df.to_string(index=False))

    # 2. Run simulation across all 8 Maharashtra ATMs
    benchmark_terminals = sorted(panel["terminal_id"].unique())
    print(f"\n[2/5] Running rolling 7-day backtest across {len(benchmark_terminals)} ATMs: {benchmark_terminals}")

    simulator = ReplenishmentSimulator(config)
    all_results = {
        "naive_last_week": [],
        "pure_exogenous": [],
        "lgb_calendar_holidays_censored": [],
        "lgb_calendar_holidays_uncensored": [],
    }

    # Also store detailed per-terminal summaries
    terminal_summaries = []

    for tid in benchmark_terminals:
        print(f"  Simulating ATM {tid}...")
        t_res = simulator.run_backtest_for_terminal(
            terminal_id=tid,
            panel=panel,
            models_to_test=list(all_results.keys()),
            load_policy="sla_99",
        )
        for m, rows in t_res.items():
            all_results[m].extend(rows)

        # Terminal-specific summary
        t_sum = simulator.summarize_simulation(t_res)
        t_sum["terminal_id"] = tid
        terminal_summaries.append(t_sum)

    # 3. Overall simulation summary
    print("\n[3/5] Aggregating overall financial & operational benchmarks...")
    overall_summary = simulator.summarize_simulation(all_results)
    overall_summary.to_csv(output_dir / "replenishment_simulation_summary.csv", index=False)
    overall_summary.to_csv(layman_out / "replenishment_simulation_summary.csv", index=False)

    print("\n=== OVERALL BENCHMARK RESULTS (440 CYCLES ACROSS 8 ATMS) ===")
    print(overall_summary.to_string(index=False))

    # 4. Generate Live Schedule Samples and Cost Curves for Frontend
    print("\n[4/5] Generating live schedule samples & cost trade-off curves for web dashboard...")
    forecaster = ReplenishmentForecaster(config, use_uncensored_target=True)
    optimizer = CashLoadOptimizer(config)

    schedules_by_atm = {}
    cost_curves_by_atm = {}

    for tid in benchmark_terminals:
        sub = panel[panel["terminal_id"] == tid].sort_values("date").reset_index(drop=True)
        feat = forecaster.build_terminal_dataset(sub)
        latest_cutoff = feat["cutoff_date"].max()

        # Generate latest 7-day forecast
        pred_res = forecaster.forecast_7day(feat, latest_cutoff, model_type="lgb_calendar_holidays")
        opt_res = optimizer.calculate_recommendation(
            predicted_7day_sum=pred_res["sum_pred"],
            daily_predictions=pred_res["daily_preds"],
            sigma_7=pred_res["sigma_7"],
            initial_cash=0.0,
            terminal_id=tid,
            service_level=0.99,
        )

        frontier = optimizer.compute_cost_frontier(
            predicted_7day_sum=pred_res["sum_pred"],
            daily_predictions=pred_res["daily_preds"],
            sigma_7=pred_res["sigma_7"],
            terminal_id=tid,
        )

        site_name = sub["loc_type"].iloc[0]
        city_name = sub["city"].iloc[0]

        schedules_by_atm[str(tid)] = {
            "terminal_id": tid,
            "site_type": site_name,
            "city": city_name,
            "cutoff_date": str(pd.Timestamp(latest_cutoff).date()),
            "dates": pred_res["dates"],
            "daily_preds": [round(x, 0) for x in pred_res["daily_preds"]],
            "sum_pred": round(pred_res["sum_pred"], 0),
            "safety_buffer": round(opt_res["safety_buffer_inr"], 0),
            "recommended_load": round(opt_res["recommended_load_inr"], 0),
            "capacity": round(opt_res["capacity_inr"], 0),
            "is_capacity_capped": opt_res["is_capacity_capped"],
            "projected_inventory": [round(x, 0) for x in opt_res["projected_inventory_trajectory"]],
            "holding_cost": round(opt_res["holding_cost_inr"], 0),
            "cit_trip_cost": round(opt_res["cit_trip_cost_inr"], 0),
            "expected_stockout_cost": round(opt_res["expected_stockout_cost_inr"], 0),
            "total_expected_cost": round(opt_res["total_expected_cost_inr"], 0),
            "stockout_probability_pct": opt_res["stockout_probability_pct"],
        }
        cost_curves_by_atm[str(tid)] = frontier

    # Save JSON files for portal integration
    summary_dict = {
        "metadata": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_atms": len(benchmark_terminals),
            "total_cycles_evaluated": len(all_results["naive_last_week"]),
            "default_holding_rate": config.holding_rate_annual,
            "default_cit_cost": config.cit_trip_cost,
            "default_stockout_penalty": config.stockout_penalty_fixed,
            "default_emergency_refill": config.emergency_refill_premium,
        },
        "overall_summary": overall_summary.to_dict(orient="records"),
        "terminal_audit": audit_df.to_dict(orient="records"),
        "schedules_by_atm": schedules_by_atm,
        "cost_curves_by_atm": cost_curves_by_atm,
    }

    # Write JSON files
    with open(output_dir / "replenishment_portal_bundle.json", "w") as f:
        json.dump(summary_dict, f, indent=2, cls=NumpyEncoder)

    with open(layman_out / "replenishment_portal_bundle.json", "w") as f:
        json.dump(summary_dict, f, indent=2, cls=NumpyEncoder)

    # Write JS bundle for direct, zero-CORS browser loading
    with open(layman_out / "replenishment_portal_bundle.js", "w", encoding="utf-8") as f:
        f.write("// ATM Replenishment & Optimization Data Bundle\n")
        f.write("window.REPLENISHMENT_PORTAL_BUNDLE = ")
        json.dump(summary_dict, f, cls=NumpyEncoder)
        f.write(";\n")

    elapsed = time.time() - start_time
    print(f"\n[5/5] Simulation complete in {elapsed:.1f}s. All outputs saved to replenishment_engine/output/ and layman experiments/output/.")


if __name__ == "__main__":
    run_full_simulation()
