from __future__ import annotations

import argparse
import json
from dataclasses import replace

from .config import DEFAULT_CONFIG
from .pipeline import run_poc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ATM cash forecasting POC")
    parser.add_argument("--as-of", default="2026-08-13", help="First forecast date (YYYY-MM-DD)")
    parser.add_argument(
        "--horizon",
        type=int,
        choices=(1, 2, 7, 14),
        default=7,
        help="Forecast horizon in days: 1=24h, 2=48h, 7 or 14",
    )
    parser.add_argument("--risk", choices=("p50", "p80", "p95"), default="p80")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dispense-only", action="store_true", help="Ignore deposits instead of recycling them")
    parser.add_argument("--lead-time", type=int, default=DEFAULT_CONFIG.lead_time_days)
    parser.add_argument("--review-period", type=int, default=DEFAULT_CONFIG.review_period_days)
    parser.add_argument("--safety-buffer", type=float, default=DEFAULT_CONFIG.safety_buffer_pct)
    parser.add_argument("--holdout-days", type=int, default=DEFAULT_CONFIG.holdout_days)
    parser.add_argument("--simulation-days", type=int, default=DEFAULT_CONFIG.simulation_days)
    parser.add_argument("--holding-cost-rate", type=float, default=DEFAULT_CONFIG.holding_cost_rate_per_day)
    parser.add_argument("--transport-cost", type=float, default=DEFAULT_CONFIG.transport_cost_per_trip)
    parser.add_argument("--shortage-penalty-rate", type=float, default=DEFAULT_CONFIG.shortage_penalty_rate)
    args = parser.parse_args()
    config = replace(
        DEFAULT_CONFIG,
        lead_time_days=args.lead_time,
        review_period_days=args.review_period,
        safety_buffer_pct=args.safety_buffer,
        holdout_days=args.holdout_days,
        simulation_days=args.simulation_days,
        holding_cost_rate_per_day=args.holding_cost_rate,
        transport_cost_per_trip=args.transport_cost,
        shortage_penalty_rate=args.shortage_penalty_rate,
    )
    summary = run_poc(
        output_dir=args.output_dir,
        as_of=args.as_of,
        horizon_days=args.horizon,
        risk_quantile=f"{args.risk}_amount",
        seed=args.seed,
        recycler_enabled=not args.dispense_only,
        config=config,
    )
    print("\nPOC completed successfully")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
