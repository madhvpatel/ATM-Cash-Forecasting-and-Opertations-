from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from atm_cash_forecast.demand_evaluation import run_walk_forward_benchmark
from atm_cash_forecast.raw_cash_files import load_daily_cash_out
from atm_cash_forecast.data_catalog import write_primary_tables
from experiments.seasonal_models import run_seasonal_challengers


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark forecasting models on raw 15-minute XLSX cash files"
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs" / "raw_cash_experiment"
    )
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--test-days", type=int, default=30)
    parser.add_argument(
        "--skip-seasonal", action="store_true", help="Run only benchmark controls"
    )
    args = parser.parse_args()

    transactions, terminals, audit = load_daily_cash_out(args.root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_primary_tables(args.data_dir, transactions, terminals)

    control_predictions, control_metrics = run_walk_forward_benchmark(
        transactions,
        terminals,
        fold_count=args.folds,
        test_days=args.test_days,
        minimum_training_days=180,
    )
    predictions = [control_predictions]
    metrics = [control_metrics]
    if not args.skip_seasonal:
        seasonal_predictions, seasonal_metrics = run_seasonal_challengers(
            transactions,
            terminals,
            fold_count=args.folds,
            test_days=args.test_days,
            minimum_training_days=180,
        )
        predictions.append(seasonal_predictions)
        metrics.append(seasonal_metrics)

    all_predictions = pd.concat(predictions, ignore_index=True)
    all_metrics = pd.concat(metrics, ignore_index=True)
    all_predictions.to_csv(args.output_dir / "all_model_predictions.csv", index=False)
    all_metrics.to_csv(args.output_dir / "all_model_metrics.csv", index=False)

    overall = all_metrics.query("fold == 'ALL'").sort_values("wape").reset_index(drop=True)
    baseline_wape = float(
        overall.loc[overall.model_name == "same_weekday_last_week", "wape"].iloc[0]
    )
    gates = overall.copy()
    gates["wape_le_20pct"] = gates.wape <= 0.20
    gates["abs_bias_le_5pct"] = gates.bias_pct.abs() <= 0.05
    gates["p80_calibrated"] = gates.p80_coverage.between(0.75, 0.85)
    gates["p95_calibrated"] = gates.p95_coverage.between(0.90, 0.98)
    gates["beats_baseline_by_1pct"] = (
        (baseline_wape - gates.wape) / baseline_wape
    ) >= 0.01
    gates["passes_all_offline_gates"] = gates[
        [
            "wape_le_20pct",
            "abs_bias_le_5pct",
            "p80_calibrated",
            "p95_calibrated",
            "beats_baseline_by_1pct",
        ]
    ].all(axis=1)
    gates.to_csv(args.output_dir / "promotion_gates.csv", index=False)

    paired = all_predictions.pivot_table(
        index=["fold", "forecast_date", "terminal_id", "denomination"],
        columns="model_name",
        values="p50_amount",
    )
    actual = (
        all_predictions.drop_duplicates(
            ["fold", "forecast_date", "terminal_id", "denomination"]
        )
        .set_index(["fold", "forecast_date", "terminal_id", "denomination"])[
            "actual_amount"
        ]
        .reindex(paired.index)
    )
    winner = str(overall.iloc[0].model_name)
    baseline = "same_weekday_last_week"
    daily = pd.DataFrame(
        {
            "actual": actual,
            "winner_error": (paired[winner] - actual).abs(),
            "baseline_error": (paired[baseline] - actual).abs(),
        }
    ).groupby(level=["fold", "forecast_date"]).sum().reset_index()
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(daily), size=(10_000, len(daily)))
    denominator = daily.actual.to_numpy()[indices].sum(axis=1)
    differences = (
        daily.winner_error.to_numpy()[indices].sum(axis=1) / denominator
        - daily.baseline_error.to_numpy()[indices].sum(axis=1) / denominator
    )

    summary = {
        "data_classification": "project-provided synthetic 15-minute XLSX cash files",
        "target": "daily cash withdrawal demand (Cash_Out)",
        "target_assumption": (
            "Per field observation, Cash_Out and withdrawals are treated as interchangeable."
        ),
        "grain": "ATM x date, represented with a ₹100 rounding unit",
        "evaluation": f"{args.folds} consecutive {args.test_days}-day rolling one-day-ahead folds",
        "audit": audit.as_dict(),
        "winner": winner,
        "winner_wape": float(overall.iloc[0].wape),
        "winner_bias_pct": float(overall.iloc[0].bias_pct),
        "same_weekday_baseline_wape": baseline_wape,
        "relative_wape_improvement_vs_baseline": float(
            (baseline_wape - overall.iloc[0].wape) / baseline_wape
        ),
        "bootstrap_wape_difference_winner_minus_baseline": {
            "lower_95pct": float(np.quantile(differences, 0.025)),
            "upper_95pct": float(np.quantile(differences, 0.975)),
            "probability_winner_better": float(np.mean(differences < 0)),
        },
        "offline_gate_passers": gates.loc[
            gates.passes_all_offline_gates, "model_name"
        ].tolist(),
        "production_validated": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
