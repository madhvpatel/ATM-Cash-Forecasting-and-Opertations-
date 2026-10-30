from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def create_plots(
    backtest_predictions: pd.DataFrame,
    simulation_detail: pd.DataFrame,
    simulation_metrics: dict[str, float | int],
    output_dir: str | Path,
) -> list[Path]:
    """Create the three decision-oriented charts required by the POC."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    daily = backtest_predictions.copy()
    daily["forecast_date"] = pd.to_datetime(daily["forecast_date"])
    daily = daily.groupby("forecast_date", as_index=False)[["actual_amount", "p50_amount", "p80_amount"]].sum()
    daily = daily.tail(60)
    fig, axis = plt.subplots(figsize=(11, 5))
    axis.plot(daily["forecast_date"], daily["actual_amount"], label="Actual", linewidth=2)
    axis.plot(daily["forecast_date"], daily["p50_amount"], label="P50 forecast", linewidth=1.6)
    axis.fill_between(
        daily["forecast_date"],
        daily["p50_amount"],
        daily["p80_amount"],
        alpha=0.18,
        label="P50–P80 range",
    )
    axis.set(title="Actual vs forecast cash demand", ylabel="Amount (INR)", xlabel="Date")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    path = output_path / "actual_vs_predicted.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    created.append(path)

    inventory = simulation_detail.copy()
    inventory["simulation_date"] = pd.to_datetime(inventory["simulation_date"])
    fig, axis = plt.subplots(figsize=(11, 5))
    for (terminal_id, denomination), group in inventory.groupby(["terminal_id", "denomination"]):
        axis.plot(
            group["simulation_date"],
            group["closing_inventory"],
            label=f"Terminal {int(terminal_id)} · INR {int(denomination):,}",
            linewidth=1.6,
        )
    axis.set(title="Simulated closing inventory", ylabel="Cash held (INR)", xlabel="Date")
    axis.legend(frameon=False, ncols=2, fontsize=8)
    axis.grid(alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    path = output_path / "inventory_over_time.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    created.append(path)

    cost_labels = ["Holding", "Transport", "Shortage", "Cash-out events"]
    cost_values = [
        float(simulation_metrics["holding_cost"]),
        float(simulation_metrics["transport_cost"]),
        float(simulation_metrics["shortage_cost"]),
        float(simulation_metrics["cashout_event_cost"]),
    ]
    fig, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(cost_labels, cost_values, color=["#4c78a8", "#f58518", "#e45756", "#b279a2"])
    axis.bar_label(bars, labels=[f"INR {value:,.0f}" for value in cost_values], padding=3)
    axis.set(title="Simulation cost breakdown", ylabel="Cost (INR)")
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path = output_path / "cost_breakdown.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    created.append(path)

    return created
