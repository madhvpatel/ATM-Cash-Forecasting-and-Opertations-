"""Replenishment Policy Simulator & Financial Backtester.

Conducts multi-cycle rolling-origin backtesting across ATMs to evaluate:
- Realized stockout frequency and duration
- Unmet customer demand shortfall (INR)
- Vault cash holding interest costs
- Fixed CIT trip and emergency refill expenditures
- Net Rupee cost savings vs. baseline heuristic policies
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from replenishment_engine.config import ReplenishmentConfig
from replenishment_engine.uncensoring import load_uncensored_panel
from replenishment_engine.forecaster import ReplenishmentForecaster
from replenishment_engine.optimizer import CashLoadOptimizer

logger = logging.getLogger(__name__)


class ReplenishmentSimulator:
    """Simulates rolling 7-day cash replenishment cycles and financial metrics."""

    def __init__(self, config: Optional[ReplenishmentConfig] = None):
        self.config = config or ReplenishmentConfig()
        self.optimizer = CashLoadOptimizer(self.config)

    def run_backtest_for_terminal(
        self,
        terminal_id: int,
        panel: pd.DataFrame,
        models_to_test: Optional[List[str]] = None,
        load_policy: str = "sla_99",  # 'sla_99' or 'cer_70'
    ) -> Dict[str, any]:
        """Run rolling replenishment simulation for a single ATM across all test origins.

        Returns:
            Dict containing detailed cycle logs and summary metrics for each model.
        """
        if models_to_test is None:
            models_to_test = [
                "naive_last_week",
                "pure_exogenous",
                "lgb_calendar_holidays_censored",
                "lgb_calendar_holidays_uncensored",
            ]

        sub = panel[panel["terminal_id"] == terminal_id].sort_values("date").reset_index(drop=True)
        dates = sub["date"].to_numpy()
        n = len(sub)
        horizon = self.config.horizon_days
        step = self.config.step_days
        min_train = self.config.min_train_days

        # Identify weekly cutoffs after warm-up
        cutoffs = [dates[i] for i in range(min_train, n - horizon, step)]

        # Forecasters
        forecaster_uncensored = ReplenishmentForecaster(self.config, use_uncensored_target=True)
        forecaster_censored = ReplenishmentForecaster(self.config, use_uncensored_target=False)

        feat_uncensored = forecaster_uncensored.build_terminal_dataset(sub)
        feat_censored = forecaster_censored.build_terminal_dataset(sub)

        # Ground truth unconstrained actuals and dates
        actual_true_dict = dict(zip(sub["date"], sub["uncensored_demand"]))

        cycle_results = {m: [] for m in models_to_test}

        for cutoff in cutoffs:
            cutoff_ts = pd.Timestamp(cutoff)
            target_dates = [cutoff_ts + pd.Timedelta(days=h) for h in range(1, horizon + 1)]
            true_demands = [actual_true_dict.get(td, 0.0) for td in target_dates]
            true_7day_sum = sum(true_demands)

            for m in models_to_test:
                # Generate forecast
                if m == "naive_last_week":
                    f_res = forecaster_uncensored.forecast_7day(feat_uncensored, cutoff_ts, model_type="naive_last_week")
                elif m == "pure_exogenous":
                    f_res = forecaster_uncensored.forecast_7day(feat_uncensored, cutoff_ts, model_type="pure_exogenous")
                elif m == "lgb_calendar_holidays_censored":
                    f_res = forecaster_censored.forecast_7day(feat_censored, cutoff_ts, model_type="lgb_calendar_holidays")
                elif m == "lgb_calendar_holidays_uncensored":
                    f_res = forecaster_uncensored.forecast_7day(feat_uncensored, cutoff_ts, model_type="lgb_calendar_holidays")
                else:
                    continue

                pred_sum = f_res["sum_pred"]
                daily_preds = f_res["daily_preds"]
                sigma_7 = f_res["sigma_7"]

                # Determine cash load under chosen policy
                if load_policy == "cer_70":
                    buffer_ratio = self.config.target_cer_buffer  # 0.70
                    unconstrained_load = pred_sum / buffer_ratio
                    load = min(self.config.get_capacity(terminal_id), max(0.0, unconstrained_load))
                else:  # 'sla_99'
                    opt_res = self.optimizer.calculate_recommendation(
                        predicted_7day_sum=pred_sum,
                        daily_predictions=daily_preds,
                        sigma_7=sigma_7,
                        initial_cash=0.0,
                        terminal_id=terminal_id,
                        service_level=self.config.target_service_level,
                    )
                    load = opt_res["recommended_load_inr"]

                # Simulate daily execution
                curr_cash = load
                daily_holding_cost = 0.0
                unmet_shortfall = 0.0
                empty_days = 0
                emergency_refills = 0

                for d_demand in true_demands:
                    if curr_cash >= d_demand:
                        curr_cash -= d_demand
                    else:
                        short = (d_demand - curr_cash)
                        unmet_shortfall += short
                        empty_days += 1
                        emergency_refills += 1
                        # Emergency carrier top-up dispatched to restore ATM operation
                        curr_cash = max(0.0, 1500000.0 - short)
                    daily_holding_cost += curr_cash * self.config.daily_holding_rate

                had_stockout = (unmet_shortfall > 0)
                trip_cost = self.config.cit_trip_cost if load > 0 else 0.0
                emergency_cost = emergency_refills * self.config.emergency_refill_premium
                penalty_cost = self.config.stockout_penalty_fixed if had_stockout else 0.0
                demand_penalty = unmet_shortfall * self.config.unmet_demand_penalty_rate

                total_cycle_cost = (
                    daily_holding_cost + trip_cost + emergency_cost + penalty_cost + demand_penalty
                )

                realized_cer = (true_7day_sum / load) if load > 0 else 1.0

                cycle_results[m].append({
                    "terminal_id": terminal_id,
                    "cutoff_date": str(cutoff_ts.date()),
                    "load_inr": load,
                    "forecast_sum_inr": pred_sum,
                    "true_sum_inr": true_7day_sum,
                    "ending_cash_inr": curr_cash,
                    "unmet_shortfall_inr": unmet_shortfall,
                    "empty_days": empty_days,
                    "had_stockout": int(had_stockout),
                    "holding_cost": daily_holding_cost,
                    "trip_cost": trip_cost,
                    "emergency_cost": emergency_cost,
                    "penalty_cost": penalty_cost,
                    "total_cost": total_cycle_cost,
                    "realized_cer": realized_cer,
                })

        return cycle_results

    def summarize_simulation(
        self,
        all_terminal_results: Dict[str, List[Dict[str, any]]],
    ) -> pd.DataFrame:
        """Aggregate backtest metrics across all evaluated terminals and cycles."""
        summary = []
        for model_name, rows in all_terminal_results.items():
            if not rows:
                continue
            df = pd.DataFrame(rows)
            total_cycles = len(df)
            stockout_cycles = df["had_stockout"].sum()
            stockout_rate_pct = (stockout_cycles / total_cycles) * 100
            total_shortfall_lakh = df["unmet_shortfall_inr"].sum() / 1e5
            total_holding_cost = df["holding_cost"].sum()
            total_cit_cost = df["trip_cost"].sum()
            total_penalties = df["penalty_cost"].sum() + df["emergency_cost"].sum()
            total_cost = df["total_cost"].sum()
            mean_load_lakh = df["load_inr"].mean() / 1e5
            avg_cer = df["realized_cer"].mean()

            # WAPE across all cycles
            wape = np.sum(np.abs(df["true_sum_inr"] - df["forecast_sum_inr"])) / np.sum(df["true_sum_inr"])

            summary.append({
                "model": model_name,
                "total_cycles": total_cycles,
                "forecast_7day_wape_pct": round(wape * 100, 2),
                "stockout_cycles": int(stockout_cycles),
                "stockout_rate_pct": round(stockout_rate_pct, 2),
                "total_shortfall_lakh": round(total_shortfall_lakh, 2),
                "mean_weekly_load_lakh": round(mean_load_lakh, 2),
                "avg_realized_cer": round(avg_cer, 3),
                "holding_cost_inr": round(total_holding_cost, 0),
                "cit_logistics_cost_inr": round(total_cit_cost, 0),
                "penalties_and_emergencies_inr": round(total_penalties, 0),
                "total_cost_inr": round(total_cost, 0),
            })

        res_df = pd.DataFrame(summary).sort_values("total_cost_inr").reset_index(drop=True)
        # Calculate Rupee savings relative to Naive Baseline
        baseline_cost = res_df.loc[res_df["model"] == "naive_last_week", "total_cost_inr"]
        if not baseline_cost.empty:
            b_val = baseline_cost.iloc[0]
            res_df["savings_vs_naive_inr"] = round(b_val - res_df["total_cost_inr"], 0)
            res_df["savings_pct"] = round((res_df["savings_vs_naive_inr"] / b_val) * 100, 2)
        else:
            res_df["savings_vs_naive_inr"] = 0.0
            res_df["savings_pct"] = 0.0

        return res_df
