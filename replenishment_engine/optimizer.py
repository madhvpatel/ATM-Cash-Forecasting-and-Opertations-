"""Cash Load and Buffer Sizing Optimizer.

Implements dual-objective optimization:
1. Target Service Level SLA (guaranteeing 95-99.5% cash availability)
2. Total Operational Rupee Cost Minimization (holding interest vs CIT trips vs stockout penalties)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional
import numpy as np

from replenishment_engine.config import ReplenishmentConfig


# Inverse normal CDF approximation (probit function) for service level z-scores
def norm_ppf(p: float) -> float:
    """Accurate rational approximation of inverse standard normal CDF."""
    if p <= 0.0 or p >= 1.0:
        raise ValueError("Probability p must be in (0, 1)")

    # Coefficients in rational approximations
    a = [
        -3.969683028665376e01, 2.209460984245205e02,
        -2.759285104469687e02, 1.383577518672690e02,
        -3.066479806614716e01, 2.506628277459239e00
    ]
    b = [
        -5.447609879822406e01, 1.615858368580409e02,
        -1.556989798598866e02, 6.680133369969707e01,
        -1.328068155288572e01
    ]
    c = [
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e00, -2.549732539343734e00,
        4.374664141464968e00, 2.938163982698783e00
    ]
    d = [
        7.784695709041462e-03, 3.224671290700398e-01,
        2.445134137142996e00, 3.754408661907416e00
    ]

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)


def norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class CashLoadOptimizer:
    """Calculates optimal cash load amounts, safety buffers, and Rupee cost trade-offs."""

    def __init__(self, config: Optional[ReplenishmentConfig] = None):
        self.config = config or ReplenishmentConfig()

    def calculate_recommendation(
        self,
        predicted_7day_sum: float,
        daily_predictions: List[float],
        sigma_7: float,
        initial_cash: float = 0.0,
        terminal_id: int = 1,
        service_level: Optional[float] = None,
        override_holding_rate: Optional[float] = None,
        override_cit_cost: Optional[float] = None,
        override_stockout_penalty: Optional[float] = None,
    ) -> Dict[str, any]:
        """Compute the recommended cash load, buffer, and cost metrics for a 7-day cycle.

        Args:
            predicted_7day_sum: Total forecasted withdrawal over 7 days.
            daily_predictions: Forecasted withdrawals for each day 1..7.
            sigma_7: Forecast error standard deviation over 7 days.
            initial_cash: Cash already present in vault at start of cycle.
            terminal_id: ATM terminal identifier (for capacity limits).
            service_level: Target availability probability (e.g. 0.99).
            override_holding_rate: Optional custom annual holding rate.
            override_cit_cost: Optional custom CIT trip cost.
            override_stockout_penalty: Optional custom stockout penalty.

        Returns:
            Dict containing recommended load, safety buffer, costs, and daily trajectory.
        """
        cfg = self.config
        sla = service_level or cfg.target_service_level
        holding_rate = override_holding_rate if override_holding_rate is not None else cfg.holding_rate_annual
        cit_cost = override_cit_cost if override_cit_cost is not None else cfg.cit_trip_cost
        stockout_penalty = (
            override_stockout_penalty if override_stockout_penalty is not None else cfg.stockout_penalty_fixed
        )
        emergency_cost = cfg.emergency_refill_premium
        capacity = cfg.get_capacity(terminal_id)

        # 1. Calculate SLA-based safety buffer
        z = norm_ppf(min(max(sla, 0.50), 0.999))
        safety_buffer = max(0.0, z * sigma_7)

        # 2. Recommended cash load
        unconstrained_load = predicted_7day_sum + safety_buffer - initial_cash
        recommended_load = min(capacity, max(0.0, unconstrained_load))

        # Check if capacity constrained
        is_capacity_capped = (unconstrained_load > capacity)

        # 3. Simulate projected daily cash inventory trajectory
        total_initial_vault = initial_cash + recommended_load
        projected_inventory = []
        curr_cash = total_initial_vault
        daily_holding_interest = holding_rate / 365.0

        total_holding_cost = 0.0
        for day_idx, pred_d in enumerate(daily_predictions):
            curr_cash -= pred_d
            ending_cash = max(0.0, curr_cash)
            projected_inventory.append(round(ending_cash, 2))
            total_holding_cost += ending_cash * daily_holding_interest

        # 4. Projected Stockout Probability & Risk
        # P(Demand > Available Cash)
        cash_margin = total_initial_vault - predicted_7day_sum
        if sigma_7 > 1e-3:
            z_score = cash_margin / sigma_7
            prob_stockout = 1.0 - norm_cdf(z_score)
        else:
            prob_stockout = 0.0 if cash_margin >= 0 else 1.0

        prob_stockout = max(0.0, min(1.0, prob_stockout))

        # Expected stockout costs
        expected_stockout_cost = prob_stockout * (stockout_penalty + emergency_cost)

        # Fixed CIT replenishment trip cost (incurred if we dispatch a vehicle)
        trip_cost = cit_cost if recommended_load > 0 else 0.0

        # Total Expected Operational Rupee Cost
        total_cost = total_holding_cost + trip_cost + expected_stockout_cost

        # Implied Cash Exhaustion Rate (CER = Forecast / Available)
        implied_cer = (
            (predicted_7day_sum / total_initial_vault)
            if total_initial_vault > 0
            else 1.0
        )

        return {
            "terminal_id": terminal_id,
            "capacity_inr": capacity,
            "target_service_level": sla,
            "service_level_z": round(z, 3),
            "predicted_7day_demand": round(predicted_7day_sum, 2),
            "sigma_7": round(sigma_7, 2),
            "safety_buffer_inr": round(safety_buffer, 2),
            "recommended_load_inr": round(recommended_load, 2),
            "total_starting_cash": round(total_initial_vault, 2),
            "is_capacity_capped": is_capacity_capped,
            "projected_inventory_trajectory": projected_inventory,
            "projected_ending_cash": round(projected_inventory[-1] if projected_inventory else 0.0, 2),
            "stockout_probability_pct": round(prob_stockout * 100, 3),
            "implied_cer": round(implied_cer, 4),
            # Rupee Cost Breakdown
            "holding_cost_inr": round(total_holding_cost, 2),
            "cit_trip_cost_inr": round(trip_cost, 2),
            "expected_stockout_cost_inr": round(expected_stockout_cost, 2),
            "total_expected_cost_inr": round(total_cost, 2),
        }

    def compute_cost_frontier(
        self,
        predicted_7day_sum: float,
        daily_predictions: List[float],
        sigma_7: float,
        terminal_id: int = 1,
        holding_rate: float = 0.065,
        cit_cost: float = 2000.0,
        stockout_penalty: float = 10000.0,
    ) -> List[Dict[str, any]]:
        """Compute cost curves across various service levels (90% to 99.9%)."""
        service_levels = [0.90, 0.92, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999]
        frontier = []
        for sla in service_levels:
            res = self.calculate_recommendation(
                predicted_7day_sum=predicted_7day_sum,
                daily_predictions=daily_predictions,
                sigma_7=sigma_7,
                initial_cash=0.0,
                terminal_id=terminal_id,
                service_level=sla,
                override_holding_rate=holding_rate,
                override_cit_cost=cit_cost,
                override_stockout_penalty=stockout_penalty,
            )
            frontier.append({
                "service_level": sla,
                "service_level_pct": f"{sla*100:.1f}%",
                "safety_buffer_lakh": round(res["safety_buffer_inr"] / 1e5, 2),
                "recommended_load_lakh": round(res["recommended_load_inr"] / 1e5, 2),
                "holding_cost": res["holding_cost_inr"],
                "stockout_cost": res["expected_stockout_cost_inr"],
                "total_cost": res["total_expected_cost_inr"],
                "stockout_prob_pct": res["stockout_probability_pct"],
            })
        return frontier
