"""Demand Uncensoring and Stockout Correction Module.

Identifies periods where historical cash withdrawals were artificially constrained
due to vault cashouts (zero-balance snapshot counts), and reconstructs unconstrained
true customer demand using active-interval run-rates.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from replenishment_engine.config import ReplenishmentConfig

logger = logging.getLogger(__name__)


def load_uncensored_panel(
    config: Optional[ReplenishmentConfig] = None,
) -> pd.DataFrame:
    """Load the daily ATM demand panel and augment with uncensored demand metrics.

    Returns:
        pd.DataFrame containing original features plus:
        - zero_balance_snapshot_count
        - hours_empty
        - active_fraction
        - is_cashout_day
        - uncensored_demand
        - lost_demand
    """
    if config is None:
        config = ReplenishmentConfig()

    atm_path = config.atm_daily_path
    qual_path = config.checkpoint_quality_path

    if not atm_path.exists():
        raise FileNotFoundError(f"ATM daily data not found at {atm_path}")
    if not qual_path.exists():
        raise FileNotFoundError(f"Checkpoint quality data not found at {qual_path}")

    panel = pd.read_csv(atm_path, parse_dates=["date"])
    panel["terminal_id"] = panel["terminal_id"].astype(int)

    qual = pd.read_csv(qual_path, parse_dates=["calendar_date"])
    qual["terminal_id"] = qual["terminal_id"].astype(int)

    qual_cols = [c for c in [
        "terminal_id",
        "calendar_date",
        "snapshot_count",
        "zero_balance_snapshot_count",
        "minimum_reported_end_cash",
        "maximum_reported_end_cash",
    ] if c in qual.columns]

    # Merge on terminal_id and date
    merged = pd.merge(
        panel,
        qual[qual_cols],
        left_on=["terminal_id", "date"],
        right_on=["terminal_id", "calendar_date"],
        how="left",
    )
    if "calendar_date" in merged.columns:
        merged = merged.drop(columns=["calendar_date"])

    # Fill missing quality metrics with clean day defaults
    merged["snapshot_count"] = merged["snapshot_count"].fillna(96)
    merged["zero_balance_snapshot_count"] = merged["zero_balance_snapshot_count"].fillna(0)

    # Calculate operational active fraction (capped at 0.20 to avoid extreme runaway multipliers)
    # 96 snapshots per day = 15-minute intervals
    zero_ratio = merged["zero_balance_snapshot_count"] / merged["snapshot_count"]
    merged["active_fraction"] = np.clip(1.0 - zero_ratio, 0.20, 1.0)

    # Calculate uncensored (unconstrained) true customer demand
    merged["uncensored_demand"] = merged["demand"] / merged["active_fraction"]
    merged["lost_demand"] = merged["uncensored_demand"] - merged["demand"]
    merged["is_cashout_day"] = (merged["zero_balance_snapshot_count"] > 0).astype(int)
    merged["hours_empty"] = merged["zero_balance_snapshot_count"] * 0.25

    merged = merged.sort_values(["terminal_id", "date"]).reset_index(drop=True)
    return merged


def get_cashout_audit_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Generate terminal-level summary of historical cashout frequency and lost demand."""
    summary = []
    for tid, grp in panel.groupby("terminal_id"):
        tot_obs = grp["demand"].sum()
        tot_unc = grp["uncensored_demand"].sum()
        lost = tot_unc - tot_obs
        n_cashout = int((grp["zero_balance_snapshot_count"] > 0).sum())
        total_days = len(grp)
        avg_empty = (
            grp.loc[grp["zero_balance_snapshot_count"] > 0, "hours_empty"].mean()
            if n_cashout > 0
            else 0.0
        )

        site_type = grp["loc_type"].iloc[0] if "loc_type" in grp.columns else "Unknown"
        city = grp["city"].iloc[0] if "city" in grp.columns else "Unknown"

        summary.append({
            "terminal_id": int(tid),
            "site_type": site_type,
            "city": city,
            "total_observed_days": total_days,
            "cashout_days": n_cashout,
            "cashout_rate_pct": round((n_cashout / total_days) * 100, 2),
            "avg_hours_empty_when_short": round(avg_empty, 2),
            "observed_demand_lakh": round(tot_obs / 1e5, 2),
            "uncensored_demand_lakh": round(tot_unc / 1e5, 2),
            "lost_demand_lakh": round(lost / 1e5, 2),
            "suppression_pct": round((lost / tot_obs) * 100, 2) if tot_obs > 0 else 0.0,
        })

    return pd.DataFrame(summary)
