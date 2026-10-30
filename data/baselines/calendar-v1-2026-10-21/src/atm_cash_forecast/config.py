from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class POCConfig:
    """All operational and model assumptions used by the POC."""

    terminal_id: int = 1001
    terminal_code: str = "ATM-MUM-001"
    history_days: int = 730
    holdout_days: int = 90
    simulation_days: int = 60
    review_period_days: int = 1
    lead_time_days: int = 1
    safety_buffer_pct: float = 0.10
    initial_inventory_ratio: float = 0.65
    historical_target_fill_ratio: float = 0.88
    historical_reorder_ratio: float = 0.28
    crm_target_fill_ratio: float = 0.65
    crm_reorder_ratio: float = 0.20
    crm_high_fill_ratio: float = 0.90
    crm_collection_max_notes: int = 3_000
    crm_collection_evacuation_ratio: float = 0.80
    crm_reject_bin_max_notes: int = 250
    holding_cost_rate_per_day: float = 0.00008
    transport_cost_per_trip: float = 2_000.0
    shortage_penalty_rate: float = 0.25
    cashout_event_penalty: float = 5_000.0
    denominations: tuple[int, ...] = (100, 200, 500, 2_000)
    cassette_max_notes: tuple[int, ...] = (2_000, 2_000, 2_000, 1_500)

    @property
    def protection_horizon_days(self) -> int:
        return self.review_period_days + self.lead_time_days

    def validate(self) -> None:
        if self.history_days < 90:
            raise ValueError("history_days must be at least 90")
        if self.holdout_days < 14 or self.holdout_days >= self.history_days - 28:
            raise ValueError("holdout_days must leave at least 28 training days")
        if self.simulation_days < 1 or self.simulation_days > self.holdout_days:
            raise ValueError("simulation_days must be between 1 and holdout_days")
        if self.review_period_days < 1 or self.lead_time_days < 0:
            raise ValueError("review_period_days must be positive and lead_time_days non-negative")
        if self.safety_buffer_pct < 0:
            raise ValueError("safety_buffer_pct cannot be negative")
        if not 0 < self.initial_inventory_ratio <= 1:
            raise ValueError("initial_inventory_ratio must be in (0, 1]")
        if len(self.denominations) != len(self.cassette_max_notes):
            raise ValueError("denominations and cassette_max_notes must have the same length")
        for field_name, value in {
            "crm_target_fill_ratio": self.crm_target_fill_ratio,
            "crm_reorder_ratio": self.crm_reorder_ratio,
            "crm_high_fill_ratio": self.crm_high_fill_ratio,
            "crm_collection_evacuation_ratio": self.crm_collection_evacuation_ratio,
        }.items():
            if not 0 < value < 1:
                raise ValueError(f"{field_name} must be in (0, 1)")
        if not self.crm_reorder_ratio < self.crm_target_fill_ratio < self.crm_high_fill_ratio:
            raise ValueError("CRM fill ratios must satisfy reorder < target < high")
        if self.crm_collection_max_notes < 1 or self.crm_reject_bin_max_notes < 1:
            raise ValueError("CRM bin capacities must be positive")


DEFAULT_CONFIG = POCConfig()
