"""Configuration and Cost Parameters for ATM Cash Replenishment Engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


HERE = Path(__file__).resolve().parent
CMS_ROOT = HERE.parent


@dataclass
class ReplenishmentConfig:
    """Configurable operational, economic, and logistical parameters."""

    # Financial / Cost Parameters (Indian Banking Industry Standards)
    holding_rate_annual: float = 0.065          # 6.5% annual opportunity cost of capital
    cit_trip_cost: float = 2000.0               # ₹2,000 fixed cost per armored CIT delivery
    stockout_penalty_fixed: float = 10000.0      # ₹10,000 RBI downtime/regulatory penalty
    emergency_refill_premium: float = 3500.0    # ₹3,500 emergency unscheduled replenishment trip
    unmet_demand_penalty_rate: float = 0.01     # 1% per-day interest/reputational penalty on unmet demand

    # Replenishment Policy & Logistics
    horizon_days: int = 7                       # Fixed weekly replenishment cycle (7 days)
    step_days: int = 7                          # Rolling review step (7 days)
    min_train_days: int = 365                   # Minimum warm-up days before first evaluation cutoff
    target_service_level: float = 0.99          # Default target availability (99.0% / z ≈ 2.33)
    target_cer_buffer: float = 0.70             # Cash Exhaustion Rate baseline target (Load = Forecast / 0.70)

    # Hardware & Physical Vault Constraints
    default_cassette_capacity: float = 6000000.0  # ₹60 Lakhs standard capacity limit
    atm_capacities: Dict[int, float] = field(default_factory=lambda: {
        1: 6000000.0,   # ATM 1: Mumbai Railway Concourse (High volume ~₹9.1L/day, capped at ₹60L max capacity)
        2: 6000000.0,   # ATM 2: Mumbai Airport Terminal (~₹7.9L/day, capped at ₹60L max capacity)
        3: 6000000.0,   # ATM 3: Pune Mall (~₹7.8L/day, capped at ₹60L max capacity)
        4: 3500000.0,   # ATM 4: Pune Education Complex (~₹2.7L/day, ₹35L capacity)
        5: 5000000.0,   # ATM 5: Nagpur Fuel Plaza (~₹4.9L/day, ₹50L capacity)
        6: 6000000.0,   # ATM 6: Nagpur Tourist Zone (~₹7.4L/day, capped at ₹60L max capacity)
        7: 6000000.0,   # ATM 7: Nashik Metro Concourse (High volume ~₹10.7L/day, capped at ₹60L max capacity)
        8: 3500000.0,   # ATM 8: Nashik Hospital (~₹3.1L/day, ₹35L capacity)
    })

    # Data file locations
    cms_root: Path = CMS_ROOT
    atm_daily_path: Path = CMS_ROOT / "simulated" / "maharashtra" / "data" / "atm_daily.csv"
    checkpoint_quality_path: Path = CMS_ROOT / "data" / "processed" / "inventory_checkpoint_quality.csv"
    output_dir: Path = HERE / "output"

    def get_capacity(self, terminal_id: int) -> float:
        """Return the physical cassette rupee limit for a specific terminal."""
        return self.atm_capacities.get(terminal_id, self.default_cassette_capacity)

    @property
    def daily_holding_rate(self) -> float:
        """Daily interest rate on idle cash."""
        return self.holding_rate_annual / 365.0
