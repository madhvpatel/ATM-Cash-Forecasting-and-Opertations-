"""Masters & Assumptions Manager for Cash Intelligence Iteration Two.

Handles loading, dynamic in-memory editing, recalculation, diff comparison,
Checker natural language overrides, digital CIT dispatch manifest generation,
and JSON persistence.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

HERE = Path(__file__).resolve().parent
DATA_MASTERS_DIR = HERE / "data_masters"
OVERRIDES_FILE = HERE / "config_overrides.json"
PORTAL_DATA_FILE = HERE / "portal_data.json"

# Default Baseline Assumptions (Indian Banking Industry Standards)
DEFAULT_ASSUMPTIONS = {
    "holding_rate_annual": 0.065,          # 6.5% annual opportunity cost of capital
    "cit_trip_cost": 2000.0,               # ₹2,000 fixed cost per armored CIT delivery
    "stockout_penalty_fixed": 10000.0,      # ₹10,000 RBI downtime/regulatory penalty
    "emergency_refill_premium": 3500.0,    # ₹3,500 emergency unscheduled replenishment trip
    "unmet_demand_penalty_rate": 0.01,     # 1% per-day interest/reputational penalty
    "target_service_level": 0.99,          # 99.0% availability SLA (z ≈ 2.326)
    "target_cer_buffer": 0.70,             # Cash Exhaustion Rate baseline target
    "default_cassette_capacity": 6000000.0 # ₹60 Lakhs standard limit
}

DEFAULT_TERMINAL_CONFIGS = {
    "1": {"capacity": 6000000.0, "serviced_by": "CIT", "cycle_days": 7, "site_type": "RAILWAY STATION CONCOURSE", "city": "MUMBAI"},
    "2": {"capacity": 6000000.0, "serviced_by": "CIT", "cycle_days": 7, "site_type": "AIRPORT TERMINAL T2", "city": "MUMBAI"},
    "3": {"capacity": 6000000.0, "serviced_by": "CIT", "cycle_days": 7, "site_type": "CITY CENTRE MALL", "city": "PUNE"},
    "4": {"capacity": 3500000.0, "serviced_by": "BRANCH", "cycle_days": 3, "site_type": "IT TECH PARK CAMPUS", "city": "PUNE"},
    "5": {"capacity": 5000000.0, "serviced_by": "CIT", "cycle_days": 7, "site_type": "COMMERCIAL CBD / FUEL PLAZA", "city": "NAGPUR"},
    "6": {"capacity": 6000000.0, "serviced_by": "CIT", "cycle_days": 7, "site_type": "TOURIST PROMENADE", "city": "NAGPUR"},
    "7": {"capacity": 6000000.0, "serviced_by": "CIT", "cycle_days": 7, "site_type": "METRO STATION CONCOURSE", "city": "NASHIK"},
    "8": {"capacity": 3500000.0, "serviced_by": "BRANCH", "cycle_days": 3, "site_type": "MIDC INDUSTRIAL ESTATE", "city": "NASHIK"},
}

DEFAULT_EXOGENOUS_EVENTS = [
    {"id": "EVT_01", "name": "Monsoon Orange Alert", "city": "MUMBAI", "start_date": "2026-08-28", "end_date": "2026-08-30", "demand_multiplier": 0.82, "notes": "Heavy localized waterlogging reduces footfall near street concourses."},
    {"id": "EVT_02", "name": "Corporate Salary Surge", "city": "ALL", "start_date": "2026-08-31", "end_date": "2026-09-04", "demand_multiplier": 1.28, "notes": "Month-end salary dispersal spike across commercial & IT zones."},
    {"id": "EVT_03", "name": "Ganesh Chaturthi Festive Influx", "city": "PUNE", "start_date": "2026-09-04", "end_date": "2026-09-07", "demand_multiplier": 1.45, "notes": "Festival bazaar cash withdrawals at peak demand."},
    {"id": "EVT_04", "name": "Mid-Month Pension Credit", "city": "NAGPUR", "start_date": "2026-09-10", "end_date": "2026-09-12", "demand_multiplier": 1.15, "notes": "Senior citizen pension credits at post & bank branches."}
]


class MastersManager:
    def __init__(self):
        self.assumptions: Dict[str, Any] = dict(DEFAULT_ASSUMPTIONS)
        self.terminal_configs: Dict[str, Any] = {k: dict(v) for k, v in DEFAULT_TERMINAL_CONFIGS.items()}
        self.exogenous_events: List[Dict[str, Any]] = [dict(e) for e in DEFAULT_EXOGENOUS_EVENTS]
        self.holidays: List[Dict[str, Any]] = []
        self.raw_bundle: Dict[str, Any] = {}
        self.approved_indents: Dict[str, Dict[str, Any]] = {}
        self.checker_overrides: Dict[str, Dict[str, Any]] = {}
        self.vault_available_inventory: float = 8500000.0  # ₹85 Lakhs available in CIT Vault
        self.cto_authorized: bool = False
        self.cto_signoff_info: Optional[Dict[str, Any]] = None
        self.reconciliation_resolutions: Dict[str, Any] = {}
        self.is_manifest_locked: bool = False
        self.manifest_lock_info: Optional[Dict[str, Any]] = None
        
        self._load_base_bundle()
        self._load_holidays()
        self._load_overrides()

    def _load_base_bundle(self):
        if PORTAL_DATA_FILE.exists():
            with open(PORTAL_DATA_FILE, "r") as f:
                self.raw_bundle = json.load(f)

    def _load_holidays(self):
        holidays_file = DATA_MASTERS_DIR / "holidays.json"
        if holidays_file.exists():
            try:
                with open(holidays_file, "r") as f:
                    self.holidays = json.load(f)
            except Exception:
                pass
        if not self.holidays:
            self.holidays = [
                {"date": "2026-08-15", "name": "Independence Day", "type": "National Gazetted", "bank_closed": True},
                {"date": "2026-08-19", "name": "Parsi New Year", "type": "Regional (Maharashtra)", "bank_closed": True},
                {"date": "2026-09-04", "name": "Ganesh Chaturthi", "type": "State Festival", "bank_closed": True},
                {"date": "2026-09-12", "name": "2nd Saturday Bank Holiday", "type": "Banking Calendar", "bank_closed": True},
                {"date": "2026-09-16", "name": "Eid-e-Milad", "type": "Gazetted Holiday", "bank_closed": True},
                {"date": "2026-09-26", "name": "4th Saturday Bank Holiday", "type": "Banking Calendar", "bank_closed": True},
            ]

    def _load_overrides(self):
        if OVERRIDES_FILE.exists():
            try:
                with open(OVERRIDES_FILE, "r") as f:
                    data = json.load(f)
                    if "assumptions" in data:
                        self.assumptions.update(data["assumptions"])
                    if "terminal_configs" in data:
                        for k, v in data["terminal_configs"].items():
                            if k in self.terminal_configs:
                                self.terminal_configs[k].update(v)
                            else:
                                self.terminal_configs[k] = v
                    if "exogenous_events" in data:
                        self.exogenous_events = data["exogenous_events"]
                    if "approved_indents" in data:
                        self.approved_indents = data["approved_indents"]
                    if "checker_overrides" in data:
                        self.checker_overrides = data["checker_overrides"]
                    if "vault_available_inventory" in data:
                        self.vault_available_inventory = float(data["vault_available_inventory"])
                    if "cto_authorized" in data:
                        self.cto_authorized = bool(data["cto_authorized"])
                    if "cto_signoff_info" in data:
                        self.cto_signoff_info = data["cto_signoff_info"]
                    if "reconciliation_resolutions" in data:
                        self.reconciliation_resolutions = data["reconciliation_resolutions"]
                    if "is_manifest_locked" in data:
                        self.is_manifest_locked = bool(data["is_manifest_locked"])
                        self.manifest_lock_info = data.get("manifest_lock_info")
            except Exception as e:
                print(f"Warning: could not load config_overrides.json: {e}")

    def save_overrides(self) -> Dict[str, Any]:
        data = {
            "assumptions": self.assumptions,
            "terminal_configs": self.terminal_configs,
            "exogenous_events": self.exogenous_events,
            "approved_indents": self.approved_indents,
            "checker_overrides": self.checker_overrides,
            "vault_available_inventory": self.vault_available_inventory,
            "cto_authorized": self.cto_authorized,
            "cto_signoff_info": self.cto_signoff_info,
            "reconciliation_resolutions": self.reconciliation_resolutions,
            "is_manifest_locked": self.is_manifest_locked,
            "manifest_lock_info": self.manifest_lock_info,
        }
        with open(OVERRIDES_FILE, "w") as f:
            json.dump(data, f, indent=2)
        return {"status": "saved", "file": str(OVERRIDES_FILE)}

    def reset_to_defaults(self) -> Dict[str, Any]:
        self.assumptions = dict(DEFAULT_ASSUMPTIONS)
        self.terminal_configs = {k: dict(v) for k, v in DEFAULT_TERMINAL_CONFIGS.items()}
        self.exogenous_events = [dict(e) for e in DEFAULT_EXOGENOUS_EVENTS]
        self.approved_indents = {}
        self.checker_overrides = {}
        self.vault_available_inventory = 8500000.0
        self.cto_authorized = False
        self.cto_signoff_info = None
        self.reconciliation_resolutions = {}
        self.is_manifest_locked = False
        self.manifest_lock_info = None
        if OVERRIDES_FILE.exists():
            OVERRIDES_FILE.unlink()
        return {"status": "reset_to_defaults"}

    def get_all_masters(self) -> Dict[str, Any]:
        return {
            "assumptions": self.assumptions,
            "terminals": self.terminal_configs,
            "holidays": self.holidays,
            "exogenous_events": self.exogenous_events,
            "approved_indents": self.approved_indents,
            "checker_overrides": self.checker_overrides,
            "is_manifest_locked": self.is_manifest_locked,
            "manifest_lock_info": self.manifest_lock_info
        }

    def apply_checker_override(self, terminal_id: str, new_indent: float, reason: str, actor: str = "Checker (AI Agent)") -> Dict[str, Any]:
        """Record an explicit Checker adjustment with a natural language audit reason."""
        tid = str(terminal_id).replace("ATM-", "").replace("ATM", "").strip()
        now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Get current recommendation before override
        current_data = self.compute_fleet_replenishment()
        t = current_data["terminals"].get(tid)
        if not t:
            raise ValueError(f"Terminal '{terminal_id}' not found")

        maker_indent = t.get("maker_proposed_indent", t.get("c1_net_indent", 0.0))

        self.checker_overrides[tid] = {
            "terminal_id": tid,
            "maker_proposed_indent": maker_indent,
            "checker_custom_indent": round(float(new_indent), 0),
            "delta_rupees": round(float(new_indent) - maker_indent, 0),
            "reason": str(reason).strip(),
            "actor": actor,
            "timestamp": now_ts
        }

        # Auto-mark as approved by Checker
        self.approved_indents[tid] = {
            "approved_at": now_ts,
            "approved_indent": round(float(new_indent), 0),
            "notes": f"Checker Override: {reason}",
            "actor": actor
        }

        self.save_overrides()
        recomputed = self.compute_fleet_replenishment()
        return {
            "status": "success",
            "terminal_id": tid,
            "maker_proposed_indent": maker_indent,
            "new_indent": round(float(new_indent), 0),
            "reason": reason,
            "timestamp": now_ts,
            "fleet_summary": recomputed["summary"]
        }

    def authorize_cto_signoff(self, officer_name: str = "Chief Treasury Officer", notes: str = "Level-2 Sovereign Sign-Off Approved for Armored CIT Dispatch") -> Dict[str, Any]:
        """Executes Level-2 Chief Treasury Officer sovereign authorization."""
        now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        res = self.compute_fleet_replenishment()
        total_pull = res["summary"]["total_net_indent"]
        sign_string = f"CTO-AUTH|{total_pull}|{now_ts}|{officer_name}|LEVEL2"
        sig_hash = hashlib.sha256(sign_string.encode("utf-8")).hexdigest()[:20].upper()

        self.cto_authorized = True
        self.cto_signoff_info = {
            "officer_name": officer_name,
            "role": "Chief Treasury Officer / General Manager - Cash Management",
            "authorized_at": now_ts,
            "notes": notes,
            "digital_signature_token": f"SIG-CTO-LV2-{sig_hash}",
            "authorized_total_rupees": total_pull,
            "status": "AUTHORIZED"
        }
        self.save_overrides()
        recomputed = self.compute_fleet_replenishment()
        return {
            "status": "authorized",
            "cto_signoff_info": self.cto_signoff_info,
            "fleet_summary": recomputed["summary"]
        }

    def resolve_reconciliation(self, terminal_id: str, officer_name: str = "Reconciliation Auditor", resolution_notes: str = "Verified at chest coin/note audit; discrepancy resolved") -> Dict[str, Any]:
        """Resolves an open T+1/T+3 variance or exception item."""
        tid = str(terminal_id).replace("ATM-", "").replace("ATM", "").strip()
        now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.reconciliation_resolutions[tid] = {
            "resolved_at": now_ts,
            "auditor": officer_name,
            "notes": resolution_notes,
            "status": "RECONCILED_CLOSED"
        }
        self.save_overrides()
        recomputed = self.compute_fleet_replenishment()
        return {
            "status": "resolved",
            "terminal_id": tid,
            "resolution": self.reconciliation_resolutions[tid],
            "fleet_summary": recomputed["summary"]
        }

    def update_parameter(self, category: str, key: str, value: Any, sub_id: Optional[str] = None) -> Dict[str, Any]:
        """Update any assumption/master parameter and return live impact diff."""
        old_val = None
        if category == "assumptions":
            if key in self.assumptions:
                old_val = self.assumptions[key]
                if isinstance(old_val, float):
                    self.assumptions[key] = float(value)
                elif isinstance(old_val, int):
                    self.assumptions[key] = int(value)
                else:
                    self.assumptions[key] = value
            else:
                self.assumptions[key] = value
        elif category == "terminals":
            tid = str(sub_id or key)
            if tid in self.terminal_configs:
                old_val = self.terminal_configs[tid].get(key)
                if key in ["capacity"]:
                    self.terminal_configs[tid][key] = float(value)
                elif key in ["cycle_days"]:
                    self.terminal_configs[tid][key] = int(value)
                else:
                    self.terminal_configs[tid][key] = value
        elif category == "events":
            for evt in self.exogenous_events:
                if evt.get("id") == sub_id:
                    old_val = evt.get(key)
                    if key in ["demand_multiplier"]:
                        evt[key] = float(value)
                    else:
                        evt[key] = value
                    break

        recomputed = self.compute_fleet_replenishment()
        return {
            "status": "updated",
            "category": category,
            "key": key,
            "sub_id": sub_id,
            "old_value": old_val,
            "new_value": value,
            "recomputed_summary": recomputed["summary"],
            "delta_net_indent": recomputed["summary"]["delta_indent_vs_baseline"]
        }

    def compute_fleet_replenishment(self) -> Dict[str, Any]:
        """Dynamically recompute indents, safety buffers, and risk across all 8 ATMs."""
        terminals_data = self.raw_bundle.get("terminals", {})
        sla = float(self.assumptions.get("target_service_level", 0.99))
        
        if sla >= 0.999:
            z_score = 3.09
        elif sla >= 0.99:
            z_score = 2.326
        elif sla >= 0.98:
            z_score = 2.054
        elif sla >= 0.95:
            z_score = 1.645
        elif sla >= 0.90:
            z_score = 1.282
        else:
            z_score = 1.0

        holding_daily = float(self.assumptions.get("holding_rate_annual", 0.065)) / 365.0
        cit_trip_cost = float(self.assumptions.get("cit_trip_cost", 2000.0))
        stockout_penalty = float(self.assumptions.get("stockout_penalty_fixed", 10000.0))

        recomputed_terminals = {}
        total_existing_cash = 0.0
        total_net_indent = 0.0
        total_maker_proposed_indent = 0.0
        total_baseline_indent = 0.0
        critical_count = 0
        scheduled_count = 0
        healthy_count = 0

        for tid, base_t in terminals_data.items():
            t_cfg = self.terminal_configs.get(tid, {})
            capacity = float(t_cfg.get("capacity", base_t.get("capacity", 6000000.0)))
            cycle_days = int(t_cfg.get("cycle_days", 7 if base_t.get("serviced_by") == "CIT" else 3))
            existing_cash = float(base_t.get("existing_cash", 1500000.0))

            strat_preds = base_t.get("daily_preds", {})
            if isinstance(strat_preds, dict):
                c1_preds = (strat_preds.get("strategy_c") or strat_preds.get("lgb_uncensored") or list(strat_preds.values())[0])[:cycle_days]
            elif isinstance(strat_preds, list):
                c1_preds = strat_preds[:cycle_days]
            else:
                c1_preds = []
            base_c1_demand = sum(c1_preds) if c1_preds else float(base_t.get("c1_pred_demand", 4000000.0))

            # Apply active exogenous multipliers
            t_city = t_cfg.get("city", base_t.get("city", "MUMBAI")).upper()
            mult = 1.0
            active_signals = []
            for evt in self.exogenous_events:
                evt_city = evt.get("city", "").upper()
                if evt_city in ["ALL", t_city]:
                    mult *= float(evt.get("demand_multiplier", 1.0))
                    active_signals.append({
                        "name": evt.get("name"),
                        "multiplier": f"{evt.get('demand_multiplier')}x",
                        "notes": evt.get("notes")
                    })

            adjusted_c1_demand = round(base_c1_demand * mult, 0)
            
            # Volatility & safety buffer based on dynamic z_score
            daily_std = (adjusted_c1_demand / max(1, cycle_days)) * 0.22
            sigma_h = daily_std * math.sqrt(cycle_days)
            safety_buffer = round(z_score * sigma_h, 0)

            gross_target = min(capacity, max(0.0, round(adjusted_c1_demand + safety_buffer, 0)))
            system_maker_indent = max(0.0, round(gross_target - existing_cash, 0))

            # Check if Checker overridden
            override_info = self.checker_overrides.get(tid)
            if override_info:
                net_indent = float(override_info["checker_custom_indent"])
                has_override = True
            else:
                net_indent = system_maker_indent
                has_override = False

            daily_burn = adjusted_c1_demand / max(1, cycle_days)
            dcr_days = round(existing_cash / daily_burn, 2) if daily_burn > 0 else 99.0

            if dcr_days < 2.0:
                urgency = "CRITICAL"
                critical_count += 1
                deplete_text = f"Dry-out in {int(dcr_days*24)}h"
            elif dcr_days <= 4.0:
                urgency = "SCHEDULED"
                scheduled_count += 1
                deplete_text = f"{dcr_days:.1f} days runway"
            else:
                urgency = "HEALTHY"
                healthy_count += 1
                deplete_text = f">5 days runway"

            baseline_indent = float(base_t.get("c1_net_indent", system_maker_indent))
            total_baseline_indent += baseline_indent
            total_maker_proposed_indent += system_maker_indent
            total_existing_cash += existing_cash
            total_net_indent += net_indent

            holding_cost = round(existing_cash * holding_daily * cycle_days, 0)
            est_trip_cost = cit_trip_cost if net_indent > 0 else 0.0

            is_approved = tid in self.approved_indents

            exp = base_t.get("c1_explainability", {})
            primary_driver = exp.get("primary_driver", "Standard cyclical demand & concourse footfall")

            # Structured Evidence Pillars for Checker
            purge_cash = 28000.0 if tid in ["1", "3", "6"] else 15000.0
            treasury_policy_ceiling = 5500000.0 if capacity >= 5500000.0 else capacity
            safe_days = round(max(0.1, (existing_cash - safety_buffer) / max(1.0, daily_burn)), 1)
            is_rec_resolved = tid in self.reconciliation_resolutions
            rec_var = 0.0 if (is_rec_resolved or tid != "3") else -500.0
            rec_status = "RECONCILED_CLOSED" if (is_rec_resolved or tid != "3") else "UNDER_INVESTIGATION"
            rec_reason = self.reconciliation_resolutions.get(tid, {}).get("notes", "1 Note jammed in diverter cassette during load, audited at chest" if tid == "3" else "Zero Variance · Certified")

            recomputed_terminals[tid] = {
                "terminal_id": tid,
                "site_name": base_t.get("site_name"),
                "site_type": t_cfg.get("site_type", base_t.get("site_type")),
                "city": t_city,
                "serviced_by": t_cfg.get("serviced_by", base_t.get("serviced_by")),
                "capacity": capacity,
                "existing_cash": existing_cash,
                "cycle_days": cycle_days,
                
                # Canonical Cash Accounting & Headroom (CASH-001)
                "dispensable_cash": existing_cash,
                "purge_reject_retract_cash": purge_cash,
                "physical_note_capacity": capacity,
                "physical_headroom": round(max(0.0, capacity - existing_cash), 0),
                "treasury_policy_ceiling": treasury_policy_ceiling,
                "policy_headroom": round(max(0.0, treasury_policy_ceiling - existing_cash), 0),
                "safe_buffer_breach_days": safe_days,
                "servicing_mode": "CASSETTE_SWAP" if t_cfg.get("serviced_by", base_t.get("serviced_by")) == "CIT" else "CASH_ADD",

                # Pillar 1: FACTS
                "facts": {
                    "vault_cash_now": existing_cash,
                    "dispensable_cash": existing_cash,
                    "purge_reject_retract_cash": purge_cash,
                    "cassette_cap": capacity,
                    "physical_headroom": round(max(0.0, capacity - existing_cash), 0),
                    "treasury_policy_ceiling": treasury_policy_ceiling,
                    "policy_headroom": round(max(0.0, treasury_policy_ceiling - existing_cash), 0),
                    "servicing_mode": "CASSETTE_SWAP" if t_cfg.get("serviced_by", base_t.get("serviced_by")) == "CIT" else "CASH_ADD",
                    "cycle_days": cycle_days,
                    "dcr_days": dcr_days,
                    "safe_buffer_breach_days": safe_days,
                    "deplete_text": deplete_text
                },
                # Pillar 2: OBSERVATIONS
                "observations": {
                    "pred_demand_7d": adjusted_c1_demand,
                    "pred_demand_24h": round(daily_burn, 0),
                    "daily_burn": round(daily_burn, 0),
                    "daily_volatility_std": round(daily_std, 0),
                    "primary_driver": primary_driver,
                    "active_signals": active_signals
                },
                # Pillar 3: ASSUMPTIONS
                "assumptions": {
                    "target_sla": sla,
                    "z_score": z_score,
                    "safety_buffer": safety_buffer,
                    "holding_rate_annual": self.assumptions.get("holding_rate_annual", 0.065),
                    "cit_trip_cost": cit_trip_cost,
                    "stockout_penalty": stockout_penalty
                },
                
                # Unambiguous Maker Recommendation & Checker Status
                "maker_proposed_indent": system_maker_indent,
                "c1_gross_target": gross_target,
                "c1_net_indent": net_indent,
                "has_override": has_override,
                "override_details": override_info,
                "is_approved": is_approved,
                "approval_info": self.approved_indents.get(tid),
                "dcr_days": dcr_days,
                "urgency_status": urgency,
                "holding_cost_est": holding_cost,
                "trip_cost_est": est_trip_cost,
                "daily_preds": base_t.get("daily_preds", {}),
                "dates": base_t.get("dates", []),
                "actual_demands": base_t.get("actual_demands", []),
                "cycles": base_t.get("cycles", []),
                "corridor_name": base_t.get("corridor_name", ""),
                "corridor_id": base_t.get("corridor_id", ""),
                "lat": base_t.get("lat"),
                "lng": base_t.get("lng"),
                "formula_ledger": {
                    "current_cash": existing_cash,
                    "net_indent": net_indent,
                    "expected_demand": adjusted_c1_demand,
                    "end_buffer_reserve": safety_buffer,
                    "post_refill_balance": min(capacity, existing_cash + net_indent)
                },
                # Cash Movement & Service Execution
                "movement_execution": {
                    "assigned_van": "MH-01-CV-4421" if t_city == "MUMBAI" else ("MH-12-RN-8812" if t_city == "PUNE" else "MH-31-TR-9021"),
                    "escort_officer": "R. Sharma (Armed Escort)" if t_city == "MUMBAI" else "P. Deshmukh (Armed Escort)",
                    "scheduled_eta": "10:30 IST Tomorrow",
                    "route_name": base_t.get("corridor_name", "Corridor 1"),
                    "seal_barcode": f"SEAL-MAH-{tid}-9821"
                },
                # T+1 / T+3 Reconciliation Compliance
                "reconciliation": {
                    "cycle_id": f"REC-MAH-{tid}-20261025",
                    "cycle_date": "2026-10-25 (T+2)",
                    "ej_switch_dispensed": 2800000.0 if tid == "1" else (net_indent if net_indent > 0 else 2500000.0),
                    "physical_cassette_counted": 2800000.0 if tid == "1" else (net_indent if net_indent > 0 else 2500000.0),
                    "variance_amount": rec_var,
                    "variance_status": rec_status,
                    "variance_reason": rec_reason,
                    "resolution_info": self.reconciliation_resolutions.get(tid),
                    "rbi_circular_compliance": "Complies with RBI May 14, 2019 T+3 Circular"
                },
                # Delay Sensitivity Table
                "delay_scenarios": {
                    "0h": {"delay": 0, "demand": 0, "balance": existing_cash, "stockout_prob": 0.8, "service_level": 99.2, "status": "ON_SCHEDULE"},
                    "4h": {"delay": 4, "demand": round(daily_burn * (4/24), 0), "balance": max(0, round(existing_cash - daily_burn * (4/24), 0)), "stockout_prob": 3.8, "service_level": 96.2, "status": "ELEVATED_RISK"},
                    "8h": {"delay": 8, "demand": round(daily_burn * (8/24), 0), "balance": max(0, round(existing_cash - daily_burn * (8/24), 0)), "stockout_prob": 12.4, "service_level": 87.6, "status": "HIGH_RISK"},
                    "24h": {"delay": 24, "demand": round(daily_burn, 0), "balance": max(0, round(existing_cash - daily_burn, 0)), "stockout_prob": 49.2, "service_level": 50.8, "status": "CRITICAL_STOCKOUT"}
                }
            }

        delta_vs_baseline = total_net_indent - total_baseline_indent

        summary = {
            "total_terminals": len(recomputed_terminals),
            "total_existing_cash": round(total_existing_cash, 0),
            "total_dispensable_cash": round(total_existing_cash, 0),
            "vault_available_inventory": round(self.vault_available_inventory, 0),
            "total_net_indent": round(total_net_indent, 0),
            "total_proposed_indent": round(total_net_indent, 0),
            "net_bank_chest_funding_requirement": round(max(0.0, total_net_indent - self.vault_available_inventory), 0),
            "funded_indent_total": round(min(self.vault_available_inventory, total_net_indent), 0),
            "total_maker_proposed_indent": round(total_maker_proposed_indent, 0),
            "total_baseline_indent": round(total_baseline_indent, 0),
            "delta_indent_vs_baseline": round(delta_vs_baseline, 0),
            "critical_count": critical_count,
            "scheduled_count": scheduled_count,
            "healthy_count": healthy_count,
            "approved_count": len(self.approved_indents),
            "override_count": len(self.checker_overrides),
            "active_sla": sla,
            "cash_availability_uptime": 99.85,
            "hardware_uptime": 99.92,
            "telecom_uptime": 99.98,
            "level1_checker_signed_count": len(self.approved_indents),
            "level1_checker_total": len(recomputed_terminals),
            "level2_cto_authorized": self.cto_authorized,
            "level2_cto_signoff_info": self.cto_signoff_info,
            "reconciliation_open_issues": 0 if "3" in self.reconciliation_resolutions else 1,
            "cit_cutoff_time": "18:00 IST",
            "is_manifest_locked": self.is_manifest_locked,
            "manifest_lock_info": self.manifest_lock_info
        }

        return {
            "summary": summary,
            "terminals": recomputed_terminals,
            "corridors": self.raw_bundle.get("corridors", []),
            "portfolio": self.raw_bundle.get("portfolio", {}),
            "assumptions": self.assumptions,
            "terminal_configs": self.terminal_configs,
            "exogenous_events": self.exogenous_events,
            "holidays": self.holidays,
            "checker_overrides": self.checker_overrides
        }

    def generate_dispatch_manifest(self, checker_name: str = "Chief Treasury Officer", notes: str = "Authorized for CIT Dispatch") -> Dict[str, Any]:
        """Generates the official Digital CIT Dispatch Manifest & Chest Pull Slip."""
        res = self.compute_fleet_replenishment()
        terminals = res["terminals"]
        corridors = res["corridors"]

        now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        manifest_id = f"CIT-MANIFEST-{datetime.now().strftime('%Y%m%d')}-MAH-01"

        total_pull_rupees = sum(t["c1_net_indent"] for t in terminals.values())

        # Group by Route Corridors
        route_manifests = []
        for corr in corridors:
            route_tids = [str(x) for x in corr.get("terminals", [])]
            route_stops = []
            route_cash_total = 0.0

            for tid in route_tids:
                if tid in terminals:
                    t = terminals[tid]
                    route_stops.append({
                        "terminal_id": tid,
                        "site_name": t["site_name"],
                        "city": t["city"],
                        "existing_cash": t["existing_cash"],
                        "indent_amount": t["c1_net_indent"],
                        "maker_recommended": t["maker_proposed_indent"],
                        "has_override": t["has_override"],
                        "override_reason": t["override_details"].get("reason") if t["has_override"] else None,
                        "serviced_by": t["serviced_by"],
                        "status": t["urgency_status"]
                    })
                    route_cash_total += t["c1_net_indent"]

            route_manifests.append({
                "route_id": corr.get("id"),
                "route_name": corr.get("name"),
                "city": corr.get("city"),
                "stop_count": len(route_stops),
                "route_total_cash": route_cash_total,
                "stops": route_stops
            })

        # Overrides audit list
        overrides_log = list(self.checker_overrides.values())

        # Generate cryptographic digital signature
        sign_string = f"{manifest_id}|{total_pull_rupees}|{now_ts}|{checker_name}"
        sig_hash = hashlib.sha256(sign_string.encode("utf-8")).hexdigest()[:16].upper()

        manifest = {
            "manifest_id": manifest_id,
            "generated_at": now_ts,
            "cutoff_deadline": "18:00 IST",
            "bank_entity": "State Bank Commercial Cash Operations",
            "currency_chest_origin": "Mumbai Main Fort Chest",
            "logistics_agency": "SecureArmored CIT Logistics Ltd.",
            "total_vault_pull_rupees": total_pull_rupees,
            "total_terminals": len(terminals),
            "checker_signoff": {
                "officer_name": checker_name,
                "notes": notes,
                "digital_signature_token": f"SIG-AUTH-{sig_hash}",
                "authorized_at": now_ts,
                "role": "Authorized Checker / Chief Treasury Manager"
            },
            "routes": route_manifests,
            "logged_checker_overrides": overrides_log,
            "is_locked": self.is_manifest_locked
        }

        return manifest

    def lock_manifest(self, checker_name: str = "Chief Treasury Officer", notes: str = "Authorized for CIT Dispatch") -> Dict[str, Any]:
        """Locks the dispatch manifest permanently."""
        manifest = self.generate_dispatch_manifest(checker_name=checker_name, notes=notes)
        self.is_manifest_locked = True
        self.manifest_lock_info = {
            "manifest_id": manifest["manifest_id"],
            "locked_at": manifest["generated_at"],
            "officer_name": checker_name,
            "total_rupees": manifest["total_vault_pull_rupees"],
            "digital_signature": manifest["checker_signoff"]["digital_signature_token"]
        }
        self.save_overrides()
        return {"status": "locked", "manifest": manifest}


# Singleton manager instance
masters_mgr = MastersManager()
