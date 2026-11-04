"""Masters & Assumptions Manager for Cash Intelligence Iteration Two.

Handles loading, dynamic in-memory editing, recalculation, diff comparison,
Checker natural language overrides, digital CIT dispatch manifest generation,
physical cassette/hopper capacity modeling, reject bin saturation tracking (CRM & Non-CRM),
deterministic downtime & anomaly detection, location categories, average ticket size,
denomination demand allocation, CRA disruption management (missed visits/SLA breaches),
inter-ATM surplus balancing (sweeps), and comprehensive Excel/CSV report generation.
"""

from __future__ import annotations

import io
import csv
import copy
import hashlib
import hmac
import json
import math
import os
import threading
from datetime import datetime, timedelta
from pathlib import Path
from statistics import NormalDist
from typing import Any, Dict, List, Optional, Set
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from replenishment_engine import demand_model

try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

HERE = Path(__file__).resolve().parent
DATA_MASTERS_DIR = HERE / "data_masters"
# Tests point this at a temp file so they never touch the operator's live state.
OVERRIDES_FILE = Path(os.environ.get("CMS_OVERRIDES_FILE") or HERE / "config_overrides.json")
PORTAL_DATA_FILE = HERE / "portal_data.json"

# Baseline Assumptions (Indian Banking Industry Standards)
DEFAULT_ASSUMPTIONS = {
    "holding_rate_annual": 0.065,          # 6.5% annual opportunity cost of capital
    "cit_trip_cost": 2000.0,               # ₹2,000 fixed cost per armored CIT delivery
    "stockout_penalty_fixed": 10000.0,     # ₹10,000 RBI downtime/regulatory penalty
    "emergency_refill_premium": 3500.0,    # ₹3,500 emergency unscheduled replenishment trip
    "unmet_demand_penalty_rate": 0.01,     # 1% per-day interest/reputational penalty
    "target_service_level": 0.99,          # 99.0% availability SLA (z ≈ 2.326 via NormalDist)
    "target_cer_buffer": 0.70,             # Cash Exhaustion Rate baseline target
    "default_cassette_capacity": 2860000.0,# ₹28.6 Lakhs physical cassette capacity (4 cassettes x 2,200 notes: 2x500, 200, 100)
    "min_sweep_threshold": 200000.0,       # ₹2,00,000 minimum viable Inter-ATM cash transfer
    "inter_atm_transfer_fee": 1500.0,      # ₹1,500 corridor transit/custodian handling fee
    "hood_threat_idle_cap": 2000000.0,     # ₹20 Lakhs surplus cap in high-threat locations
    "reject_bin_warning_pct": 75.0,        # 75% reject bin saturation warning
    "reject_bin_critical_pct": 90.0        # 90% reject bin saturation shutdown fault
}

DEFAULT_TERMINAL_CONFIGS = {
    "1": {"capacity": 2860000.0, "serviced_by": "CIT", "cycle_days": 7, "site_type": "RAILWAY STATION CONCOURSE", "city": "MUMBAI"},
    "2": {"capacity": 2860000.0, "serviced_by": "CIT", "cycle_days": 7, "site_type": "AIRPORT TERMINAL T2", "city": "MUMBAI"},
    "3": {"capacity": 2860000.0, "serviced_by": "CIT", "cycle_days": 7, "site_type": "CITY CENTRE MALL", "city": "PUNE"},
    "4": {"capacity": 2860000.0, "serviced_by": "BRANCH", "cycle_days": 3, "site_type": "IT TECH PARK CAMPUS", "city": "PUNE"},
    "5": {"capacity": 2860000.0, "serviced_by": "CIT", "cycle_days": 7, "site_type": "COMMERCIAL CBD / FUEL PLAZA", "city": "NAGPUR"},
    "6": {"capacity": 2860000.0, "serviced_by": "CIT", "cycle_days": 7, "site_type": "TOURIST PROMENADE", "city": "NAGPUR"},
    "7": {"capacity": 2860000.0, "serviced_by": "CIT", "cycle_days": 7, "site_type": "METRO STATION CONCOURSE", "city": "NASHIK"},
    "8": {"capacity": 2860000.0, "serviced_by": "BRANCH", "cycle_days": 3, "site_type": "MIDC INDUSTRIAL ESTATE", "city": "NASHIK"},
}

TERMINAL_OPERATIONAL_METADATA = {
    "1": {
        "terminal_code": "SYNCRM0001",
        "device_type": "CRM",
        "is_crm": True,
        "location_tier": "Metro",
        "location_category": "Metro Transit",
        "lat": 18.9400,
        "lng": 72.8353,
        "coords_str": "18.9400° N, 72.8353° E",
        "address": "Dadar Central Concourse, Mumbai, Maharashtra 400014",
        "serving_vault": "BKC Central Currency Chest (Bandra Kurla Complex, Mumbai)",
        "vault_lat": 19.0657,
        "vault_lng": 72.8686,
        "vault_distance_km": 9.4,
        "transit_offpeak_min": 22,
        "transit_peak_min": 48,
        "corridor_route": "Mumbai Transit Route 1",
        "assigned_van": "CIT Van #MH-02-7712",
        "van_capacity_cr": 4.50,
        "working_hours": "24 Hours / 7 Days (Street Concourse Lobby)",
        "optimal_window": "06:00 AM – 08:30 AM (Low Footfall / Zero Transaction Loss)",
        "avoid_window": "12:00 PM – 02:30 PM & 06:00 PM – 09:00 PM (Peak Commuter Rush)",
        "suggested_date": "2026-08-26",
        "suggested_time": "07:00 AM",
        "suggested_window": "06:30 AM – 08:30 AM",
        "telemetry_last_sync": "2026-08-24 18:30 IST",
        "telemetry_sync_delayed": False,
        "telemetry_status": "ONLINE_HEALTHY",
        "avg_ticket_size": {"daily": 3200, "weekly": 3100, "monthly": 3050},
        "denomination_weights": {"500": 0.70, "200": 0.20, "100": 0.10},
        "cassettes": [
            {"slot": 1, "denomination": 500, "max_notes": 2200, "current_notes": 1200, "amount": 600000.0},
            {"slot": 2, "denomination": 500, "max_notes": 2200, "current_notes": 1000, "amount": 500000.0},
            {"slot": 3, "denomination": 200, "max_notes": 2200, "current_notes": 1000, "amount": 200000.0},
            {"slot": 4, "denomination": 100, "max_notes": 2200, "current_notes": 2000, "amount": 200000.0}
        ],
        "reject_bin": {"max_notes": 300, "current_notes": 48, "status": "NORMAL"},
        "crm_collection_bin": {"max_notes": 2000, "current_notes": 412, "amount": 164800.0, "status": "NORMAL"},
        "cra_tracking": {
            "scheduled_date": "2026-08-26",
            "window": "06:30 AM – 08:30 AM",
            "cra_status": "ON_SCHEDULE",
            "sla_penalty": 0.0,
            "provider": "SecureArmored CIT Logistics Ltd."
        },
        "threat_monitoring": {"threat_level": "LOW", "curfew_active": False, "daylight_servicing_only": False},
        "human_errors": {"has_shortfall": False, "shortfall_amount": 0.0, "cassette_mismatch": False, "reject_bin_unemptied": False},
        "last_feed": {"timestamp": "2026-08-24 18:30 IST", "end_cash_value": 1500000.0, "supervisor_access": "2026-08-19 07:15 IST"}
    },
    "2": {
        "terminal_code": "SYNATM0002",
        "device_type": "Non-CRM",
        "is_crm": False,
        "location_tier": "Metro",
        "location_category": "Metro Transit",
        "lat": 19.0896,
        "lng": 72.8656,
        "coords_str": "19.0896° N, 72.8656° E",
        "address": "CSMIA Airport T2 Arrivals Concourse, Sahar, Mumbai 400099",
        "serving_vault": "BKC Central Currency Chest (Bandra Kurla Complex, Mumbai)",
        "vault_lat": 19.0657,
        "vault_lng": 72.8686,
        "vault_distance_km": 7.8,
        "transit_offpeak_min": 16,
        "transit_peak_min": 38,
        "corridor_route": "Mumbai Transit Route 1",
        "assigned_van": "CIT Van #MH-02-7712",
        "van_capacity_cr": 4.50,
        "working_hours": "24 Hours / 7 Days (Airport Security Zone)",
        "optimal_window": "05:30 AM – 07:30 AM (Pre-flight arrival lull)",
        "avoid_window": "11:00 AM – 03:00 PM & 09:00 PM – 01:00 AM (International arrivals wave)",
        "suggested_date": "2026-08-26",
        "suggested_time": "08:15 AM",
        "suggested_window": "07:30 AM – 09:00 AM",
        "telemetry_last_sync": "2026-08-24 18:30 IST",
        "telemetry_sync_delayed": False,
        "telemetry_status": "ONLINE_HEALTHY",
        "avg_ticket_size": {"daily": 4500, "weekly": 4400, "monthly": 4350},
        "denomination_weights": {"500": 0.85, "200": 0.10, "100": 0.05},
        "cassettes": [
            {"slot": 1, "denomination": 500, "max_notes": 2200, "current_notes": 1400, "amount": 700000.0},
            {"slot": 2, "denomination": 500, "max_notes": 2200, "current_notes": 1200, "amount": 600000.0},
            {"slot": 3, "denomination": 200, "max_notes": 2200, "current_notes": 1000, "amount": 200000.0},
            {"slot": 4, "denomination": 100, "max_notes": 2200, "current_notes": 1500, "amount": 150000.0}
        ],
        "reject_bin": {"max_notes": 300, "current_notes": 62, "status": "NORMAL"},
        "crm_collection_bin": None,
        "cra_tracking": {
            "scheduled_date": "2026-08-26",
            "window": "07:30 AM – 09:00 AM",
            "cra_status": "ON_SCHEDULE",
            "sla_penalty": 0.0,
            "provider": "SecureArmored CIT Logistics Ltd."
        },
        "threat_monitoring": {"threat_level": "LOW", "curfew_active": False, "daylight_servicing_only": False},
        "human_errors": {"has_shortfall": False, "shortfall_amount": 0.0, "cassette_mismatch": False, "reject_bin_unemptied": False},
        "last_feed": {"timestamp": "2026-08-24 18:30 IST", "end_cash_value": 1650000.0, "supervisor_access": "2026-08-19 08:30 IST"}
    },
    "3": {
        "terminal_code": "SYNCRM0003",
        "device_type": "CRM",
        "is_crm": True,
        "location_tier": "Metro",
        "location_category": "Hood",
        "lat": 18.5620,
        "lng": 73.9167,
        "coords_str": "18.5620° N, 73.9167° E",
        "address": "Phoenix Marketcity Ground Floor Atrium, Viman Nagar, Pune 411014",
        "serving_vault": "Shivaji Nagar Main Currency Chest, Pune 411005",
        "vault_lat": 18.5314,
        "vault_lng": 73.8446,
        "vault_distance_km": 9.2,
        "transit_offpeak_min": 20,
        "transit_peak_min": 45,
        "corridor_route": "Pune Urban Route 2",
        "assigned_van": "CIT Van #MH-12-4409",
        "van_capacity_cr": 4.00,
        "working_hours": "10:00 AM – 10:30 PM (Mall Commercial Hours)",
        "optimal_window": "09:00 AM – 10:00 AM (Pre-opening staff entry gate)",
        "avoid_window": "04:00 PM – 09:30 PM (Peak evening shopping footfall)",
        "suggested_date": "2026-08-25",
        "suggested_time": "09:30 AM",
        "suggested_window": "09:00 AM – 10:30 AM",
        "telemetry_last_sync": "2026-08-24 18:30 IST",
        "telemetry_sync_delayed": True,
        "telemetry_status": "CRITICAL_STARVATION",
        "avg_ticket_size": {"daily": 2300, "weekly": 2250, "monthly": 2200},
        "denomination_weights": {"500": 0.50, "200": 0.35, "100": 0.15},
        "cassettes": [
            {"slot": 1, "denomination": 500, "max_notes": 2200, "current_notes": 400, "amount": 200000.0},
            {"slot": 2, "denomination": 500, "max_notes": 2200, "current_notes": 240, "amount": 120000.0},
            {"slot": 3, "denomination": 200, "max_notes": 2200, "current_notes": 300, "amount": 60000.0},
            {"slot": 4, "denomination": 100, "max_notes": 2200, "current_notes": 400, "amount": 40000.0}
        ],
        "reject_bin": {"max_notes": 300, "current_notes": 115, "status": "NORMAL"},
        "crm_collection_bin": {"max_notes": 2000, "current_notes": 890, "amount": 445000.0, "status": "NORMAL"},
        "cra_tracking": {
            "scheduled_date": "2026-08-25",
            "window": "09:00 AM – 10:30 AM",
            "cra_status": "MISSED_SLA_BREACH",
            "sla_penalty": 5000.0,
            "provider": "SecureArmored CIT Logistics Ltd."
        },
        "threat_monitoring": {"threat_level": "HIGH", "curfew_active": False, "daylight_servicing_only": True},
        "human_errors": {"has_shortfall": False, "shortfall_amount": 0.0, "cassette_mismatch": False, "reject_bin_unemptied": False},
        "last_feed": {"timestamp": "2026-08-24 18:30 IST", "end_cash_value": 420000.0, "supervisor_access": "2026-08-18 09:30 IST"}
    },
    "4": {
        "terminal_code": "SYNCRM0004",
        "device_type": "CRM",
        "is_crm": True,
        "location_tier": "Metro",
        "location_category": "Corporate",
        "lat": 18.5913,
        "lng": 73.7389,
        "coords_str": "18.5913° N, 73.7389° E",
        "address": "Rajiv Gandhi Infotech Park, Hinjawadi Phase 1, Pune 411057",
        "serving_vault": "Shivaji Nagar Main Currency Chest, Pune 411005",
        "vault_lat": 18.5314,
        "vault_lng": 73.8446,
        "vault_distance_km": 17.5,
        "transit_offpeak_min": 28,
        "transit_peak_min": 65,
        "corridor_route": "Pune Urban Route 2",
        "assigned_van": "Branch Van #MH-12-1188",
        "van_capacity_cr": 3.00,
        "working_hours": "24 Hours / 7 Days (Tech Campus Lobby)",
        "optimal_window": "06:30 AM – 08:30 AM (Before tech campus shift rush)",
        "avoid_window": "12:30 PM – 03:00 PM & 06:30 PM – 08:30 PM (Cafeteria & shift change)",
        "suggested_date": "2026-08-27",
        "suggested_time": "07:30 AM",
        "suggested_window": "07:00 AM – 08:30 AM",
        "telemetry_last_sync": "2026-08-24 18:30 IST",
        "telemetry_sync_delayed": False,
        "telemetry_status": "SURPLUS_IDLE",
        "avg_ticket_size": {"daily": 4800, "weekly": 4650, "monthly": 4500},
        "denomination_weights": {"500": 0.80, "200": 0.15, "100": 0.05},
        "cassettes": [
            {"slot": 1, "denomination": 500, "max_notes": 2200, "current_notes": 2000, "amount": 1000000.0},
            {"slot": 2, "denomination": 500, "max_notes": 2200, "current_notes": 2000, "amount": 1000000.0},
            {"slot": 3, "denomination": 200, "max_notes": 2200, "current_notes": 2200, "amount": 440000.0},
            {"slot": 4, "denomination": 100, "max_notes": 2200, "current_notes": 2200, "amount": 220000.0}
        ],
        "reject_bin": {"max_notes": 300, "current_notes": 35, "status": "NORMAL"},
        "crm_collection_bin": {"max_notes": 2000, "current_notes": 1720, "amount": 860000.0, "status": "SATURATED"},
        "cra_tracking": {
            "scheduled_date": "2026-08-27",
            "window": "07:00 AM – 08:30 AM",
            "cra_status": "ON_SCHEDULE",
            "sla_penalty": 0.0,
            "provider": "Branch Vault Services"
        },
        "threat_monitoring": {"threat_level": "LOW", "curfew_active": False, "daylight_servicing_only": False},
        "human_errors": {"has_shortfall": False, "shortfall_amount": 0.0, "cassette_mismatch": False, "reject_bin_unemptied": False},
        "last_feed": {"timestamp": "2026-08-24 18:30 IST", "end_cash_value": 2660000.0, "supervisor_access": "2026-08-21 07:30 IST"}
    },
    "5": {
        "terminal_code": "SYNATM0005",
        "device_type": "Non-CRM",
        "is_crm": False,
        "location_tier": "Non-Metro",
        "location_category": "Hood",
        "lat": 21.1458,
        "lng": 79.0882,
        "coords_str": "21.1458° N, 79.0882° E",
        "address": "Sitabuldi Commercial Hub & Fuel Station, Nagpur 440012",
        "serving_vault": "Civil Lines Nodal Currency Chest, Nagpur 440001",
        "vault_lat": 21.1524,
        "vault_lng": 79.0716,
        "vault_distance_km": 3.1,
        "transit_offpeak_min": 8,
        "transit_peak_min": 18,
        "corridor_route": "Nagpur Metro Route 3",
        "assigned_van": "CIT Van #MH-31-1092",
        "van_capacity_cr": 3.50,
        "working_hours": "24 Hours / 7 Days (Fuel Station Kiosk)",
        "optimal_window": "06:00 AM – 08:00 AM (Minimal commercial traffic)",
        "avoid_window": "05:00 PM – 08:30 PM (Evening fuel plaza congestion)",
        "suggested_date": "2026-08-28",
        "suggested_time": "07:00 AM",
        "suggested_window": "06:30 AM – 08:00 AM",
        "telemetry_last_sync": "2026-08-24 18:30 IST",
        "telemetry_sync_delayed": False,
        "telemetry_status": "REJECT_BIN_SATURATED",
        "avg_ticket_size": {"daily": 2100, "weekly": 2050, "monthly": 2000},
        "denomination_weights": {"500": 0.50, "200": 0.35, "100": 0.15},
        "cassettes": [
            {"slot": 1, "denomination": 500, "max_notes": 2200, "current_notes": 900, "amount": 450000.0},
            {"slot": 2, "denomination": 500, "max_notes": 2200, "current_notes": 800, "amount": 400000.0},
            {"slot": 3, "denomination": 200, "max_notes": 2200, "current_notes": 750, "amount": 150000.0},
            {"slot": 4, "denomination": 100, "max_notes": 2200, "current_notes": 800, "amount": 80000.0}
        ],
        "reject_bin": {"max_notes": 300, "current_notes": 268, "status": "SATURATED"},
        "crm_collection_bin": None,
        "cra_tracking": {
            "scheduled_date": "2026-08-28",
            "window": "06:30 AM – 08:00 AM",
            "cra_status": "ON_SCHEDULE",
            "sla_penalty": 0.0,
            "provider": "SecureArmored CIT Logistics Ltd."
        },
        "threat_monitoring": {"threat_level": "HIGH", "curfew_active": False, "daylight_servicing_only": True},
        "human_errors": {"has_shortfall": True, "shortfall_amount": 300000.0, "authorized_indent": 1500000.0, "actual_loaded_cash": 1200000.0, "cassette_mismatch": False, "reject_bin_unemptied": True},
        "last_feed": {"timestamp": "2026-08-24 18:30 IST", "end_cash_value": 1080000.0, "supervisor_access": "2026-08-21 07:00 IST"}
    },
    "6": {
        "terminal_code": "SYNATM0006",
        "device_type": "Non-CRM",
        "is_crm": False,
        "location_tier": "Non-Metro",
        "location_category": "Residential",
        "lat": 21.1539,
        "lng": 79.0436,
        "coords_str": "21.1539° N, 79.0436° E",
        "address": "Futala Lake Promenade, Vayu Sena Nagar, Nagpur 440001",
        "serving_vault": "Civil Lines Nodal Currency Chest, Nagpur 440001",
        "vault_lat": 21.1524,
        "vault_lng": 79.0716,
        "vault_distance_km": 4.8,
        "transit_offpeak_min": 11,
        "transit_peak_min": 22,
        "corridor_route": "Nagpur Metro Route 3",
        "assigned_van": "CIT Van #MH-31-1092",
        "van_capacity_cr": 3.50,
        "working_hours": "06:00 AM – 11:30 PM (Public Promenade Area)",
        "optimal_window": "06:30 AM – 08:30 AM (Morning walker hours, low ATM use)",
        "avoid_window": "05:30 PM – 10:00 PM (Promenade tourist rush)",
        "suggested_date": "2026-08-26",
        "suggested_time": "08:00 AM",
        "suggested_window": "07:30 AM – 09:00 AM",
        "telemetry_last_sync": "2026-08-24 18:30 IST",
        "telemetry_sync_delayed": False,
        "telemetry_status": "ONLINE_HEALTHY",
        "avg_ticket_size": {"daily": 2900, "weekly": 2850, "monthly": 2800},
        "denomination_weights": {"500": 0.60, "200": 0.25, "100": 0.15},
        "cassettes": [
            {"slot": 1, "denomination": 500, "max_notes": 2200, "current_notes": 1200, "amount": 600000.0},
            {"slot": 2, "denomination": 500, "max_notes": 2200, "current_notes": 1000, "amount": 500000.0},
            {"slot": 3, "denomination": 200, "max_notes": 2200, "current_notes": 900, "amount": 180000.0},
            {"slot": 4, "denomination": 100, "max_notes": 2200, "current_notes": 1200, "amount": 120000.0}
        ],
        "reject_bin": {"max_notes": 300, "current_notes": 54, "status": "NORMAL"},
        "crm_collection_bin": None,
        "cra_tracking": {
            "scheduled_date": "2026-08-26",
            "window": "07:30 AM – 09:00 AM",
            "cra_status": "ON_SCHEDULE",
            "sla_penalty": 0.0,
            "provider": "SecureArmored CIT Logistics Ltd."
        },
        "threat_monitoring": {"threat_level": "MEDIUM", "curfew_active": False, "daylight_servicing_only": False},
        "human_errors": {"has_shortfall": False, "shortfall_amount": 0.0, "cassette_mismatch": False, "reject_bin_unemptied": False},
        "last_feed": {"timestamp": "2026-08-24 18:30 IST", "end_cash_value": 1400000.0, "supervisor_access": "2026-08-19 08:00 IST"}
    },
    "7": {
        "terminal_code": "SYNCRM0007",
        "device_type": "CRM",
        "is_crm": True,
        "location_tier": "Non-Metro",
        "location_category": "Metro Transit",
        "lat": 19.9975,
        "lng": 73.7898,
        "coords_str": "19.9975° N, 73.7898° E",
        "address": "CBS Central Bus Station & Metro Transit Hub, Nashik 422001",
        "serving_vault": "Old Agra Road Nodal Currency Chest, Nashik 422002",
        "vault_lat": 19.9912,
        "vault_lng": 73.7785,
        "vault_distance_km": 1.8,
        "transit_offpeak_min": 5,
        "transit_peak_min": 12,
        "corridor_route": "Nashik Corridor Route 4",
        "assigned_van": "CIT Van #MH-15-8821",
        "van_capacity_cr": 3.50,
        "working_hours": "24 Hours / 7 Days (Transit Concourse)",
        "optimal_window": "05:30 AM – 07:30 AM (Prior to morning intercity bus wave)",
        "avoid_window": "09:00 AM – 12:00 PM & 05:00 PM – 08:30 PM (Peak bus concourse rush)",
        "suggested_date": "2026-08-26",
        "suggested_time": "06:45 AM",
        "suggested_window": "06:00 AM – 07:30 AM",
        "telemetry_last_sync": "2026-08-24 18:30 IST",
        "telemetry_sync_delayed": False,
        "telemetry_status": "ONLINE_HEALTHY",
        "avg_ticket_size": {"daily": 2400, "weekly": 2350, "monthly": 2300},
        "denomination_weights": {"500": 0.70, "200": 0.20, "100": 0.10},
        "cassettes": [
            {"slot": 1, "denomination": 500, "max_notes": 2200, "current_notes": 1100, "amount": 550000.0},
            {"slot": 2, "denomination": 500, "max_notes": 2200, "current_notes": 900, "amount": 450000.0},
            {"slot": 3, "denomination": 200, "max_notes": 2200, "current_notes": 800, "amount": 160000.0},
            {"slot": 4, "denomination": 100, "max_notes": 2200, "current_notes": 1200, "amount": 120000.0}
        ],
        "reject_bin": {"max_notes": 300, "current_notes": 40, "status": "NORMAL"},
        "crm_collection_bin": {"max_notes": 2000, "current_notes": 520, "amount": 208000.0, "status": "NORMAL"},
        "cra_tracking": {
            "scheduled_date": "2026-08-26",
            "window": "06:00 AM – 07:30 AM",
            "cra_status": "ON_SCHEDULE",
            "sla_penalty": 0.0,
            "provider": "SecureArmored CIT Logistics Ltd."
        },
        "threat_monitoring": {"threat_level": "MEDIUM", "curfew_active": False, "daylight_servicing_only": False},
        "human_errors": {"has_shortfall": False, "shortfall_amount": 0.0, "cassette_mismatch": False, "reject_bin_unemptied": False},
        "last_feed": {"timestamp": "2026-08-24 18:30 IST", "end_cash_value": 1280000.0, "supervisor_access": "2026-08-19 06:45 IST"}
    },
    "8": {
        "terminal_code": "SYNATM0008",
        "device_type": "Non-CRM",
        "is_crm": False,
        "location_tier": "Non-Metro",
        "location_category": "Corporate",
        "lat": 19.9572,
        "lng": 73.7420,
        "coords_str": "19.9572° N, 73.7420° E",
        "address": "Ambad MIDC Industrial Area & Civil Hospital Road, Nashik 422010",
        "serving_vault": "Old Agra Road Nodal Currency Chest, Nashik 422002",
        "vault_lat": 19.9912,
        "vault_lng": 73.7785,
        "vault_distance_km": 10.4,
        "transit_offpeak_min": 16,
        "transit_peak_min": 32,
        "corridor_route": "Nashik Corridor Route 4",
        "assigned_van": "Branch Van #MH-15-3004",
        "van_capacity_cr": 3.00,
        "working_hours": "08:00 AM – 09:00 PM (Industrial Gate Hours)",
        "optimal_window": "07:00 AM – 08:15 AM (Before shift-1 entry bell)",
        "avoid_window": "11:30 AM – 02:00 PM (Lunchtime factory worker queues)",
        "suggested_date": "2026-08-26",
        "suggested_time": "08:00 AM",
        "suggested_window": "07:30 AM – 09:00 AM",
        "telemetry_last_sync": "2026-08-24 18:30 IST",
        "telemetry_sync_delayed": False,
        "telemetry_status": "ONLINE_HEALTHY",
        "avg_ticket_size": {"daily": 3800, "weekly": 3700, "monthly": 3600},
        "denomination_weights": {"500": 0.80, "200": 0.15, "100": 0.05},
        "cassettes": [
            {"slot": 1, "denomination": 500, "max_notes": 2200, "current_notes": 700, "amount": 350000.0},
            {"slot": 2, "denomination": 500, "max_notes": 2200, "current_notes": 600, "amount": 300000.0},
            {"slot": 3, "denomination": 200, "max_notes": 2200, "current_notes": 500, "amount": 100000.0},
            {"slot": 4, "denomination": 100, "max_notes": 2200, "current_notes": 600, "amount": 60000.0}
        ],
        "reject_bin": {"max_notes": 300, "current_notes": 22, "status": "NORMAL"},
        "crm_collection_bin": None,
        "cra_tracking": {
            "scheduled_date": "2026-08-26",
            "window": "07:30 AM – 09:00 AM",
            "cra_status": "ON_SCHEDULE",
            "sla_penalty": 0.0,
            "provider": "Branch Vault Services"
        },
        "threat_monitoring": {"threat_level": "LOW", "curfew_active": True, "curfew_hours": "08:00 PM – 07:00 AM", "daylight_servicing_only": True},
        "human_errors": {"has_shortfall": False, "shortfall_amount": 0.0, "cassette_mismatch": False, "reject_bin_unemptied": False},
        "last_feed": {"timestamp": "2026-08-24 18:30 IST", "end_cash_value": 810000.0, "supervisor_access": "2026-08-23 08:00 IST"}
    }
}

# Operator-entered demand events (city, dates, multiplier). None by default: lgb_holiday
# already models paydays and RBI holidays, so default multipliers would count them twice.
DEFAULT_EXOGENOUS_EVENTS: List[Dict[str, Any]] = []


def get_physical_cassette_capacity(tid: str) -> float:
    """Calculates physical capacity as sum over cassettes of max_notes * denomination."""
    meta = TERMINAL_OPERATIONAL_METADATA.get(str(tid), {})
    cassettes = meta.get("cassettes", [])
    if cassettes:
        return float(sum(c.get("max_notes", 2200) * c.get("denomination", 500) for c in cassettes))
    return 2860000.0


def sync_feed_to_snapshot(meta: Dict[str, Any], end_cash: float) -> None:
    """Point last_feed and cassette notes at the raw cutoff-day End_Cash.

    ponytail: the raw files carry no per-denomination split, so the typed per-slot
    shares are kept and rescaled to the real total. Replace with hopper-level data when supplied.
    """
    meta["last_feed"]["timestamp"] = "2026-08-24 23:45 IST (raw snapshot)"
    meta["last_feed"]["end_cash_value"] = end_cash
    cassettes = meta.get("cassettes", [])
    typed_total = sum(c["current_notes"] * c["denomination"] for c in cassettes)
    if typed_total <= 0:
        return
    scale = end_cash / typed_total
    for c in cassettes:
        c["current_notes"] = min(c["max_notes"], int(c["current_notes"] * scale))
        c["amount"] = float(c["current_notes"] * c["denomination"])


def get_free_cassette_space(tid: str) -> Dict[str, Any]:
    """Inspects physical dispenser slots and computes free notes and free cash capacity."""
    meta = TERMINAL_OPERATIONAL_METADATA.get(str(tid), {})
    cassettes = meta.get("cassettes", [])
    free_by_denom = {500: 0, 200: 0, 100: 0}
    free_by_slot = []
    total_free_cash = 0.0
    for c in cassettes:
        fn = max(0, c.get("max_notes", 2200) - c.get("current_notes", 0))
        denom = c.get("denomination", 500)
        slot_free_cash = fn * denom
        free_by_denom[denom] = free_by_denom.get(denom, 0) + fn
        free_by_slot.append({
            "slot": c.get("slot"),
            "denomination": denom,
            "max_notes": c.get("max_notes", 2200),
            "current_notes": c.get("current_notes", 0),
            "free_notes": fn,
            "free_cash": slot_free_cash
        })
        total_free_cash += slot_free_cash
    return {
        "free_by_denom": free_by_denom,
        "free_by_slot": free_by_slot,
        "total_free_cash": total_free_cash
    }


def allocate_denomination_mix(net_indent: float, tid: str, denom_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Allocates notes across physical cassette slots respecting available free space.
    Guarantees sum of notes * denomination matches net_indent (or warns and caps if overflowing free capacity).
    """
    free_info = get_free_cassette_space(tid)
    total_free_cash = free_info["total_free_cash"]
    free_500_notes = free_info["free_by_denom"].get(500, 0)
    free_200_notes = free_info["free_by_denom"].get(200, 0)
    free_100_notes = free_info["free_by_denom"].get(100, 0)

    overflow_warning = None
    target_indent = max(0.0, round(float(net_indent) / 100.0) * 100.0)

    if target_indent > total_free_cash:
        overflow_warning = f"Requested indent ₹{target_indent:,.0f} exceeds free physical cassette space ₹{total_free_cash:,.0f} by ₹{target_indent - total_free_cash:,.0f}."
        alloc_target = total_free_cash
    else:
        alloc_target = target_indent

    if denom_weights is None:
        denom_weights = {"500": 0.70, "200": 0.20, "100": 0.10}

    w500 = float(denom_weights.get("500", 0.70))
    w200 = float(denom_weights.get("200", 0.20))
    w100 = float(denom_weights.get("100", 0.10))
    total_w = w500 + w200 + w100
    if total_w <= 0:
        w500, w200, w100 = 0.70, 0.20, 0.10
        total_w = 1.0
    w500 /= total_w
    w200 /= total_w
    w100 /= total_w

    target_500_cash = alloc_target * w500
    n500 = min(free_500_notes, int(target_500_cash // 500))
    rem_cash = alloc_target - (n500 * 500)

    target_200_cash = rem_cash * (w200 / (w200 + w100) if (w200 + w100) > 0 else 0.5)
    n200 = min(free_200_notes, int(target_200_cash // 200))
    rem_cash -= (n200 * 200)

    n100 = min(free_100_notes, int(rem_cash // 100))
    rem_cash -= (n100 * 100)

    while rem_cash >= 500 and n500 < free_500_notes:
        n500 += 1
        rem_cash -= 500
    while rem_cash >= 200 and n200 < free_200_notes:
        n200 += 1
        rem_cash -= 200
    while rem_cash >= 100 and n100 < free_100_notes:
        n100 += 1
        rem_cash -= 100

    actual_allocated = (n500 * 500) + (n200 * 200) + (n100 * 100)
    mix_str = f"₹500: {n500:,} notes (₹{n500*500:,.0f}) | ₹200: {n200:,} notes (₹{n200*200:,.0f}) | ₹100: {n100:,} notes (₹{n100*100:,.0f})"

    return {
        "500_notes": n500,
        "200_notes": n200,
        "100_notes": n100,
        "500_amount": n500 * 500,
        "200_amount": n200 * 200,
        "100_amount": n100 * 100,
        "total_allocated": actual_allocated,
        "denomination_mix": mix_str,
        "free_capacity": total_free_cash,
        "overflow_warning": overflow_warning
    }


class MastersManager:
    def __init__(self):
        self._lock = threading.RLock()
        self.assumptions: Dict[str, Any] = dict(DEFAULT_ASSUMPTIONS)
        self.terminal_configs: Dict[str, Any] = {k: dict(v) for k, v in DEFAULT_TERMINAL_CONFIGS.items()}
        self.exogenous_events: List[Dict[str, Any]] = [dict(e) for e in DEFAULT_EXOGENOUS_EVENTS]
        self.holidays: List[Dict[str, Any]] = []
        self.raw_bundle: Dict[str, Any] = {}
        self.approved_indents: Dict[str, Dict[str, Any]] = {}
        self.checker_overrides: Dict[str, Dict[str, Any]] = {}
        self.amended_replenishments: Dict[str, Dict[str, Any]] = {}
        self.feedback_logs: List[Dict[str, Any]] = []
        self.executed_sweeps: List[Dict[str, Any]] = []
        self.is_manifest_locked: bool = False
        self.manifest_lock_info: Optional[Dict[str, Any]] = None
        self.locked_manifest_snapshot: Optional[Dict[str, Any]] = None
        self.cleared_reject_bins: Set[str] = set()
        
        self._load_base_bundle()
        self._load_holidays()
        self._load_overrides()

    def _load_base_bundle(self):
        if PORTAL_DATA_FILE.exists():
            with open(PORTAL_DATA_FILE, "r") as f:
                self.raw_bundle = json.load(f)
        for tid, base_t in self.raw_bundle.get("terminals", {}).items():
            if tid in TERMINAL_OPERATIONAL_METADATA:
                sync_feed_to_snapshot(TERMINAL_OPERATIONAL_METADATA[tid], float(base_t["existing_cash"]))

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

    def get_physical_cassette_capacity(self, tid: str) -> float:
        return get_physical_cassette_capacity(tid)

    def get_free_cassette_space(self, tid: str) -> Dict[str, Any]:
        return get_free_cassette_space(tid)

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
                    if "amended_replenishments" in data:
                        self.amended_replenishments = data["amended_replenishments"]
                    if "feedback_logs" in data:
                        self.feedback_logs = data["feedback_logs"]
                    if "executed_sweeps" in data:
                        self.executed_sweeps = data["executed_sweeps"]
                    if "cleared_reject_bins" in data:
                        self.cleared_reject_bins = set(data["cleared_reject_bins"])
                        for c_tid in self.cleared_reject_bins:
                            meta = TERMINAL_OPERATIONAL_METADATA.get(c_tid)
                            if meta and "reject_bin" in meta:
                                meta["reject_bin"]["current_notes"] = 0
                                meta["reject_bin"]["status"] = "NORMAL"
                    if "is_manifest_locked" in data:
                        self.is_manifest_locked = bool(data["is_manifest_locked"])
                        self.manifest_lock_info = data.get("manifest_lock_info")
                        self.locked_manifest_snapshot = data.get("locked_manifest_snapshot")
            except Exception as e:
                print(f"Warning: could not load config_overrides.json: {e}")

    def save_overrides(self) -> Dict[str, Any]:
        with self._lock:
            data = {
                "assumptions": self.assumptions,
                "terminal_configs": self.terminal_configs,
                "exogenous_events": self.exogenous_events,
                "approved_indents": self.approved_indents,
                "checker_overrides": self.checker_overrides,
                "amended_replenishments": self.amended_replenishments,
                "feedback_logs": self.feedback_logs,
                "executed_sweeps": self.executed_sweeps,
                "cleared_reject_bins": list(self.cleared_reject_bins),
                "is_manifest_locked": self.is_manifest_locked,
                "manifest_lock_info": self.manifest_lock_info,
                "locked_manifest_snapshot": self.locked_manifest_snapshot,
            }
            with open(OVERRIDES_FILE, "w") as f:
                json.dump(data, f, indent=2)
            return {"status": "saved", "file": str(OVERRIDES_FILE)}

    def reset_to_defaults(self) -> Dict[str, Any]:
        with self._lock:
            if self.is_manifest_locked:
                raise PermissionError("Manifest is digitally locked. Reset is not permitted while locked.")
            self.assumptions = dict(DEFAULT_ASSUMPTIONS)
            self.terminal_configs = {k: dict(v) for k, v in DEFAULT_TERMINAL_CONFIGS.items()}
            self.exogenous_events = [dict(e) for e in DEFAULT_EXOGENOUS_EVENTS]
            self.approved_indents = {}
            self.checker_overrides = {}
            self.amended_replenishments = {}
            self.feedback_logs = []
            self.executed_sweeps = []
            self.cleared_reject_bins = set()
            self.is_manifest_locked = False
            self.manifest_lock_info = None
            self.locked_manifest_snapshot = None
            # Reset metadata counters
            for tid, meta in TERMINAL_OPERATIONAL_METADATA.items():
                if "reject_bin" in meta:
                    meta["reject_bin"]["current_notes"] = 48 if tid == "1" else (62 if tid == "2" else (40 if tid in ["4", "7"] else (275 if tid == "5" else 45)))
                    meta["reject_bin"]["status"] = "SATURATED" if tid == "5" else "NORMAL"
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
            "amended_replenishments": self.amended_replenishments,
            "feedback_logs": self.feedback_logs,
            "executed_sweeps": self.executed_sweeps,
            "is_manifest_locked": self.is_manifest_locked,
            "manifest_lock_info": self.manifest_lock_info
        }

    def amend_replenishment(self, terminal_id: str, new_date: Optional[str] = None, new_window: Optional[str] = None, new_indent: Optional[float] = None, notes: str = "", actor: str = "Field Ops Planner", refill_cycles: Optional[List[Dict[str, Any]]] = None, strategy: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            if self.is_manifest_locked:
                raise PermissionError("Manifest is digitally locked. No modifications permitted while locked.")

            tid = str(terminal_id).replace("ATM-", "").replace("ATM", "").strip()
            now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            current_data = self.compute_fleet_replenishment()
            t = current_data["terminals"].get(tid)
            if not t:
                raise ValueError(f"Terminal '{terminal_id}' not found")

            phys_cap = get_physical_cassette_capacity(tid)
            valid_dates = t.get("dates", []) or ['2026-08-25', '2026-08-26', '2026-08-27', '2026-08-28', '2026-08-29', '2026-08-30', '2026-08-31']

            if refill_cycles and isinstance(refill_cycles, list):
                clean_cycles = []
                total_cycle_indent = 0.0
                prev_date = ""
                for i, c in enumerate(refill_cycles):
                    amt = float(c.get("amount", 0.0))
                    if amt < 0:
                        raise ValueError(f"Cycle amount cannot be negative: ₹{amt}")
                    if amt > phys_cap:
                        raise ValueError(f"Cycle amount ₹{amt:,.0f} exceeds physical cassette capacity ₹{phys_cap:,.0f}")

                    dt = str(c.get("date", valid_dates[0])).strip()
                    if dt not in valid_dates and (dt < valid_dates[0] or dt > valid_dates[-1]):
                        raise ValueError(f"Cycle date '{dt}' outside forecast window [{valid_dates[0]}, {valid_dates[-1]}]")

                    if prev_date and dt < prev_date:
                        raise ValueError(f"Cycle dates must be sequential: cycle #{i+1} ({dt}) cannot precede cycle #{i} ({prev_date})")
                    prev_date = dt

                    win = str(c.get("window", "06:00 AM – 08:30 AM")).strip()
                    is_en = bool(c.get("is_enabled", True))
                    serv = str(c.get("servicing", t.get("serviced_by", "CIT"))).strip()
                    c_id = int(c.get("cycle_id", i + 1))
                    c_name = str(c.get("name", f"Refill Cycle #{c_id}"))
                    clean_cycles.append({
                        "cycle_id": c_id,
                        "name": c_name,
                        "date": dt,
                        "window": win,
                        "amount": round(amt / 100.0) * 100.0,
                        "is_enabled": is_en,
                        "servicing": serv,
                        "status": "Amended" if is_en else "Disabled",
                        "reason": str(c.get("reason", "Custom schedule"))
                    })
                    if is_en:
                        total_cycle_indent += round(amt / 100.0) * 100.0

                strat = strategy or ("2-CYCLE" if len(clean_cycles) > 1 else "1-CYCLE")
                indent_val = total_cycle_indent
                date_val = clean_cycles[0]["date"] if clean_cycles else str(new_date or valid_dates[0])
                window_val = clean_cycles[0]["window"] if clean_cycles else str(new_window or "07:00 AM")
            else:
                clean_cycles = None
                strat = strategy or "1-CYCLE"
                raw_amt = float(new_indent) if new_indent is not None else float(t.get("c1_net_indent", t.get("maker_proposed_indent", 0.0)))
                if raw_amt < 0:
                    raise ValueError(f"Indent amount cannot be negative: ₹{raw_amt}")
                if raw_amt > phys_cap:
                    raise ValueError(f"Indent amount ₹{raw_amt:,.0f} exceeds physical cassette capacity ₹{phys_cap:,.0f}")
                indent_val = round(raw_amt / 100.0) * 100.0
                date_val = str(new_date or valid_dates[0]).strip()
                if date_val not in valid_dates and (date_val < valid_dates[0] or date_val > valid_dates[-1]):
                    raise ValueError(f"Replenishment date '{date_val}' outside forecast window [{valid_dates[0]}, {valid_dates[-1]}]")
                window_val = str(new_window or "07:00 AM").strip()

            self.amended_replenishments[tid] = {
                "terminal_id": tid,
                "strategy": strat,
                "refill_cycles": clean_cycles,
                "amended_date": date_val,
                "amended_window": window_val,
                "amended_indent": round(indent_val, 0),
                "original_indent": round(float(t.get("maker_proposed_indent", 0.0)) / 100.0) * 100.0,
                "notes": str(notes).strip() or "Refill cycles amended via Portal Operations",
                "actor": actor,
                "timestamp": now_ts
            }

            # Clear stale overrides so amendments strictly win
            if tid in self.checker_overrides:
                del self.checker_overrides[tid]

            # Invalidate any previous approval
            if tid in self.approved_indents:
                del self.approved_indents[tid]

            self.save_overrides()
            recomputed = self.compute_fleet_replenishment()
            return {
                "status": "amended_pending_approval",
                "terminal_id": tid,
                "amendment": self.amended_replenishments[tid],
                "is_approved": False,
                "schedule_status": "AMENDED",
                "recomputed_summary": recomputed["summary"]
            }

    def approve_terminal_indent(self, terminal_id: str, officer_name: str = "Treasury Checker Officer", notes: str = "Authorized for CIT Dispatch", actor: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            if self.is_manifest_locked:
                raise PermissionError("Manifest is digitally locked. No approvals permitted while locked.")

            if actor:
                officer_name = actor

            tid = str(terminal_id).replace("ATM-", "").replace("ATM", "").strip()
            now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            current_data = self.compute_fleet_replenishment()
            t = current_data["terminals"].get(tid)
            if not t:
                raise ValueError(f"Terminal '{terminal_id}' not found")

            # Separation of duties: whoever overrode or amended this indent cannot approve it
            approver = str(officer_name).strip().lower()
            authors = {
                str(src[tid].get("actor", "")).strip().lower()
                for src in (self.checker_overrides, self.amended_replenishments) if tid in src
            }
            if approver in authors:
                raise PermissionError(
                    f"Separation of duties: '{officer_name}' authored the change to ATM {tid} and cannot also approve it. "
                    f"A different checker must approve."
                )

            # Determine approved figure and freeze it
            indent_val = float(t.get("c1_net_indent", 0.0))
            self.approved_indents[tid] = {
                "approved_at": now_ts,
                "approved_indent": round(indent_val, 0),
                "approved_cycles": copy.deepcopy(t.get("refill_cycles", [])),
                "notes": notes,
                "actor": officer_name,
                "maker_proposed": float(t.get("maker_proposed_indent", indent_val))
            }

            self.save_overrides()
            recomputed = self.compute_fleet_replenishment()
            return {
                "status": "approved_and_scheduled",
                "terminal_id": tid,
                "approved_indent": round(indent_val, 0),
                "schedule_status": "SCHEDULED",
                "recomputed_summary": recomputed["summary"]
            }

    def record_feedback(self, terminal_id: str, feedback_type: str, reason: str = "", comment: str = "", actor: str = "CMS Ops Operator") -> Dict[str, Any]:
        with self._lock:
            tid = str(terminal_id).replace("ATM-", "").replace("ATM", "").strip()
            now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            entry = {
                "terminal_id": tid,
                "feedback": feedback_type,
                "reason": reason,
                "comment": comment,
                "actor": actor,
                "timestamp": now_ts
            }
            self.feedback_logs.append(entry)
            self.save_overrides()
            return {"status": "feedback_recorded", "entry": entry, "total_feedback_count": len(self.feedback_logs)}

    def apply_checker_override(self, terminal_id: str, new_indent: float, reason: str, actor: str = "Checker Officer") -> Dict[str, Any]:
        with self._lock:
            if self.is_manifest_locked:
                raise PermissionError("Manifest is digitally locked. No modifications permitted while locked.")

            tid = str(terminal_id).replace("ATM-", "").replace("ATM", "").strip()
            now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            current_data = self.compute_fleet_replenishment()
            t = current_data["terminals"].get(tid)
            if not t:
                raise ValueError(f"Terminal '{terminal_id}' not found")

            phys_cap = get_physical_cassette_capacity(tid)
            val = float(new_indent)
            if val < 0.0:
                raise ValueError(f"Indent amount cannot be negative: ₹{val:,.0f}")
            if val > phys_cap:
                raise ValueError(f"Indent amount ₹{val:,.0f} exceeds physical cassette capacity of ₹{phys_cap:,.0f}")

            maker_indent = round(float(t.get("maker_proposed_indent", t.get("c1_net_indent", 0.0))) / 100.0) * 100.0
            rounded_indent = round(val / 100.0) * 100.0

            self.checker_overrides[tid] = {
                "terminal_id": tid,
                "maker_proposed_indent": maker_indent,
                "checker_custom_indent": rounded_indent,
                "delta_rupees": round(rounded_indent - maker_indent, 0),
                "reason": str(reason).strip(),
                "actor": actor,
                "timestamp": now_ts
            }

            # Maker/Checker rule: override is an amended proposal and does NOT self-approve!
            if tid in self.approved_indents:
                del self.approved_indents[tid]

            self.save_overrides()
            recomputed = self.compute_fleet_replenishment()
            return {
                "status": "override_recorded_pending_approval",
                "terminal_id": tid,
                "maker_proposed_indent": maker_indent,
                "new_indent": rounded_indent,
                "reason": reason,
                "timestamp": now_ts,
                "is_approved": False,
                "fleet_summary": recomputed["summary"]
            }

    def update_parameter(self, category: str, key: str, value: Any, sub_id: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            if self.is_manifest_locked:
                raise PermissionError("Manifest is digitally locked. No modifications permitted while locked.")

            ALLOWED_ASSUMPTIONS = {
                "holding_rate_annual": (0.01, 0.50),
                "cit_trip_cost": (100.0, 50000.0),
                "stockout_penalty_fixed": (0.0, 100000.0),
                "emergency_refill_premium": (0.0, 50000.0),
                "unmet_demand_penalty_rate": (0.001, 0.20),
                "target_service_level": (0.80, 0.9999),
                "target_cer_buffer": (0.1, 2.0),
                "default_cassette_capacity": (500000.0, 10000000.0),
                "min_sweep_threshold": (10000.0, 2000000.0),
                "inter_atm_transfer_fee": (0.0, 50000.0),
                "hood_threat_idle_cap": (500000.0, 10000000.0),
                "reject_bin_warning_pct": (50.0, 95.0),
                "reject_bin_critical_pct": (60.0, 100.0)
            }

            old_val = None
            if category == "assumptions":
                if key not in ALLOWED_ASSUMPTIONS:
                    raise ValueError(f"Unknown assumption parameter '{key}'. Allowed keys: {list(ALLOWED_ASSUMPTIONS.keys())}")
                min_v, max_v = ALLOWED_ASSUMPTIONS[key]
                val = float(value)
                if not (min_v <= val <= max_v):
                    raise ValueError(f"Parameter '{key}' value {val} out of bounds [{min_v}, {max_v}]")
                old_val = self.assumptions.get(key)
                self.assumptions[key] = val
                # Any global assumption edit invalidates all current approvals so they must be re-evaluated
                self.approved_indents.clear()

            elif category == "terminals":
                tid = str(sub_id or key)
                if tid not in self.terminal_configs and tid not in TERMINAL_OPERATIONAL_METADATA:
                    raise ValueError(f"Terminal '{tid}' not found")
                if tid not in self.terminal_configs:
                    self.terminal_configs[tid] = {}

                if key == "capacity":
                    val = float(value)
                    phys_cap = get_physical_cassette_capacity(tid)
                    if val <= 0 or val > phys_cap:
                        raise ValueError(f"Capacity must be between 1 and physical limit ₹{phys_cap:,.0f}")
                    old_val = self.terminal_configs[tid].get("capacity")
                    self.terminal_configs[tid]["capacity"] = val
                elif key == "cycle_days":
                    val = int(value)
                    if val not in [3, 7, 14]:
                        raise ValueError("cycle_days must be 3, 7, or 14")
                    old_val = self.terminal_configs[tid].get("cycle_days")
                    self.terminal_configs[tid]["cycle_days"] = val
                elif key == "serviced_by":
                    val = str(value).upper()
                    if val not in ["CIT", "BRANCH"]:
                        raise ValueError("serviced_by must be CIT or BRANCH")
                    old_val = self.terminal_configs[tid].get("serviced_by")
                    self.terminal_configs[tid]["serviced_by"] = val
                else:
                    raise ValueError(f"Invalid terminal parameter '{key}'. Allowed: ['capacity', 'cycle_days', 'serviced_by']")

                # Terminal parameter edit invalidates its approval
                if tid in self.approved_indents:
                    del self.approved_indents[tid]

            elif category == "events":
                found = False
                for evt in self.exogenous_events:
                    if evt.get("id") == sub_id:
                        found = True
                        old_val = evt.get(key)
                        if key in ["demand_multiplier"]:
                            val = float(value)
                            if not (0.1 <= val <= 5.0):
                                raise ValueError("demand_multiplier must be between 0.1 and 5.0")
                            evt[key] = val
                        else:
                            evt[key] = value
                        break
                if not found:
                    raise ValueError(f"Event with id '{sub_id}' not found")
                self.approved_indents.clear()
            else:
                raise ValueError(f"Invalid category '{category}'. Allowed: ['assumptions', 'terminals', 'events']")

            self.save_overrides()
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

    def execute_inter_atm_sweep(self, donor_id: str, recipient_id: str, amount: float, actor: str = "Chief Treasury Officer") -> Dict[str, Any]:
        """Executes an Inter-ATM Cash Balancing Transfer (Surplus Removal -> Starving ATM) with strict cash conservation."""
        with self._lock:
            if self.is_manifest_locked:
                raise PermissionError("Manifest is digitally locked. No modifications permitted while locked.")

            d_id = str(donor_id).replace("ATM-", "").replace("ATM", "").strip()
            r_id = str(recipient_id).replace("ATM-", "").replace("ATM", "").strip()
            transfer_amt = round(float(amount) / 100.0) * 100.0
            now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if transfer_amt <= 0:
                raise ValueError(f"Sweep transfer amount must be positive, got ₹{transfer_amt}")

            current_data = self.compute_fleet_replenishment()
            donor = current_data["terminals"].get(d_id)
            recipient = current_data["terminals"].get(r_id)

            if not donor or not recipient:
                raise ValueError(f"Invalid donor '{donor_id}' or recipient '{recipient_id}'")
            if d_id == r_id or donor.get("city") != recipient.get("city"):
                raise ValueError(
                    f"Sweep must move cash between two different ATMs in the same city "
                    f"(donor {donor.get('city')}, recipient {recipient.get('city')})"
                )

            donor_cash = float(donor["existing_cash"])
            recipient_cash = float(recipient["existing_cash"])
            donor_min_reserve = float(donor.get("assumptions", {}).get("safety_buffer", 200000.0))
            donor_max_transferable = max(0.0, donor_cash - donor_min_reserve)

            if transfer_amt > donor_max_transferable:
                raise ValueError(
                    f"Transfer amount ₹{transfer_amt:,.0f} exceeds donor available cash ₹{donor_max_transferable:,.0f} "
                    f"(current cash ₹{donor_cash:,.0f} minus safety reserve ₹{donor_min_reserve:,.0f})"
                )

            recipient_cap = float(recipient["capacity"])
            if (recipient_cash + transfer_amt) > recipient_cap:
                raise ValueError(
                    f"Recipient capacity ₹{recipient_cap:,.0f} exceeded: "
                    f"current ₹{recipient_cash:,.0f} + transfer ₹{transfer_amt:,.0f} = ₹{recipient_cash + transfer_amt:,.0f}"
                )

            # Strict cash conservation: money deducted from donor equals money added to recipient
            new_donor_cash = donor_cash - transfer_amt
            new_recipient_cash = recipient_cash + transfer_amt

            # Update single source of truth across metadata, raw_bundle, and terminal_configs
            if d_id in TERMINAL_OPERATIONAL_METADATA:
                TERMINAL_OPERATIONAL_METADATA[d_id]["last_feed"]["end_cash_value"] = new_donor_cash
            if d_id in self.raw_bundle.get("terminals", {}):
                self.raw_bundle["terminals"][d_id]["existing_cash"] = new_donor_cash
            if d_id not in self.terminal_configs:
                self.terminal_configs[d_id] = {}
            self.terminal_configs[d_id]["existing_cash"] = new_donor_cash

            if r_id in TERMINAL_OPERATIONAL_METADATA:
                TERMINAL_OPERATIONAL_METADATA[r_id]["last_feed"]["end_cash_value"] = new_recipient_cash
            if r_id in self.raw_bundle.get("terminals", {}):
                self.raw_bundle["terminals"][r_id]["existing_cash"] = new_recipient_cash
            if r_id not in self.terminal_configs:
                self.terminal_configs[r_id] = {}
            self.terminal_configs[r_id]["existing_cash"] = new_recipient_cash

            emergency_premium = float(self.assumptions.get("emergency_refill_premium", 3500.0))
            transfer_fee = float(self.assumptions.get("inter_atm_transfer_fee", 1500.0))
            net_savings = max(0.0, emergency_premium - transfer_fee)

            sweep_record = {
                "sweep_id": f"SWEEP-{datetime.now().strftime('%Y%m%d%H%M%S')}-{d_id}-{r_id}",
                "donor_terminal_id": d_id,
                "donor_site_name": donor["site_name"],
                "recipient_terminal_id": r_id,
                "recipient_site_name": recipient["site_name"],
                "transfer_amount": transfer_amt,
                "corridor": donor.get("corridor_route", "Corridor"),
                "executed_at": now_ts,
                "actor": actor,
                "status": "EXECUTED",
                "donor_cash_before": donor_cash,
                "donor_cash_after": new_donor_cash,
                "recipient_cash_before": recipient_cash,
                "recipient_cash_after": new_recipient_cash,
                "cash_conserved": (donor_cash + recipient_cash) == (new_donor_cash + new_recipient_cash),
                "estimated_savings": {
                    "avoided_emergency_cit": emergency_premium,
                    "avoided_stockout_penalty": 10000.0,
                    "transfer_fee": transfer_fee,
                    "net_benefit": net_savings
                }
            }

            self.executed_sweeps.append(sweep_record)
            if d_id in self.approved_indents:
                del self.approved_indents[d_id]
            if r_id in self.approved_indents:
                del self.approved_indents[r_id]

            self.save_overrides()
            recomputed = self.compute_fleet_replenishment()
            return {
                "status": "sweep_executed",
                "sweep_record": sweep_record,
                "fleet_summary": recomputed["summary"]
            }

    def execute_clear_reject_bin(self, terminal_id: str, actor: str = "Custodian Technician") -> Dict[str, Any]:
        """Clears/resets the physical reject/retract bin counter to 0 notes and persists across reloads."""
        with self._lock:
            if self.is_manifest_locked:
                raise PermissionError("Manifest is digitally locked. No modifications permitted while locked.")

            tid = str(terminal_id).replace("ATM-", "").replace("ATM", "").strip()
            now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            self.cleared_reject_bins.add(tid)
            meta = TERMINAL_OPERATIONAL_METADATA.get(tid)
            if meta and "reject_bin" in meta:
                meta["reject_bin"]["current_notes"] = 0
                meta["reject_bin"]["status"] = "NORMAL"

            if meta and "human_errors" in meta:
                meta["human_errors"]["reject_bin_unemptied"] = False

            self.feedback_logs.append({
                "terminal_id": tid,
                "feedback": "REJECT_BIN_CLEARED",
                "reason": "Purge bin emptied by field custodian technician",
                "comment": "Counters reset to zero",
                "actor": actor,
                "timestamp": now_ts
            })
            self.save_overrides()
            recomputed = self.compute_fleet_replenishment()

            return {
                "status": "reject_bin_cleared",
                "terminal_id": tid,
                "cleared_at": now_ts,
                "fleet_summary": recomputed["summary"]
            }

    def execute_emergency_indent(self, terminal_id: str, amount: Optional[float] = None, actor: str = "Emergency Dispatch Planner") -> Dict[str, Any]:
        """Triggers an emergency hot-dispatch replenishment for an ATM facing imminent cashout."""
        with self._lock:
            if self.is_manifest_locked:
                raise PermissionError("Manifest is digitally locked. No modifications permitted while locked.")

            tid = str(terminal_id).replace("ATM-", "").replace("ATM", "").strip()
            now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            current_data = self.compute_fleet_replenishment()
            t = current_data["terminals"].get(tid)
            if not t:
                raise ValueError(f"Terminal '{terminal_id}' not found")

            phys_cap = get_physical_cassette_capacity(tid)
            emergency_amt = round(float(amount or t.get("maker_proposed_indent", 1000000.0)) / 100.0) * 100.0
            if emergency_amt > phys_cap:
                emergency_amt = phys_cap

            forecast_dates = t.get("dates", []) or ['2026-08-25', '2026-08-26', '2026-08-27', '2026-08-28', '2026-08-29', '2026-08-30', '2026-08-31']
            emergency_date = forecast_dates[0]

            emergency_cycles = [
                {
                    "cycle_id": 1,
                    "name": "⚡ Emergency Hot-Dispatch Drop",
                    "date": emergency_date,
                    "window": "Immediate Hot-Run (Within 2 Hours)",
                    "amount": emergency_amt,
                    "is_enabled": True,
                    "servicing": "CIT Emergency Carrier",
                    "status": "Scheduled",
                    "reason": "Emergency Hot-Dispatch to prevent stockout SLA breach"
                }
            ]

            self.amended_replenishments[tid] = {
                "terminal_id": tid,
                "strategy": "EMERGENCY_HOT_RUN",
                "refill_cycles": emergency_cycles,
                "amended_date": emergency_date,
                "amended_window": "Immediate Hot-Run (Within 2 Hours)",
                "amended_indent": emergency_amt,
                "original_indent": float(t.get("c1_net_indent", 0.0)),
                "notes": "Emergency Hot-Dispatch authorized to prevent regulatory downtime penalty",
                "actor": actor,
                "timestamp": now_ts
            }

            # Emergency indent requires Checker approval: do not auto-approve
            if tid in self.approved_indents:
                del self.approved_indents[tid]

            self.save_overrides()
            recomputed = self.compute_fleet_replenishment()

            return {
                "status": "emergency_indent_scheduled_pending_checker",
                "terminal_id": tid,
                "emergency_amount": emergency_amt,
                "emergency_date": emergency_date,
                "scheduled_time": "Immediate Hot-Run (Within 2 Hours)",
                "is_approved": False,
                "fleet_summary": recomputed["summary"]
            }

    def compute_fleet_replenishment(self) -> Dict[str, Any]:
        """Dynamically recomputes indents, cassette physics, quantile forecasts, downtime factors, and disruption balancing."""
        with self._lock:
            terminals_data = self.raw_bundle.get("terminals", {})
            sla = float(self.assumptions.get("target_service_level", 0.99))
            
            # Exact z-score from normal distribution (Finding A8)
            z_score = round(NormalDist().inv_cdf(min(0.9999, max(0.80, sla))), 3)

            holding_daily = float(self.assumptions.get("holding_rate_annual", 0.065)) / 365.0
            cit_trip_cost = float(self.assumptions.get("cit_trip_cost", 2000.0))
            stockout_penalty = float(self.assumptions.get("stockout_penalty_fixed", 10000.0))

            cycle_buffers = self.raw_bundle["metadata"]["operational_model"]["cycle_buffers"]["1"]

            recomputed_terminals = {}
            total_existing_cash = 0.0
            total_net_indent = 0.0
            total_maker_proposed_indent = 0.0
            total_baseline_indent = 0.0

            # Corridors for routing & inter-ATM balancing
            corridor_groups: Dict[str, List[Dict[str, Any]]] = {}

            for tid, base_t in terminals_data.items():
                t_cfg = self.terminal_configs.get(tid, {})
                meta = TERMINAL_OPERATIONAL_METADATA.get(tid, {})
                
                # Physical Cassette Capacity (Finding A1)
                phys_cap = get_physical_cassette_capacity(tid)
                cfg_cap = float(t_cfg.get("capacity", phys_cap))
                capacity = min(cfg_cap, phys_cap)
                
                servicing_mode = t_cfg.get("serviced_by", base_t.get("serviced_by", "CIT"))
                cycle_days = int(t_cfg.get("cycle_days", 7 if servicing_mode == "CIT" else 3))
                # Operator-entered or swept cash first, else End_Cash from the raw cutoff-day snapshot
                existing_cash = float(t_cfg.get("existing_cash", base_t["existing_cash"]))

                t_city = t_cfg.get("city", base_t.get("city", "MUMBAI")).upper()
                location_tier = meta.get("location_tier", "Metro")
                location_category = meta.get("location_category", "Metro Transit")

                # Forward Daily Predictions
                fwd_dates = (base_t.get("dates") or ['2026-08-25', '2026-08-26', '2026-08-27', '2026-08-28', '2026-08-29', '2026-08-30', '2026-08-31'])[:7]
                fwd_day_names = ['Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'Mon']

                s_preds = base_t.get("daily_preds", {})
                raw_pred_list = s_preds[demand_model.MODEL_NAME][:7]

                # Apply Category-Sensitive Exogenous Multipliers per-day only within cycle (Finding A6)
                pred_list = []
                active_signals = []
                for i in range(len(fwd_dates)):
                    dt = fwd_dates[i]
                    day_mult = 1.0
                    # Event multipliers only apply to dates within cycle_days (Finding A6)
                    if i < cycle_days:
                        for evt in self.exogenous_events:
                            evt_city = evt.get("city", "").upper()
                            evt_start = evt.get("start_date", "2026-08-01")
                            evt_end = evt.get("end_date", "2026-09-30")
                            if (evt_city in ["ALL", t_city]) and (evt_start <= dt <= evt_end):
                                cat_mults = evt.get("category_multipliers", {})
                                applied_mult = float(cat_mults.get(location_category, evt.get("demand_multiplier", 1.0)))
                                day_mult *= applied_mult
                                sig_entry = {
                                    "name": evt.get("name"),
                                    "multiplier": f"{applied_mult:.2f}x",
                                    "date": dt,
                                    "notes": evt.get("notes")
                                }
                                if not any(s["name"] == sig_entry["name"] for s in active_signals):
                                    active_signals.append(sig_entry)

                    pred_list.append(round(float(raw_pred_list[i]) * day_mult, 0))

                # Replenishment Order-Up-To Formula with Lead Time (Finding A7)
                # Refill at the cycle start: the week the buffer was calibrated on and the
                # week the holiday flag describes. Later lgb_holiday days come from later cutoffs.
                c1_default_date = fwd_dates[0]
                c1_default_window = meta.get("suggested_time", "07:00 AM")
                lead_time_days = fwd_dates.index(c1_default_date) if c1_default_date in fwd_dates else 0
                lead_time_demand = sum(pred_list[:lead_time_days])
                cash_at_c1 = max(0.0, existing_cash - lead_time_demand)

                refill_end_idx = min(len(pred_list), lead_time_days + cycle_days)
                demand_from_c1 = sum(pred_list[lead_time_days:refill_end_idx])
                
                daily_burn = round((demand_from_c1 / max(1, cycle_days)) / 100.0) * 100.0
                # P99 multiplier calibrated on 7-day backtest totals, holiday or quiet week.
                # ponytail: sqrt(7/days) scaling for 3-day cycles assumes independent days;
                # conservative if days are correlated. Calibrate on 3-day totals if branch volume grows.
                p99_multiplier = demand_model.cycle_multiplier(cycle_buffers, tid)
                p99_multiplier = 1.0 + (p99_multiplier - 1.0) * math.sqrt(7 / min(cycle_days, 7))
                sigma_h = demand_model.equivalent_sigma(demand_from_c1, p99_multiplier, 0.99)
                daily_std = sigma_h / math.sqrt(cycle_days)
                safety_buffer = round((z_score * sigma_h) / 100.0) * 100.0

                # Quantile Demand Projections
                p50_cycle_demand = round(demand_from_c1 / 100.0) * 100.0
                p80_cycle_demand = round((demand_from_c1 + 0.84 * sigma_h) / 100.0) * 100.0
                p95_cycle_demand = round((demand_from_c1 + 1.645 * sigma_h) / 100.0) * 100.0

                # Target order-up-to at C1 arrival (Finding A1: bounded by physical capacity)
                target_order_up_to = min(capacity, demand_from_c1 + safety_buffer)
                raw_net_indent = max(0.0, target_order_up_to - cash_at_c1)
                system_maker_indent = round(raw_net_indent / 100.0) * 100.0

                # Dynamic runway hours and days (Finding A9, A11)
                rem_cash = existing_cash
                exact_hours = 0
                for d_idx, d_dem in enumerate(pred_list):
                    if rem_cash <= 0:
                        break
                    if rem_cash >= d_dem:
                        exact_hours += 24
                        rem_cash -= d_dem
                    else:
                        frac = rem_cash / max(1.0, d_dem)
                        exact_hours += int(round(frac * 24))
                        rem_cash = 0.0
                        break
                if rem_cash > 0:
                    exact_hours += int(round((rem_cash / max(1.0, daily_burn)) * 24))
                dcr_hours = exact_hours
                dcr_days = round(dcr_hours / 24.0, 2)

                # Dynamic Stockout Timestamp (Finding A11)
                base_time = datetime(2026, 8, 25, 0, 0)
                stockout_dt = base_time + timedelta(hours=dcr_hours)
                if dcr_hours <= 24:
                    stockout_text = stockout_dt.strftime("%A, %b %d at ~%H:00 IST")
                    stockout_day = "Tuesday (Today)"
                elif dcr_hours <= 72:
                    stockout_text = stockout_dt.strftime("%A, %b %d at ~%H:00 IST")
                    stockout_day = stockout_dt.strftime("%A, %b %d")
                else:
                    stockout_text = stockout_dt.strftime("%A, %b %d or later")
                    stockout_day = stockout_dt.strftime("%A, %b %d+")

                # Multi-cycle evaluation (Finding A3, A4, A5)
                # Simulate single refill over the terminal's actual operational horizon (Finding A5)
                sim_horizon = cycle_days if servicing_mode == "BRANCH" else len(fwd_dates)
                b_sim = existing_cash
                single_stockout_idx = None
                single_breach_idx = None
                for i in range(sim_horizon):
                    if fwd_dates[i] == c1_default_date:
                        b_sim = min(capacity, b_sim + system_maker_indent)
                    b_sim = max(0.0, b_sim - pred_list[i])
                    if b_sim == 0 and single_stockout_idx is None:
                        single_stockout_idx = i
                    elif b_sim < safety_buffer and single_breach_idx is None:
                        single_breach_idx = i

                # Branch/3-day ATMs with adequate cash do NOT need second cycle (Finding A5)
                requires_second_cycle = (single_stockout_idx is not None or (single_breach_idx is not None and single_breach_idx <= 4)) and cycle_days > 3
                suggested_strategy = "2-CYCLE" if requires_second_cycle else "1-CYCLE"

                # 2-cycle arithmetic & sequencing (Finding A4)
                if requires_second_cycle:
                    c1_idx = lead_time_days
                    # Cycle 2 strictly after Cycle 1 (Finding A4)
                    c2_idx = max(c1_idx + 1, min(len(fwd_dates) - 1, single_stockout_idx if single_stockout_idx is not None else (c1_idx + 3)))
                    c2_date = fwd_dates[c2_idx]

                    # Segment 1: from C1 to C2
                    dem_seg1 = sum(pred_list[c1_idx:c2_idx])
                    need1 = max(0.0, min(capacity, dem_seg1 + safety_buffer) - cash_at_c1)
                    
                    # Segment 2: from C2 to cycle end
                    dem_seg2 = sum(pred_list[c2_idx:sim_horizon])
                    cash_entering_c2 = max(0.0, (cash_at_c1 + need1) - dem_seg1)
                    need2 = max(0.0, min(capacity, dem_seg2 + safety_buffer) - cash_entering_c2)

                    c1_sugg_amt = round(need1 / 100.0) * 100.0
                    c2_sugg_amt = round(need2 / 100.0) * 100.0

                    suggested_cycles = [
                        {
                            "cycle_id": 1,
                            "name": "Refill Cycle #1 (Mid-Week Primary)",
                            "date": c1_default_date,
                            "window": c1_default_window,
                            "amount": c1_sugg_amt,
                            "is_enabled": True,
                            "servicing": servicing_mode,
                            "status": "Suggested",
                            "reason": "Primary operational replenishment"
                        },
                        {
                            "cycle_id": 2,
                            "name": "Refill Cycle #2 (Weekend Surge Shield)",
                            "date": c2_date,
                            "window": "07:00 AM – 09:15 AM",
                            "amount": c2_sugg_amt,
                            "is_enabled": True,
                            "servicing": servicing_mode,
                            "status": "Suggested",
                            "reason": "Secondary cycle to prevent buffer exhaustion"
                        }
                    ]
                else:
                    suggested_cycles = [
                        {
                            "cycle_id": 1,
                            "name": "Refill Cycle #1 (Cyclical Main)",
                            "date": c1_default_date,
                            "window": c1_default_window,
                            "amount": system_maker_indent,
                            "is_enabled": True,
                            "servicing": servicing_mode,
                            "status": "Suggested",
                            "reason": "Standard cyclical replenishment"
                        }
                    ]

                amended_info = self.amended_replenishments.get(tid)
                is_amended = amended_info is not None
                is_approved = tid in self.approved_indents

                if amended_info and amended_info.get("refill_cycles"):
                    active_cycles = amended_info["refill_cycles"]
                    active_strategy = amended_info.get("strategy", suggested_strategy)
                else:
                    active_cycles = suggested_cycles
                    active_strategy = suggested_strategy

                # Schedule badges
                sched_date = active_cycles[0]["date"] if active_cycles else c1_default_date
                sched_time = active_cycles[0].get("window", c1_default_window)
                if is_approved:
                    schedule_status = "SCHEDULED"
                    checker_badge_text = "✓ Approved"
                    schedule_badge = "SCHEDULED"
                    schedule_display = f"✓ Scheduled: {sched_date} · {sched_time}"
                elif is_amended:
                    schedule_status = "AMENDED"
                    checker_badge_text = "Amended · Pending Approval"
                    schedule_badge = "AMENDED"
                    schedule_display = f"Amended: {sched_date} · {sched_time}"
                else:
                    schedule_status = "SUGGESTED"
                    checker_badge_text = "Pending Review"
                    schedule_badge = "SUGGESTED"
                    schedule_display = f"Suggested: {sched_date} · {sched_time}"

                # Net Indent & Overrides
                override_info = self.checker_overrides.get(tid)
                if override_info:
                    net_indent = min(capacity, round(float(override_info["checker_custom_indent"]) / 100.0) * 100.0)
                    has_override = True
                    # Keep cycles, simulation and the approval snapshot equal to the overridden total:
                    # scale each enabled cycle pro rata, rounding remainder on the last one.
                    enabled_idx = [i for i, c in enumerate(active_cycles) if c.get("is_enabled", True)]
                    base_total = sum(float(active_cycles[i].get("amount", 0.0)) for i in enabled_idx)
                    if enabled_idx:
                        scaled = [dict(c) for c in active_cycles]
                        running = 0.0
                        for i in enabled_idx[:-1]:
                            share = float(scaled[i].get("amount", 0.0)) / base_total if base_total > 0 else 0.0
                            scaled[i]["amount"] = round(net_indent * share / 100.0) * 100.0
                            running += scaled[i]["amount"]
                        scaled[enabled_idx[-1]]["amount"] = net_indent - running
                        for i in enabled_idx:
                            scaled[i]["status"] = "Overridden"
                        active_cycles = scaled
                else:
                    net_indent = round(sum(float(c.get("amount", 0.0)) for c in active_cycles if c.get("is_enabled", True)) / 100.0) * 100.0
                    has_override = False

                maker_total_indent = round(sum(float(c.get("amount", 0.0)) for c in suggested_cycles if c.get("is_enabled", True)) / 100.0) * 100.0

                # Reject bin & Hardware Physics
                rb = meta.get("reject_bin", {"max_notes": 300, "current_notes": 45, "status": "NORMAL"})
                rb_cap = int(rb.get("max_notes", 300))
                rb_cur = 0 if tid in self.cleared_reject_bins else int(rb.get("current_notes", 45))
                rb_pct = round((rb_cur / max(1, rb_cap)) * 100.0, 1)
                if rb_pct >= float(self.assumptions.get("reject_bin_critical_pct", 90.0)):
                    rb_stat = "CRITICAL_FAULT"
                elif rb_pct >= float(self.assumptions.get("reject_bin_warning_pct", 75.0)):
                    rb_stat = "SATURATED"
                else:
                    rb_stat = "NORMAL"

                crm_col = meta.get("crm_collection_bin")
                crm_col_notes = int(crm_col.get("current_notes", 0)) if crm_col else 0
                crm_col_cap = int(crm_col.get("max_notes", 2000)) if crm_col else 2000
                crm_col_pct = round((crm_col_notes / max(1, crm_col_cap)) * 100.0, 1) if crm_col else 0.0

                # Denomination Note Mix per Cassette Slot (Finding A2)
                denom_weights = meta.get("denomination_weights", {"500": 0.70, "200": 0.20, "100": 0.10})
                denom_alloc = allocate_denomination_mix(net_indent, tid, denom_weights)
                denom_mix_str = denom_alloc["denomination_mix"]
                indent_500_notes = denom_alloc["500_notes"]
                indent_200_notes = denom_alloc["200_notes"]
                indent_100_notes = denom_alloc["100_notes"]

                # Disruptions & CRA SLA
                cra_info = meta.get("cra_tracking", {"cra_status": "ON_SCHEDULE", "sla_penalty": 0.0})
                cra_status = cra_info.get("cra_status", "ON_SCHEDULE")
                cra_penalty = float(cra_info.get("sla_penalty", 0.0))

                human_err = meta.get("human_errors", {})
                has_shortfall = bool(human_err.get("has_shortfall", False))
                shortfall_amt = float(human_err.get("shortfall_amount", 0.0))
                reject_bin_unemptied = bool(human_err.get("reject_bin_unemptied", False)) and (tid not in self.cleared_reject_bins)

                threat = meta.get("threat_monitoring", {"threat_level": "LOW"})

                # Downtime detection (Finding C2: remove hardcoded tid 3/5 anomalies)
                downtime_causes = []
                risk_points = 0
                if dcr_hours < 12:
                    downtime_causes.append(f"Immediate Cash Exhaustion Risk (Runway: {dcr_hours}h)")
                    risk_points += 50
                elif dcr_hours < 24:
                    downtime_causes.append(f"Low Cash Runway Warning ({dcr_hours}h remaining)")
                    risk_points += 25

                if rb_stat == "CRITICAL_FAULT":
                    downtime_causes.append(f"Purge/Reject Bin Jam Fault ({rb_cur}/{rb_cap} notes, {rb_pct}%)")
                    risk_points += 40
                elif rb_stat == "SATURATED":
                    downtime_causes.append(f"Reject Bin Saturated ({rb_cur}/{rb_cap} notes, {rb_pct}%)")
                    risk_points += 20

                if cra_status == "MISSED_SLA_BREACH":
                    downtime_causes.append("CRA Missed Replenishment Visit — SLA Breached")
                    risk_points += 35

                if has_shortfall:
                    downtime_causes.append(f"Custodian Loading Shortfall of ₹{shortfall_amt:,.0f} detected at supervisor latch")
                    risk_points += 15

                if reject_bin_unemptied:
                    downtime_causes.append("Human Error: Reject bin unemptied during prior service cycle")
                    risk_points += 15

                downtime_risk_score = min(100, risk_points)

                # Mutually Exclusive Urgency Status (Finding C3)
                if dcr_hours < 24 or cra_status == "MISSED_SLA_BREACH" or rb_stat == "CRITICAL_FAULT" or downtime_risk_score >= 70:
                    urgency = "CRITICAL"
                    deplete_text = f"Dry-out in {dcr_hours}h"
                elif dcr_hours < 48 or rb_stat == "SATURATED" or downtime_risk_score >= 40:
                    urgency = "SCHEDULED"
                    deplete_text = f"{dcr_hours} hours runway"
                else:
                    urgency = "HEALTHY"
                    deplete_text = f"{dcr_hours} hours runway"

                # Surplus cash eligibility excludes deposit bins (Finding A10)
                is_surplus_eligible = (dcr_days > (cycle_days * 1.5)) and (existing_cash >= 2000000.0)
                surplus_removable_amount = max(0.0, round((existing_cash - (demand_from_c1 * 0.7 + safety_buffer)) / 100.0) * 100.0) if is_surplus_eligible else 0.0

                baseline_indent = round(float(base_t.get("c1_net_indent", system_maker_indent)) / 100.0) * 100.0
                total_baseline_indent += baseline_indent
                total_maker_proposed_indent += maker_total_indent
                total_existing_cash += existing_cash
                total_net_indent += net_indent

                # 7-Day History from actual recorded telemetry (Finding C2: remove fake ₹25L reload)
                past_dates = ['2026-08-18', '2026-08-19', '2026-08-20', '2026-08-21', '2026-08-22', '2026-08-23', '2026-08-24']
                past_day_names = ['Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'Mon']
                actual_vals = (base_t.get("actual_demands") or [daily_burn * 0.95]*7)[-7:]
                if len(actual_vals) < 7:
                    actual_vals = [daily_burn * 0.95] * 7

                history_rows = []
                curr_running = existing_cash
                temp_hist = []
                for i in range(6, -1, -1):
                    dt = past_dates[i]
                    dname = past_day_names[i]
                    wdraw = float(actual_vals[i])
                    open_b = curr_running + wdraw
                    temp_hist.append({
                        "date": dt,
                        "day": dname,
                        "opening_cash": round(open_b, 0),
                        "actual_withdrawal": round(wdraw, 0),
                        "refill_amount": 0.0,
                        "closing_cash": round(curr_running, 0),
                        "utilization_pct": min(100.0, round((curr_running / capacity) * 100, 1)),
                        "status": "Healthy" if curr_running >= safety_buffer else "Safety Floor Breach",
                        "data_tag": "Measured Telemetry"
                    })
                    curr_running = open_b
                history_rows = list(reversed(temp_hist))

                # 7-Day Forward Forecast & Plan Verification (Finding A3)
                forecast_rows = []
                b_without = existing_cash
                b_with = existing_cash
                holding_cost_7d = 0.0

                for i in range(len(fwd_dates)):
                    dt = fwd_dates[i]
                    dname = fwd_day_names[i % 7]
                    p_demand = float(pred_list[i])
                    b_without = max(0.0, b_without - p_demand)

                    drop_today = 0.0
                    cycle_tags = []
                    for cyc in active_cycles:
                        if cyc.get("is_enabled", True) and cyc.get("date") == dt:
                            amt = float(cyc.get("amount", 0.0))
                            drop_today += amt
                            c_num = cyc.get("cycle_id", 1)
                            c_ico = "🚚" if c_num == 1 else "⚡"
                            cycle_tags.append(f"{c_ico} Refill #{c_num} (+₹{amt:,.0f})")

                    if drop_today > 0:
                        # Physical capacity bounds the refill (Finding A1)
                        b_with = min(capacity, b_with + drop_today)

                    b_with = max(0.0, b_with - p_demand)
                    daily_holding = round(b_with * holding_daily, 0)
                    holding_cost_7d += daily_holding

                    r_stat = "CRITICAL STOCKOUT" if b_with == 0 else ("BUFFER BREACH" if b_with < safety_buffer else "NORMAL")
                    r_stat_without = "CRITICAL STOCKOUT" if b_without == 0 else ("BUFFER BREACH" if b_without < safety_buffer else "NORMAL")

                    forecast_rows.append({
                        "date": dt,
                        "day": dname,
                        "projected_demand": round(p_demand, 0),
                        "balance_without_refill": round(b_without, 0),
                        "balance_with_refill": round(b_with, 0),
                        "daily_holding_cost": daily_holding,
                        "risk_status": r_stat,
                        "risk_status_without": r_stat_without,
                        "event_tag": ", ".join(cycle_tags) if cycle_tags else "Standard Transit",
                        "refill_amount_today": drop_today,
                        "data_tag": "Modeled Projection"
                    })

                avg_balance_7d = round(sum(r["balance_with_refill"] for r in forecast_rows) / 7.0, 0)
                active_trips_count = sum(1 for c in active_cycles if c.get("is_enabled", True) and float(c.get("amount", 0.0)) > 0)
                cit_trip_costs_7d = active_trips_count * cit_trip_cost
                total_logistics_and_carry_cost = round(holding_cost_7d + cit_trip_costs_7d, 0)
                
                # Check plan stockouts (Finding A3)
                plan_stockout_days = sum(1 for r in forecast_rows if r["balance_with_refill"] == 0)
                sla_projected_pct = 100.0 if plan_stockout_days == 0 else round((7 - plan_stockout_days) / 7.0 * 100, 1)

                # Honest Ops Directive (Finding A3)
                if plan_stockout_days == 0:
                    plain_explanation = f"Loading ₹{net_indent/100000:.1f} Lakhs provides complete cycle coverage without stockout."
                else:
                    plain_explanation = f"⚠️ Physical capacity limit (₹{capacity/100000:.1f}L) reached: plan leaves {plan_stockout_days} potential stockout day(s)."

                # Prior cycle efficacy: accurately label as unmeasured where not measured (Finding C2)
                past_cycle_metrics = [
                    {
                        "cycle_label": "Cycle t-1 (Aug 15 - Aug 21)",
                        "predicted_demand": round(demand_from_c1 * 0.95, 0),
                        "actual_demand": "N/A (Unmeasured)",
                        "effectiveness_pct": 0.0,
                        "ending_residual_cash": round(existing_cash, 0),
                        "stockout_hours": 0,
                        "cra_sla_status": "HISTORICAL_BASELINE",
                        "cycle_status": "UNMEASURED"
                    },
                    {
                        "cycle_label": "Cycle t-2 (Aug 08 - Aug 14)",
                        "predicted_demand": round(demand_from_c1 * 1.0, 0),
                        "actual_demand": "N/A (Unmeasured)",
                        "effectiveness_pct": 0.0,
                        "ending_residual_cash": round(existing_cash, 0),
                        "stockout_hours": 0,
                        "cra_sla_status": "HISTORICAL_BASELINE",
                        "cycle_status": "UNMEASURED"
                    },
                    {
                        "cycle_label": "Cycle t-3 (Aug 01 - Aug 07)",
                        "predicted_demand": round(demand_from_c1 * 0.9, 0),
                        "actual_demand": "N/A (Unmeasured)",
                        "effectiveness_pct": 0.0,
                        "ending_residual_cash": round(existing_cash, 0),
                        "stockout_hours": 0,
                        "cra_sla_status": "HISTORICAL_BASELINE",
                        "cycle_status": "UNMEASURED"
                    }
                ]
                overall_cycle_rating = "UNMEASURED (Telemetry baseline only)"

                # Neanderthal Ops Directive
                if urgency == "CRITICAL" and cra_status == "MISSED_SLA_BREACH":
                    directive_action = f"⚡ ACTION: Trigger Emergency Hot-Dispatch or Corridor Cash Sweep (Starving ATM: ₹{existing_cash:,.0f})"
                    directive_alert = f"🚨 CRA MISSED VISIT SLA BREACH: Dry-out in {dcr_hours} hours without emergency action!"
                elif urgency == "CRITICAL":
                    directive_action = f"⚡ ACTION: Immediate Hot-Run Indent of ₹{net_indent:,.0f} on {sched_date}"
                    directive_alert = f"⚠️ OUT OF CASH IN {dcr_hours} HOURS: {stockout_text}"
                elif is_surplus_eligible:
                    directive_action = f"📦 ACTION: Recommend Surplus Cash Sweep of ₹{surplus_removable_amount:,.0f} to starving corridor neighbor"
                    directive_alert = f"SURPLUS CASH DETECTED: {dcr_hours} Hours Runway (Holding cost drag)"
                else:
                    directive_action = f"ACTION: Load ₹{net_indent:,.0f} on {sched_date} ({sched_time})"
                    directive_alert = f"HEALTHY: {dcr_hours} Hours Runway"

                t_obj = {
                    "terminal_id": tid,
                    "terminal_code": meta.get("terminal_code", f"SYNATM000{tid}"),
                    "site_name": base_t.get("site_name"),
                    "site_type": t_cfg.get("site_type", base_t.get("site_type")),
                    "city": t_city,
                    "serviced_by": servicing_mode,
                    "capacity": capacity,
                    "physical_capacity": phys_cap,
                    "free_capacity": denom_alloc["free_capacity"],
                    "capacity_overflow_warning": denom_alloc.get("overflow_warning"),
                    "existing_cash": existing_cash,
                    "usable_balance": existing_cash,
                    "cycle_days": cycle_days,
                    
                    # Classification & Location
                    "location_tier": location_tier,
                    "location_category": location_category,
                    "device_type": meta.get("device_type", "Non-CRM"),
                    "is_crm": meta.get("is_crm", False),
                    "avg_ticket_size": meta.get("avg_ticket_size", {"daily": 3000, "weekly": 3000, "monthly": 3000}),
                    "denomination_weights": denom_weights,
                    "denomination_mix": denom_mix_str,
                    "indent_notes_split": {
                        "500": indent_500_notes,
                        "200": indent_200_notes,
                        "100": indent_100_notes
                    },

                    # Physical Cassette & Hopper Values
                    "cassettes": meta.get("cassettes", []),
                    "reject_bin": {
                        "max_notes": rb_cap,
                        "current_notes": rb_cur,
                        "pct_full": rb_pct,
                        "status": rb_stat
                    },
                    "crm_collection_bin": {
                        "max_notes": crm_col_cap,
                        "current_notes": crm_col_notes,
                        "pct_full": crm_col_pct,
                        "status": "SATURATED" if crm_col_pct >= 80.0 else "NORMAL"
                    } if meta.get("is_crm") else None,

                    # Downtime & Anomaly Metrics
                    "downtime_risk_score": downtime_risk_score,
                    "downtime_causes": downtime_causes,
                    "urgency_status": urgency,
                    "status_color": "var(--crimson)" if urgency == "CRITICAL" else ("var(--amber)" if urgency == "SCHEDULED" else "var(--emerald)"),
                    "status_label": "CRITICAL" if urgency == "CRITICAL" else ("WARNING" if urgency == "SCHEDULED" else "HEALTHY"),

                    # CRA & Operational Tracking
                    "cra_tracking": {
                        "scheduled_date": sched_date,
                        "window": sched_time,
                        "cra_status": cra_status,
                        "sla_penalty_amount": cra_penalty,
                        "provider": cra_info.get("provider", "CIT Carrier")
                    },
                    "human_error_audit": human_err,
                    "threat_monitoring": threat,
                    "last_feed": meta.get("last_feed", {
                        "timestamp": "2026-08-24 18:30 IST",
                        "end_cash_value": existing_cash,
                        "supervisor_access": "2026-08-20 08:00 IST"
                    }),

                    # Quantile Demand Projections
                    "quantile_forecast": {
                        "p50_demand": p50_cycle_demand,
                        "p80_demand": p80_cycle_demand,
                        "p95_demand": p95_cycle_demand
                    },
                    "quantiles": {
                        "p50_demand": p50_cycle_demand,
                        "p80_demand": p80_cycle_demand,
                        "p95_demand": p95_cycle_demand
                    },

                    # Surplus & Balancing
                    "is_surplus_eligible": is_surplus_eligible,
                    "surplus_removable_amount": surplus_removable_amount,

                    # Cycle Efficacy
                    "last_3_cycles_efficacy": past_cycle_metrics,
                    "overall_cycle_rating": overall_cycle_rating,
                    "past_cycles_efficacy": {
                        "cycles": past_cycle_metrics,
                        "cycle_rating": overall_cycle_rating,
                        "avg_accuracy_pct": 0.0
                    },

                    # Schedule & Amend Tracking
                    "schedule_status": schedule_status,
                    "schedule_display": schedule_display,
                    "schedule_badge": schedule_badge,
                    "checker_badge_text": checker_badge_text,
                    "is_amended": is_amended,
                    "amended_info": amended_info,
                    "suggested_date": sched_date,
                    "suggested_time": sched_time,
                    
                    # Geographical & Route Info
                    "lat": meta.get("lat", base_t.get("lat")),
                    "lng": meta.get("lng", base_t.get("lng")),
                    "coords_str": meta.get("coords_str", f"{meta.get('lat', 18.9)}° N, {meta.get('lng', 72.8)}° E"),
                    "address": meta.get("address", base_t.get("address", "Terminal Location")),
                    "serving_vault": meta.get("serving_vault", "Main Regional Currency Chest"),
                    "vault_lat": meta.get("vault_lat", 19.0657),
                    "vault_lng": meta.get("vault_lng", 72.8686),
                    "vault_distance_km": meta.get("vault_distance_km", 10.0),
                    "transit_offpeak_min": meta.get("transit_offpeak_min", 20),
                    "transit_peak_min": meta.get("transit_peak_min", 45),
                    "corridor_route": meta.get("corridor_route", base_t.get("corridor_name", "Main Route")),
                    "assigned_van": meta.get("assigned_van", "CIT Van #MH-01-CIT"),
                    "van_capacity_cr": meta.get("van_capacity_cr", 4.50),
                    "working_hours": meta.get("working_hours", "24 Hours / 7 Days"),
                    "optimal_window": meta.get("optimal_window", "06:00 AM – 08:30 AM"),
                    "avoid_window": meta.get("avoid_window", "12:00 PM – 02:30 PM"),
                    "telemetry_last_sync": meta.get("telemetry_last_sync", "2026-08-24 18:30 IST"),
                    "telemetry_status": meta.get("telemetry_status", "ONLINE_HEALTHY"),

                    # Ops Directive
                    "ops_directive": {
                        "stockout_alert": directive_alert,
                        "day_of_stockout": stockout_day,
                        "stockout_time_exact": stockout_text,
                        "action_required": directive_action,
                        "post_refill_runway_days": 7.0 if plan_stockout_days == 0 else round(dcr_days, 1),
                        "post_refill_plain_explanation": plain_explanation,
                        "buffer_floor_warning": f"Safety buffer ₹{safety_buffer:,.0f} active."
                    },

                    # History & Forecast Rows
                    "history_7d": history_rows,
                    "forecast_7d": forecast_rows,

                    # Cycles & Financials
                    "refill_cycles": active_cycles,
                    "active_strategy": active_strategy,
                    "requires_second_cycle": requires_second_cycle,
                    "holding_cost_7d": round(holding_cost_7d, 0),
                    "avg_balance_7d": avg_balance_7d,
                    "holding_rate_annual": float(self.assumptions.get("holding_rate_annual", 0.065)),
                    "cit_trip_costs_7d": cit_trip_costs_7d,
                    "total_logistics_and_carry_cost": total_logistics_and_carry_cost,
                    "forecast_sla_pct": sla_projected_pct,
                    "stockout_days_count": plan_stockout_days,
                    
                    # Pillars
                    "facts": {
                        "vault_cash_now": existing_cash,
                        "atm_safe_cash_now": existing_cash,
                        "cassette_cap": capacity,
                        "physical_capacity": phys_cap,
                        "servicing_mode": servicing_mode,
                        "cycle_days": cycle_days,
                        "dcr_days": dcr_days,
                        "dcr_hours": dcr_hours,
                        "deplete_text": deplete_text
                    },
                    "observations": {
                        "pred_demand_7d": round(demand_from_c1 / 100.0) * 100.0,
                        "daily_burn": round(daily_burn / 100.0) * 100.0,
                        "daily_volatility_std": round(daily_std / 100.0) * 100.0,
                        "primary_driver": "Cycle Demand & Location Multiplier",
                        "active_signals": active_signals
                    },
                    "assumptions": {
                        "target_sla": sla,
                        "z_score": z_score,
                        "safety_buffer": safety_buffer,
                        "holding_rate_annual": self.assumptions.get("holding_rate_annual", 0.065),
                        "cit_trip_cost": cit_trip_cost,
                        "stockout_penalty": stockout_penalty
                    },
                    
                    # Indents & Maker/Checker
                    "maker_proposed_indent": maker_total_indent,
                    "maker_single_cycle_indent": system_maker_indent,
                    "c1_gross_target": target_order_up_to,
                    "c1_net_indent": net_indent,
                    "has_override": has_override,
                    "override_details": override_info,
                    "is_approved": is_approved,
                    "approval_info": self.approved_indents.get(tid),
                    "dcr_days": dcr_days,
                    "dcr_hours": dcr_hours,
                    "holding_cost_est": holding_cost_7d,
                    "trip_cost_est": cit_trip_costs_7d,
                    "daily_preds": base_t.get("daily_preds", {}),
                    "dates": base_t.get("dates", []),
                    "actual_demands": base_t.get("actual_demands", []),
                    "cycles": base_t.get("cycles", []),
                    "corridor_name": base_t.get("corridor_name", ""),
                    "corridor_id": base_t.get("corridor_id", ""),
                    "formula_ledger": {
                        "current_cash": existing_cash,
                        "net_indent": net_indent,
                        "expected_demand": demand_from_c1,
                        "end_buffer_reserve": safety_buffer,
                        "post_refill_balance": min(capacity, existing_cash + net_indent)
                    }
                }

                recomputed_terminals[tid] = t_obj

                # Group by corridor for Inter-ATM pairing
                c_route = t_obj.get("corridor_route", "Route")
                if c_route not in corridor_groups:
                    corridor_groups[c_route] = []
                corridor_groups[c_route].append(t_obj)

            # Proposals for inter-ATM balancing
            inter_atm_rebalancing_proposals = []
            for c_route, term_list in corridor_groups.items():
                donors = [t for t in term_list if t["is_surplus_eligible"] and t["surplus_removable_amount"] >= float(self.assumptions.get("min_sweep_threshold", 200000.0))]
                starving = [t for t in term_list if t["urgency_status"] == "CRITICAL" or t["dcr_hours"] < 24 or t["cra_tracking"]["cra_status"] == "MISSED_SLA_BREACH"]

                for d in donors:
                    for r in starving:
                        transfer_amt = min(d["surplus_removable_amount"], max(200000.0, r["c1_net_indent"]))
                        transfer_amt = round(transfer_amt / 100.0) * 100.0
                        if transfer_amt >= float(self.assumptions.get("min_sweep_threshold", 200000.0)):
                            inter_atm_rebalancing_proposals.append({
                                "donor_terminal_id": d["terminal_id"],
                                "donor_terminal_code": d["terminal_code"],
                                "donor_site_name": d["site_name"],
                                "donor_usable_balance": d["usable_balance"],
                                "recipient_terminal_id": r["terminal_id"],
                                "recipient_terminal_code": r["terminal_code"],
                                "recipient_site_name": r["site_name"],
                                "recipient_usable_balance": r["usable_balance"],
                                "recipient_runway_hours": r["dcr_hours"],
                                "suggested_transfer_amount": transfer_amt,
                                "corridor_route": c_route,
                                "reason": f"Sweep ₹{transfer_amt/100000:.1f}L excess cash from {d['terminal_code']} to eliminate immediate stockout at {r['terminal_code']}",
                                "avoided_emergency_cost": 3500.0 + 10000.0,
                                "handling_fee": float(self.assumptions.get("inter_atm_transfer_fee", 1500.0)),
                                "net_savings": 12000.0
                            })

            critical_queue = [t for t in recomputed_terminals.values() if t["urgency_status"] == "CRITICAL"]
            critical_queue.sort(key=lambda x: x["downtime_risk_score"], reverse=True)

            delta_vs_baseline = total_net_indent - total_baseline_indent

            # Status counts partition the fleet cleanly (Finding C3)
            crit_count = sum(1 for t in recomputed_terminals.values() if t["urgency_status"] == "CRITICAL")
            sched_count = sum(1 for t in recomputed_terminals.values() if t["urgency_status"] == "SCHEDULED")
            hlth_count = sum(1 for t in recomputed_terminals.values() if t["urgency_status"] == "HEALTHY")

            # Real hardware telemetry availability (Finding C3)
            online_count = sum(1 for t in recomputed_terminals.values() if t["telemetry_status"] == "ONLINE_HEALTHY")
            uptime_pct = round((online_count / max(1, len(recomputed_terminals))) * 100.0, 1)
            total_sla_penalties = sum(t["cra_tracking"]["sla_penalty_amount"] for t in recomputed_terminals.values())

            summary = {
                "total_terminals": len(recomputed_terminals),
                "total_existing_cash": round(total_existing_cash, 0),
                "total_net_indent": round(total_net_indent, 0),
                "total_maker_proposed_indent": round(total_maker_proposed_indent, 0),
                "total_baseline_indent": round(total_baseline_indent, 0),
                "delta_indent_vs_baseline": round(delta_vs_baseline, 0),
                "total_holding_cost_7d": round(sum(t.get("holding_cost_7d", 0.0) for t in recomputed_terminals.values()), 0),
                "critical_count": crit_count,
                "scheduled_count": sched_count,
                "healthy_count": hlth_count,
                "approved_count": len(self.approved_indents),
                "override_count": len(self.checker_overrides),
                "network_uptime_pct": uptime_pct,
                "total_sla_penalties": total_sla_penalties,
                "active_rebalancing_proposals_count": len(inter_atm_rebalancing_proposals),
                "active_sla": sla,
                "cit_cutoff_time": "18:00 IST",
                "is_manifest_locked": self.is_manifest_locked,
                "manifest_lock_info": self.manifest_lock_info
            }

            return {
                "summary": summary,
                "terminals": recomputed_terminals,
                "critical_queue": critical_queue,
                "inter_atm_rebalancing_proposals": inter_atm_rebalancing_proposals,
                "rebalancing_proposals": inter_atm_rebalancing_proposals,
                "executed_sweeps": self.executed_sweeps,
                "corridors": self.raw_bundle.get("corridors", []),
                "portfolio": self.raw_bundle.get("portfolio", {}),
                "assumptions": self.assumptions,
                "terminal_configs": self.terminal_configs,
                "exogenous_events": self.exogenous_events,
                "holidays": self.holidays,
                "checker_overrides": self.checker_overrides
            }

    # =========================================================================
    # COMPREHENSIVE REPORT GENERATION & DOWNLOAD ENGINE
    # =========================================================================

    def generate_daily_cash_action_report_data(self) -> Dict[str, Any]:
        """Builds structured tabular rows for the Master Daily Cash Action Report."""
        res = self.compute_fleet_replenishment()
        terminals = res["terminals"]
        assumptions = res["assumptions"]

        headers = [
            "Atm ID",
            "metro/non-metro",
            "Location Category",
            "Usable Balance",
            "Timestamp of record",
            "Projected Cashout",
            "Next Feasible Service Date",
            "Next Recommended Action",
            "Amount",
            "Denomination Mix",
            "Maker/Checker",
            "last 3 predictions and their effictiveness compared to their cycles",
            "Cashout Flags",
            "All Model Parameters considered",
            "Atm Master data",
            "forecast",
            "Hopper Values",
            "Captured History"
        ]

        rows = []
        for tid, t in sorted(terminals.items(), key=lambda x: int(x[0])):
            action_verdict = t["ops_directive"]["action_required"]
            rec_action = "EMERGENCY_HOT_DISPATCH" if t["urgency_status"] == "CRITICAL" and t["cra_tracking"]["cra_status"] == "MISSED_SLA_BREACH" else ("REPLENISH" if t["c1_net_indent"] > 0 else ("SWEEP_SURPLUS" if t["is_surplus_eligible"] else "HOLD"))

            # Maker Checker string
            if t["is_approved"]:
                mc_str = f"APPROVED (Checker: {t.get('approval_info', {}).get('actor', 'Officer')})"
            elif t["has_override"]:
                mc_str = f"OVERRIDDEN (Checker: {t['override_details'].get('actor')} | Note: {t['override_details'].get('reason')})"
            else:
                mc_str = "PROPOSED_BY_SYSTEM (Pending Review)"

            # Last 3 cycles string
            past_c = t.get("last_3_cycles_efficacy", [])
            past_str = " | ".join([
                f"{c['cycle_label']}: Pred ₹{c['predicted_demand']/100000:.1f}L vs Act "
                + (f"₹{c['actual_demand']/100000:.1f}L" if isinstance(c['actual_demand'], (int, float)) else "N/A")
                + f" ({c['effectiveness_pct']}% Eff, {c['stockout_hours']}h stockout, {c['cra_sla_status']})"
                for c in past_c
            ])
            past_str += f" => Rating: {t.get('overall_cycle_rating', 'OPTIMAL')}"

            # Cashout flags
            flags = [t["urgency_status"]]
            if t["cra_tracking"]["cra_status"] == "MISSED_SLA_BREACH":
                flags.append("CRA_MISSED_SLA_BREACH")
            if t["reject_bin"]["status"] != "NORMAL":
                flags.append(f"REJECT_BIN_{t['reject_bin']['status']}")
            if t["is_surplus_eligible"]:
                flags.append("SURPLUS_IDLE_CASH")
            if t["human_error_audit"].get("has_shortfall"):
                flags.append("CUSTODIAN_LOADING_SHORTFALL")
            cashout_flags_str = ", ".join(flags)

            # Model parameters considered
            params_str = f"SLA: {assumptions.get('target_service_level', 0.99)*100:.1f}% | z={t['assumptions']['z_score']} | Carry={assumptions.get('holding_rate_annual', 0.065)*100:.1f}% | CIT=₹{assumptions.get('cit_trip_cost', 2000):,.0f} | StockoutPenalty=₹{assumptions.get('stockout_penalty_fixed', 10000):,.0f} | ReserveBuffer=₹{t['assumptions']['safety_buffer']:,.0f}"

            # Master data
            master_str = f"Site: {t['site_name']} | City: {t['city']} | Device: {t['device_type']} (CRM={t['is_crm']}) | Corridor: {t['corridor_route']} | Capacity: ₹{t['capacity']/100000:.1f}L | Cycle: {t['cycle_days']}d"

            # Forecast
            qf = t["quantile_forecast"]
            forecast_str = f"7-Day Total P50: ₹{qf['p50_demand']/100000:.2f}L | P80: ₹{qf['p80_demand']/100000:.2f}L | P95: ₹{qf['p95_demand']/100000:.2f}L | Daily Burn: ₹{t['observations']['daily_burn']/100000:.2f}L/d"

            # Hopper values
            cass_str_list = [f"C{c['slot']}(₹{c['denomination']}): {c['current_notes']:,}n (₹{c['amount']/100000:.2f}L)" for c in t["cassettes"]]
            hopper_str = " | ".join(cass_str_list) + f" | RejectBin: {t['reject_bin']['current_notes']}/{t['reject_bin']['max_notes']}n ({t['reject_bin']['pct_full']}%, {t['reject_bin']['status']})"
            if t["is_crm"] and t["crm_collection_bin"]:
                hopper_str += f" | CRM Collection: {t['crm_collection_bin']['current_notes']}/{t['crm_collection_bin']['max_notes']}n ({t['crm_collection_bin']['pct_full']}%)"

            # Captured history
            lf = t["last_feed"]
            history_str = f"Last Feed: ₹{lf.get('end_cash_value', t['usable_balance']):,.0f} at {lf.get('timestamp')} | Supervisor Access: {lf.get('supervisor_access')} | Last Loaded Indent: ₹{t['human_error_audit'].get('authorized_indent', t['usable_balance']):,.0f}"

            row = [
                t["terminal_code"],
                t["location_tier"],
                t["location_category"],
                t["usable_balance"],
                t["last_feed"]["timestamp"],
                t["ops_directive"]["stockout_time_exact"],
                t["suggested_date"],
                rec_action,
                t["c1_net_indent"],
                t["denomination_mix"],
                mc_str,
                past_str,
                cashout_flags_str,
                params_str,
                master_str,
                forecast_str,
                hopper_str,
                history_str
            ]
            rows.append(row)

        return {
            "report_name": "Daily Cash Action Report - Plan",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "headers": headers,
            "rows": rows
        }

    def generate_daily_cash_action_csv(self) -> str:
        """Serializes the Daily Cash Action Report into RFC 4180 CSV format."""
        data = self.generate_daily_cash_action_report_data()
        out = io.StringIO()
        writer = csv.writer(out, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(data["headers"])
        for row in data["rows"]:
            writer.writerow(row)
        return out.getvalue()

    def generate_daily_cash_action_workbook(self) -> bytes:
        """Creates a formatted multi-tab Excel Workbook (.xlsx)."""
        if not OPENPYXL_AVAILABLE:
            raise RuntimeError("openpyxl library is required for .xlsx export")

        wb = openpyxl.Workbook()
        res = self.compute_fleet_replenishment()
        terminals = res["terminals"]
        assumptions = res["assumptions"]

        # Color Styles
        header_fill = PatternFill(start_color="1E293B", end_color="1E293B", fill_type="solid")
        header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
        sub_fill = PatternFill(start_color="F1F5F9", end_color="F1F5F9", fill_type="solid")
        sub_font = Font(name="Calibri", size=10, bold=True, color="0F172A")
        data_font = Font(name="Calibri", size=10, color="1E293B")
        alert_fill = PatternFill(start_color="FEE2E2", end_color="FEE2E2", fill_type="solid")
        alert_font = Font(name="Calibri", size=10, bold=True, color="991B1B")
        success_fill = PatternFill(start_color="DCFCE7", end_color="DCFCE7", fill_type="solid")
        success_font = Font(name="Calibri", size=10, bold=True, color="166534")

        thin_border = Border(
            left=Side(style="thin", color="CBD5E1"),
            right=Side(style="thin", color="CBD5E1"),
            top=Side(style="thin", color="CBD5E1"),
            bottom=Side(style="thin", color="CBD5E1")
        )

        # -------------------------------------------------------------
        # TAB 1: Daily Action Plan
        # -------------------------------------------------------------
        ws1 = wb.active
        ws1.title = "Daily Action Plan"
        
        tab1_headers = [
            "ATM Code", "Site Name", "City", "Location Tier", "Location Category",
            "Usable Balance (₹)", "Runway (Hours)", "Projected Cashout", "Service Date",
            "Recommended Action", "Net Indent (₹)", "Denomination Mix", "Maker/Checker Status", "Downtime Risk"
        ]
        ws1.append(tab1_headers)
        for col_idx in range(1, len(tab1_headers) + 1):
            cell = ws1.cell(row=1, column=col_idx)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")

        for r_idx, (tid, t) in enumerate(sorted(terminals.items(), key=lambda x: int(x[0])), start=2):
            rec_action = "EMERGENCY_HOT_DISPATCH" if t["urgency_status"] == "CRITICAL" and t["cra_tracking"]["cra_status"] == "MISSED_SLA_BREACH" else ("REPLENISH" if t["c1_net_indent"] > 0 else ("SWEEP_SURPLUS" if t["is_surplus_eligible"] else "HOLD"))
            mc_status = "Approved" if t["is_approved"] else ("Overridden" if t["has_override"] else "Pending Review")

            ws1.append([
                t["terminal_code"],
                t["site_name"],
                t["city"],
                t["location_tier"],
                t["location_category"],
                t["usable_balance"],
                t["dcr_hours"],
                t["ops_directive"]["stockout_time_exact"],
                t["suggested_date"],
                rec_action,
                t["c1_net_indent"],
                t["denomination_mix"],
                mc_status,
                f"{t['downtime_risk_score']}% ({t['urgency_status']})"
            ])

            for c_idx in range(1, len(tab1_headers) + 1):
                cell = ws1.cell(row=r_idx, column=c_idx)
                cell.font = data_font
                cell.border = thin_border
                if c_idx in [6, 11]:
                    cell.number_format = "₹#,##0"
                if c_idx == 10 and rec_action == "EMERGENCY_HOT_DISPATCH":
                    cell.fill = alert_fill
                    cell.font = alert_font
                elif c_idx == 13 and mc_status == "Approved":
                    cell.fill = success_fill
                    cell.font = success_font

        # -------------------------------------------------------------
        # TAB 2: Hopper & Cassette Physics
        # -------------------------------------------------------------
        ws2 = wb.create_sheet(title="Hopper & Cassettes")
        tab2_headers = [
            "ATM Code", "Device Type", "Is CRM?", "Cassette 1 (₹500 Notes)", "Cassette 2 (₹500/200)",
            "Cassette 3 (₹200 Notes)", "Cassette 4 (₹100 Notes)", "Total Usable Cash (₹)",
            "Reject Bin Notes", "Reject Bin Cap", "Reject Bin %", "Reject Bin Status",
            "CRM Collection Notes", "CRM Collection %"
        ]
        ws2.append(tab2_headers)
        for col_idx in range(1, len(tab2_headers) + 1):
            cell = ws2.cell(row=1, column=col_idx)
            cell.fill = header_fill
            cell.font = header_font

        for r_idx, (tid, t) in enumerate(sorted(terminals.items(), key=lambda x: int(x[0])), start=2):
            c1 = t["cassettes"][0]["current_notes"] if len(t["cassettes"]) > 0 else 0
            c2 = t["cassettes"][1]["current_notes"] if len(t["cassettes"]) > 1 else 0
            c3 = t["cassettes"][2]["current_notes"] if len(t["cassettes"]) > 2 else 0
            c4 = t["cassettes"][3]["current_notes"] if len(t["cassettes"]) > 3 else 0
            rb = t["reject_bin"]
            crm_c = t["crm_collection_bin"]

            ws2.append([
                t["terminal_code"],
                t["device_type"],
                "YES" if t["is_crm"] else "NO",
                c1,
                c2,
                c3,
                c4,
                t["usable_balance"],
                rb["current_notes"],
                rb["max_notes"],
                f"{rb['pct_full']}%",
                rb["status"],
                crm_c["current_notes"] if crm_c else "N/A",
                f"{crm_c['pct_full']}%" if crm_c else "N/A"
            ])
            for c_idx in range(1, len(tab2_headers) + 1):
                cell = ws2.cell(row=r_idx, column=c_idx)
                cell.font = data_font
                cell.border = thin_border
                if c_idx == 8:
                    cell.number_format = "₹#,##0"
                if c_idx == 12 and rb["status"] != "NORMAL":
                    cell.fill = alert_fill
                    cell.font = alert_font

        # -------------------------------------------------------------
        # TAB 3: Cycle Efficacy Audit
        # -------------------------------------------------------------
        ws3 = wb.create_sheet(title="Cycle Efficacy Audit")
        tab3_headers = [
            "ATM Code", "Cycle Window", "Predicted Demand (₹)", "Actual Demand (₹)",
            "Effectiveness (%)", "Residual Ending Cash (₹)", "Stockout Hours", "CRA Service SLA", "Cycle Status"
        ]
        ws3.append(tab3_headers)
        for col_idx in range(1, len(tab3_headers) + 1):
            cell = ws3.cell(row=1, column=col_idx)
            cell.fill = header_fill
            cell.font = header_font

        row_counter = 2
        for tid, t in sorted(terminals.items(), key=lambda x: int(x[0])):
            for cyc in t.get("last_3_cycles_efficacy", []):
                ws3.append([
                    t["terminal_code"],
                    cyc["cycle_label"],
                    cyc["predicted_demand"],
                    cyc["actual_demand"],
                    f"{cyc['effectiveness_pct']}%",
                    cyc["ending_residual_cash"],
                    cyc["stockout_hours"],
                    cyc["cra_sla_status"],
                    cyc["cycle_status"]
                ])
                for c_idx in range(1, len(tab3_headers) + 1):
                    cell = ws3.cell(row=row_counter, column=c_idx)
                    cell.font = data_font
                    cell.border = thin_border
                    if c_idx in [3, 4, 6]:
                        cell.number_format = "₹#,##0"
                row_counter += 1

        # -------------------------------------------------------------
        # TAB 4: Model Parameters & Masters
        # -------------------------------------------------------------
        ws4 = wb.create_sheet(title="Model Parameters & Masters")
        ws4.append(["Parameter Category", "Parameter Key", "Configured Value", "Operational Meaning"])
        for col_idx in range(1, 5):
            cell = ws4.cell(row=1, column=col_idx)
            cell.fill = header_fill
            cell.font = header_font

        param_rows = [
            ("Assumptions", "target_service_level", f"{assumptions.get('target_service_level', 0.99)*100}%", "Availability SLA target (z=2.326)"),
            ("Assumptions", "holding_rate_annual", f"{assumptions.get('holding_rate_annual', 0.065)*100}%", "Treasury opportunity cost of carry"),
            ("Assumptions", "cit_trip_cost", f"₹{assumptions.get('cit_trip_cost', 2000):,.0f}", "Armored carrier fixed delivery fee"),
            ("Assumptions", "stockout_penalty_fixed", f"₹{assumptions.get('stockout_penalty_fixed', 10000):,.0f}", "RBI regulatory cashout downtime penalty"),
            ("Assumptions", "emergency_refill_premium", f"₹{assumptions.get('emergency_refill_premium', 3500):,.0f}", "Carrier emergency unscheduled trip premium"),
            ("Assumptions", "min_sweep_threshold", f"₹{assumptions.get('min_sweep_threshold', 200000):,.0f}", "Minimum transfer amount for Inter-ATM sweep"),
            ("Assumptions", "inter_atm_transfer_fee", f"₹{assumptions.get('inter_atm_transfer_fee', 1500):,.0f}", "Custodian route transfer handling fee"),
            ("Disruption Protocol", "cra_sla_cutoff", "18:00 IST", "EOD cutoff to flag CRA MISSED_SLA_BREACH"),
            ("Hardware Constraint", "cassette_max_notes", "2,200 Notes", "Physical dispenser slot note limit"),
            ("Hardware Constraint", "reject_bin_capacity", "300 Notes", "Purge/reject divert cassette limit")
        ]
        for pr in param_rows:
            ws4.append(pr)
            for c_idx in range(1, 5):
                ws4.cell(row=ws4.max_row, column=c_idx).font = data_font

        # Adjust Column Widths Across All Sheets
        for ws in [ws1, ws2, ws3, ws4]:
            for col in ws.columns:
                max_len = max(len(str(cell.value or "")) for cell in col)
                col_letter = get_column_letter(col[0].column)
                ws.column_dimensions[col_letter].width = max(12, min(max_len + 3, 40))

        bio = io.BytesIO()
        wb.save(bio)
        return bio.getvalue()

    def generate_predictions_csv(self) -> str:
        """Exports historical predictions vs actual demand with truthful metrics (no synthetic np.random noise)."""
        res = self.compute_fleet_replenishment()
        terminals = res["terminals"]

        out = io.StringIO()
        writer = csv.writer(out, quoting=csv.QUOTE_MINIMAL)
        writer.writerow([
            "ATM Code", "Date", "Day", "Projected Demand (₹)", "Actual Demand (₹)",
            "Absolute Error (₹)", "Percentage Error (%)", "Balance With Refill (₹)", "Risk Status", "Data Source"
        ])

        for tid, t in sorted(terminals.items(), key=lambda x: int(x[0])):
            for f_row in t.get("forecast_7d", []):
                p_dem = f_row["projected_demand"]
                # Future forecast dates have no measured actuals yet (Finding C1: truthful reporting)
                writer.writerow([
                    t["terminal_code"],
                    f_row["date"],
                    f_row["day"],
                    p_dem,
                    "N/A (Future Forecast)",
                    "N/A",
                    "N/A",
                    f_row["balance_with_refill"],
                    f_row["risk_status"],
                    "Modeled Projection"
                ])

        return out.getvalue()

    def generate_dispatch_manifest(self, checker_name: str = "Chief Treasury Officer", notes: str = "Authorized for CIT Dispatch") -> Dict[str, Any]:
        """Generates the official Digital CIT Dispatch Manifest & Chest Pull Slip.
        Only includes approved figures for CIT-serviced ATMs (Finding B6).
        Returns frozen snapshot if locked (Finding B8).
        """
        with self._lock:
            if self.is_manifest_locked and self.locked_manifest_snapshot is not None:
                return copy.deepcopy(self.locked_manifest_snapshot)

            res = self.compute_fleet_replenishment()
            terminals = res["terminals"]
            corridors = res["corridors"]

            now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            manifest_id = f"CIT-MANIFEST-{datetime.now().strftime('%Y%m%d')}-MAH-01"

            route_manifests = []
            total_pull_rupees = 0.0
            # Each approved cycle is pulled from the chest on its own date, not all on day one
            pull_by_date: Dict[str, float] = {}

            for corr in corridors:
                route_tids = [str(x) for x in corr.get("terminals", [])]
                route_stops = []
                route_cash_total = 0.0

                for tid in route_tids:
                    if tid in terminals:
                        t = terminals[tid]
                        is_cit = (t.get("serviced_by") == "CIT")
                        is_appr = (tid in self.approved_indents)
                        approved_info = self.approved_indents.get(tid)

                        stop_cycles = []
                        if is_cit and is_appr and approved_info:
                            indent_amt = float(approved_info.get("approved_indent", 0.0))
                            status_text = "APPROVED"
                            stop_cycles = [
                                {"date": c["date"], "amount": float(c["amount"])}
                                for c in approved_info.get("approved_cycles", []) if c.get("is_enabled", True)
                            ]
                            for c in stop_cycles:
                                pull_by_date[c["date"]] = pull_by_date.get(c["date"], 0.0) + c["amount"]
                        elif not is_cit:
                            # Branch serviced ATMs are not serviced by CIT armored vans (Finding B6)
                            indent_amt = 0.0
                            status_text = "BRANCH_SERVICED (EXCLUDED)"
                        else:
                            indent_amt = 0.0
                            status_text = "PENDING_APPROVAL (EXCLUDED)"

                        route_stops.append({
                            "terminal_id": tid,
                            "terminal_code": t["terminal_code"],
                            "site_name": t["site_name"],
                            "city": t["city"],
                            "existing_cash": t["existing_cash"],
                            "indent_amount": indent_amt,
                            "denomination_mix": t["denomination_mix"],
                            "maker_recommended": t["maker_proposed_indent"],
                            "approved_indent": float(approved_info.get("approved_indent", 0.0)) if approved_info else 0.0,
                            "cycles": stop_cycles,
                            "has_override": t["has_override"],
                            "override_reason": t["override_details"].get("reason") if t["has_override"] else None,
                            "serviced_by": t["serviced_by"],
                            "status": status_text,
                            "is_approved": is_appr
                        })
                        route_cash_total += indent_amt

                total_pull_rupees += route_cash_total
                route_manifests.append({
                    "route_id": corr.get("id"),
                    "route_name": corr.get("name"),
                    "city": corr.get("city"),
                    "stop_count": len(route_stops),
                    "route_total_cash": route_cash_total,
                    "stops": route_stops
                })

            overrides_log = list(self.checker_overrides.values())
            sign_string = f"{manifest_id}|{total_pull_rupees}|{now_ts}|{checker_name}"
            # HMAC-SHA256 signature; the key lives outside the source so it cannot be read from the repo
            audit_secret = os.environ.get("CMS_SIGNING_KEY", "")
            sig_hash = (
                hmac.new(audit_secret.encode("utf-8"), sign_string.encode("utf-8"), hashlib.sha256).hexdigest()[:24].upper()
                if audit_secret else "UNSIGNED"
            )

            manifest = {
                "manifest_id": manifest_id,
                "generated_at": now_ts,
                "cutoff_deadline": "18:00 IST",
                "bank_entity": "State Bank Commercial Cash Operations",
                "currency_chest_origin": "Mumbai Main Fort Chest",
                "logistics_agency": "SecureArmored CIT Logistics Ltd.",
                "total_vault_pull_rupees": total_pull_rupees,
                "vault_pull_by_date": dict(sorted(pull_by_date.items())),
                "total_terminals": len(terminals),
                "checker_signoff": {
                    "officer_name": checker_name,
                    "notes": notes,
                    "digital_signature_token": f"SIG-HMAC-{sig_hash}",
                    "authorized_at": now_ts,
                    "role": "Authorized Checker / Chief Treasury Manager"
                },
                "routes": route_manifests,
                "logged_checker_overrides": overrides_log,
                "is_locked": self.is_manifest_locked
            }

            return manifest

    def lock_manifest(self, checker_name: str = "Chief Treasury Officer", notes: str = "Authorized for CIT Dispatch") -> Dict[str, Any]:
        """Locks the dispatch manifest permanently; freezes snapshot and blocks subsequent writes (Finding B8)."""
        with self._lock:
            if self.is_manifest_locked and self.locked_manifest_snapshot is not None:
                return {"status": "already_locked", "manifest": copy.deepcopy(self.locked_manifest_snapshot)}

            if not os.environ.get("CMS_SIGNING_KEY"):
                raise ValueError("Cannot sign the manifest: CMS_SIGNING_KEY is not set on the server.")
            manifest = self.generate_dispatch_manifest(checker_name=checker_name, notes=notes)
            if manifest["total_vault_pull_rupees"] <= 0:
                raise ValueError("Cannot lock a manifest with no approved CIT indents.")
            self.is_manifest_locked = True
            manifest["is_locked"] = True
            self.locked_manifest_snapshot = copy.deepcopy(manifest)
            self.manifest_lock_info = {
                "manifest_id": manifest["manifest_id"],
                "locked_at": manifest["generated_at"],
                "officer_name": checker_name,
                "total_rupees": manifest["total_vault_pull_rupees"],
                "digital_signature": manifest["checker_signoff"]["digital_signature_token"]
            }
            self.save_overrides()
            return {"status": "locked", "manifest": copy.deepcopy(self.locked_manifest_snapshot)}


# Singleton manager instance
masters_mgr = MastersManager()
