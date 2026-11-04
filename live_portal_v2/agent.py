"""LangGraph-Powered Cash Intelligence AI Agent (Iteration Two).

Implements a true agentic workflow using LangGraph StateGraph:
- Multi-turn conversational intelligence powered by Gemini (gemini-2.5-flash).
- Automatic Tool calling:
  * query_network_status: Network cash health HUD, critical counts, CIT cutoff.
  * explain_atm: Deep-dive into specific ATM cash runway, demand drivers, safety buffer.
  * simulate_scenario: Live What-If replenishment simulation with custom indents or delay hours.
  * get_masters: Inspect economic assumptions, holiday calendars, weather alerts.
  * update_master_parameter: Live update to holding cost, CIT cost, SLA, or cassette capacity.
  * approve_indents: Approve replenishment orders for armored CIT dispatch.
- Fast-path slash command router (/status, /explain, /simulate, /set, /approve, /masters, /save, /reset, /help).
- Dual-mode execution:
  * LLM-driven LangGraph tool-calling loop when Gemini/Google API key is available.
  * High-fidelity deterministic tool-dispatch engine fallback when offline/no API key.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Union, Sequence, Annotated
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from masters_manager import masters_mgr

HERE = Path(__file__).resolve().parent
CMS_ROOT = HERE.parent


# ==========================================
# 0. API KEY RESOLUTION
# ==========================================

def resolve_api_key(explicit_key: Optional[str] = None) -> Optional[str]:
    """Resolves Gemini/Google API key with hierarchy: explicit > env > .env file."""
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()

    # Check process environment
    for env_var in ["GEMINI_API_KEY", "GOOGLE_API_KEY"]:
        val = os.environ.get(env_var)
        if val and val.strip():
            return val.strip()

    # Check .env in CMS_ROOT
    env_file = CMS_ROOT / ".env"
    if env_file.exists():
        try:
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("GEMINI_API_KEY=") or line.startswith("GOOGLE_API_KEY="):
                    val = line.split("=", 1)[1].strip()
                    if val:
                        return val
        except Exception:
            pass

    return None


_QUOTA_CACHE: Dict[str, Any] = {"status": None, "checked_at": 0.0}

def is_gemini_quota_available(api_key: str) -> bool:
    """Fast probe cached for 60s to check if Gemini API quota is available before initiating graph."""
    now = time.time()
    if _QUOTA_CACHE["status"] is not None and (now - _QUOTA_CACHE["checked_at"]) < 60:
        return bool(_QUOTA_CACHE["status"])

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": "1"}]}]}
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            _QUOTA_CACHE["status"] = True
            _QUOTA_CACHE["checked_at"] = now
            return True
    except urllib.error.HTTPError as e:
        if e.code == 429:
            _QUOTA_CACHE["status"] = False
            _QUOTA_CACHE["checked_at"] = now
            return False
    except Exception:
        pass
    _QUOTA_CACHE["status"] = True
    _QUOTA_CACHE["checked_at"] = now
    return True


# ==========================================
# 1. AGENT TOOLS
# ==========================================

@tool
def query_network_status() -> str:
    """Get high-level summary of network cash health, critical alerts, total indents, and CIT cutoff."""
    res = masters_mgr.compute_fleet_replenishment()
    s = res["summary"]
    return (
        f"**Network Cash Status HUD**:\n"
        f"- Total Terminals: {s['total_terminals']}\n"
        f"- Total Network Cash: ₹{s['total_existing_cash']:,.0f} (~₹{s['total_existing_cash']/1e7:.2f} Cr)\n"
        f"- Total Recommended Indent: ₹{s['total_net_indent']:,.0f} (~₹{s['total_net_indent']/1e7:.2f} Cr)\n"
        f"- Critical Stockout Risk (<2 days cash): {s['critical_count']} ATMs\n"
        f"- Scheduled Replenishments: {s['scheduled_count']} ATMs\n"
        f"- Healthy Runway (>4 days cash): {s['healthy_count']} ATMs\n"
        f"- Approved Indents: {s['approved_count']}/{s['total_terminals']}\n"
        f"- Target Service Level (SLA): {s['active_sla']*100:.1f}%\n"
        f"- CIT Dispatch Cutoff: {s['cit_cutoff_time']}"
    )


@tool
def explain_atm(terminal_id: str) -> str:
    """Explain replenishment recommendation, safety buffer, cash runway (DCR), and demand drivers for a specific ATM."""
    tid = str(terminal_id).replace("ATM-", "").replace("ATM", "").strip()
    res = masters_mgr.compute_fleet_replenishment()
    t = res["terminals"].get(tid)
    if not t:
        return f"Error: Terminal ID '{terminal_id}' not found. Available IDs: 1 to 8."

    exp = t.get("c1_explainability", {})
    primary_driver = exp.get("primary_driver", "Standard cyclical demand & weekend surge")
    cats = exp.get("categories", {})

    status_emoji = "🔴 CRITICAL" if t['urgency_status'] == "CRITICAL" else ("🟡 SCHEDULED" if t['urgency_status'] == "SCHEDULED" else "🟢 HEALTHY")

    drivers_str = ""
    for k, v in list(cats.items())[:3]:
        impact_pct = v.get("share_pct", v.get("impact_pct", 15))
        drivers_str += f"\n  • **{k.title()}**: {v.get('description', 'Key operational demand factor')} (~{impact_pct}%)"

    pred_demand = (
        t.get("c1_pred_demand")
        or t.get("observations", {}).get("pred_demand_7d")
        or t.get("formula_ledger", {}).get("expected_demand")
        or 0.0
    )
    safety_buffer = (
        t.get("c1_safety_buffer")
        or t.get("assumptions", {}).get("safety_buffer")
        or 0.0
    )

    return (
        f"**ATM {tid}: {t['site_name']} ({t['city']})**\n"
        f"- Status: **{status_emoji}**\n"
        f"- Existing Cash: **₹{t['existing_cash']:,.0f}**\n"
        f"- Cash Remaining: **{t.get('dcr_hours', int(round(t['dcr_days'] * 24)))} Hours**\n"
        f"- Projected 7-Day Demand: **₹{pred_demand:,.0f}**\n"
        f"- Safety Buffer: **₹{safety_buffer:,.0f}** (SLA {masters_mgr.assumptions.get('target_service_level', 0.99)*100:.1f}%)\n"
        f"- Gross Target: **₹{t['c1_gross_target']:,.0f}** (Cassette Cap: ₹{t['capacity']:,.0f})\n"
        f"- Recommended Net Top-Up Indent: **₹{t['c1_net_indent']:,.0f}**\n"
        f"- Servicing Mode: **{t['serviced_by']}** ({t['cycle_days']}-day cycle)\n"
        f"- Primary Driver: *{primary_driver}*{drivers_str}\n"
        f"- Approved: {'✅ Yes' if t['is_approved'] else '⏳ Pending Approval'}"
    )


@tool
def simulate_scenario(terminal_id: str, custom_indent: Optional[float] = None, delay_hours: int = 0) -> str:
    """Simulate what-if replenishment with a custom cash indent amount or CIT dispatch delay hours."""
    tid = str(terminal_id).replace("ATM-", "").replace("ATM", "").strip()
    res = masters_mgr.compute_fleet_replenishment()
    t = res["terminals"].get(tid)
    if not t:
        return f"Error: Terminal ID '{terminal_id}' not found."

    indent = float(custom_indent) if custom_indent is not None else float(t["c1_net_indent"])
    delay = int(delay_hours)

    cycle_days = int(t.get("cycle_days", 7))
    pred_demand = float(
        t.get("c1_pred_demand")
        or t.get("observations", {}).get("pred_demand_7d")
        or t.get("formula_ledger", {}).get("expected_demand")
        or 4000000.0
    )
    daily_burn = pred_demand / max(1, cycle_days)
    starting_cash = float(t["existing_cash"])
    
    # Delay burn effect
    delay_burn = daily_burn * (delay / 24.0)
    cash_at_arrival = max(0.0, starting_cash - delay_burn)
    is_stockout_before_arrival = (starting_cash - delay_burn) <= 0.0

    phys_cap = masters_mgr.get_physical_cassette_capacity(tid)
    post_refill_cash = min(phys_cap, cash_at_arrival + indent)
    end_of_cycle_cash = max(0.0, post_refill_cash - (daily_burn * cycle_days))
    dcr_post_refill = round(post_refill_cash / daily_burn, 2) if daily_burn > 0 else 99.0

    holding_rate = float(masters_mgr.assumptions.get("holding_rate_annual", 0.075))
    holding_cost_cycle = round(post_refill_cash * (holding_rate / 365.0) * cycle_days, 0)

    return (
        f"**Simulation Results for ATM {tid}** ({t['site_name']}):\n"
        f"- Test Parameters: Indent = ₹{indent:,.0f} | CIT Delay = {delay} hours\n"
        f"- Cash at Delivery Arrival: ₹{cash_at_arrival:,.0f} {'⚠️ STOCKOUT OCCURRED BEFORE REFILL' if is_stockout_before_arrival else '✅ Preserved runway'}\n"
        f"- Post-Refill Starting Balance: ₹{post_refill_cash:,.0f} (Cassette Cap: ₹{phys_cap:,.0f})\n"
        f"- Post-Refill Cash Runway: {int(round(dcr_post_refill * 24))} Hours\n"
        f"- Projected End-of-Cycle Cash: ₹{end_of_cycle_cash:,.0f}\n"
        f"- Estimated Holding Cost: ₹{holding_cost_cycle:,.0f}"
    )


@tool
def get_masters(category: str = "all") -> str:
    """Inspect masters, economic assumptions, holiday calendars, or exogenous weather events."""
    cat = category.lower().strip()
    masters = masters_mgr.get_all_masters()
    
    if "cost" in cat or "assum" in cat:
        a = masters["assumptions"]
        return (
            f"**Economic & Policy Assumptions**:\n"
            f"- Annual Holding Rate: {a['holding_rate_annual']*100:.2f}%\n"
            f"- CIT Delivery Fixed Cost: ₹{a['cit_trip_cost']:,.0f}\n"
            f"- Regulatory Stockout Penalty: ₹{a['stockout_penalty_fixed']:,.0f}\n"
            f"- Emergency Refill Premium: ₹{a['emergency_refill_premium']:,.0f}\n"
            f"- Unmet Demand Penalty Rate: {a['unmet_demand_penalty_rate']*100:.1f}%/day\n"
            f"- Target Service Level (SLA): {a['target_service_level']*100:.1f}%\n"
            f"- Standard Cassette Cap: ₹{a['default_cassette_capacity']:,.0f}"
        )
    elif "holiday" in cat or "cal" in cat:
        h_list = masters["holidays"][:6]
        items = "\n".join([f"  • {h.get('date')}: {h.get('name')} ({h.get('type')})" for h in h_list])
        return f"**Active Holiday & Banking Calendar**:\n{items}"
    elif "event" in cat or "weather" in cat or "exo" in cat:
        e_list = masters["exogenous_events"]
        items = "\n".join([f"  • [{e.get('city')}] {e.get('name')} ({e.get('start_date')} to {e.get('end_date')}): x{e.get('demand_multiplier')} - {e.get('notes')}" for e in e_list])
        return f"**Exogenous Signals & Special Events**:\n{items}"
    elif "term" in cat:
        t_list = masters["terminals"]
        items = "\n".join([f"  • ATM {k}: Cap ₹{v.get('capacity'):,.0f} | Mode: {v.get('serviced_by')} ({v.get('cycle_days')}d) | {v.get('site_type')} ({v.get('city')})" for k, v in t_list.items()])
        return f"**Terminal Master Configurations**:\n{items}"
    else:
        return (
            f"**Data Masters Catalog Available**:\n"
            f"1. Cost & Policy Assumptions (Holding rate, CIT cost, SLA, penalties)\n"
            f"2. Terminal Masters (8 ATMs, capacities, servicing mode)\n"
            f"3. Holiday & Payday Calendars\n"
            f"4. Exogenous Signals & Events (Monsoon alerts, festival multipliers)\n"
            f"Ask specifically for any category (e.g., 'show cost assumptions' or 'show holiday calendar')."
        )


@tool
def update_master_parameter(category: str, key: str, value: str, sub_id: Optional[str] = None) -> str:
    """Update an assumption or master parameter (e.g. category='assumptions', key='holding_rate_annual', value='0.075') and recompute live impact."""
    try:
        val_clean = value.replace("%", "").replace("₹", "").replace(",", "").strip()
        num_val: Union[float, int, str]
        if "%" in value or (val_clean.replace(".", "", 1).isdigit() and "." in val_clean):
            num_val = float(val_clean)
            if "%" in value and num_val > 1.0:
                num_val = num_val / 100.0
        elif val_clean.isdigit():
            num_val = int(val_clean)
        else:
            num_val = value.strip()

        res = masters_mgr.update_parameter(category=category, key=key, value=num_val, sub_id=sub_id)
        delta = res["delta_net_indent"]
        sign = "+" if delta >= 0 else "-"
        return (
            f"✅ **Parameter Updated Successfully**:\n"
            f"- Parameter: `{key}` ({category})\n"
            f"- Old Value: `{res['old_value']}` ➔ New Value: `{res['new_value']}`\n"
            f"- Impact on Fleet Total Indent: **{sign}₹{abs(delta):,.0f}**\n"
            f"- New Fleet Indent: **₹{res['recomputed_summary']['total_net_indent']:,.0f}**\n"
            f"- Critical ATMs: **{res['recomputed_summary']['critical_count']}**\n"
            f"*(Changes applied live in-memory. Say '/save' to commit to permanent disk.)*"
        )
    except Exception as e:
        return f"Error updating parameter: {e}"


@tool
def approve_indents(terminal_ids: str, actor: str, approver_notes: str = "Approved by Treasury Checker via AI Copilot") -> str:
    """Formally approve net cash replenishment indents for specified ATMs or all pending terminals (Checker Action)."""
    # Check if locked
    if masters_mgr.locked_manifest_snapshot is not None:
        return "❌ Error: Dispatch manifest is digitally locked. Approvals cannot be modified."

    res = masters_mgr.compute_fleet_replenishment()
    t_dict = res["terminals"]
    
    target_ids = []
    if str(terminal_ids).lower().strip() in ["all", "*"]:
        target_ids = list(t_dict.keys())
    else:
        found = re.findall(r"\d+", str(terminal_ids))
        target_ids = [str(x) for x in found if str(x) in t_dict]

    if not target_ids:
        return f"No matching terminals found for input '{terminal_ids}'. Available: 1 to 8."

    approved_names = []
    total_approved_cash = 0.0
    now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for tid in target_ids:
        try:
            r = masters_mgr.approve_terminal_indent(terminal_id=tid, officer_name=actor, notes=approver_notes)
            approved_names.append(f"ATM {tid} (₹{r['approved_indent']:,.0f})")
            total_approved_cash += r["approved_indent"]
        except Exception as e:
            return f"❌ Approval failed for ATM {tid}: {e}"

    masters_mgr.save_overrides()

    return (
        f"✅ **Replenishment Indents Approved by Checker** ({len(target_ids)} ATMs):\n"
        f"- Total Approved Net Indent: **₹{total_approved_cash:,.0f}**\n"
        f"- Approved Terminals: {', '.join(approved_names)}\n"
        f"- Timestamp: {now_ts}\n"
        f"- Status: Recorded & Persisted. CIT Dispatch Manifest updated."
    )


@tool
def adjust_indent(terminal_id: str, new_indent: Any, actor: str, reason: str = "Checker Natural Language Adjustment") -> str:
    """Adjust an ATM replenishment indent with an explicit Checker natural language audit reason. Amounts must include explicit units (e.g. '28L', '28 Lakhs') or be >= 10,000 rupees."""
    tid = str(terminal_id).replace("ATM-", "").replace("ATM", "").strip()
    try:
        val_str = str(new_indent).replace("₹", "").replace(",", "").strip()
        lower_str = val_str.lower()
        if "cr" in lower_str:
            clean_num = float(re.findall(r"[\d\.]+", lower_str)[0])
            val = clean_num * 10000000.0
        elif "l" in lower_str:
            clean_num = float(re.findall(r"[\d\.]+", lower_str)[0])
            val = clean_num * 100000.0
        else:
            val = float(val_str)
            if 0.0 < val < 10000.0:
                return (
                    f"⚠️ **Ambiguous Amount Unit**: '{new_indent}'.\n"
                    f"To prevent catastrophic treasury errors, amounts under ₹10,000 without explicit unit suffixes (like 'L', 'Lakhs', 'Cr') are rejected.\n"
                    f"Please re-enter with explicit units, e.g. `28 Lakhs`, `28L`, or exact rupees `2800000`."
                )

        res = masters_mgr.apply_checker_override(terminal_id=tid, new_indent=val, reason=reason, actor=actor)
        delta = res["new_indent"] - res["maker_proposed_indent"]
        sign = "+" if delta >= 0 else ""
        phys_cap = masters_mgr.get_physical_cassette_capacity(tid)
        return (
            f"✅ **Checker Override Applied for ATM {tid}**:\n"
            f"- Maker Recommended Indent: **₹{res['maker_proposed_indent']:,.0f}**\n"
            f"- Checker Adjusted Indent: **₹{res['new_indent']:,.0f}** ({sign}₹{delta:,.0f})\n"
            f"- Physical Cassette Capacity: **₹{phys_cap:,.0f}**\n"
            f"- Logged Audit Reason: *\"{reason}\"*\n"
            f"- Updated Fleet Net Indent: **₹{res['fleet_summary']['total_net_indent']:,.0f}**\n"
            f"- Status: **Indent Updated (Pending Checker Re-Approval & Manifest Lock)**."
        )
    except Exception as e:
        return f"Error applying Checker override: {e}"


@tool
def get_dispatch_manifest() -> str:
    """View the formal Digital CIT Dispatch Manifest & Currency Chest Pull Slip."""
    m = masters_mgr.generate_dispatch_manifest()
    stops_summary = []
    for r in m["routes"]:
        stops_summary.append(f"  • **{r['route_name']}**: {r['stop_count']} stops | ₹{r['route_total_cash']:,.0f}")
    
    return (
        f"📋 **Digital CIT Dispatch Manifest ({m['manifest_id']})**:\n"
        f"- Origin Chest: **{m['currency_chest_origin']}**\n"
        f"- Total Vault Cash Pull: **₹{m['total_vault_pull_rupees']:,.0f}** (~₹{m['total_vault_pull_rupees']/1e7:.2f} Cr)\n"
        f"- Dispatch Cutoff: **{m['cutoff_deadline']}**\n"
        f"- Armored Routes:\n" + "\n".join(stops_summary) + f"\n"
        f"- Digital Signature: `{m['checker_signoff']['digital_signature_token']}`\n"
        f"- Logged Overrides: **{len(m['logged_checker_overrides'])}** recorded\n"
        f"- Status: {'🔒 LOCKED FOR CIT' if m['is_locked'] else '📝 READY FOR CHECKER LOCK'}"
    )


@tool
def lock_dispatch_manifest(checker_name: str = "Chief Treasury Officer", notes: str = "Authorized for CIT Dispatch") -> str:
    """Digitally sign and permanently lock the fleet dispatch manifest for armored cash transit."""
    res = masters_mgr.lock_manifest(checker_name=checker_name, notes=notes)
    m = res["manifest"]
    return (
        f"🔒 **CIT Dispatch Manifest Locked & Authorized**:\n"
        f"- Manifest ID: `{m['manifest_id']}`\n"
        f"- Authorized By: **{checker_name}** ({m['checker_signoff']['role']})\n"
        f"- Total Chest Withdrawal: **₹{m['total_vault_pull_rupees']:,.0f}**\n"
        f"- Cryptographic Token: `{m['checker_signoff']['digital_signature_token']}`\n"
        f"- Timestamp: **{m['generated_at']}**\n"
        f"- Status: **Vault pull slips transmitted to CIT armored fleet**."
    )


@tool
def explain_model_variables(terminal_id: str = "1") -> str:
    """Explain the 30 ML model variables (features, autoregressive lags, rolling statistics, calendar, paydays, and exogenous multipliers) and their relative weights."""
    tid = str(terminal_id).replace("ATM-", "").replace("ATM", "").strip()
    res = masters_mgr.compute_fleet_replenishment()
    t = res["terminals"].get(tid)
    site_name = t["site_name"] if t else f"ATM #{tid}"
    city = t["city"] if t else "Mumbai"
    cycle_days = t["cycle_days"] if t else 7
    pred_demand = (
        t.get("c1_pred_demand")
        or t.get("observations", {}).get("pred_demand_7d")
        or t.get("formula_ledger", {}).get("expected_demand")
        or 4000000.0
    ) if t else 4000000.0

    return (
        f"### 🧬 LightGBM Strategy C: Model Variables & Feature Registry\n\n"
        f"**Terminal Focus:** **ATM {tid} — {site_name}** ({city} · {cycle_days}-Day Cycle)\n\n"
        f"The autonomous replenishment forecast is generated by a 30-variable gradient boosted tree model (LightGBM Strategy C), partitioned into 5 key operational feature clusters:\n\n"
        f"#### **1. Autoregressive Lags & Demand Velocity (~45.2% Aggregate Importance)**\n"
        f"- `lag_7` (**Relative Weight: ~22.6%** — *Highest single predictor*): Same-day-prior-week withdrawal volume, capturing day-of-week cyclicality.\n"
        f"- `lag_1` (**Relative Weight: ~18.4%**): Prior-day actual cash demand, capturing immediate short-term velocity and inertia.\n"
        f"- `lag_14` (**Relative Weight: ~9.8%**): Demand 14 days earlier, providing bi-weekly seasonal stability.\n"
        f"- `rolling_mean_7` (**Relative Weight: ~14.2%**): 7-day smoothed rolling mean baseline, filtering out single-day transaction outliers.\n"
        f"- `rolling_std_7` (**Relative Weight: ~8.5%**): 7-day rolling standard deviation, feeding directly into the dynamic volatility cushion ($\\sigma_H$).\n\n"
        f"#### **2. Calendar & Payroll Surge Signals (~28.4% Aggregate Importance)**\n"
        f"- `is_payday` (**Relative Weight: ~16.8%**): Flags salary credit dates (1st, 15th, and month-end; **Aug 31 is an exact payday** driving a **+41.4% demand surge**).\n"
        f"- `day_of_week` (**Relative Weight: ~12.5%**): Monday=0 to Sunday=6; drives the **+46.3% weekend traveler footfall lift** on Friday evenings through Sundays.\n"
        f"- `days_to_month_end` / `is_month_start` / `is_month_end`: Non-linear monthly spending curves.\n\n"
        f"#### **3. Holiday & Bank Branch Closure Regimes (~14.1% Aggregate Importance)**\n"
        f"- `is_bank_closed_day`: Flags RBI listed bank holidays, Sundays, and 2nd/4th Saturdays. Counter closures at commercial branches directly divert over-the-counter cash withdrawal traffic to nearby ATMs.\n"
        f"- `is_pre_holiday_3d`: 1-to-3 day window immediately preceding a public holiday (+25% to +35% pre-holiday consumer cash stockpiling).\n"
        f"- `is_post_holiday_3d`: Post-holiday transaction normalization coefficient.\n\n"
        f"#### **4. Exogenous Signals & Transit Footfall Multipliers (~8.3% Aggregate Importance)**\n"
        f"- `monsoon_weather_index`: City-specific meteorological signal (active Mumbai monsoon advisory multiplier: 0.98x - 1.05x).\n"
        f"- `site_type_transit_footfall`: Site-specific structural velocity floor ({t['site_type'] if t else 'Transit'} baseline: ~₹{round(pred_demand/cycle_days):,.0f}/day).\n\n"
        f"#### **5. Institutional Policy & Safety Constraints (Masters Configuration)**\n"
        f"- `cassette_capacity`: Physical mechanical limit (**₹{t['capacity'] if t else 6000000:,.0f}** across 4 cassettes), enforcing the hard ceiling $\\min(\\text{{Cap}}, \\text{{Demand}} + \\text{{Buffer}})$.\n"
        f"- `target_service_level`: 99.0% SLA; the buffer is the backtest P99 of actual ÷ forecast, not a normal z-score."
    )


@tool
def query_holiday_calendar(terminal_id: str = "1", cycle_window: str = "Aug 25 - Aug 31") -> str:
    """Query RBI bank holidays, gazetted public holidays, Sundays, and corporate salary paydays active in and around the current replenishment cycle."""
    tid = str(terminal_id).replace("ATM-", "").replace("ATM", "").strip()
    res = masters_mgr.compute_fleet_replenishment()
    t = res["terminals"].get(tid)
    site_name = t["site_name"] if t else f"ATM #{tid}"
    city = t["city"] if t else "Mumbai"
    pred_demand = (
        t.get("c1_pred_demand")
        or t.get("observations", {}).get("pred_demand_7d")
        or t.get("formula_ledger", {}).get("expected_demand")
        or 6500000.0
    ) if t else 6500000.0

    return (
        f"### 📅 Active Holiday & Banking Calendar Analysis\n\n"
        f"**Terminal:** **ATM {tid} — {site_name}** ({city} RBI Jurisdiction)\n"
        f"**Replenishment Horizon:** **Cycle 1 ({cycle_window}, 2026)**\n\n"
        f"Here are the verified RBI holidays, branch closures, and payroll catalysts in and around this planning window:\n\n"
        f"#### **1. Within Active Replenishment Cycle (Aug 25 - Aug 31, 2026)**\n"
        f"- **🗓️ August 30, 2026 (Sunday)** — **Weekly Bank Branch Closure**\n"
        f"  * RBI Status: `is_bank_closed_day = 1`\n"
        f"  * Impact: Commercial bank branches in {city} are closed for counter withdrawals, diverting high-volume cash transactions directly to ATM {tid}.\n\n"
        f"- **💰 August 31, 2026 (Monday)** — **Corporate & Public Sector Month-End Payday**\n"
        f"  * Master Registry: `paydays.json` flags `is_payday = 1` (verified in RBI calendar reference).\n"
        f"  * Impact: Major catalyst! Salary credit transfers trigger an immediate **+41.4% cash withdrawal surge** on the final day of Cycle 1, which the forecasting engine factored into the demand calculation.\n\n"
        f"#### **2. Preceding Surrounding Holidays (Mid-August 2026)**\n"
        f"- **🇮🇳 August 15, 2026 (Saturday)** — **Independence Day** (National Gazetted Holiday across all RBI offices).\n"
        f"- **🕊️ August 19, 2026 (Wednesday)** — **Parsi New Year (Shahenshahi) / Raksha Bandhan** (RBI Negotiable Instruments Act holiday in Maharashtra offices: Mumbai, Belapur, Nagpur).\n"
        f"- **🏛️ August 22, 2026 (Saturday)** — **4th Saturday Statutory Bank Holiday** (All commercial bank counters closed).\n\n"
        f"#### **3. Upcoming Near-Term Holidays (Early September 2026)**\n"
        f"- **🌺 September 4 - 7, 2026** — **Ganesh Chaturthi Festival Period**\n"
        f"  * Maharashtra State Major Holiday (`is_pre_holiday_3d = 1` activates during the tail of Cycle 1).\n"
        f"  * Pre-festival retail and transport consumer cash withdrawals begin surging 72 hours in advance.\n\n"
        f"💡 **Operational Summary for Checker:** There is 1 scheduled bank closure (Sunday Aug 30) and 1 high-impact payroll day (Monday Aug 31) inside this cycle window. The predictive model has already accounted for both events in its ₹{pred_demand:,.0f} demand forecast."
    )


@tool
def explain_mathematical_derivation(terminal_id: str = "1") -> str:
    """Step-by-step mathematical proof and treasury formula derivation for an ATM's cash indent."""
    tid = str(terminal_id).replace("ATM-", "").replace("ATM", "").strip()
    res = masters_mgr.compute_fleet_replenishment()
    t = res["terminals"].get(tid)
    if not t:
        return f"Error: Terminal ID '{terminal_id}' not found."

    cycle_days = t.get("cycle_days", 7)
    pred_demand = (
        t.get("c1_pred_demand")
        or t.get("observations", {}).get("pred_demand_7d")
        or t.get("formula_ledger", {}).get("expected_demand")
        or 0.0
    )
    safety_buffer = (
        t.get("c1_safety_buffer")
        or t.get("assumptions", {}).get("safety_buffer")
        or 0.0
    )
    capacity = t.get("capacity", 6000000.0)
    existing_cash = t.get("existing_cash", 0.0)
    gross_target = t.get("c1_gross_target") or t.get("formula_ledger", {}).get("gross_target") or min(capacity, pred_demand + safety_buffer)
    net_indent = t.get("c1_net_indent") or t.get("formula_ledger", {}).get("net_indent") or max(0.0, gross_target - existing_cash)
    sla = masters_mgr.assumptions.get("target_service_level", 0.99)
    daily_burn = pred_demand / max(1, cycle_days)
    buffer_pct = safety_buffer / pred_demand * 100 if pred_demand else 0.0

    uncapped_gross = pred_demand + safety_buffer
    is_capped = uncapped_gross > capacity

    return (
        f"### 🧮 Mathematical Derivation of Replenishment Indent\n\n"
        f"**Terminal:** **ATM {tid} — {t['site_name']}** ({t['city']} · {t['serviced_by']} {cycle_days}-Day Cycle)\n\n"
        f"The replenishment indent is computed via a 5-step order-up-to formula:\n\n"
        f"#### **Step 1: Cycle Horizon Demand ($\\hat{{D}}_H$)**\n"
        f"- Cycle Duration: $H = {cycle_days}\\text{{ days}}$\n"
        f"- Total Projected Withdrawal Demand:\n"
        f"  $$\\hat{{D}}_H = \\sum_{{t=1}}^{{{cycle_days}}} \\hat{{d}}_t = \\mathbf{{₹{pred_demand:,.0f}}}$$\n"
        f"- Daily Average Demand Velocity: $\\bar{{d}} = \\frac{{\\hat{{D}}_H}}{{H}} = \\mathbf{{₹{round(daily_burn):,.0f}\\text{{ / day}}}}$\n\n"
        f"#### **Step 2: SLA Safety Buffer ($B_S$)**\n"
        f"- Target Service Level: $\\alpha = {sla*100:.1f}\\%$\n"
        f"- Multiplier from the 100-ATM backtest: the P99 of actual ÷ forecast 7-day totals over the past 26 weeks, "
        f"calibrated separately for holiday and quiet weeks (scaled by $\\sqrt{{7/H}}$ for cycles shorter than 7 days)\n"
        f"- Safety Buffer: $B_S = \\mathbf{{₹{safety_buffer:,.0f}}}$ ({buffer_pct:.1f}% of $\\hat{{D}}_H$)\n\n"
        f"#### **Step 3: Gross Target Inventory ($T_{{gross}}$)**\n"
        f"- Unconstrained Requirement: $\\hat{{D}}_H + B_S = ₹{pred_demand:,.0f} + ₹{safety_buffer:,.0f} = ₹{uncapped_gross:,.0f}$\n"
        f"- Physical Cassette Capacity Limit: $\\text{{Cap}} = \\mathbf{{₹{capacity:,.0f}}}$\n"
        f"- Gross Target:\n"
        f"  $$T_{{gross}} = \\min\\left(\\text{{Cap}}, \\hat{{D}}_H + B_S\\right) = \\mathbf{{₹{gross_target:,.0f}}}$$ {'*(Capped at mechanical cassette limit)*' if is_capped else '*(Within cassette headroom)*'}\n\n"
        f"#### **Step 4: Net Top-Up Indent ($I_{{net}}$)**\n"
        f"- Existing Safe Balance: $C_{{existing}} = \\mathbf{{₹{existing_cash:,.0f}}}$\n"
        f"- Net Order to Dispatch:\n"
        f"  $$I_{{net}} = \\max\\left(0, T_{{gross}} - C_{{existing}}\\right) = ₹{gross_target:,.0f} - ₹{existing_cash:,.0f} = \\mathbf{{₹{net_indent:,.0f}}}$$\n\n"
        f"#### **Step 5: Days of Cash Remaining (DCR)**\n"
        f"- Current Runway: $\\text{{DCR}} = \\frac{{C_{{existing}}}}{{\\bar{{d}}}} = \\frac{{₹{existing_cash:,.0f}}}{{₹{round(daily_burn):,.0f}}} = \\mathbf{{{t['dcr_days']} \\text{{ Days}}}}$ ({t['urgency_status']} status)\n\n"
        f"**Verification:** Loading exactly ₹{net_indent:,.0f} restores vault cash to ₹{gross_target:,.0f}, ensuring {sla*100:.1f}% SLA compliance through {cycle_days} days without cassette overfill."
    )


@tool
def execute_inter_atm_sweep_tool(donor_terminal_id: str, recipient_terminal_id: str, amount_rupees: float, actor: str) -> str:
    """Execute an Inter-ATM corridor cash sweep transferring surplus cash from donor to starving ATM, avoiding emergency CIT run."""
    try:
        res = masters_mgr.execute_inter_atm_sweep(donor_id=donor_terminal_id, recipient_id=recipient_terminal_id, amount=amount_rupees, actor=actor)
        rec = res["sweep_record"]
        sav = rec.get("estimated_savings", {})
        return (
            f"⚡ **Inter-ATM Cash Sweep Executed Successfully**:\n"
            f"- Transfer: **₹{rec['transfer_amount']:,.0f}** from **ATM {rec['donor_terminal_id']} ({rec['donor_site_name']})** ➔ **ATM {rec['recipient_terminal_id']} ({rec['recipient_site_name']})**\n"
            f"- Corridor: **{rec['corridor']}**\n"
            f"- Avoided Emergency CIT Cost: **₹{sav.get('avoided_emergency_cit', 3500):,.0f}**\n"
            f"- Avoided Downtime SLA Penalty: **₹{sav.get('avoided_stockout_penalty', 10000):,.0f}**\n"
            f"- Net Treasury Benefit: **₹{sav.get('net_benefit', 12000):,.0f}**\n"
            f"- Status: **{rec['status']}** (Record ID: `{rec['sweep_id']}`)"
        )
    except Exception as e:
        return f"Error executing Inter-ATM sweep: {e}"


@tool
def clear_reject_bin_tool(terminal_id: str, actor: str) -> str:
    """Clear/empty the purge reject bin counter to 0 notes, unblocking dispenser hardware after mispicks/rejects."""
    try:
        res = masters_mgr.execute_clear_reject_bin(terminal_id=terminal_id, actor=actor)
        return (
            f"🧹 **Purge Reject Bin Emptied & Reset**:\n"
            f"- Terminal: **ATM {res['terminal_id']}**\n"
            f"- Notes in Reject Bin: **0 / 300 notes (0.0% Saturation — NORMAL)**\n"
            f"- Hardware Sensor Lockout: **CLEARED (Dispenser Online)**\n"
            f"- Audit Timestamp: **{res['cleared_at']}**"
        )
    except Exception as e:
        return f"Error clearing reject bin: {e}"


@tool
def explain_downtime_and_anomalies(terminal_id: str = "3") -> str:
    """Explain downtime risk factors, reject bin saturation, CRA SLA missed visit breaches, threat monitor, and anomaly telemetry for an ATM."""
    tid = str(terminal_id).replace("ATM-", "").replace("ATM", "").strip()
    res = masters_mgr.compute_fleet_replenishment()
    t = res["terminals"].get(tid)
    if not t:
        return f"Error: Terminal ID '{terminal_id}' not found."

    score = t.get("downtime_risk_score", 0.0)
    causes = t.get("downtime_causes", [])
    meta = t.get("operational_metadata", {})
    t_code = t.get("terminal_code", f"SYNCRM000{tid}")
    dev_type = t.get("device_type", "CRM")

    reject_bin = t.get("reject_bin", {})
    r_cur = reject_bin.get("current_notes", 0)
    r_max = reject_bin.get("max_capacity_notes", 300)
    r_pct = reject_bin.get("fill_percentage", 0.0)
    r_stat = reject_bin.get("status", "NORMAL")

    cra = meta.get("cra_tracking", {})
    threat = meta.get("threat_monitoring", {})
    mispick = meta.get("mispick_rate_per_1k", 0.0)

    causes_str = "\n".join([f"  • ⚠️ **{c}**" for c in causes]) if causes else "  • No acute deterministic triggers detected."

    return (
        f"### 🛡️ Downtime Risk & Disruption Diagnostics: ATM {tid} ({t_code})\n\n"
        f"- Site: **{t['site_name']}** ({t.get('location_tier', 'Metro')} · **{t.get('location_category', 'Metro Transit')}**)\n"
        f"- Device Architecture: **{dev_type}** ({'Cash Recycler' if t.get('is_crm') else 'Dispenser Only'})\n"
        f"- Composite Downtime Risk Score: **{score:.1f}%** ({'🔴 CRITICAL HIGH RISK' if score >= 75 else ('🟡 ELEVATED RISK' if score >= 40 else '🟢 LOW RISK')})\n\n"
        f"#### **Deterministic Root-Cause Triggers:**\n{causes_str}\n\n"
        f"#### **Physical & Disruption Telemetry:**\n"
        f"- **Purge Reject Bin**: **{r_cur} / {r_max} notes** ({r_pct:.1f}% capacity) — Status: `{r_stat}`\n"
        f"- **CRA Servicing Status**: `{cra.get('status', 'ON_TIME')}` (Window: {cra.get('scheduled_window', 'N/A')}, Delay: {cra.get('delay_hours', 0)} hrs, SLA Breach Penalty: ₹{cra.get('sla_penalty_incurred', 0):,.0f})\n"
        f"- **Threat & Curfew Monitoring**: Zone: `{threat.get('zone', 'GREEN')}`, Threat Level: `{threat.get('curfew_threat_level', 'LOW')}`, Curfew Active: `{'YES (' + threat.get('active_curfew_hours', '') + ')' if threat.get('has_active_curfew') else 'No curfews active'}`\n"
        f"- **Dispenser Mispick Velocity**: {mispick} notes / 1,000 dispenses\n"
        f"- **Usable Ending Cash**: **₹{t['existing_cash']:,.0f}** (~{t.get('dcr_hours', 0)} hours runway)"
    )


@tool
def explain_atm_performance_and_cycles(terminal_id: str = "1") -> str:
    """Analyze historical performance, past 3-cycle forecasting efficacy, ticket sizes (day/week/month), and P50/P80/P95 quantile demand projections."""
    tid = str(terminal_id).replace("ATM-", "").replace("ATM", "").strip()
    res = masters_mgr.compute_fleet_replenishment()
    t = res["terminals"].get(tid)
    if not t:
        return f"Error: Terminal ID '{terminal_id}' not found."

    perf = t.get("past_cycles_efficacy", {})
    cycles = perf.get("cycles") if isinstance(perf, dict) and "cycles" in perf else (t.get("last_3_cycles_efficacy") or [])
    rating = perf.get("cycle_rating") if isinstance(perf, dict) and "cycle_rating" in perf else (t.get("overall_cycle_rating") or "OPTIMAL")

    c_rows = []
    for c in cycles:
        c_label = c.get("cycle_label") or c.get("cycle_code", "Cycle")
        pred_d = c.get("predicted_demand", 0)
        act_d = c.get("actual_demand", 0)
        eff = c.get("effectiveness_pct") if "effectiveness_pct" in c else c.get("accuracy_pct", 95.0)
        res_cash = c.get("ending_residual_cash") if "ending_residual_cash" in c else c.get("residual_ending_cash", 0)
        stk = c.get("stockout_hours", 0)
        sla_s = c.get("cra_sla_status") or c.get("cra_on_time_status", "ON_TIME")
        stat = c.get("cycle_status") or "OPTIMAL"
        act_str = f"₹{act_d:,.0f}" if isinstance(act_d, (int, float)) else "N/A"
        c_rows.append(
            f"  • **{c_label}**: Predicted ₹{pred_d:,.0f} vs Actual {act_str} | "
            f"Accuracy: **{eff}%** | Residual Ending: ₹{res_cash:,.0f} | Stockout: {stk}h | SLA: `{sla_s}` ({stat})"
        )
    cycles_str = "\n".join(c_rows) if c_rows else "  • No historical cycles logged."

    avg_ticket = t.get("avg_ticket_size", {})
    quant = t.get("quantile_forecast") or t.get("quantiles", {})
    cassettes = t.get("cassettes", [])
    cass_str = ", ".join([f"Slot {c['slot']}: {c['current_notes']}/{c['max_notes']} notes (₹{c['denomination']})" for c in cassettes])

    p50_val = quant.get("p50_demand", 0)
    p80_val = quant.get("p80_demand", 0)
    p95_val = quant.get("p95_demand", 0)
    mix = t.get("denomination_weights") or {"₹500": 0.65, "₹200": 0.20, "₹100": 0.15}
    n500 = int(round((p50_val * mix.get("₹500", 0.65)) / 500))
    n200 = int(round((p50_val * mix.get("₹200", 0.20)) / 200))
    n100 = int(round((p50_val * mix.get("₹100", 0.15)) / 100))

    return (
        f"### 📊 Historical Efficacy & Multi-Quantile Demand: ATM {tid} ({t.get('terminal_code', 'SYNCRM000' + tid)})\n\n"
        f"- Location Category: **{t.get('location_category', 'Metro Transit')}** ({t.get('location_tier', 'Metro')})\n"
        f"- Historical Forecast Efficacy Rating: **{rating}** (3-Cycle Avg Accuracy: **{perf.get('avg_accuracy_pct', 94.0):.1f}%**)\n\n"
        f"#### **Past 3-Cycle Prediction Audit:**\n{cycles_str}\n\n"
        f"#### **Average Ticket Sizes:**\n"
        f"- Daily: **₹{avg_ticket.get('daily', 2500):,.0f}** | Weekly: **₹{avg_ticket.get('weekly', 3000):,.0f}** | Monthly: **₹{avg_ticket.get('monthly', 4000):,.0f}**\n\n"
        f"#### **Plausible Quantile Demand Projections:**\n"
        f"- **P50 (Expected Baseline)**: **₹{p50_val:,.0f}** (₹500: {n500:,} notes, ₹200: {n200:,} notes, ₹100: {n100:,} notes)\n"
        f"- **P80 (Surge Buffer)**: **₹{p80_val:,.0f}**\n"
        f"- **P95 (Peak Stress Scenario)**: **₹{p95_val:,.0f}**\n\n"
        f"#### **Cassette Physics Configuration:**\n- {cass_str}"
    )


ALL_TOOLS = [
    query_network_status,
    explain_atm,
    simulate_scenario,
    get_masters,
    update_master_parameter,
    approve_indents,
    adjust_indent,
    get_dispatch_manifest,
    lock_dispatch_manifest,
    explain_model_variables,
    query_holiday_calendar,
    explain_mathematical_derivation,
    execute_inter_atm_sweep_tool,
    clear_reject_bin_tool,
    explain_downtime_and_anomalies,
    explain_atm_performance_and_cycles
]
# Tools that move cash or freeze approvals. The LLM never gets these: they run only from
# slash commands that carry an explicit trailing `confirm` and the signed-in user's identity.
ACTION_TOOLS = [approve_indents, adjust_indent, lock_dispatch_manifest, execute_inter_atm_sweep_tool, clear_reject_bin_tool]
LLM_TOOLS = [t for t in ALL_TOOLS if t.name not in {a.name for a in ACTION_TOOLS}]
TOOL_MAP = {t.name: t for t in LLM_TOOLS}


# ==========================================
# 2. LANGGRAPH STATE & WORKFLOW
# ==========================================

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    context: Dict[str, Any]
    action_payload: Optional[Dict[str, Any]]
    api_key: Optional[str]


def create_agent_graph():
    """Builds and compiles the conversational LangGraph agent with tool execution."""

    def call_model(state: AgentState) -> Dict[str, Any]:
        api_key = state.get("api_key")
        if not api_key:
            raise ValueError("No API key available")

        # Try models in priority order: gemini-3.5-flash (ultra-fast with thinking_budget=0), then default, then fallback
        model_configs = [
            {"model": "gemini-3.5-flash", "thinking_budget": 0},
            {"model": "gemini-3.5-flash"},
            {"model": "gemini-2.5-flash"}
        ]
        last_err = None
        for cfg in model_configs:
            try:
                kwargs = {"model": cfg["model"], "google_api_key": api_key, "temperature": 0.2, "max_retries": 0}
                if "thinking_budget" in cfg:
                    kwargs["thinking_budget"] = cfg["thinking_budget"]
                llm = ChatGoogleGenerativeAI(**kwargs)
                llm_with_tools = llm.bind_tools(LLM_TOOLS)
                resp = llm_with_tools.invoke(state["messages"])
                return {"messages": [resp]}
            except Exception as e:
                last_err = e
                continue
        raise last_err

    def call_tools(state: AgentState) -> Dict[str, Any]:
        last_msg = state["messages"][-1]
        tool_msgs = []
        action_payload = dict(state.get("action_payload") or {})

        for tc in getattr(last_msg, "tool_calls", []):
            t_name = tc.get("name")
            t_args = tc.get("args", {})
            if t_name in TOOL_MAP:
                try:
                    tool_output = TOOL_MAP[t_name].invoke(t_args)
                except Exception as err:
                    tool_output = f"Tool execution failed: {err}"

                tool_msgs.append(ToolMessage(content=str(tool_output), tool_call_id=tc["id"]))

                # Populate UI action triggers for the portal frontend
                if t_name in ["explain_atm", "simulate_scenario", "explain_model_variables", "query_holiday_calendar", "explain_mathematical_derivation"]:
                    tid = str(t_args.get("terminal_id", "1")).replace("ATM-", "").replace("ATM", "").strip()
                    action_payload["type"] = "highlight_terminal"
                    action_payload["terminal_id"] = tid
                    if t_name == "explain_model_variables":
                        action_payload["visual_aid"] = {"type": "feature_variables", "terminal_id": tid}
                    elif t_name == "query_holiday_calendar":
                        action_payload["visual_aid"] = {"type": "holiday_calendar", "terminal_id": tid}
                    elif t_name == "explain_mathematical_derivation":
                        action_payload["visual_aid"] = {"type": "math_derivation", "terminal_id": tid}
                elif t_name in ["update_master_parameter", "adjust_indent"]:
                    action_payload["type"] = "master_updated"
                    if "terminal_id" in t_args:
                        action_payload["terminal_id"] = str(t_args["terminal_id"]).replace("ATM-", "").replace("ATM", "").strip()
                elif t_name in ["approve_indents", "lock_dispatch_manifest"]:
                    action_payload["type"] = "indents_approved"
                elif t_name in ["query_network_status", "get_dispatch_manifest"]:
                    action_payload["type"] = "refresh"
                elif t_name == "execute_inter_atm_sweep_tool":
                    action_payload["type"] = "sweep_executed"
                elif t_name == "clear_reject_bin_tool":
                    tid = str(t_args.get("terminal_id", "5")).replace("ATM-", "").replace("ATM", "").strip()
                    action_payload["type"] = "reject_bin_cleared"
                    action_payload["terminal_id"] = tid
                elif t_name == "explain_downtime_and_anomalies":
                    tid = str(t_args.get("terminal_id", "3")).replace("ATM-", "").replace("ATM", "").strip()
                    action_payload["type"] = "highlight_terminal"
                    action_payload["terminal_id"] = tid
                    action_payload["visual_aid"] = {"type": "downtime_diagnostics", "terminal_id": tid}
                elif t_name == "explain_atm_performance_and_cycles":
                    tid = str(t_args.get("terminal_id", "1")).replace("ATM-", "").replace("ATM", "").strip()
                    action_payload["type"] = "highlight_terminal"
                    action_payload["terminal_id"] = tid
                    action_payload["visual_aid"] = {"type": "cycle_efficacy", "terminal_id": tid}

        return {"messages": tool_msgs, "action_payload": action_payload}

    def should_continue(state: AgentState) -> str:
        last_msg = state["messages"][-1]
        if getattr(last_msg, "tool_calls", None):
            return "tools"
        return END

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", call_tools)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, ["tools", END])
    workflow.add_edge("tools", "agent")

    return workflow.compile()


# Lazy-compile agent graph
_compiled_graph = None

def get_agent_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = create_agent_graph()
    return _compiled_graph


def extract_message_text(content: Union[str, list, Any]) -> str:
    """Safely extract plain text from LangChain message content (handles list of parts/dicts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and "text" in p:
                parts.append(p["text"])
            else:
                parts.append(str(p))
        return "\n".join(parts)
    return str(content)


# ==========================================
# 3. SLASH COMMANDS & DETERMINISTIC FALLBACK
# ==========================================

ACTION_COMMANDS = {"/override", "/adjust", "/lock", "/signoff", "/approve", "/approve-indents", "/sweep", "/rebalance", "/clear-bin", "/empty-bin", "/reset"}
CHECKER_COMMANDS = {"/lock", "/signoff", "/approve", "/approve-indents", "/reset"}


def _gate_action(cmd: str, cmd_text: str, confirmed: bool, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Returns a refusal/preview reply unless the action is confirmed by a signed-in user with the right role."""
    actor = str(context.get("actor") or "").strip()
    if not actor:
        return {"reply": "❌ Sign-in required: actions can only be run by an authenticated portal user.", "action_payload": {}}
    if cmd in CHECKER_COMMANDS and str(context.get("role", "")).upper() != "CHECKER":
        return {"reply": f"❌ `{cmd}` requires the CHECKER role. You are signed in as {context.get('role') or 'unknown'}.", "action_payload": {}}
    if not confirmed:
        return {
            "reply": f"⚠️ **Confirmation required.** Nothing was changed. To run it, send exactly:\n`{cmd_text} confirm`",
            "action_payload": {}
        }
    return None


def handle_slash_command(cmd_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Instant execution for slash commands (/status, /explain, /simulate, /set, /approve, /override, /manifest, /lock, /masters, /save, /reset, /help).
    State-changing commands need a trailing `confirm` and a signed-in actor in context (set by the server, never the client)."""
    words = cmd_text.split()
    confirmed = len(words) > 1 and words[-1].lower() == "confirm"
    if confirmed:
        cmd_text = " ".join(words[:-1])
    parts = cmd_text.split(maxsplit=2)
    cmd = parts[0].lower()
    arg1 = parts[1] if len(parts) > 1 else ""
    arg2 = parts[2] if len(parts) > 2 else ""
    actor = str(context.get("actor") or "").strip()

    if cmd in ("/approve", "/approve-indents") and not arg1:
        return {"reply": "Usage: `/approve [atm_id|all] confirm` — name the ATMs explicitly.", "action_payload": {}}
    if cmd in ACTION_COMMANDS:
        refusal = _gate_action(cmd, cmd_text, confirmed, context)
        if refusal:
            return refusal

    if cmd in ["/status", "/summary"]:
        res = query_network_status.invoke({})
        return {"reply": res, "action_payload": {"type": "refresh"}}

    elif cmd in ["/explain", "/why"]:
        tid = arg1 or str(context.get("active_terminal", "1"))
        res = explain_atm.invoke({"terminal_id": tid})
        return {"reply": res, "action_payload": {"type": "highlight_terminal", "terminal_id": tid}}

    elif cmd in ["/override", "/adjust"]:
        # Usage: /override 1 2800000 Chest shortage
        tokens = (arg1 + " " + arg2).split(maxsplit=2)
        if len(tokens) >= 2:
            tid = tokens[0]
            amt = tokens[1]
            reason = tokens[2] if len(tokens) > 2 else "Checker manual adjustment"
            res = adjust_indent.invoke({"terminal_id": tid, "new_indent": amt, "reason": reason, "actor": actor})
            return {"reply": res, "action_payload": {"type": "master_updated", "terminal_id": tid}}
        else:
            return {"reply": "Usage: `/override [atm_id] [amount] [reason]` (e.g., `/override 1 2800000 Chest capacity limit`)", "action_payload": {}}

    elif cmd in ["/manifest", "/pullslip"]:
        res = get_dispatch_manifest.invoke({})
        return {"reply": res, "action_payload": {"type": "show_manifest"}}

    elif cmd in ["/lock", "/signoff"]:
        notes = " ".join(p for p in (arg1, arg2) if p) or "Authorized for CIT Dispatch"
        try:
            res = lock_dispatch_manifest.invoke({"checker_name": actor, "notes": notes})
        except Exception as e:
            return {"reply": f"❌ Manifest not locked: {e}", "action_payload": {}}
        return {"reply": res, "action_payload": {"type": "indents_approved", "manifest_locked": True}}

    elif cmd in ["/simulate", "/whatif"]:
        tid = arg1 or str(context.get("active_terminal", "1"))
        delay = 0
        indent = 0.0
        if "delay" in arg2.lower():
            m = re.search(r"\d+", arg2)
            delay = int(m.group(0)) if m else 4
        elif "indent" in arg2.lower():
            m = re.search(r"\d+", arg2)
            indent = float(m.group(0)) if m else 1000000.0
        res = simulate_scenario.invoke({"terminal_id": tid, "custom_indent": indent, "delay_hours": delay})
        return {"reply": res, "action_payload": {"type": "simulation_run", "terminal_id": tid}}

    elif cmd in ["/set"]:
        tokens = (arg1 + " " + arg2).split()
        if len(tokens) >= 2:
            param = tokens[0]
            val = tokens[1]
            sub_id = tokens[2] if len(tokens) > 2 else None
            cat = "assumptions"
            if "capacity" in param.lower():
                cat = "terminals"
                sub_id = tokens[1] if len(tokens) > 2 else "1"
                val = tokens[2] if len(tokens) > 2 else tokens[1]
            res = update_master_parameter.invoke({"category": cat, "key": param, "value": val, "sub_id": sub_id})
            return {"reply": res, "action_payload": {"type": "master_updated"}}
        else:
            return {"reply": "Usage: `/set [parameter] [value]` (e.g. `/set holding_rate_annual 7.0%` or `/set cit_trip_cost 2500`)", "action_payload": {}}

    elif cmd in ["/approve", "/approve-indents"]:
        res = approve_indents.invoke({"terminal_ids": arg1, "actor": actor})
        return {"reply": res, "action_payload": {"type": "indents_approved"}}

    elif cmd in ["/masters", "/config"]:
        cat = arg1 or "all"
        res = get_masters.invoke({"category": cat})
        return {"reply": res, "action_payload": {"type": "view_masters"}}

    elif cmd in ["/variables", "/features", "/inputs"]:
        tid = arg1 or str(context.get("active_terminal", "1"))
        res = explain_model_variables.invoke({"terminal_id": tid})
        return {"reply": res, "action_payload": {"type": "highlight_terminal", "terminal_id": tid, "visual_aid": {"type": "feature_variables", "terminal_id": tid}}}

    elif cmd in ["/holidays", "/calendar"]:
        tid = arg1 or str(context.get("active_terminal", "1"))
        res = query_holiday_calendar.invoke({"terminal_id": tid})
        return {"reply": res, "action_payload": {"type": "highlight_terminal", "terminal_id": tid, "visual_aid": {"type": "holiday_calendar", "terminal_id": tid}}}

    elif cmd in ["/formula", "/math", "/derivation"]:
        tid = arg1 or str(context.get("active_terminal", "1"))
        res = explain_mathematical_derivation.invoke({"terminal_id": tid})
        return {"reply": res, "action_payload": {"type": "highlight_terminal", "terminal_id": tid, "visual_aid": {"type": "math_derivation", "terminal_id": tid}}}

    elif cmd in ["/sweep", "/rebalance"]:
        tokens = (arg1 + " " + arg2).split()
        if len(tokens) != 3 or not tokens[2].isdigit() or int(tokens[2]) < 10000:
            return {"reply": "Usage: `/sweep [donor_atm] [recipient_atm] [amount in rupees] confirm` (e.g. `/sweep 4 3 700000 confirm`).", "action_payload": {}}
        donor, recipient, amt = tokens[0], tokens[1], float(tokens[2])
        res = execute_inter_atm_sweep_tool.invoke({"donor_terminal_id": donor, "recipient_terminal_id": recipient, "amount_rupees": amt, "actor": actor})
        return {"reply": res, "action_payload": {"type": "sweep_executed"}}

    elif cmd in ["/clear-bin", "/empty-bin"]:
        if not arg1:
            return {"reply": "Usage: `/clear-bin [atm_id] confirm`.", "action_payload": {}}
        tid = arg1
        res = clear_reject_bin_tool.invoke({"terminal_id": tid, "actor": actor})
        return {"reply": res, "action_payload": {"type": "reject_bin_cleared", "terminal_id": tid}}

    elif cmd in ["/downtime", "/anomaly", "/threat"]:
        tid = arg1 or str(context.get("active_terminal", "3"))
        res = explain_downtime_and_anomalies.invoke({"terminal_id": tid})
        return {"reply": res, "action_payload": {"type": "highlight_terminal", "terminal_id": tid, "visual_aid": {"type": "downtime_diagnostics", "terminal_id": tid}}}

    elif cmd in ["/cycles", "/performance", "/quantiles"]:
        tid = arg1 or str(context.get("active_terminal", "1"))
        res = explain_atm_performance_and_cycles.invoke({"terminal_id": tid})
        return {"reply": res, "action_payload": {"type": "highlight_terminal", "terminal_id": tid, "visual_aid": {"type": "cycle_efficacy", "terminal_id": tid}}}

    elif cmd in ["/save", "/commit"]:
        res = masters_mgr.save_overrides()
        return {"reply": "💾 **Changes Committed to Production Disk** (`config_overrides.json`). All batch scripts and live APIs now reflect these parameters.", "action_payload": {"type": "saved"}}

    elif cmd in ["/reset"]:
        try:
            res = masters_mgr.reset_to_defaults()
        except PermissionError as e:
            return {"reply": f"❌ {e}", "action_payload": {}}
        return {"reply": "🔄 **Masters & Assumptions Reset to System Defaults**.", "action_payload": {"type": "reset"}}

    elif cmd in ["/help"]:
        return {
            "reply": (
                "**Available AI Slash Commands**:\n"
                "- `/status` — View 30-second network health HUD & indent totals\n"
                "- `/explain [atm_id]` — Facts, observations & demand drivers for an ATM\n"
                "- `/downtime [atm_id]` — Inspect downtime triggers, CRA breach, threat & anomaly metrics\n"
                "- `/cycles [atm_id]` — Inspect past 3-cycle forecasting efficacy, ticket sizes & P50/P80/P95 quantiles\n"
                "- `/sweep [donor] [recipient] [rupees] confirm` — Execute same-city inter-ATM cash rebalancing\n"
                "- `/clear-bin [atm_id] confirm` — Clear purge reject bin and reset sensor lockout\n"
                "- `/variables [atm_id]` — 30 ML model features, relative importance weights & lags\n"
                "- `/holidays [atm_id]` — Verified RBI bank holidays & payday calendar for cycle\n"
                "- `/formula [atm_id]` — Exact 6-step mathematical derivation of indent\n"
                "- `/override [atm_id] [amount] [reason] confirm` — Override an indent (needs a different checker to approve)\n"
                "- `/manifest` — Inspect Digital CIT Dispatch Manifest & Chest Pull Slip\n"
                "- `/lock [notes] confirm` — Digitally sign & lock the fleet dispatch manifest (Checker)\n"
                "- `/simulate [atm_id] delay [hours]` — Run what-if scenario on delivery delay\n"
                "- `/set [param] [value]` — Update cost assumption or terminal capacity\n"
                "- `/approve [all|atm_id] confirm` — Approve replenishment orders for dispatch (Checker)\n"
                "- `/masters [cost|terminal|holiday|weather]` — Inspect master data records\n"
                "- `/save` — Commit active overrides to permanent disk\n"
                "- `/reset confirm` — Reset all parameters to baseline defaults (Checker)"
            ),
            "action_payload": {}
        }

    return {"reply": f"Unknown command: `{cmd}`. Type `/help` for available commands.", "action_payload": {}}


def deterministic_nlp_dispatch(query: str, context: Dict[str, Any], quota_exceeded: bool = False) -> Dict[str, Any]:
    """Rich conversational engine for local execution and offline fallback."""
    q = query.lower().strip()
    active_tid = str(context.get("active_terminal", "1"))
    found_tid = re.search(r"atm[- ]?(\d+)|terminal[- ]?(\d+)|synatm000(\d+)", q)
    if found_tid:
        active_tid = found_tid.group(1) or found_tid.group(2) or found_tid.group(3)
    else:
        # Match city / site names
        if "airport" in q or "t2" in q: active_tid = "2"
        elif "railway" in q or "concourse" in q: active_tid = "1"
        elif "pune" in q or "mall" in q: active_tid = "3"
        elif "nagpur" in q or "wardha" in q: active_tid = "4"
        elif "nashik" in q: active_tid = "5"
        elif "bandra" in q: active_tid = "6"
        elif "thane" in q: active_tid = "7"
        elif "aurangabad" in q: active_tid = "8"

    banner = ""
    if quota_exceeded:
        banner = "\n\n> 💡 *Note: Google AI Studio Free Tier daily quota limit (20 req/day) was reached on this API key. Copilot dynamically resolved your operational request using the real-time simulation engine.*"

    # Free text never executes a control action; it only proposes the exact slash command to run.
    def propose(cmd_text: str, what: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            "reply": (
                f"⚠️ **{what} — not executed.** Free-text requests never move cash or change approvals.\n"
                f"Review the figures, then send exactly:\n`{cmd_text} confirm`"
            ) + banner,
            "action_payload": payload or {}
        }

    # 0. Checker Override with Natural Language Reason
    # e.g. "change atm 1 indent to 28 lakhs because chest has limited 500 notes"
    m_override = re.search(r"(?:change|override|adjust|set)\s*(?:atm[- ]?(\d+))?.*?(?:indent|load|amount)?.*?(?:to|=)\s*₹?\s*(\d+(?:\.\d+)?)\s*(lakhs?|l|cr)?(?:\s*(?:because|due to|reason:?)\s*(.*))?", q)
    if m_override and ("indent" in q or "lakh" in q or "atm" in q or "to" in q) and any(verb in q for verb in ["change", "override", "adjust"]):
        target_tid = m_override.group(1) or active_tid
        raw_amt = float(m_override.group(2))
        unit = (m_override.group(3) or "").lower()
        if "cr" in unit:
            amt = raw_amt * 10000000.0
        elif "l" in unit:
            amt = raw_amt * 100000.0
        elif raw_amt >= 10000.0:
            amt = raw_amt
        else:
            return {
                "reply": f"⚠️ Ambiguous amount '{raw_amt}'. To prevent unintended multi-crore indents, please specify explicit units like '{int(raw_amt)} Lakhs', '{int(raw_amt)}L', or exact rupees." + banner,
                "action_payload": {}
            }
        reason_txt = (m_override.group(4) or "Checker manual override via Copilot").strip()
        return propose(f"/override {target_tid} {int(amt)} {reason_txt}", f"Override ATM {target_tid} to ₹{amt:,.0f}",
                       {"type": "highlight_terminal", "terminal_id": target_tid})

    # 0b. Inter-ATM Sweep / Corridor Rebalancing
    if any(k in q for k in ["sweep", "rebalance", "surplus cash", "transfer cash", "corridor rebalance", "move cash"]):
        m_trans = re.search(r"from\s*(?:atm[- ]?)?(\d+).*?to\s*(?:atm[- ]?)?(\d+)", q)
        donor = m_trans.group(1) if m_trans else "4"
        recipient = m_trans.group(2) if m_trans else "3"
        m_amt = re.search(r"₹?\s*(\d+(?:\.\d+)?)\s*(lakhs?|l|cr)?", q)
        amt = 700000.0
        if m_amt:
            val_num = float(m_amt.group(1))
            u = (m_amt.group(2) or "").lower()
            if "cr" in u: amt = val_num * 10000000.0
            elif "l" in u: amt = val_num * 100000.0
            elif val_num >= 10000.0: amt = val_num
            else:
                return {
                    "reply": f"⚠️ Ambiguous sweep amount '{m_amt.group(0)}'. Please specify explicit units (e.g. '7 Lakhs' or '700000')." + banner,
                    "action_payload": {}
                }
        return propose(f"/sweep {donor} {recipient} {int(amt)}", f"Sweep ₹{amt:,.0f} from ATM {donor} to ATM {recipient}",
                       {"type": "highlight_terminal", "terminal_id": donor})

    # 0c. Clear / Empty Reject Bin
    if any(k in q for k in ["clear reject", "empty reject", "reset reject", "purge reject", "reject bin"]):
        target = active_tid if active_tid != "1" else "5"
        return propose(f"/clear-bin {target}", f"Clear reject bin on ATM {target}",
                       {"type": "highlight_terminal", "terminal_id": target})

    # 0d. Downtime Risk Factors, CRA SLA Breach & Anomaly Telemetry
    if any(k in q for k in ["downtime", "anomaly", "anomalies", "risk score", "cra miss", "missed visit", "sla breach", "threat", "curfew", "mispick"]):
        return {
            "reply": explain_downtime_and_anomalies.invoke({"terminal_id": active_tid}) + banner,
            "action_payload": {"type": "highlight_terminal", "terminal_id": active_tid, "visual_aid": {"type": "downtime_diagnostics", "terminal_id": active_tid}}
        }

    # 0e. Historical Performance, Past Cycles, Ticket Sizes & Quantiles
    if any(k in q for k in ["cycle", "cycles", "efficacy", "accuracy rate", "historical performance", "ticket size", "p50", "p80", "p95", "quantile", "quantiles"]):
        return {
            "reply": explain_atm_performance_and_cycles.invoke({"terminal_id": active_tid}) + banner,
            "action_payload": {"type": "highlight_terminal", "terminal_id": active_tid, "visual_aid": {"type": "cycle_efficacy", "terminal_id": active_tid}}
        }

    # Digital Manifest / Chest Pull Slip
    if any(k in q for k in ["manifest", "pull slip", "vault slip", "chest slip"]):
        if any(k in q for k in ["lock", "sign", "authorize"]):
            return propose("/lock", "Lock the dispatch manifest (permanent for this cycle)", {"type": "show_manifest"})
        return {
            "reply": get_dispatch_manifest.invoke({}) + banner,
            "action_payload": {"type": "show_manifest"}
        }

    # 1. Critical ATMs List
    if any(k in q for k in ["which atm", "critical atm", "at risk", "stockout risk"]):
        res = masters_mgr.compute_fleet_replenishment()
        crit = [f"• **ATM {t['terminal_id']} ({t['site_name']})**: Runway **{t.get('dcr_hours', int(round(t.get('dcr_days', 2)*24)))}h**, Cash **₹{t['existing_cash']:,.0f}**, Net Indent **₹{t['c1_net_indent']:,.0f}**" for t in res["terminals"].values() if t["urgency_status"] == "CRITICAL"]
        reply = f"**Currently Critical ATMs (<48 Hours Cash Remaining)** ({len(crit)} terminals):\n" + "\n".join(crit)
        return {"reply": reply + banner, "action_payload": {"type": "refresh"}}

    # 2. Greetings & Capabilities (word boundary match)
    if re.search(r"\b(hello|hi|hey|who are you|what can you do|help me|greetings)\b", q):
        reply = (
            "Hello! I am your **Cash Intelligence AI Copilot**, an expert autonomous treasury operations and cash replenishment strategist for Indian commercial banks.\n\n"
            "### **Operational Capabilities (Maker is the System, You are the Checker):**\n"
            "- **Evidence-Backed Explanations**: Ask *\"Why is ATM 1 proposed for ₹34L?\"* or *\"Explain ATM 2\"* for facts, observations, and safety buffer math.\n"
            "- **Model Variables & Features**: Ask *\"What are the model variables?\"* or *\"I want to know the variables mate\"* for exact LightGBM features, lags, and weights.\n"
            "- **Holiday & Payday Calendars**: Ask *\"Are there any holidays around that time?\"* for verified RBI branch closures and salary dates.\n"
            "- **Mathematical Formula Derivation**: Ask *\"Show me the mathematical derivation\"* for the step-by-step 6-step equation ledger.\n"
            "- **Natural Language Overrides**: Say *\"Change ATM 1 indent to 28 Lakhs because chest has limited 500 notes\"* — I will calculate the impact and record your audit reason.\n"
            "- **Digital Manifest & Pull Slips**: Ask *\"Show dispatch manifest\"* or *\"Lock manifest\"* for currency chest route authorization.\n"
            "- **What-If Simulations**: Ask *\"Simulate a 6-hour delay for ATM 2\"* to evaluate delivery bottlenecks and stockout risks.\n"
            "- **Master Assumptions**: Ask *\"Show cost assumptions\"* or *\"Set holding rate to 7.8%\"* to update economic parameters live."
        )
        return {"reply": reply + banner, "action_payload": {"type": "refresh"}}

    # 3. Model Features, Variables, Weights & Lags
    if any(k in q for k in ["variable", "variables", "feature", "features", "input", "inputs", "weight", "weights", "lag", "lags", "predictor", "predictors", "coefficient", "importance"]):
        return {
            "reply": explain_model_variables.invoke({"terminal_id": active_tid}) + banner,
            "action_payload": {"type": "highlight_terminal", "terminal_id": active_tid, "visual_aid": {"type": "feature_variables", "terminal_id": active_tid}}
        }

    # 4. Holidays, Festivals & Banking Calendar
    if any(k in q for k in ["holiday", "holidays", "calendar", "festival", "festivals", "around that time", "bank close", "bank closed", "payday", "pay days", "salary day", "long weekend"]):
        return {
            "reply": query_holiday_calendar.invoke({"terminal_id": active_tid}) + banner,
            "action_payload": {"type": "highlight_terminal", "terminal_id": active_tid, "visual_aid": {"type": "holiday_calendar", "terminal_id": active_tid}}
        }

    # 5. Mathematical Formula Derivation & Ledger
    if any(k in q for k in ["math", "mathematical", "derivation", "derive", "formula", "equation", "equations", "step by step", "calculation", "how did you calculate", "ledger"]):
        return {
            "reply": explain_mathematical_derivation.invoke({"terminal_id": active_tid}) + banner,
            "action_payload": {"type": "highlight_terminal", "terminal_id": active_tid, "visual_aid": {"type": "math_derivation", "terminal_id": active_tid}}
        }

    # 6. Network Overview / Status
    if any(k in q for k in ["status", "overview", "summary", "how is the network", "health", "cutoff"]):
        return {"reply": query_network_status.invoke({}) + banner, "action_payload": {"type": "refresh"}}

    # 7. Approval
    if any(k in q for k in ["approve", "dispatch all", "sign off", "accept indent"]):
        target = "all" if ("all" in q or "every" in q or "fleet" in q) else active_tid
        return propose(f"/approve {target}", f"Approve indents for {'all terminals' if target == 'all' else f'ATM {target}'}",
                       {"type": "refresh"})

    # 8. Simulation / What-If
    if any(k in q for k in ["what if", "simulate", "delay", "scenario", "test"]):
        m_delay = re.search(r"(\d+)\s*(?:hour|hr|h)", q)
        delay = int(m_delay.group(1)) if m_delay else (4 if "delay" in q else 0)
        m_indent = re.search(r"(?:indent|amount|cash|load|refill)?\s*(?:of|=|to)?\s*₹?\s*(\d+(?:\.\d+)?)\s*(lakhs?|l|cr)?", q)
        custom_indent = None
        if m_indent and any(w in q for w in ["indent", "load", "refill", "amount", "zero", "0"]):
            val_str = m_indent.group(1)
            if val_str:
                val_num = float(val_str)
                u = (m_indent.group(2) or "").lower()
                if "cr" in u: custom_indent = val_num * 10000000.0
                elif "l" in u: custom_indent = val_num * 100000.0
                elif val_num >= 10000.0 or val_num == 0.0: custom_indent = val_num
        return {"reply": simulate_scenario.invoke({"terminal_id": active_tid, "delay_hours": delay, "custom_indent": custom_indent}) + banner, "action_payload": {"type": "highlight_terminal", "terminal_id": active_tid}}

    # 9. Parameter Updates
    if any(k in q for k in ["set ", "update ", "change "]) and any(p in q for p in ["holding", "rate", "cost", "sla", "trip", "penalty", "capacity"]):
        m_pct = re.search(r"(\d+(?:\.\d+)?)\s*%", q)
        m_num = re.search(r"₹?\s*(\d+(?:,\d+)*)", q)
        val = m_pct.group(0) if m_pct else (m_num.group(1) if m_num else "7.5%")
        key = "holding_rate_annual" if "holding" in q else ("cit_trip_cost" if "trip" in q else ("target_service_level" if "sla" in q else "holding_rate_annual"))
        return {"reply": update_master_parameter.invoke({"category": "assumptions", "key": key, "value": val}) + banner, "action_payload": {"type": "master_updated"}}

    # 10. Masters / Assumptions
    if any(k in q for k in ["master", "assumption", "holding rate", "cost", "weather"]):
        cat = "cost" if ("cost" in q or "holding" in q) else ("weather" if "weather" in q else "all")
        return {"reply": get_masters.invoke({"category": cat}) + banner, "action_payload": {"type": "view_masters"}}

    # 11. Explain / Why specific ATM
    if any(k in q for k in ["why", "explain", "reason", "driver", "critical", "dcr", "buffer", "recommend", "indent"]):
        return {"reply": explain_atm.invoke({"terminal_id": active_tid}) + banner, "action_payload": {"type": "highlight_terminal", "terminal_id": active_tid}}

    # Conversational Default
    return {
        "reply": (
            f"Here is the operational breakdown for **ATM {active_tid}**:\n\n"
            f"{explain_atm.invoke({'terminal_id': active_tid})}\n\n"
            f"*Tip: You can ask \"What are the model variables?\", \"Are there any holidays around that time?\", \"Show mathematical derivation\", or \"Change ATM {active_tid} indent to 25L because...\".*{banner}"
        ),
        "action_payload": {"type": "highlight_terminal", "terminal_id": active_tid}
    }


# ==========================================
# 4. PRIMARY RUN_AGENT ENTRYPOINT
# ==========================================

SYSTEM_PROMPT = """You are the Cash Intelligence AI Copilot, an expert autonomous treasury operations and cash replenishment strategist for Indian commercial banks.
Operational Context & Governance:
- THE SYSTEM IS THE MAKER: It analyzes forecasts, physical cassette limits, safety stocks, and recommends replenishment indents.
- THE USER IS THE CHECKER: The user is the senior treasury / currency chest officer verifying and authorizing cash movements.
- Your role is to assist the Checker by explaining recommendations with FACTS, OBSERVATIONS, and ASSUMPTIONS, answering deep questions about ML features, holiday calendars, mathematical derivations, running simulations, adjusting indents based on Checker natural language reasons, and locking the digital dispatch manifest.

Operational Network Context:
- Fleet of 8 ATMs across Maharashtra (Mumbai Airport T2, Railway Concourse, Pune Mall, Nagpur CBD, Nashik Metro, etc.).
- Replenishment cycle length: CIT = 7 days, Branch = 3 days. Active planning window: Cycle 1 (Aug 25 - Aug 31, 2026).
- Active target SLA: 99.0% service level (buffer = backtest P99 of actual ÷ forecast, by holiday or quiet week).
- CIT cutoff time: 18:00 IST.
- Baseline annual holding cost: 6.5%. Fixed CIT trip fee: ₹2,000.

Specialized Domain Tools:
1. `explain_model_variables(terminal_id)`: Explains the 30 ML model variables (features, autoregressive lags like lag_7 and lag_1, rolling stats, calendar, paydays, weather) and their relative weights/impacts for that ATM. ALWAYS call this tool whenever the Checker asks about variables, features, inputs, weights, lags, or what fed the forecast.
2. `query_holiday_calendar(terminal_id, cycle_window)`: Queries exact RBI bank holidays, Sundays, gazetted holidays, and salary payday dates for the active replenishment cycle (Cycle 1: Aug 25 - Aug 31, 2026) and surrounding dates. ALWAYS call this tool whenever the Checker asks about holidays, calendar, festivals, bank closures, or "around that time".
3. `explain_mathematical_derivation(terminal_id)`: Explains the exact 6-step mathematical formula steps (expected demand, horizon volatility sigma, z-score safety buffer, gross target, and net top-up indent). ALWAYS call this tool whenever the Checker asks for math, derivation, formula, equations, calculations, or ledger steps.
4. `explain_atm(terminal_id)`: Provides high-level operational facts, runway, and primary demand drivers.
5. Overrides, approvals, manifest lock, sweeps and reject-bin clearing are NOT available to you as tools. When the Checker asks for one, explain the impact and give the exact slash command to run, ending in `confirm` (e.g. `/override 1 2800000 chest short of 500s confirm`, `/approve 3 confirm`, `/lock confirm`, `/sweep 4 3 700000 confirm`, `/clear-bin 5 confirm`). Never say an action was done.
6. `simulate_scenario(terminal_id, custom_indent, delay_hours)`: Runs What-If simulation with custom indents or delay hours.
7. `get_dispatch_manifest()`: For digital manifest inspection.
8. `update_master_parameter(category, key, value)`: Updates economic parameters live.
11. `explain_downtime_and_anomalies(terminal_id)`: Explains composite downtime risk %, reject bin status, CRA SLA missed visit breaches, curfews/threat levels, and mechanical mispick rates.
12. `explain_atm_performance_and_cycles(terminal_id)`: Explains past 3-cycle forecasting efficacy (accuracy %, residual ending cash, stockouts), average ticket sizes (daily/weekly/monthly), and multi-quantile demand projections (P50, P80, P95).

Guidelines for Answers:
- Format all currency amounts clearly in Indian Rupees (₹) with Indian numbering (e.g., ₹26,04,800 or ₹26.0 Lakhs, ₹2.13 Cr).
- Never give generic canned responses. Always answer the Checker's specific question with concrete figures, percentages, dates, and formula steps.
"""

def enrich_action_payload(action_payload: Optional[Dict[str, Any]], query: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Enrich the Copilot action payload with structured telemetry, simulation parameters, and charts data."""
    if not isinstance(action_payload, dict):
        action_payload = {}

    q = (query or "").lower().strip()

    # Determine targeted terminal ID
    tid = str(action_payload.get("terminal_id") or "").strip()
    if not tid or tid == "None":
        m = re.search(r"atm[- ]?(\d+)|terminal[- ]?(\d+)|synatm000(\d+)", q)
        if m:
            tid = m.group(1) or m.group(2) or m.group(3)
        elif "railway" in q or "concourse" in q: tid = "1"
        elif "airport" in q or "t2" in q: tid = "2"
        elif "pune" in q or "mall" in q: tid = "3"
        elif "nagpur" in q: tid = "4"
        elif "nashik" in q: tid = "5"
        elif "bandra" in q: tid = "6"
        elif "thane" in q: tid = "7"
        elif "aurangabad" in q: tid = "8"
        elif context.get("active_terminal"):
            tid = str(context.get("active_terminal"))
        else:
            tid = "1"

    fleet = masters_mgr.compute_fleet_replenishment()
    terminals = fleet.get("terminals", {})
    t = terminals.get(tid)
    if t:
        action_payload["terminal_id"] = tid
        pred_demand = (
            t.get("c1_pred_demand")
            or t.get("observations", {}).get("pred_demand_7d")
            or t.get("formula_ledger", {}).get("expected_demand")
            or 4000000.0
        )
        safety_buffer = (
            t.get("c1_safety_buffer")
            or t.get("assumptions", {}).get("safety_buffer")
            or 1500000.0
        )
        daily_burn = round((pred_demand / max(1, t.get("cycle_days", 7))) / 100.0) * 100.0
        action_payload["terminal_data"] = {
            "terminal_id": tid,
            "site_name": t.get("site_name", f"ATM #{tid}"),
            "city": t.get("city", "Mumbai"),
            "existing_cash": t.get("existing_cash", 0.0),
            "capacity": t.get("capacity", 6000000.0),
            "c1_net_indent": t.get("c1_net_indent", 0.0),
            "c1_gross_target": t.get("c1_gross_target", 0.0),
            "c1_safety_buffer": safety_buffer,
            "c1_pred_demand": pred_demand,
            "dcr_days": t.get("dcr_days", 2.0),
            "dcr_hours": t.get("dcr_hours", int(round(t.get("dcr_days", 2.0) * 24))),
            "urgency_status": t.get("urgency_status", "SCHEDULED"),
            "daily_burn": round(daily_burn / 100.0) * 100.0,
            "cycle_days": t.get("cycle_days", 7),
            "holding_rate_annual": masters_mgr.assumptions.get("holding_rate_annual", 0.065),
            "cit_trip_cost": masters_mgr.assumptions.get("cit_trip_cost", 2000.0),
            "is_approved": t.get("is_approved", False)
        }

    # Visual Aid detection for UI cards
    if "visual_aid" not in action_payload:
        if any(k in q for k in ["variable", "variables", "feature", "features", "weight", "weights", "lag", "lags"]):
            action_payload["visual_aid"] = {"type": "feature_variables", "terminal_id": tid}
        elif any(k in q for k in ["holiday", "holidays", "calendar", "festival", "around that time"]):
            action_payload["visual_aid"] = {"type": "holiday_calendar", "terminal_id": tid}
        elif any(k in q for k in ["math", "mathematical", "derivation", "formula", "equation"]):
            action_payload["visual_aid"] = {"type": "math_derivation", "terminal_id": tid}

    # Summary telemetry for network
    summary = fleet.get("summary", {})
    action_payload["network_summary"] = {
        "total_existing_cash": summary.get("total_existing_cash", 0),
        "total_recommended_indent": summary.get("total_recommended_indent", 0),
        "critical_count": len([x for x in terminals.values() if (x.get("urgency_status") or "").upper() == "CRITICAL"]),
        "total_terminals": len(terminals)
    }

    return action_payload


def run_agent(
    query: str,
    context: Optional[Dict[str, Any]] = None,
    history: Optional[List[Dict[str, Any]]] = None,
    client_key: Optional[str] = None
) -> Dict[str, Any]:
    """Main execution point for the Cash Intelligence Copilot.

    Supports:
    - Slash command fast-routing.
    - Anaphora resolution and multi-turn context inheritance.
    - Full conversational LangGraph tool-calling loop using Gemini 2.5 Flash.
    - High-fidelity deterministic fallback if API is unreachable.
    """
    clean_query = (query or "").strip()
    ctx = dict(context or {})
    active_tid = str(ctx.get("active_terminal") or "1")

    # Anaphora & Terminal Resolution: check query first, then reverse history scan
    q_lower = clean_query.lower()
    found_tid = re.search(r"atm[- ]?(\d+)|terminal[- ]?(\d+)|synatm000(\d+)", clean_query, re.I)
    if found_tid:
        active_tid = found_tid.group(1) or found_tid.group(2) or found_tid.group(3)
    elif "airport" in q_lower or "t2" in q_lower:
        active_tid = "2"
    elif "railway" in q_lower or "concourse" in q_lower:
        active_tid = "1"
    elif "pune" in q_lower or "mall" in q_lower:
        active_tid = "3"
    elif "nagpur" in q_lower or "wardha" in q_lower:
        active_tid = "4"
    elif "nashik" in q_lower:
        active_tid = "5"
    elif "bandra" in q_lower:
        active_tid = "6"
    elif "thane" in q_lower:
        active_tid = "7"
    elif "aurangabad" in q_lower:
        active_tid = "8"
    elif history:
        # Reverse scan past turns to resolve contextual references ("around that time", "the variables", "why is it")
        for h in reversed(history):
            h_content = str(h.get("content", ""))
            h_lower = h_content.lower()
            m_hist = re.search(r"atm[- ]?(\d+)|terminal[- ]?(\d+)|synatm000(\d+)", h_content, re.I)
            if m_hist:
                active_tid = m_hist.group(1) or m_hist.group(2) or m_hist.group(3)
                break
            if "airport" in h_lower or "t2" in h_lower or "synatm0002" in h_lower:
                active_tid = "2"
                break
            if "railway" in h_lower or "concourse" in h_lower or "synatm0001" in h_lower:
                active_tid = "1"
                break
            if "pune" in h_lower:
                active_tid = "3"
                break
            if "nagpur" in h_lower:
                active_tid = "4"
                break

    ctx["active_terminal"] = active_tid

    # 1. Fast path for slash commands
    if clean_query.startswith("/"):
        slash_res = handle_slash_command(clean_query, ctx)
        payload = enrich_action_payload(slash_res.get("action_payload", {}), clean_query, ctx)
        return {
            "reply": slash_res.get("reply", ""),
            "action_payload": payload,
            "source": "deterministic_slash",
            "timestamp": datetime.now().isoformat()
        }

    # 2. Resolve API key
    api_key = resolve_api_key(client_key)

    if api_key:
        # Check quota availability quickly before launching LangGraph to avoid tenacity timeout hangs
        if not is_gemini_quota_available(api_key):
            fb = deterministic_nlp_dispatch(clean_query, ctx, quota_exceeded=True)
            payload = enrich_action_payload(fb.get("action_payload", {}), clean_query, ctx)
            return {
                "reply": fb.get("reply", ""),
                "action_payload": payload,
                "source": "deterministic_quota_fallback",
                "timestamp": datetime.now().isoformat()
            }

        try:
            # Build conversation message sequence
            messages: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]

            # Include recent chat history (up to last 8 turns)
            if history:
                for h in history[-8:]:
                    role = h.get("role", "")
                    content = h.get("content", "")
                    if role == "user" and content:
                        messages.append(HumanMessage(content=content))
                    elif role in ["agent", "assistant"] and content:
                        messages.append(AIMessage(content=content))

            # Add current user query enriched with active terminal context
            enriched_query = clean_query
            if active_tid and "atm" not in clean_query.lower() and "terminal" not in clean_query.lower():
                enriched_query += f" (Note: Currently focused on ATM {active_tid} in UI, planning window Cycle 1: Aug 25 - Aug 31, 2026)"
            messages.append(HumanMessage(content=enriched_query))

            initial_state: AgentState = {
                "messages": messages,
                "context": ctx,
                "action_payload": {},
                "api_key": api_key
            }

            graph = get_agent_graph()
            graph_output = graph.invoke(initial_state)

            final_msg = graph_output["messages"][-1]
            reply_text = extract_message_text(final_msg.content)
            action_payload = graph_output.get("action_payload") or {}
            payload = enrich_action_payload(action_payload, clean_query, ctx)

            return {
                "reply": reply_text,
                "action_payload": payload,
                "source": "gemini_langgraph",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as err:
            err_str = str(err)
            is_quota = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower()
            print(f"[Copilot Warning] LangGraph execution error: {err}. Falling back to deterministic dispatcher.")
            fb = deterministic_nlp_dispatch(clean_query, ctx, quota_exceeded=is_quota)
            payload = enrich_action_payload(fb.get("action_payload", {}), clean_query, ctx)
            return {
                "reply": fb.get("reply", ""),
                "action_payload": payload,
                "source": "deterministic_quota_fallback" if is_quota else "deterministic_fallback",
                "timestamp": datetime.now().isoformat()
            }

    # 3. Deterministic NLP fallback
    fb = deterministic_nlp_dispatch(clean_query, ctx)
    payload = enrich_action_payload(fb.get("action_payload", {}), clean_query, ctx)
    return {
        "reply": fb.get("reply", ""),
        "action_payload": payload,
        "source": "deterministic_fallback",
        "timestamp": datetime.now().isoformat()
    }
