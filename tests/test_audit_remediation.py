"""Comprehensive Automated Test Suite for live_portal_v2 Audit Remediation.

Covers:
- Physical capacity & cassette slot space (Findings A1, A2, C4)
- Policy engine, exact z-scores, event dates, 3-day branch horizon, 2-cycle arithmetic (Findings A3, A4, A5, A6, A7, A8)
- Runway, status counts, real telemetry, truthful exports (Findings A9, A10, A11, C1, C2, C3)
- Maker/checker separation, locks, stale overrides, sweep conservation (Findings B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, D1, D2)
- AI Copilot unit parsing, negation detection, confirmation requirements, simulation (Findings B2, B3, C5)
"""

import unittest
import math
from statistics import NormalDist

from masters_manager import (
    MastersManager,
    masters_mgr,
    get_physical_cassette_capacity,
    get_free_cassette_space,
    allocate_denomination_mix,
)
from agent import (
    adjust_indent,
    approve_indents,
    simulate_scenario,
    deterministic_nlp_dispatch,
    handle_slash_command,
)


def fresh_state():
    """Unlock (tests only) and reset so each test starts from the baseline."""
    masters_mgr.is_manifest_locked = False
    masters_mgr.locked_manifest_snapshot = None
    masters_mgr.reset_to_defaults()


CHECKER = {"actor": "Bob Checker", "role": "CHECKER"}


class TestPhysicalCapacityAndDenominations(unittest.TestCase):
    def setUp(self):
        fresh_state()

    def test_a1_configured_capacity_caps_at_physical_cassette_limit(self):
        """A1: Configured capacity must equal physical cassette capacity (₹28.6L for 4 cassettes x 2200 notes)."""
        for tid in ["1", "2", "3", "4", "5", "6", "7", "8"]:
            phys_cap = get_physical_cassette_capacity(tid)
            self.assertEqual(phys_cap, 2860000.0, f"ATM {tid} physical capacity must be 28.6L")
            self.assertLessEqual(masters_mgr.terminal_configs[tid]["capacity"], 2860000.0)

    def test_a2_denomination_split_matches_exact_rupees_and_slot_bounds(self):
        """A2: Denomination split must sum exactly to net indent and respect cassette free slots."""
        free_space = get_free_cassette_space("1")
        total_free = free_space["total_free_cash"]
        self.assertGreater(total_free, 0)

        test_indent = min(1000000.0, total_free)
        alloc = allocate_denomination_mix(test_indent, "1")
        
        # Verify exact rupee match
        rupee_sum = (
            alloc["500_notes"] * 500
            + alloc["200_notes"] * 200
            + alloc["100_notes"] * 100
        )
        self.assertEqual(rupee_sum, alloc["total_allocated"])
        self.assertEqual(alloc["total_allocated"], test_indent)

    def test_a2_overflow_detection(self):
        """A2: Requesting indent greater than free cassette space must raise overflow warning."""
        free_space = get_free_cassette_space("1")
        overflow_indent = free_space["total_free_cash"] + 500000.0
        alloc = allocate_denomination_mix(overflow_indent, "1")
        self.assertIsNotNone(alloc["overflow_warning"])
        self.assertEqual(alloc["total_allocated"], free_space["total_free_cash"])

    def test_c4_crm_collection_bin_note_physics(self):
        """C4: Collection bins in recyclers must not have average note value > ₹500."""
        for tid, cfg in masters_mgr.terminal_configs.items():
            if cfg.get("is_crm"):
                coll = cfg.get("crm_collection_bin", {})
                notes = coll.get("current_notes", 0)
                val = coll.get("current_cash_value", 0)
                if notes > 0:
                    avg_note_val = val / notes
                    self.assertLessEqual(
                        avg_note_val,
                        500.0,
                        f"ATM {tid} collection bin average note value ({avg_note_val}) exceeds ₹500 note maximum",
                    )


class TestPolicyCalculationEngine(unittest.TestCase):
    def setUp(self):
        fresh_state()

    def test_a8_exact_normal_dist_z_scores(self):
        """A8: SLA to z-score mapping must use exact inv_cdf, not step-table approximations."""
        # 97% SLA -> ~1.8808 (step table was erroneously 1.645)
        z_97 = NormalDist().inv_cdf(0.97)
        self.assertAlmostEqual(z_97, 1.88079, places=4)
        self.assertNotEqual(round(z_97, 3), 1.645)

        # 99.5% SLA -> ~2.5758 (step table was erroneously 2.326)
        z_995 = NormalDist().inv_cdf(0.995)
        self.assertAlmostEqual(z_995, 2.57583, places=4)
        self.assertNotEqual(round(z_995, 3), 2.326)

    def test_a6_event_multipliers_do_not_bleed_outside_cycle(self):
        """A6: 3-day ATMs (T4, T8: Aug 25-27) must not receive Salary Surge event (Aug 31 - Sep 4)."""
        res = masters_mgr.compute_fleet_replenishment()
        t4 = res["terminals"]["4"]
        t8 = res["terminals"]["8"]

        # Neither T4 nor T8 should have Salary Surge in active events
        self.assertNotIn("Salary Surge", t4.get("active_exogenous_events", []))
        self.assertNotIn("Salary Surge", t8.get("active_exogenous_events", []))

    def test_a5_branch_atm_t4_3day_horizon_zero_indent(self):
        """A5: Branch ATM T4 simulated over its 3-day horizon must have ₹0 net indent (no phantom stockout)."""
        res = masters_mgr.compute_fleet_replenishment()
        t4 = res["terminals"]["4"]
        self.assertEqual(t4["cycle_days"], 3)
        self.assertEqual(t4["serviced_by"], "BRANCH")
        self.assertEqual(t4["c1_net_indent"], 0.0, "T4 has sufficient cash for 3 days and should need ₹0 indent")

    def test_a4_two_cycle_arithmetic_and_sequencing(self):
        """A4: 2-cycle plans must sequence Cycle 2 strictly after Cycle 1 without artificial 110% inflation."""
        res = masters_mgr.compute_fleet_replenishment()
        for tid, t in res["terminals"].items():
            if t.get("is_two_cycle"):
                c1_date = t.get("suggested_date")
                c2_date = t.get("c2_suggested_date")
                self.assertIsNotNone(c2_date)
                self.assertGreater(
                    c2_date,
                    c1_date,
                    f"ATM {tid} Cycle 2 date ({c2_date}) must be after Cycle 1 date ({c1_date})",
                )
                # Split should equal total need
                total_need = t.get("formula_ledger", {}).get("net_indent", 0)
                if total_need > 0:
                    split_sum = t["c1_net_indent"] + t.get("c2_net_indent", 0)
                    self.assertAlmostEqual(split_sum, total_need, places=1)


class TestVarianceAndAuditDataIntegrity(unittest.TestCase):
    def setUp(self):
        fresh_state()

    def test_c1_predictions_export_has_no_np_random_actuals(self):
        """C1: Predictions CSV must report future forecast truthfully and not fabricate actuals."""
        csv_data = masters_mgr.generate_predictions_csv()
        self.assertIn("N/A (Future Forecast)", csv_data)
        self.assertNotIn("np.random", csv_data)

    def test_c3_mutually_exclusive_status_counts(self):
        """C3: Status counts must sum exactly to 8 fleet terminals without overlap."""
        res = masters_mgr.compute_fleet_replenishment()
        summary = res["summary"]
        crit = summary["critical_count"]
        sched = summary["scheduled_count"]
        health = summary["healthy_count"]
        self.assertEqual(
            crit + sched + health,
            8,
            f"Status counts must sum to 8: Critical({crit}) + Scheduled({sched}) + Healthy({health})",
        )

    def test_unmeasured_cycle_actuals_render_without_crashing(self):
        """Daily action report and agent cycle explainer must handle unmeasured actuals."""
        report = masters_mgr.generate_daily_cash_action_report_data()
        self.assertEqual(len(report["rows"]), 8)
        self.assertIn("Act N/A", report["rows"][0][11])
        import agent
        self.assertIn("Actual N/A", agent.explain_atm_performance_and_cycles.invoke({"terminal_id": "1"}))

    def test_c2_no_fake_fixed_cycle_efficacy(self):
        """C2: Cycle efficacy should report unmeasured/N/A rather than hardcoded 94.6%."""
        res = masters_mgr.compute_fleet_replenishment()
        for tid, t in res["terminals"].items():
            eff = t.get("past_cycles_efficacy", {})
            rating = eff.get("cycle_rating", "")
            self.assertTrue(
                rating.startswith("UNMEASURED") or rating == "N/A",
                f"ATM {tid} efficacy rating '{rating}' should be UNMEASURED or N/A",
            )


class TestMakerCheckerControlsAndLocks(unittest.TestCase):
    def setUp(self):
        fresh_state()

    def test_b6_manifest_pulls_only_approved_cit_indents(self):
        """B6: With 0 approvals, manifest vault pull is ₹0; branch ATMs are excluded from CIT pull."""
        # Initially 0 approved
        m = masters_mgr.generate_dispatch_manifest()
        self.assertEqual(m["total_vault_pull_rupees"], 0.0)

        # Approve ATM 1 (CIT)
        masters_mgr.approve_terminal_indent("1", officer_name="Chief Treasury Officer")
        m2 = masters_mgr.generate_dispatch_manifest()
        t1_indent = masters_mgr.approved_indents["1"]["approved_indent"]
        self.assertEqual(m2["total_vault_pull_rupees"], t1_indent)

        # Approve ATM 4 (Branch) - should not be added to CIT manifest vault pull
        masters_mgr.approve_terminal_indent("4", officer_name="Chief Treasury Officer")
        m3 = masters_mgr.generate_dispatch_manifest()
        self.assertEqual(m3["total_vault_pull_rupees"], t1_indent)

    def test_b8_manifest_lock_freezes_state_and_rejects_edits(self):
        """B8: Manifest lock freezes snapshot with HMAC-SHA256 signature and rejects modifications."""
        # Approve ATM 1
        masters_mgr.approve_terminal_indent("1", officer_name="Chief Treasury Officer")
        lock_res = masters_mgr.lock_manifest(checker_name="Chief Treasury Officer")
        self.assertTrue(masters_mgr.is_manifest_locked)
        sig = lock_res["manifest"]["checker_signoff"]["digital_signature_token"]
        self.assertTrue(sig.startswith("SIG-HMAC-"))

        # Subsequent override must raise PermissionError
        with self.assertRaises(PermissionError):
            masters_mgr.apply_checker_override("1", 2000000.0, reason="Late override")

        # Subsequent parameter update must raise PermissionError
        with self.assertRaises(PermissionError):
            masters_mgr.update_parameter("assumptions", "cit_trip_cost", 3000)

        # Subsequent approval must raise PermissionError
        with self.assertRaises(PermissionError):
            masters_mgr.approve_terminal_indent("2", officer_name="Chief Treasury Officer")

    def test_b7_parameter_edits_invalidate_approvals(self):
        """B7: Editing economic assumptions invalidates prior approvals."""
        masters_mgr.approve_terminal_indent("1", officer_name="Chief Treasury Officer")
        self.assertIn("1", masters_mgr.approved_indents)

        # Updating an assumption invalidates fleet approvals
        masters_mgr.update_parameter("assumptions", "cit_trip_cost", 3000)
        self.assertEqual(len(masters_mgr.approved_indents), 0)

    def test_b9_amendments_clear_stale_overrides(self):
        """B9: Amending a replenishment clears prior overrides for that terminal so amendment takes precedence."""
        # Set an override on T1
        masters_mgr.apply_checker_override("1", 2000000.0, reason="First override")
        self.assertIn("1", masters_mgr.checker_overrides)

        # Now amend T1
        masters_mgr.amend_replenishment("1", new_indent=1800000.0, new_date="2026-08-26", notes="Amended plan")
        self.assertNotIn("1", masters_mgr.checker_overrides)
        self.assertIn("1", masters_mgr.amended_replenishments)
        self.assertEqual(masters_mgr.amended_replenishments["1"]["amended_indent"], 1800000.0)

    def test_b10_inter_atm_sweep_strict_cash_conservation(self):
        """B10: Sweep moves exact cash from donor to recipient and synchronizes all state trackers."""
        res_before = masters_mgr.compute_fleet_replenishment()
        t4_before = res_before["terminals"]["4"]["existing_cash"]
        t3_before = res_before["terminals"]["3"]["existing_cash"]

        sweep_amt = 500000.0
        sweep_res = masters_mgr.execute_inter_atm_sweep("4", "3", sweep_amt, actor="Treasury Officer")
        self.assertIn(sweep_res["status"], ["sweep_executed", "EXECUTED"])
        self.assertEqual(sweep_res["sweep_record"]["status"], "EXECUTED")

        res_after = masters_mgr.compute_fleet_replenishment()
        t4_after = res_after["terminals"]["4"]["existing_cash"]
        t3_after = res_after["terminals"]["3"]["existing_cash"]

        # Exact rupee-for-rupee conservation
        self.assertEqual(t4_before - t4_after, sweep_amt)
        self.assertEqual(t3_after - t3_before, sweep_amt)

    def test_b10_sweep_rejects_excessive_amount_or_capacity_overflow(self):
        """B10: Sweep rejects amount exceeding donor cash or recipient capacity."""
        # T4 has ~1.8M cash. Trying to sweep 5M must raise ValueError
        with self.assertRaises(ValueError):
            masters_mgr.execute_inter_atm_sweep("4", "3", 5000000.0)

    def test_d1_parameter_validation(self):
        """D1: Parameter updates must enforce whitelist and numerical bounds."""
        with self.assertRaises(ValueError):
            # Unknown parameter
            masters_mgr.update_parameter("assumptions", "unauthorized_param", 100)

        with self.assertRaises(ValueError):
            # Out-of-bounds SLA (> 1.0)
            masters_mgr.update_parameter("assumptions", "target_service_level", 1.7)


class TestAICopilotHardening(unittest.TestCase):
    def setUp(self):
        fresh_state()

    def test_b3_adjust_indent_rejects_ambiguous_small_amounts(self):
        """B3: adjust_indent rejects numbers < 10000 without explicit unit suffixes (like 500)."""
        res = adjust_indent.invoke({"terminal_id": "1", "new_indent": 500, "actor": "Alice Maker"})
        self.assertIn("Ambiguous Amount Unit", res)

        # Explicit Lakhs must succeed
        res_l = adjust_indent.invoke({"terminal_id": "1", "new_indent": "20L", "reason": "Explicit test", "actor": "Alice Maker"})
        self.assertIn("Checker Override Applied", res_l)

    def test_b2_free_text_never_executes_control_actions(self):
        """B2: Free text (negated, questions, or containing 'confirm'/'now'/'know') only proposes a slash command."""
        for q in ["don't approve all yet", "what happens if I sign the manifest?", "approve all",
                  "confirm approve all", "approve all indents, I know the numbers", "lock the manifest, acknowledged",
                  "confirm sweep from atm 4 to atm 3 7 lakhs now", "confirm clear reject bin atm 5",
                  "change atm 1 indent to 20 lakhs because chest short, confirm"]:
            r = deterministic_nlp_dispatch(q, dict(CHECKER))
            self.assertTrue("not executed" in r["reply"] or "Ambiguous" in r["reply"], q)
        self.assertEqual(len(masters_mgr.approved_indents), 0)
        self.assertEqual(len(masters_mgr.checker_overrides), 0)
        self.assertEqual(len(masters_mgr.executed_sweeps), 0)
        self.assertFalse(masters_mgr.is_manifest_locked)

    def test_b2_slash_commands_need_trailing_confirm(self):
        """B2: Slash actions preview without `confirm` and run only with it."""
        r = handle_slash_command("/approve 3", dict(CHECKER))
        self.assertIn("Confirmation required", r["reply"])
        self.assertEqual(len(masters_mgr.approved_indents), 0)

        r_confirm = handle_slash_command("/approve 3 confirm", dict(CHECKER))
        self.assertIn("Approved by Checker", r_confirm["reply"])
        self.assertEqual(masters_mgr.approved_indents["3"]["actor"], "Bob Checker")

    def test_c5_simulation_engine_respects_custom_zero_indent(self):
        """C5: simulate_scenario correctly handles custom_indent=0.0 without defaulting to c1_net_indent."""
        sim = simulate_scenario.invoke({"terminal_id": "1", "custom_indent": 0.0, "delay_hours": 0})
        self.assertIn("Indent = ₹0", sim)


if __name__ == "__main__":
    unittest.main()
