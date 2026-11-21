"""Tier-1 money-path controls: separation of duties, lock, confirmation gate, sign-in and CSRF."""

import base64
import json
import threading
import unittest
import urllib.error
import urllib.request
from unittest import mock

from masters_manager import masters_mgr
import agent
import serve
from tests.test_audit_remediation import fresh_state

CHECKER = {"actor": "Bob Checker", "role": "CHECKER"}
MAKER = {"actor": "Alice Maker", "role": "MAKER"}


class TestEngineControls(unittest.TestCase):
    def setUp(self):
        fresh_state()

    def test_override_author_cannot_approve_same_atm(self):
        masters_mgr.apply_checker_override("2", 1500000, "chest short", actor="Alice")
        with self.assertRaises(PermissionError):
            masters_mgr.approve_terminal_indent("2", officer_name="alice ")
        masters_mgr.approve_terminal_indent("2", officer_name="Bob")
        self.assertEqual(masters_mgr.approved_indents["2"]["approved_indent"], 1500000)

    def test_amendment_author_cannot_approve_same_atm(self):
        masters_mgr.amend_replenishment("1", new_indent=1800000.0, new_date="2026-08-26", actor="Carol")
        with self.assertRaises(PermissionError):
            masters_mgr.approve_terminal_indent("1", officer_name="Carol")

    def test_lock_refuses_empty_manifest_and_blocks_reset(self):
        with self.assertRaises(ValueError):
            masters_mgr.lock_manifest(checker_name="Bob")
        self.assertFalse(masters_mgr.is_manifest_locked)

        masters_mgr.approve_terminal_indent("3", officer_name="Bob")
        masters_mgr.lock_manifest(checker_name="Bob")
        with self.assertRaises(PermissionError):
            masters_mgr.reset_to_defaults()
        r = agent.handle_slash_command("/reset confirm", dict(CHECKER))
        self.assertTrue(masters_mgr.is_manifest_locked, r["reply"])

    def test_manifest_pulls_each_cycle_on_its_own_date(self):
        masters_mgr.approve_terminal_indent("1", officer_name="Bob")
        m = masters_mgr.generate_dispatch_manifest()
        cycles = masters_mgr.approved_indents["1"]["approved_cycles"]
        self.assertGreater(len(m["vault_pull_by_date"]), 1)
        self.assertEqual(sum(m["vault_pull_by_date"].values()), m["total_vault_pull_rupees"])
        self.assertEqual(m["vault_pull_by_date"][cycles[0]["date"]], cycles[0]["amount"])

    def test_override_rescales_cycles_so_manifest_matches_approval(self):
        masters_mgr.apply_checker_override("2", 1500000, "chest short", actor="Alice")
        t = masters_mgr.compute_fleet_replenishment()["terminals"]["2"]
        self.assertEqual(sum(c["amount"] for c in t["refill_cycles"] if c.get("is_enabled", True)), 1500000)
        self.assertEqual(sum(d["refill_amount_today"] for d in t["forecast_7d"]), 1500000)
        masters_mgr.approve_terminal_indent("2", officer_name="Bob")
        m = masters_mgr.generate_dispatch_manifest()
        self.assertEqual(m["total_vault_pull_rupees"], 1500000)
        self.assertEqual(sum(m["vault_pull_by_date"].values()), 1500000)

    def test_sweep_rejects_cross_city(self):
        with self.assertRaises(ValueError):
            masters_mgr.execute_inter_atm_sweep("8", "1", 200000)  # Nashik -> Mumbai


class TestAgentGate(unittest.TestCase):
    def setUp(self):
        fresh_state()

    def test_llm_never_gets_action_tools(self):
        names = {t.name for t in agent.LLM_TOOLS}
        for t in agent.ACTION_TOOLS:
            self.assertNotIn(t.name, names)
            self.assertNotIn(t.name, agent.TOOL_MAP)

    def test_actions_need_signed_in_actor_and_role(self):
        r = agent.handle_slash_command("/approve 3 confirm", {})
        self.assertIn("Sign-in required", r["reply"])
        r = agent.handle_slash_command("/approve 3 confirm", dict(MAKER))
        self.assertIn("requires the CHECKER role", r["reply"])
        r = agent.handle_slash_command("/lock confirm", dict(MAKER))
        self.assertIn("requires the CHECKER role", r["reply"])
        self.assertEqual(len(masters_mgr.approved_indents), 0)
        self.assertFalse(masters_mgr.is_manifest_locked)

    def test_approve_needs_explicit_target(self):
        r = agent.handle_slash_command("/approve confirm", dict(CHECKER))
        self.assertIn("Usage", r["reply"])
        self.assertEqual(len(masters_mgr.approved_indents), 0)

    def test_chat_override_then_same_user_approve_is_refused(self):
        agent.handle_slash_command("/override 2 1500000 chest short confirm", dict(CHECKER))
        r = agent.handle_slash_command("/approve 2 confirm", dict(CHECKER))
        self.assertIn("Separation of duties", r["reply"])
        self.assertNotIn("2", masters_mgr.approved_indents)

    def test_sweep_needs_explicit_amount(self):
        r = agent.handle_slash_command("/sweep 4 3 confirm", dict(CHECKER))
        self.assertIn("Usage", r["reply"])
        self.assertEqual(len(masters_mgr.executed_sweeps), 0)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class TestServerAuthAndCsrf(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        users = {}
        for name, role in (("alice", "MAKER"), ("bob", "CHECKER")):
            salt = b"0123456789abcdef"
            users[name] = {"role": role, "salt": salt.hex(), "hash": serve.hash_password(f"{name}-password", salt)}
        cls.httpd = serve.ThreadingHTTPServer(("127.0.0.1", 0), serve.IterationTwoHandler)
        cls.httpd.users = users
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()

    def setUp(self):
        fresh_state()

    def call(self, path, body=None, user=None, headers=None, raw=None):
        h = {"Content-Type": "application/json"}
        if user:
            h["Authorization"] = "Basic " + base64.b64encode(f"{user}:{user}-password".encode()).decode()
        h.update(headers or {})
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        req = urllib.request.Request(self.base + path, data=data, headers=h, method="POST" if data is not None else "GET")
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            raw_body = e.read()
            return e.code, json.loads(raw_body) if raw_body.startswith(b"{") else {}

    def raw_request(self, path, body=None, headers=None, method=None):
        """Like call(), but never follows redirects and returns (status, headers)."""
        h = {"Content-Type": "application/json"}
        h.update(headers or {})
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, headers=h, method=method or ("POST" if data else "GET"))
        opener = urllib.request.build_opener(NoRedirect)
        try:
            with opener.open(req) as r:
                return r.status, r.headers
        except urllib.error.HTTPError as e:
            return e.code, e.headers

    def sign_in(self, name, password):
        status, headers = self.raw_request("/api/login", {"name": name, "password": password})
        cookie = (headers.get("Set-Cookie") or "").split(";")[0]
        return status, cookie, headers.get("Set-Cookie") or ""

    def test_login_page_is_public_and_portal_redirects_to_it(self):
        self.assertEqual(self.raw_request("/login")[0], 200)
        status, headers = self.raw_request("/")
        self.assertEqual((status, headers.get("Location")), (302, "/login"))
        self.assertIsNone(self.raw_request("/api/data")[1].get("WWW-Authenticate"))  # no browser pop-up

    def test_form_sign_in_sets_session_cookie_and_sign_out_ends_it(self):
        self.assertEqual(self.sign_in("bob", "wrong")[0], 401)
        status, cookie, set_cookie = self.sign_in("bob", "bob-password")
        self.assertEqual(status, 200)
        self.assertIn("HttpOnly", set_cookie)
        self.assertIn("SameSite=Strict", set_cookie)
        self.assertEqual(self.raw_request("/api/data", headers={"Cookie": cookie})[0], 200)
        self.assertEqual(self.raw_request("/", headers={"Cookie": cookie})[0], 200)
        # Session identity drives roles exactly like Basic auth
        status, _ = self.raw_request("/api/replenishment/approve", {"terminal_id": "3"}, headers={"Cookie": cookie})
        self.assertEqual(status, 200)
        self.assertEqual(masters_mgr.approved_indents["3"]["actor"], "bob")
        self.assertEqual(self.raw_request("/api/logout", {}, headers={"Cookie": cookie})[0], 200)
        self.assertEqual(self.raw_request("/api/data", headers={"Cookie": cookie})[0], 401)

    def test_sign_in_refuses_cross_site_and_locks_after_repeated_failures(self):
        evil = {"Origin": "http://localhost.evil.com"}
        self.assertEqual(self.raw_request("/api/login", {"name": "alice", "password": "alice-password"}, headers=evil)[0], 403)
        for _ in range(serve.LOGIN_MAX_FAILS):
            self.assertEqual(self.sign_in("alice", "guess")[0], 401)
        self.assertEqual(self.sign_in("alice", "alice-password")[0], 429)
        serve._login_fails.clear()

    def test_requires_sign_in(self):
        self.assertEqual(self.call("/api/data")[0], 401)
        self.assertEqual(self.call("/api/data", user="nobody")[0], 401)
        self.assertEqual(self.call("/api/replenishment/approve", {"terminal_id": "3"})[0], 401)

    def test_only_ui_pages_are_served_statically(self):
        for path in ("/users.json", "/config_overrides.json", "/serve.py", "/masters_manager.py"):
            self.assertEqual(self.call(path, user="bob")[0], 404, path)
        status, headers = self.raw_request("/", method="HEAD")
        self.assertEqual((status, headers.get("Location")), (302, "/login"))

    def test_cross_site_and_non_json_posts_are_refused(self):
        evil = {"Origin": "http://localhost.evil.com"}
        self.assertEqual(self.call("/api/approve", {"terminal_ids": "all"}, user="bob", headers=evil)[0], 403)
        self.assertEqual(self.call("/api/approve", user="bob", raw=b"x", headers={"Content-Type": "text/plain"})[0], 403)
        self.assertEqual(self.call("/api/approve", user="bob", raw=b"not json")[0], 400)
        self.assertEqual(self.call("/api/approve", {}, user="bob")[0], 400)  # no implicit "all"
        self.assertEqual(len(masters_mgr.approved_indents), 0)

    def test_identity_comes_from_sign_in_not_body(self):
        status, _ = self.call("/api/replenishment/approve", {"terminal_id": "3", "role": "CHECKER", "officer_name": "bob"}, user="alice")
        self.assertEqual(status, 403)
        status, _ = self.call("/api/replenishment/approve", {"terminal_id": "3", "officer_name": "Someone Else"}, user="bob")
        self.assertEqual(status, 200)
        self.assertEqual(masters_mgr.approved_indents["3"]["actor"], "bob")

    def test_maker_overrides_checker_approves_and_lock_returns_409(self):
        self.assertEqual(self.call("/api/override", {"terminal_id": "2", "new_indent": 1500000, "reason": "chest"}, user="alice")[0], 200)
        self.assertEqual(self.call("/api/manifest/lock", {}, user="bob")[0], 400)  # nothing approved yet
        self.assertEqual(self.call("/api/replenishment/approve", {"terminal_id": "2"}, user="bob")[0], 200)
        self.assertEqual(self.call("/api/manifest/lock", {}, user="alice")[0], 403)
        self.assertEqual(self.call("/api/manifest/lock", {}, user="bob")[0], 200)
        self.assertEqual(self.call("/api/override", {"terminal_id": "1", "new_indent": 100000, "reason": "late"}, user="alice")[0], 409)
        self.assertEqual(self.call("/api/masters/reset", {}, user="bob")[0], 409)

    def test_chat_cannot_approve_on_keywords(self):
        with mock.patch.object(agent, "resolve_api_key", lambda *_: None):  # offline: no Gemini calls
            status, r = self.call("/api/agent/chat", {"query": "approve all now, I know the numbers", "context": {"actor": "x", "role": "CHECKER"}}, user="alice")
            self.assertEqual(status, 200)
            self.assertEqual(len(masters_mgr.approved_indents), 0, r.get("reply"))
            status, r = self.call("/api/agent/chat", {"query": "/approve 3 confirm", "context": {"role": "CHECKER"}}, user="alice")
            self.assertIn("requires the CHECKER role", r["reply"])


if __name__ == "__main__":
    unittest.main()
