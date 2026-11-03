"""Live Server for Cash Intelligence Iteration Two Portal.

Serves the simplified, glanceable 3-tab portal on port 8089 and provides:
- GET / : Iteration Two UI (Glanceable 30s HUD, 3-stage pipeline, no text clutter)
- GET /api/data : Live recalculated fleet data bundle
- GET /api/masters : Complete masters, assumptions, holidays, and exogenous events catalog
- POST /api/masters/update : Live in-memory master update with instant recalculation
- POST /api/masters/save : Persist changes to config_overrides.json
- POST /api/masters/reset : Reset to default baseline
- POST /api/agent/chat : LangGraph agent conversational & action endpoint
- POST /api/approve : Indent approval workflow
- POST /api/simulate : Live What-If sensitivity trajectory calculation
"""

from __future__ import annotations

import os
import sys
import json
import base64
import getpass
import hashlib
import hmac
import secrets
from pathlib import Path
from http.server import SimpleHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from typing import Any, Dict, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from masters_manager import masters_mgr
from agent import run_agent, ALL_TOOLS

# Local user store: {"name": {"role": "MAKER"|"CHECKER", "salt": hex, "hash": hex}}. Never commit it.
USERS_FILE = Path(os.environ.get("CMS_USERS_FILE") or HERE / "users.json")
PBKDF2_ROUNDS = 200_000
STATIC_PAGES = ("/cash_intelligence_ui_ux_mockups.html",)
ROLES = ("MAKER", "CHECKER")
MAX_BODY_BYTES = 1_000_000


def hash_password(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ROUNDS).hex()


def load_users() -> Dict[str, Dict[str, str]]:
    if not USERS_FILE.exists():
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)


def add_user(name: str, role: str, password: str) -> None:
    role = role.upper()
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    salt = secrets.token_bytes(16)
    users = load_users()
    users[name] = {"role": role, "salt": salt.hex(), "hash": hash_password(password, salt)}
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)
    os.chmod(USERS_FILE, 0o600)


def check_basic_auth(header: str, users: Dict[str, Dict[str, str]]) -> Optional[Dict[str, str]]:
    """Returns {"name", "role"} for valid HTTP Basic credentials, else None."""
    if not header.startswith("Basic "):
        return None
    try:
        name, _, password = base64.b64decode(header[6:]).decode("utf-8").partition(":")
    except Exception:
        return None
    rec = users.get(name)
    if not rec:
        return None
    if not hmac.compare_digest(hash_password(password, bytes.fromhex(rec["salt"])), rec["hash"]):
        return None
    return {"name": name, "role": rec["role"]}


class IterationTwoHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(HERE), **kwargs)

    def _send_json(self, status: int, obj: Dict[str, Any]) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(obj).encode("utf-8"))

    def _authenticate(self) -> bool:
        """HTTP Basic auth against the local user store; the browser shows its own sign-in box."""
        self.user = check_basic_auth(self.headers.get("Authorization", ""), self.server.users)
        if self.user:
            return True
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="CMS2 Cash Portal", charset="UTF-8"')
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"error": "Sign-in required"}')
        return False

    def _is_same_origin_json(self) -> bool:
        """Blocks cross-site request forgery: browsers cannot send a cross-site JSON POST without
        a CORS preflight (which this server never grants), and they always send Origin on one."""
        if not self.headers.get("Content-Type", "").lower().startswith("application/json"):
            return False
        origin = self.headers.get("Origin")
        port = self.server.server_address[1]
        return origin is None or origin in (f"http://127.0.0.1:{port}", f"http://localhost:{port}")

    def _require_checker_role(self, body: Dict[str, Any]) -> bool:
        """Maker/Checker separation: only signed-in CHECKER users may approve, lock or reset."""
        if self.user["role"] == "CHECKER":
            return True
        self._send_json(403, {"error": f"Forbidden: '{self.user['name']}' is a {self.user['role']}. Checker authorization required."})
        return False

    def _check_manifest_unlocked(self) -> bool:
        """Rejects writes when manifest is locked (Finding B8)."""
        if masters_mgr.is_manifest_locked:
            self.send_response(409)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "error": "Conflict: CIT Dispatch Manifest is digitally locked. All modifications are rejected."
            }).encode("utf-8"))
            return False
        return True

    def do_GET(self):
        if not self._authenticate():
            return
        parsed = urlparse(self.path)

        if parsed.path in ["/", "/index.html"]:
            self.path = "/index.html"
            return super().do_GET()

        if parsed.path == "/api/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "service": "Cash Intelligence Portal v2", "version": "2.0.0"}).encode("utf-8"))
            return

        if parsed.path == "/api/data":
            data = masters_mgr.compute_fleet_replenishment()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))
            return

        if parsed.path == "/api/masters":
            masters = masters_mgr.get_all_masters()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(masters).encode("utf-8"))
            return

        if parsed.path == "/api/manifest":
            manifest = masters_mgr.generate_dispatch_manifest()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(manifest).encode("utf-8"))
            return

        if parsed.path == "/api/reports/daily-action-plan":
            qs = parse_qs(parsed.query)
            fmt = qs.get("format", ["csv"])[0].lower()
            if fmt == "json":
                rep_data = masters_mgr.generate_daily_cash_action_report_data()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(rep_data).encode("utf-8"))
            else:
                csv_bytes = masters_mgr.generate_daily_cash_action_csv().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", 'attachment; filename="Daily_Cash_Action_Report.csv"')
                self.end_headers()
                self.wfile.write(csv_bytes)
            return

        if parsed.path == "/api/reports/workbook":
            try:
                wb_bytes = masters_mgr.generate_daily_cash_action_workbook()
                self.send_response(200)
                self.send_header("Content-Type", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                self.send_header("Content-Disposition", 'attachment; filename="Daily_Cash_Action_Report_Workbook.xlsx"')
                self.end_headers()
                self.wfile.write(wb_bytes)
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        if parsed.path == "/api/reports/predictions":
            csv_bytes = masters_mgr.generate_predictions_csv().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", 'attachment; filename="Historical_Predictions_Accuracy_Log.csv"')
            self.end_headers()
            self.wfile.write(csv_bytes)
            return

        # Only the pages the UI links to; never users.json, config_overrides.json or source files
        if parsed.path in STATIC_PAGES:
            return super().do_GET()
        self.send_error(404, "File not found")

    def do_HEAD(self):
        if not self._authenticate():
            return
        if urlparse(self.path).path in ("/", "/index.html") + STATIC_PAGES:
            return super().do_HEAD()
        self.send_error(404, "File not found")

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        if content_len > MAX_BODY_BYTES:
            self.close_connection = True
            self._send_json(413, {"error": "Request body too large"})
            return
        # Read the body before any early reply: closing with unread data resets the connection
        raw = self.rfile.read(content_len) if content_len > 0 else b"{}"
        if not self._authenticate():
            return
        if not self._is_same_origin_json():
            self._send_json(403, {"error": "Forbidden: POST must be same-origin with Content-Type application/json"})
            return
        try:
            body = json.loads(raw.decode("utf-8"))
        except ValueError:
            body = None
        if not isinstance(body, dict):
            self._send_json(400, {"error": "Request body must be a JSON object"})
            return
        # Identity always comes from the sign-in, never from the request body
        body.update(officer_name=self.user["name"], actor=self.user["name"], role=self.user["role"])
        try:
            self._handle_post(urlparse(self.path), body)
        except PermissionError as e:
            self._send_json(409, {"error": str(e)})
        except ValueError as e:
            self._send_json(400, {"error": str(e)})

    def _handle_post(self, parsed, body: Dict[str, Any]) -> None:
        if parsed.path == "/api/masters/update":
            if not self._check_manifest_unlocked(): return
            category = body.get("category", "assumptions")
            key = body.get("key", "")
            value = body.get("value")
            sub_id = body.get("sub_id")
            
            try:
                result = masters_mgr.update_parameter(category=category, key=key, value=value, sub_id=sub_id)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode("utf-8"))
            except (ValueError, PermissionError) as e:
                self.send_response(409 if isinstance(e, PermissionError) else 400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        if parsed.path == "/api/masters/save":
            if not self._check_manifest_unlocked(): return
            result = masters_mgr.save_overrides()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))
            return

        if parsed.path == "/api/masters/reset":
            if not self._require_checker_role(body): return
            if not self._check_manifest_unlocked(): return
            result = masters_mgr.reset_to_defaults()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))
            return

        if parsed.path in ["/api/agent/chat", "/api/agent/command"]:
            query = (body.get("query") or body.get("command") or "").strip()
            context = body.get("context") if isinstance(body.get("context"), dict) else {}
            context.update(actor=self.user["name"], role=self.user["role"])
            if not context.get("active_terminal") and body.get("active_terminal"):
                context["active_terminal"] = body.get("active_terminal")
            history = body.get("history", [])

            client_key = body.get("api_key")

            # Execute LangGraph agent
            reply_data = run_agent(query=query, context=context, history=history, client_key=client_key)
            if not isinstance(reply_data, dict):
                reply_data = {"reply": str(reply_data), "status": "ok"}
            if "status" not in reply_data:
                reply_data["status"] = "ok"
            if "response" not in reply_data and "reply" in reply_data:
                reply_data["response"] = reply_data["reply"]

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(reply_data).encode("utf-8"))
            return

        if parsed.path == "/api/approve":
            if not self._require_checker_role(body): return
            if not self._check_manifest_unlocked(): return
            terminal_ids = str(body.get("terminal_ids") or "").strip()
            if not terminal_ids:
                self._send_json(400, {"error": "terminal_ids is required (an ATM id, a list of ids, or 'all')"})
                return
            notes = body.get("notes", "Approved via Portal UI")
            from agent import approve_indents
            msg = approve_indents.invoke({"terminal_ids": terminal_ids, "actor": self.user["name"], "approver_notes": notes})
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success", "message": msg, "data": masters_mgr.compute_fleet_replenishment()}).encode("utf-8"))
            return

        if parsed.path == "/api/simulate":
            tid = str(body.get("terminal_id", "1"))
            custom_indent = float(body.get("custom_indent", 0.0))
            delay_hours = int(body.get("delay_hours", 0))
            from agent import simulate_scenario
            msg = simulate_scenario.invoke({"terminal_id": tid, "custom_indent": custom_indent, "delay_hours": delay_hours})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success", "message": msg}).encode("utf-8"))
            return

        if parsed.path == "/api/override":
            if not self._check_manifest_unlocked(): return
            tid = str(body.get("terminal_id", "1"))
            new_indent = float(body.get("new_indent", body.get("override_indent", 0.0)))
            reason = str(body.get("reason", body.get("reason_code", "Checker override via portal")))
            officer = str(body.get("officer_name", "Checker Officer"))
            try:
                res = masters_mgr.apply_checker_override(terminal_id=tid, new_indent=new_indent, reason=reason, actor=officer)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(res).encode("utf-8"))
            except (ValueError, PermissionError) as e:
                self.send_response(409 if isinstance(e, PermissionError) else 400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        if parsed.path == "/api/replenishment/amend":
            if not self._check_manifest_unlocked(): return
            tid = str(body.get("terminal_id", "1"))
            new_date = str(body.get("amended_date", "2026-08-26"))
            new_window = str(body.get("amended_window", "07:00 AM"))
            new_indent = float(body["amended_indent"]) if "amended_indent" in body and body["amended_indent"] is not None else None
            notes = str(body.get("notes", "Rescheduled via Portal Roster"))
            actor = str(body.get("actor", "Ops Dispatch Planner"))
            refill_cycles = body.get("refill_cycles")
            strategy = body.get("strategy")
            try:
                res = masters_mgr.amend_replenishment(
                    terminal_id=tid,
                    new_date=new_date,
                    new_window=new_window,
                    new_indent=new_indent,
                    notes=notes,
                    actor=actor,
                    refill_cycles=refill_cycles,
                    strategy=strategy
                )
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(res).encode("utf-8"))
            except (ValueError, PermissionError) as e:
                self.send_response(409 if isinstance(e, PermissionError) else 400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        if parsed.path == "/api/replenishment/approve":
            if not self._require_checker_role(body): return
            if not self._check_manifest_unlocked(): return
            tid = str(body.get("terminal_id", "1"))
            officer = str(body.get("officer_name", "Treasury Checker Officer"))
            notes = str(body.get("notes", "Authorized for CIT Dispatch"))
            try:
                res = masters_mgr.approve_terminal_indent(terminal_id=tid, officer_name=officer, notes=notes)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(res).encode("utf-8"))
            except (ValueError, PermissionError) as e:
                self.send_response(409 if isinstance(e, PermissionError) else 400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        if parsed.path == "/api/feedback":
            tid = str(body.get("terminal_id", "1"))
            fb_type = str(body.get("feedback", "thumbs_up"))
            reason = str(body.get("reason", ""))
            comment = str(body.get("comment", ""))
            actor = str(body.get("actor", "CMS Ops Operator"))
            res = masters_mgr.record_feedback(terminal_id=tid, feedback_type=fb_type, reason=reason, comment=comment, actor=actor)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))
            return

        if parsed.path == "/api/manifest/lock":
            if not self._require_checker_role(body): return
            officer = str(body.get("officer_name", "Chief Treasury Officer"))
            notes = str(body.get("notes", "Authorized for CIT Dispatch"))
            res = masters_mgr.lock_manifest(checker_name=officer, notes=notes)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))
            return

        if parsed.path == "/api/actions/sweep":
            if not self._check_manifest_unlocked(): return
            donor_id = str(body.get("donor_id", "")).strip()
            recipient_id = str(body.get("recipient_id", "")).strip()
            amount = float(body.get("amount", 200000.0))
            actor = str(body.get("actor", "Chief Treasury Officer"))
            try:
                res = masters_mgr.execute_inter_atm_sweep(donor_id=donor_id, recipient_id=recipient_id, amount=amount, actor=actor)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(res).encode("utf-8"))
            except (ValueError, PermissionError) as e:
                self.send_response(409 if isinstance(e, PermissionError) else 400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        if parsed.path == "/api/actions/clear-reject-bin":
            if not self._check_manifest_unlocked(): return
            tid = str(body.get("terminal_id", "1")).strip()
            actor = str(body.get("actor", "Field Custodian Technician"))
            res = masters_mgr.execute_clear_reject_bin(terminal_id=tid, actor=actor)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))
            return

        if parsed.path == "/api/actions/emergency-indent":
            if not self._check_manifest_unlocked(): return
            tid = str(body.get("terminal_id", "1")).strip()
            amt = float(body["amount"]) if "amount" in body and body["amount"] is not None else None
            actor = str(body.get("actor", "Emergency Hot-Dispatch Planner"))
            try:
                res = masters_mgr.execute_emergency_indent(terminal_id=tid, amount=amt, actor=actor)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(res).encode("utf-8"))
            except (ValueError, PermissionError) as e:
                self.send_response(409 if isinstance(e, PermissionError) else 400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        self.send_error(404, "Endpoint not found")


def run_server(port: int = 8089):
    # Bind to localhost (127.0.0.1) instead of public 0.0.0.0 (Finding B1)
    server_address = ("127.0.0.1", port)
    httpd = ThreadingHTTPServer(server_address, IterationTwoHandler)
    httpd.users = load_users()
    if not httpd.users:
        sys.exit(f"No portal users in {USERS_FILE}. Add one first: python serve.py add-user <name> <MAKER|CHECKER>")
    if not os.environ.get("CMS_SIGNING_KEY"):
        print("⚠️  CMS_SIGNING_KEY is not set: manifests cannot be locked until it is.")
    print(f"🚀 Cash Intelligence Iteration Two serving securely on http://127.0.0.1:{port}/")
    print(f"   Architecture: Glanceable 30s HUD + 3 Tabs + Masters & Config + LangGraph AI Agent (Multi-Threaded)")
    httpd.serve_forever()


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "add-user":
        pw = getpass.getpass(f"Password for {sys.argv[2]}: ")
        if len(pw) < 10 or pw != getpass.getpass("Repeat password: "):
            sys.exit("Passwords must match and be at least 10 characters.")
        add_user(sys.argv[2], sys.argv[3], pw)
        print(f"Added {sys.argv[2]} ({sys.argv[3].upper()}) to {USERS_FILE}")
        sys.exit(0)
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8089
    run_server(port)
