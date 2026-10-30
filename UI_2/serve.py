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
from pathlib import Path
from http.server import SimpleHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from masters_manager import masters_mgr
from agent import run_agent, ALL_TOOLS


class IterationTwoHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(HERE), **kwargs)

    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, HEAD")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With")

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path in ["/", "/index.html"]:
            self.path = "/index.html"
            return super().do_GET()

        if parsed.path == "/api/data":
            data = masters_mgr.compute_fleet_replenishment()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))
            return

        if parsed.path == "/api/masters":
            masters = masters_mgr.get_all_masters()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(masters).encode("utf-8"))
            return

        if parsed.path == "/api/manifest":
            manifest = masters_mgr.generate_dispatch_manifest()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(manifest).encode("utf-8"))
            return

        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        content_len = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
        
        try:
            body = json.loads(post_body)
        except Exception:
            body = {}

        if parsed.path == "/api/masters/update":
            category = body.get("category", "assumptions")
            key = body.get("key", "")
            value = body.get("value")
            sub_id = body.get("sub_id")
            
            result = masters_mgr.update_parameter(category=category, key=key, value=value, sub_id=sub_id)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))
            return

        if parsed.path == "/api/masters/save":
            result = masters_mgr.save_overrides()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))
            return

        if parsed.path == "/api/masters/reset":
            result = masters_mgr.reset_to_defaults()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))
            return

        if parsed.path in ["/api/agent/chat", "/api/agent/command"]:
            query = (body.get("query") or body.get("command") or "").strip()
            context = body.get("context", {})
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
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(reply_data).encode("utf-8"))
            return

        if parsed.path == "/api/approve":
            terminal_ids = body.get("terminal_ids", "all")
            notes = body.get("notes", "Approved via Portal UI")
            from agent import approve_indents
            msg = approve_indents.invoke({"terminal_ids": terminal_ids, "approver_notes": notes})
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
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
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success", "message": msg}).encode("utf-8"))
            return

        if parsed.path == "/api/override":
            tid = str(body.get("terminal_id", "1"))
            new_indent = float(body.get("new_indent", 0.0))
            reason = str(body.get("reason", "Checker override via portal"))
            officer = str(body.get("officer_name", "Checker Officer"))
            res = masters_mgr.apply_checker_override(terminal_id=tid, new_indent=new_indent, reason=reason, actor=officer)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))
            return

        if parsed.path == "/api/manifest/lock":
            officer = str(body.get("officer_name", "Chief Treasury Officer"))
            notes = str(body.get("notes", "Authorized for CIT Dispatch"))
            res = masters_mgr.lock_manifest(checker_name=officer, notes=notes)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))
            return

        if parsed.path == "/api/approve/cto":
            officer = str(body.get("officer_name", "Chief Treasury Officer"))
            notes = str(body.get("notes", "Level-2 Sovereign Sign-Off Approved for Armored CIT Dispatch"))
            res = masters_mgr.authorize_cto_signoff(officer_name=officer, notes=notes)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))
            return

        if parsed.path == "/api/reconciliation/resolve":
            tid = str(body.get("terminal_id", "3"))
            officer = str(body.get("officer_name", "Senior Reconciliation Auditor"))
            notes = str(body.get("notes", "Physical count confirmed at chest sorting counter. Reconciled and closed."))
            res = masters_mgr.resolve_reconciliation(terminal_id=tid, officer_name=officer, resolution_notes=notes)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))
            return

        self.send_error(404, "Endpoint not found")


def run_server(port: int = 8092):
    server_address = ("0.0.0.0", port)
    httpd = ThreadingHTTPServer(server_address, IterationTwoHandler)
    print(f"🚀 Cash Intelligence UI_2 (Operating System) serving on http://localhost:{port}/")
    print(f"   Architecture: Operational Activity Driven (Network Ops, Planning, Approvals, Dispatch, Rec, Analytics, Contextual AI)")
    httpd.serve_forever()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8092
    run_server(port)
