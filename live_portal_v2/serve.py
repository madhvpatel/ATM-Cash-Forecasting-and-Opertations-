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

        if parsed.path == "/api/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "service": "Cash Intelligence Portal v2", "version": "2.0.0"}).encode("utf-8"))
            return

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

        if parsed.path == "/api/reports/daily-action-plan":
            qs = parse_qs(parsed.query)
            fmt = qs.get("format", ["csv"])[0].lower()
            if fmt == "json":
                rep_data = masters_mgr.generate_daily_cash_action_report_data()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps(rep_data).encode("utf-8"))
            else:
                csv_bytes = masters_mgr.generate_daily_cash_action_csv().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", 'attachment; filename="Daily_Cash_Action_Report.csv"')
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(csv_bytes)
            return

        if parsed.path == "/api/reports/workbook":
            try:
                wb_bytes = masters_mgr.generate_daily_cash_action_workbook()
                self.send_response(200)
                self.send_header("Content-Type", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                self.send_header("Content-Disposition", 'attachment; filename="Daily_Cash_Action_Report_Workbook.xlsx"')
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(wb_bytes)
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        if parsed.path == "/api/reports/predictions":
            csv_bytes = masters_mgr.generate_predictions_csv().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", 'attachment; filename="Historical_Predictions_Accuracy_Log.csv"')
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(csv_bytes)
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

        if parsed.path == "/api/replenishment/amend":
            tid = str(body.get("terminal_id", "1"))
            new_date = str(body.get("amended_date", "2026-08-26"))
            new_window = str(body.get("amended_window", "07:00 AM"))
            new_indent = float(body["amended_indent"]) if "amended_indent" in body and body["amended_indent"] is not None else None
            notes = str(body.get("notes", "Rescheduled via Portal Roster"))
            actor = str(body.get("actor", "Ops Dispatch Planner"))
            refill_cycles = body.get("refill_cycles")
            strategy = body.get("strategy")
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
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))
            return

        if parsed.path == "/api/replenishment/approve":
            tid = str(body.get("terminal_id", "1"))
            officer = str(body.get("officer_name", "Treasury Checker Officer"))
            notes = str(body.get("notes", "Authorized for CIT Dispatch"))
            res = masters_mgr.approve_terminal_indent(terminal_id=tid, officer_name=officer, notes=notes)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))
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

        if parsed.path == "/api/actions/sweep":
            donor_id = str(body.get("donor_id", "")).strip()
            recipient_id = str(body.get("recipient_id", "")).strip()
            amount = float(body.get("amount", 200000.0))
            actor = str(body.get("actor", "Chief Treasury Officer"))
            try:
                res = masters_mgr.execute_inter_atm_sweep(donor_id=donor_id, recipient_id=recipient_id, amount=amount, actor=actor)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps(res).encode("utf-8"))
            except Exception as e:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        if parsed.path == "/api/actions/clear-reject-bin":
            tid = str(body.get("terminal_id", "1")).strip()
            actor = str(body.get("actor", "Field Custodian Technician"))
            res = masters_mgr.execute_clear_reject_bin(terminal_id=tid, actor=actor)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))
            return

        if parsed.path == "/api/actions/emergency-indent":
            tid = str(body.get("terminal_id", "1")).strip()
            amt = float(body["amount"]) if "amount" in body and body["amount"] is not None else None
            actor = str(body.get("actor", "Emergency Hot-Dispatch Planner"))
            res = masters_mgr.execute_emergency_indent(terminal_id=tid, amount=amt, actor=actor)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))
            return

        self.send_error(404, "Endpoint not found")


def run_server(port: int = 8089):
    server_address = ("0.0.0.0", port)
    httpd = ThreadingHTTPServer(server_address, IterationTwoHandler)
    print(f"🚀 Cash Intelligence Iteration Two serving on http://localhost:{port}/")
    print(f"   Architecture: Glanceable 30s HUD + 3 Tabs + Masters & Config + LangGraph AI Agent (Multi-Threaded)")
    httpd.serve_forever()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8089
    run_server(port)
