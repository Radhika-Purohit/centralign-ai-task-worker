"""Generalization/regression evals. The SAME agent code runs different tasks; success is judged
against sandbox ground truth (not the agent's own claim).

    GEMINI_API_KEY=... python -m evals.run_evals [--only id] [--provider gemini|anthropic]
"""
import argparse
import os
import sys
import threading

from werkzeug.serving import make_server

from sandbox.app import State, create_app
from worker.agent import Agent
from worker.browser import Browser
from worker.llm import make_llm
from worker.tools import Toolset


class EvalUI:
    """Auto-approves, answers clarification questions from a script, records interactions."""

    def __init__(self, answers=()):
        self.answers, self.asked, self.approvals = list(answers), [], []

    def thought(self, t): pass
    def action(self, n, a): pass
    def observation(self, t, ok): pass

    def approve(self, d):
        self.approvals.append(d)
        return True

    def ask(self, q):
        self.asked.append(q)
        return self.answers.pop(0) if self.answers else "Please use your best judgment"


def bill(state, number):
    return next((b for b in state.bills if b["number"] == number), None)


TASKS = [
    dict(id="enter_acme_industries", answers=[],
         task="Find the latest invoice from Acme Industries in the vendor portal, extract the amount and due date, "
              "enter it into the ERP as a vendor bill, and tell me once it is done.",
         check=lambda s, ui: (bill(s, "AI-2026-0517") or {}).get("amount") == "124500.00"
         and bill(s, "AI-2026-0517")["due"] == "2026-10-14" and len(s.bills) == 3),
    dict(id="enter_globex", answers=[],
         task="Get Globex's most recent invoice from the portal and book it in the ERP.",
         check=lambda s, ui: (bill(s, "GX-9002") or {}).get("amount") == "175320.75"
         and bill(s, "GX-9002")["vendor"] == "Globex Corporation" and len(s.bills) == 3),
    dict(id="ambiguous_acme_must_ask", answers=["Acme Logistics"],
         task="Enter the latest invoice from Acme into the ERP.",
         check=lambda s, ui: len(ui.asked) >= 1 and (bill(s, "LG-7731") or {}).get("amount") == "31800.50"
         and len(s.bills) == 3),
    dict(id="read_only_dedupe", answers=[],
         task="Is Globex's invoice GX-9001 already entered in the ERP? Do NOT change anything, just report.",
         check=lambda s, ui: len(s.bills) == 2 and not ui.approvals),
    dict(id="already_exists_no_duplicate", answers=[],
         task="Make sure Acme Industries' invoice AI-2026-0412 is in the ERP. Add it only if it is missing.",
         check=lambda s, ui: len(s.bills) == 2),
]


def run_one(t, model, provider):
    state = State()
    srv = make_server("127.0.0.1", 0, create_app(state), threaded=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    ui = EvalUI(t["answers"])
    agent = Agent(make_llm(provider, model), Toolset(Browser([base]), workspace="workspace"), ui,
                  context=f"Company tools at {base} (home page links to vendor portal and ERP).",
                  max_steps=30, auto_approve=True, trace_root="runs/evals")
    res = agent.run(t["task"])
    srv.shutdown()
    ok = bool(t["check"](state, ui))
    return ok, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--model")
    ap.add_argument("--provider", choices=["gemini", "anthropic"])
    a = ap.parse_args()
    rows, passed = [], 0
    for t in TASKS:
        if a.only and t["id"] != a.only:
            continue
        ok, res = run_one(t, a.model, a.provider)
        passed += ok
        rows.append((t["id"], "PASS" if ok else "FAIL", res.status, res.steps, res.tokens["in"] + res.tokens["out"]))
        print(f"{rows[-1][0]:32} {rows[-1][1]}  agent_status={res.status:9} steps={res.steps:3} tokens={rows[-1][4]}", flush=True)
    print(f"\n{passed}/{len(rows)} passed (judged on sandbox ground truth)")


if __name__ == "__main__":
    main()
