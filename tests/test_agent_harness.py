FORM = dict(form=1, fields=dict(vendor="Acme Industries Pvt Ltd", invoice_number="AI-2026-0517",
                                amount="124500.00", due_date="2026-10-14"))


def open_(u): return ("browser_open", {"url": u})


def test_premature_finish_rejected_until_state_is_reread(make_agent, env):
    state, base, _ = env
    script = [
        open_(base + "/erp/bills/new"), ("browser_submit_form", FORM),
        ("finish", dict(status="completed", summary="done", evidence=["Bill BILL-0003 created"])),  # rejected: no re-read
        open_(base + "/erp/bills"),
        ("finish", dict(status="completed", summary="done", evidence=["BILL-0003 | Acme Industries Pvt Ltd | AI-2026-0517 | 124500.00 | 2026-10-14"])),
    ]
    agent, ui = make_agent(script)
    r = agent.run("enter invoice")
    assert r.status == "completed" and r.steps == 5
    assert r.evidence[0][1] is True                       # evidence is grounded in observed pages
    assert len(ui.approvals) == 1 and state.bills[-1]["number"] == "AI-2026-0517"


def test_denied_approval_blocks_the_write(make_agent, env):
    state, base, _ = env
    script = [open_(base + "/erp/bills/new"), ("browser_submit_form", FORM),
              ("finish", dict(status="blocked", summary="human denied the write"))]
    agent, ui = make_agent(script, ui=__import__("tests.conftest", fromlist=["FakeUI"]).FakeUI(approve=False))
    r = agent.run("enter invoice")
    assert r.status == "blocked" and len(state.bills) == 2


def test_failed_write_cannot_be_reported_as_completed(make_agent, env):
    state, base, _ = env
    bad = dict(form=1, fields={**FORM["fields"], "amount": "Rs 1,24,500.00"})   # ERP returns 422
    script = [open_(base + "/erp/bills/new"), ("browser_submit_form", bad),
              ("finish", dict(status="completed", summary="done", evidence=["x"])),   # rejected
              ("finish", dict(status="failed", summary="could not enter the bill"))]
    agent, _ = make_agent(script)
    r = agent.run("enter invoice")
    assert r.status == "failed" and len(state.bills) == 2


def test_ask_user_and_memory_and_ungrounded_evidence(make_agent, env):
    _, base, _ = env
    from tests.conftest import FakeUI
    script = [("ask_user", dict(question="Acme Industries or Acme Logistics?")),
              ("remember", dict(key="vendor", value="Acme Industries")),
              open_(base + "/portal/invoices?q=acme"),
              ("finish", dict(status="completed", summary="read-only", evidence=["a quote that was never on any page"]))]
    ui = FakeUI(answers=["Acme Industries"])
    agent, _ = make_agent(script, ui=ui)
    r = agent.run("what is the latest Acme invoice?")
    assert ui.asked and agent.tools.memory["vendor"] == "Acme Industries"
    assert r.status == "completed" and r.evidence[0][1] is False        # flagged as unverified claim


def test_step_budget(make_agent, env):
    _, base, _ = env
    agent, _ = make_agent([open_(base + "/")] * 3, max_steps=3)
    r = agent.run("loop forever")
    assert r.status == "failed" and "budget" in r.summary


def test_old_observations_are_compacted():
    from worker.agent import Agent
    msgs = [{"role": "user", "content": "goal"}]
    for i in range(10):
        msgs += [{"role": "assistant", "content": []},
                 {"role": "user", "content": [{"type": "tool_result", "tool_use_id": str(i), "content": "x" * 5000}]}]
    out = Agent._compact(msgs)
    assert len(out[2]["content"][0]["content"]) < 500 and len(out[-1]["content"][0]["content"]) == 5000
