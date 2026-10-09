"""Sandbox company environment: a flaky vendor portal + an internal ERP.

Everything here is mock data. It deliberately contains realistic friction so an
agent has something real to recover from:
  * invoices are listed unsorted, so "latest" must be derived from dates
  * the invoice detail page returns HTTP 503 for the first few hits
  * the portal shows dates like "14 Sep 2026" and amounts like "Rs 1,24,500.00";
    the ERP only accepts ISO dates and plain decimals (422 otherwise)
  * ERP vendor names differ from portal names ("Acme Industries" vs "Acme Industries Pvt Ltd")
  * the ERP rejects duplicate invoice numbers (409)
  * "Acme" is ambiguous (two vendors) -> the agent should ask the human
"""
import re
import threading
from datetime import datetime

from flask import Flask, make_response, render_template_string, request

PAGE = """<!doctype html><html><head><title>{{ title }}</title></head><body>
<nav><a href="/">Home</a> | <a href="/portal/invoices">Vendor Portal</a> | <a href="/erp/bills">ERP Bills</a></nav>
<h1>{{ title }}</h1>{{ body|safe }}</body></html>"""

ERP_VENDORS = ["Acme Industries Pvt Ltd", "Acme Logistics", "Globex Corporation", "Initech"]


def inr(x: float) -> str:
    whole, frac = f"{x:.2f}".split(".")
    if len(whole) > 3:
        head, tail, groups = whole[:-3], whole[-3:], []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join(groups + [tail])
    return f"Rs {whole}.{frac}"


def pretty_date(iso: str) -> str:
    return datetime.strptime(iso, "%Y-%m-%d").strftime("%d %b %Y")


class State:
    def __init__(self, flaky_hits: int = 3):
        # deliberately NOT sorted by date
        self.invoices = [
            dict(id="inv-1003", vendor="Globex", number="GX-9001", issued="2026-09-01", due="2026-10-01", amount=250000.00),
            dict(id="inv-1001", vendor="Acme Industries", number="AI-2026-0412", issued="2026-08-03", due="2026-09-02", amount=98250.00),
            dict(id="inv-1005", vendor="Initech", number="IT-554", issued="2026-09-10", due="2026-10-10", amount=18400.00),
            dict(id="inv-1002", vendor="Acme Industries", number="AI-2026-0517", issued="2026-09-14", due="2026-10-14", amount=124500.00),
            dict(id="inv-1004", vendor="Globex", number="GX-9002", issued="2026-09-28", due="2026-10-28", amount=175320.75),
            dict(id="inv-1006", vendor="Acme Logistics", number="LG-7731", issued="2026-09-20", due="2026-10-20", amount=31800.50),
            dict(id="inv-1007", vendor="Acme Industries", number="AI-2026-0388", issued="2026-07-01", due="2026-07-31", amount=45000.00),
        ]
        self.bills = [
            dict(id="BILL-0001", vendor="Globex Corporation", number="GX-9001", amount="250000.00", due="2026-10-01"),
            dict(id="BILL-0002", vendor="Acme Industries Pvt Ltd", number="AI-2026-0412", amount="98250.00", due="2026-09-02"),
        ]
        self.detail_hits = 0
        self.flaky_hits = flaky_hits
        self.lock = threading.Lock()


def create_app(state: State | None = None) -> Flask:
    state = state or State()
    app = Flask(__name__)
    app.state = state

    def page(title, body, status=200):
        return make_response(render_template_string(PAGE, title=title, body=body), status)

    @app.get("/")
    def home():
        return page("Acme Corp Internal Tools", '<p>Tools: <a href="/portal/invoices">Vendor Invoice Portal</a>, '
                    '<a href="/erp/bills">Internal ERP</a></p>')

    # ---------------- vendor portal ----------------
    @app.get("/portal/invoices")
    def portal_list():
        q = request.args.get("q", "").strip().lower()
        rows = [i for i in state.invoices if q in i["vendor"].lower() or q in i["number"].lower()]
        trs = "".join(f"<tr><td>{i['vendor']}</td><td>{i['number']}</td><td>{pretty_date(i['issued'])}</td>"
                      f"<td><a href='/portal/invoices/{i['id']}'>View</a></td></tr>" for i in rows)
        form = ("<form method='get' action='/portal/invoices'>Search vendor or invoice no: "
                f"<input type='text' name='q' value='{q}'><input type='submit' value='Search'></form>")
        table = ("<table><tr><th>Vendor</th><th>Invoice No</th><th>Issued</th><th></th></tr>" + trs + "</table>"
                 if rows else "<p>No invoices found.</p>")
        return page("Vendor Invoice Portal", form + f"<p>{len(rows)} invoice(s)</p>" + table)

    @app.get("/portal/invoices/<inv_id>")
    def portal_detail(inv_id):
        with state.lock:
            state.detail_hits += 1
            hit = state.detail_hits
        if hit <= state.flaky_hits:
            r = page("503 Service Unavailable", "<p>The invoice service is temporarily unavailable. Please retry.</p>", 503)
            r.headers["Retry-After"] = "1"
            return r
        inv = next((i for i in state.invoices if i["id"] == inv_id), None)
        if not inv:
            return page("404 Not Found", "<p>No such invoice.</p>", 404)
        body = (f"<table><tr><th>Vendor</th><td>{inv['vendor']}</td></tr>"
                f"<tr><th>Invoice number</th><td>{inv['number']}</td></tr>"
                f"<tr><th>Issue date</th><td>{pretty_date(inv['issued'])}</td></tr>"
                f"<tr><th>Due date</th><td>{pretty_date(inv['due'])}</td></tr>"
                f"<tr><th>Amount payable</th><td>{inr(inv['amount'])}</td></tr></table>"
                "<p><a href='/portal/invoices'>Back to list</a></p>")
        return page(f"Invoice {inv['number']}", body)

    # ---------------- internal ERP ----------------
    def bills_table():
        trs = "".join(f"<tr><td>{b['id']}</td><td>{b['vendor']}</td><td>{b['number']}</td>"
                      f"<td>{b['amount']}</td><td>{b['due']}</td></tr>" for b in state.bills)
        return ("<table><tr><th>Bill ID</th><th>Vendor</th><th>Invoice No</th><th>Amount (INR)</th><th>Due date</th></tr>"
                + trs + "</table>")

    @app.get("/erp/bills")
    def erp_list():
        return page("ERP - Vendor Bills", f"<p><a href='/erp/bills/new'>Enter new bill</a></p>{bills_table()}")

    def bill_form(values=None, errors=None):
        v = values or {}
        opts = "".join(f"<option value='{o}'>{o}</option>" for o in ERP_VENDORS)
        err = "".join(f"<li>{e}</li>" for e in (errors or []))
        err = f"<p>Errors:</p><ul>{err}</ul>" if err else ""
        return (err + "<form method='post' action='/erp/bills'>"
                f"Vendor: <select name='vendor'>{opts}</select><br>"
                f"Invoice number: <input type='text' name='invoice_number' value='{v.get('invoice_number', '')}'><br>"
                f"Amount (INR, plain decimal e.g. 1234.50): <input type='text' name='amount' value='{v.get('amount', '')}'><br>"
                f"Due date (YYYY-MM-DD): <input type='text' name='due_date' value='{v.get('due_date', '')}'><br>"
                "<input type='submit' value='Create bill'></form>")

    @app.get("/erp/bills/new")
    def erp_new():
        return page("ERP - New Vendor Bill", bill_form())

    @app.post("/erp/bills")
    def erp_create():
        f = {k: request.form.get(k, "").strip() for k in ("vendor", "invoice_number", "amount", "due_date")}
        errors = []
        if f["vendor"] not in ERP_VENDORS:
            errors.append(f"Unknown vendor '{f['vendor']}'. Choose one of: {', '.join(ERP_VENDORS)}")
        if not f["invoice_number"]:
            errors.append("Invoice number is required")
        if not re.fullmatch(r"\d+(\.\d{1,2})?", f["amount"]):
            errors.append("Amount must be a plain decimal like 1234.50 (no currency symbol or commas)")
        try:
            datetime.strptime(f["due_date"], "%Y-%m-%d")
        except ValueError:
            errors.append("Due date must be in YYYY-MM-DD format")
        if errors:
            return page("ERP - New Vendor Bill (validation failed)", bill_form(f, errors), 422)
        with state.lock:
            if any(b["number"] == f["invoice_number"] for b in state.bills):
                return page("ERP - Duplicate", f"<p>Bill for invoice {f['invoice_number']} already exists.</p>", 409)
            bill = dict(id=f"BILL-{len(state.bills) + 1:04d}", vendor=f["vendor"], number=f["invoice_number"],
                        amount=f["amount"], due=f["due_date"])
            state.bills.append(bill)
        return page("ERP - Bill created", f"<p>Bill {bill['id']} created for invoice {bill['number']}.</p>"
                    "<p><a href='/erp/bills'>View all bills</a></p>", 201)

    return app


if __name__ == "__main__":
    create_app().run(port=8765)
