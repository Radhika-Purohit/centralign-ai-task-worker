from sandbox.app import State, create_app
from worker.browser import Browser


def test_detail_is_flaky_then_recovers():
    c = create_app(State(flaky_hits=3)).test_client()
    assert [c.get("/portal/invoices/inv-1002").status_code for _ in range(4)] == [503, 503, 503, 200]


def test_erp_validation_and_duplicate():
    c = create_app().test_client()
    bad = dict(vendor="Acme Industries", invoice_number="N1", amount="Rs 1,00", due_date="14 Sep")
    assert c.post("/erp/bills", data=bad).status_code == 422
    good = dict(vendor="Acme Industries Pvt Ltd", invoice_number="N1", amount="100.00", due_date="2026-10-14")
    assert c.post("/erp/bills", data=good).status_code == 201
    assert c.post("/erp/bills", data=good).status_code == 409


def test_browser_retries_get_but_not_post(env):
    state, base, _ = env
    b = Browser([base], backoff=0.01)
    p = b.open(base + "/portal/invoices/inv-1002")
    assert p.status == 503 and "gave up after 2 attempts" in p.format()      # hits 1-2
    p = b.open(base + "/portal/invoices/inv-1002")
    assert p.status == 200 and "succeeded on attempt 2" in p.format()        # hit 3 fails, hit 4 ok
    assert "Rs 1,24,500.00" in p.text


def test_browser_origin_block_and_form_parsing(env):
    import pytest
    from worker.browser import BrowserError
    _, base, _ = env
    b = Browser([base], backoff=0.01)
    with pytest.raises(BrowserError):
        b.open("http://example.com")
    b.open(base + "/erp/bills/new")
    method, action, data = b.prepare_submit(1, {"invoice_number": "X"})
    assert method == "POST" and data["vendor"] == "Acme Industries Pvt Ltd" and data["invoice_number"] == "X"
    with pytest.raises(BrowserError):
        b.prepare_submit(1, {"nope": "1"})
