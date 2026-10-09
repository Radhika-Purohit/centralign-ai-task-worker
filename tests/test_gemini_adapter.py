"""GeminiLLM tested against a strict fake of the Gemini REST API (no network, no key)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from tests.conftest import FakeUI
from worker.agent import Agent
from worker.browser import Browser
from worker.llm import GeminiLLM, LLMError, make_llm
from worker.tools import Toolset


def walk(o):
    if isinstance(o, dict):
        yield o
        for v in o.values():
            yield from walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from walk(v)


def make_fake(script):
    """script: list of turns; each turn is a list of (name, args) calls (len>1 = parallel calls)."""
    log = {"requests": [], "turn": 0, "last_calls": None}

    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code); self.send_header("content-type", "application/json"); self.end_headers(); self.wfile.write(data)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            log["requests"].append(body)
            if self.headers.get("x-goog-api-key") != "good-key":
                return self._send(400, {"error": {"message": "API key not valid"}})
            if ":generateContent" not in self.path or "/v1beta/models/" not in self.path:
                return self._send(404, {"error": {"message": "bad path"}})
            # schema must stay inside Gemini's OpenAPI subset
            tools = body["tools"][0]["functionDeclarations"]
            for node in walk(tools):
                if "additionalProperties" in node:
                    return self._send(400, {"error": {"message": "Unknown name additionalProperties"}})
                if node.get("type") == "object" and "properties" not in node:
                    return self._send(400, {"error": {"message": "OBJECT type must have properties"}})
            contents = body["contents"]
            if log["last_calls"] is not None:               # Gemini 3 strictness
                model_parts = contents[-2]["parts"]
                sent_calls = [p for p in model_parts if "functionCall" in p]
                if not sent_calls or "thoughtSignature" not in sent_calls[0]:
                    return self._send(400, {"error": {"message": "Function call is missing a thought_signature"}})
                resp_parts = contents[-1]["parts"]
                if [p["functionResponse"]["id"] for p in resp_parts] != [c["functionCall"]["id"] for c in sent_calls]:
                    return self._send(400, {"error": {"message": "function responses do not match function calls"}})
            if log["turn"] >= len(script):
                return self._send(500, {"error": {"message": "script exhausted"}})
            parts = []
            for i, (name, args) in enumerate(script[log["turn"]]):
                part = {"functionCall": {"name": name, "args": args, "id": f"fc{log['turn']}_{i}"}}
                if i == 0:
                    part["thoughtSignature"] = f"SIG{log['turn']}"       # first call carries the signature
                parts.append(part)
            log["last_calls"] = parts
            log["turn"] += 1
            self._send(200, {"candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": "STOP"}],
                             "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 4, "thoughtsTokenCount": 2}})

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, log


@pytest.fixture
def gemini_env(monkeypatch):
    servers = []

    def _start(script, key="good-key"):
        srv, log = make_fake(script)
        servers.append(srv)
        monkeypatch.setenv("GEMINI_BASE_URL", f"http://127.0.0.1:{srv.server_port}")
        monkeypatch.setenv("GEMINI_API_KEY", key)
        return log
    yield _start
    for s in servers:
        s.shutdown()


def test_full_task_through_gemini_adapter(env, gemini_env, tmp_path):
    state, base, _ = env
    kv = lambda d: [{"key": k, "value": v} for k, v in d.items()]
    fields = dict(vendor="Acme Industries Pvt Ltd", invoice_number="AI-2026-0517", amount="124500.00", due_date="2026-10-14")
    log = gemini_env([
        [("browser_open", {"url": base + "/erp/bills/new"})],
        [("browser_open", {"url": base + "/"}), ("remember", {"key": "x", "value": "y"})],   # parallel: 2nd must be skipped
        [("browser_open", {"url": base + "/erp/bills/new"})],
        [("browser_submit_form", {"form": 1, "fields": kv(fields)})],                         # map arrives as key/value list
        [("browser_open", {"url": base + "/erp/bills"})],
        [("finish", {"status": "completed", "summary": "entered", "evidence": ["BILL-0003"]})],
    ])
    ui = FakeUI()
    agent = Agent(GeminiLLM(), Toolset(Browser([base], backoff=0.01), workspace=str(tmp_path / "ws")), ui,
                  context=base, trace_root=str(tmp_path / "runs"))
    res = agent.run("enter the latest Acme invoice")
    assert res.status == "completed" and res.steps == 6
    assert state.bills[-1]["number"] == "AI-2026-0517" and state.bills[-1]["amount"] == "124500.00"
    assert len(ui.approvals) == 1
    assert agent.tools.memory == {}            # the parallel 'remember' call was skipped, not executed
    assert res.tokens["out"] == 6 * 6          # (4 + 2 thinking tokens) per turn are counted


def test_schema_translation_has_no_unsupported_constructs():
    from worker.tools import SPECS
    decls = GeminiLLM()._declarations(SPECS)
    assert not any("additionalProperties" in n for n in walk(decls))
    assert all("properties" in n for n in walk(decls) if n.get("type") == "object")
    fields = next(d for d in decls if d["name"] == "browser_submit_form")["parameters"]["properties"]["fields"]
    assert fields["type"] == "array"


def test_bad_key_gives_actionable_error(gemini_env):
    gemini_env([], key="bad")
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        GeminiLLM().complete("s", [{"role": "user", "content": "hi"}], [])


def test_rate_limit_waits_for_server_retry_delay(monkeypatch):
    import worker.llm as llm
    sleeps, calls = [], {"n": 0}

    class R:
        def __init__(self, code, text, js=None): self.status_code, self.text, self._js = code, text, js
        def json(self): return self._js

    ok = {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]}
    def post(*a, **k):
        calls["n"] += 1
        return R(429, '{"error":{"details":[{"retryDelay":"21s"}]}}') if calls["n"] == 1 else R(200, "", ok)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    g = GeminiLLM(); g.http.post = post
    monkeypatch.setattr(llm.time, "sleep", lambda s: sleeps.append(s))
    assert g.complete("s", [{"role": "user", "content": "x"}], []).content[0]["text"] == "hi"
    assert sleeps == [22.0]


def test_factory_picks_provider_from_available_key(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "LLM_PROVIDER"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(SystemExit):
        make_llm()
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert isinstance(make_llm(), GeminiLLM)
