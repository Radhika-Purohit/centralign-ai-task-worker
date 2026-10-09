import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from worker.llm import AnthropicLLM, LLMError

SEEN = []


class Fake(BaseHTTPRequestHandler):
    calls = 0

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        SEEN.append((self.path, dict(self.headers), body))
        Fake.calls += 1
        if self.headers.get("x-api-key") == "bad":
            code, out = 401, {"error": "invalid x-api-key"}
        elif self.headers.get("x-api-key") == "flaky" and Fake.calls == 1:
            code, out = 529, {"error": "overloaded"}
        else:
            code, out = 200, {"content": [{"type": "text", "text": "thinking"},
                                          {"type": "tool_use", "id": "tu_1", "name": "browser_open", "input": {"url": "/"}}],
                              "stop_reason": "tool_use", "usage": {"input_tokens": 11, "output_tokens": 5}}
        data = json.dumps(out).encode()
        self.send_response(code); self.send_header("content-type", "application/json"); self.end_headers(); self.wfile.write(data)

    def log_message(self, *a): pass


@pytest.fixture
def fake_api(monkeypatch):
    srv = HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("ANTHROPIC_BASE_URL", f"http://127.0.0.1:{srv.server_port}")
    Fake.calls = 0
    yield
    srv.shutdown()


def test_request_shape_and_parsing(fake_api, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    r = AnthropicLLM().complete("sys", [{"role": "user", "content": "hi"}], [{"name": "t", "input_schema": {}}])
    path, headers, body = SEEN[-1]
    assert path == "/v1/messages" and headers["x-api-key"] == "k" and headers["anthropic-version"] == "2023-06-01"
    assert body["system"] == "sys" and body["tool_choice"]["disable_parallel_tool_use"] is True
    assert [b["type"] for b in r.content] == ["text", "tool_use"] and r.usage == {"in": 11, "out": 5}


def test_bad_key_fails_fast_with_clear_error(fake_api, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "bad")
    with pytest.raises(LLMError, match="401"):
        AnthropicLLM().complete("s", [], [])
    assert Fake.calls == 1          # no pointless retries on auth errors


def test_overload_is_retried(fake_api, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "flaky")
    monkeypatch.setattr("worker.llm.time.sleep", lambda s: None)
    assert AnthropicLLM().complete("s", [], []).content[1]["name"] == "browser_open"
    assert Fake.calls == 2
