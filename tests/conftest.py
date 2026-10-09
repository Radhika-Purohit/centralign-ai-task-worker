import threading

import pytest
from werkzeug.serving import make_server

from sandbox.app import State, create_app
from worker.agent import Agent
from worker.browser import Browser
from worker.llm import LLMResponse
from worker.tools import Toolset


class ScriptedLLM:
    """Test double: replays a fixed list of tool calls. Used ONLY to test the harness
    (approval, verify gate, budgets...), never to demonstrate autonomy."""

    def __init__(self, script):
        self.script, self.i = list(script), 0

    def complete(self, system, messages, tools):
        name, args = self.script[self.i]
        self.i += 1
        return LLMResponse([{"type": "tool_use", "id": f"t{self.i}", "name": name, "input": args}], "tool_use", {"in": 1, "out": 1})


class FakeUI:
    def __init__(self, approve=True, answers=()):
        self._approve, self.answers, self.asked, self.approvals = approve, list(answers), [], []

    def thought(self, t): pass
    def action(self, n, a): pass
    def observation(self, t, ok): pass

    def approve(self, d):
        self.approvals.append(d)
        return self._approve

    def ask(self, q):
        self.asked.append(q)
        return self.answers.pop(0)


@pytest.fixture
def env(tmp_path):
    state = State()
    srv = make_server("127.0.0.1", 0, create_app(state), threaded=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    yield state, base, tmp_path
    srv.shutdown()


@pytest.fixture
def make_agent(env):
    state, base, tmp = env

    def _make(script, ui=None, **kw):
        tools = Toolset(Browser([base], backoff=0.01), workspace=str(tmp / "ws"))
        ui = ui or FakeUI()
        return Agent(ScriptedLLM(script), tools, ui, context=base, trace_root=str(tmp / "runs"), **kw), ui
    return _make
