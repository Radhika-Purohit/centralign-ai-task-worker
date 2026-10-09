import argparse
import os
import sys
import threading

from werkzeug.serving import make_server

from sandbox.app import State, create_app
from .agent import Agent
from .browser import Browser
from .tools import Toolset
from .ui import ConsoleUI


def start_sandbox(port):
    state = State()
    srv = make_server("127.0.0.1", port, create_app(state), threaded=True)
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return state, srv


def main(argv=None):
    ap = argparse.ArgumentParser(description="Autonomous AI task worker")
    ap.add_argument("task", help="Natural language task")
    ap.add_argument("--auto-approve", action="store_true", help="Skip human approval prompts (demo/eval only)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--provider", choices=["gemini", "anthropic"], default=None, help="Default: whichever API key is set")
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--base-url", default=None, help="Use an already-running environment instead of the built-in sandbox")
    args = ap.parse_args(argv)

    from .llm import LLMError, make_llm
    llm = make_llm(args.provider, args.model)

    state = srv = None
    base = args.base_url or f"http://127.0.0.1:{args.port}"
    if not args.base_url:
        state, srv = start_sandbox(args.port)
    context = (f"Company tools are reachable at {base}  (home page: {base}/ ; it links to the vendor invoice portal "
               f"and the internal ERP). Files: workspace/ folder.")
    agent = Agent(llm, Toolset(Browser([base])), ConsoleUI(), context=context,
                  max_steps=args.max_steps, auto_approve=args.auto_approve)
    try:
        res = agent.run(args.task)
    except LLMError as e:
        if srv:
            srv.shutdown()
        sys.exit(f"\nLLM API error: {e}")

    print(f"\n{'=' * 60}\nSTATUS: {res.status}\nSUMMARY: {res.summary}")
    for t, g in res.evidence:
        print(f"  evidence [{'ok' if g else 'UNVERIFIED'}]: {t}")
    print(f"steps={res.steps} tokens={res.tokens}\ntrace: {res.run_dir}/trace.jsonl")
    if state:
        print("\nGROUND TRUTH (ERP bills, read straight from the sandbox, not from the agent):")
        for b in state.bills:
            print("  ", b)
    if srv:
        srv.shutdown()


if __name__ == "__main__":
    main()
