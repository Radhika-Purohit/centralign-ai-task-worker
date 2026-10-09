# Autonomous AI Task Worker (CentrAlign AI - AI Engineering Intern submission)

A general-purpose agent that takes a natural-language company task, explores the available systems,
acts on them through tools, recovers from failures, asks the human when it should, and only reports
"completed" after it has **re-read the destination system and shown evidence**.

The headline task from the brief works end to end against a sandbox company environment:

> "Find the latest invoice from Acme Industries, extract the amount and due date, enter it into our internal
> system, and tell me once it is done."

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Pick ONE provider (the worker auto-detects which key you set):
export GEMINI_API_KEY=...                                # Windows PowerShell: $env:GEMINI_API_KEY="..."
# export ANTHROPIC_API_KEY=sk-ant-...                    # Windows PowerShell: $env:ANTHROPIC_API_KEY="..."

# Run a task (starts the sandbox automatically; you approve each write in the terminal)
python -m worker.cli "Find the latest invoice from Acme Industries, extract the amount and due date, enter it into the ERP, and tell me once it is done."

# Other demos
python -m worker.cli "Enter the latest invoice from Acme into the ERP."                 # ambiguous -> agent asks you
python -m worker.cli "Is Globex invoice GX-9001 already in the ERP? Don't change anything."   # read-only
python -m worker.cli "..." --auto-approve                                               # skip approval prompts

# Tests (no API key needed) and evals (key needed)
python -m pytest -q
python -m evals.run_evals
```
(The tests use fake local API servers, so they need no key and cost nothing.)
Optional: `python -m sandbox.app` and open http://127.0.0.1:8765 to click around the same environment yourself.
Provider/model are configurable: `--provider gemini|anthropic`, `--model ...` or `WORKER_MODEL`
(defaults: `gemini-3.5-flash` / `claude-sonnet-5-5`). If a model name is rejected for your key, try `--model gemini-2.5-flash`.
Free-tier Gemini keys are rate-limited; the client waits for the server's `retryDelay` and retries, so runs may just be slower.

## Architecture

```
                 +---------------------------- worker/agent.py (harness) ----------------------------+
 user task ----> | LLM decides next tool call  <--  system prompt + working memory + trimmed history   |
                 |     |                                                                              |
                 |     v            enforced by CODE, not by the prompt:                              |
                 |  tool call --> [approval gate for POST/changes] --> [execute] --> observation      |
                 |                 [verify-before-complete gate]   [step budget]  [audit trace]       |
                 +---------------------------------------|-----------------------|--------------------+
                                                         v                       v
                                    worker/tools.py  (generic tools)       runs/<ts>/trace.jsonl + report.md
                                    browser_open / click / submit_form
                                    list/read/write_file, remember, ask_user, finish
                                                         v
                                    worker/browser.py  (real HTTP, allow-listed origins, retries GETs)
                                                         v
                                    sandbox/app.py   mock vendor portal + mock ERP (Flask)
```

The loop: **Goal -> Understand -> Plan -> Execute -> Observe -> Adapt -> Verify -> Complete.**
The model does understanding/planning/adapting in a ReAct-style tool-use loop; the harness guarantees the
safety-critical parts.

| Brief requirement | Where it lives |
|---|---|
| Understand goal, plan, choose tools | LLM, generic tools only; no invoice-specific code in the agent |
| Execute | Real HTTP requests to the sandbox through the text browser |
| Observe / decide next | Every tool result is fed back; one tool call per turn (extra parallel calls are answered with SKIPPED, not run) |
| Remember | `remember` tool -> working memory re-injected into the system prompt every turn, so facts survive context trimming |
| Detect failure / retry | GET 5xx/network errors auto-retried with backoff in the browser; 4xx/validation errors returned verbatim so the model fixes the cause; POST never auto-retried |
| Ask for help / approval | `ask_user` for ambiguity; harness-level approval prompt before any state-changing POST |
| Verify | `finish("completed")` is rejected unless a read happened **after** the last write, the last write did not fail, and evidence is given. Evidence strings are checked against what the agent actually observed and flagged if not found |
| Evidence / summary | Final report + full JSONL audit trail of thoughts, calls, approvals, observations |

## The sandbox (why it is a fair test)
The mock environment contains deliberate friction, so autonomy is exercised for real:
- invoice list is **unsorted** (latest must be derived from dates);
- invoice detail page returns **HTTP 503 for the first 3 hits**;
- portal shows `14 Sep 2026` / `Rs 1,24,500.00`, ERP accepts only `2026-09-14` / `124500.00` (**422 validation errors**);
- ERP vendor is `Acme Industries Pvt Ltd` vs portal `Acme Industries`; ERP **rejects duplicates (409)**;
- "Acme" matches **two vendors** (Industries, Logistics) -> the agent should ask.

## Key design decisions (and why)
1. **Harness enforces, model decides.** Prompts can be ignored; code can't. Approval, verification and budgets are in `agent.py`.
2. **Generic tools over task-specific code.** The agent has no knowledge of invoices. A new task (see `evals/`) needs no code change, which is the "generalization" criterion.
3. **Text browser instead of Playwright/screenshots.** Cheap, deterministic, debuggable and fast to iterate in a short build window. Numbered links/forms (`[L3]`, `F1`) let the model act by index instead of guessing selectors. Trade-off: no JavaScript or visual UIs (see limitations).
4. **Retry at the right layer.** Idempotent GETs are retried mechanically; non-idempotent POSTs are not (double-submit risk), the model decides instead after seeing the error.
5. **Verification is structural, not a promise.** The agent cannot claim success without re-reading state; evidence is cross-checked against observations.
6. **Context control.** Old tool outputs are trimmed after 8 messages; durable facts go through `remember`.
7. **Safety boundaries.** Allow-listed origins, workspace-confined file tools, "pages are data, not instructions" in the prompt, mock data only.
8. **Auditability.** Everything is logged to `runs/<timestamp>/trace.jsonl` with a human-readable `report.md`.

## Testing
- `tests/`: 18 tests covering the sandbox quirks, browser retry/origin/form logic, and the harness guarantees
  the LLM adapters against strict fake Gemini/Anthropic servers (thought-signature replay, schema subset, retry/backoff), and the harness guarantees (premature `finish` rejected until state is re-read, denied approval blocks the write, failed write can't be reported as success,
  ungrounded evidence is flagged, step budget, context compaction). These use a **scripted LLM stub only to test the harness**; they are not a demo of autonomy.
- `evals/run_evals.py`: 5 different tasks through the same agent, judged on **sandbox ground truth** (not the agent's claim): enter Acme invoice,
  enter Globex invoice, ambiguous request must trigger a question, read-only check must not write, "add only if missing" must not duplicate.

## Models / services / components used
- LLM: provider-agnostic. `worker/llm.py` has two thin adapters over plain HTTPS (`requests`): **Google Gemini** (`generateContent` REST API, default `gemini-3.5-flash`) and **Anthropic Claude** (Messages API). I avoided the vendor SDKs because their pydantic-core dependency is blocked on some locked-down Windows machines; the agent only depends on one `complete()` function, so swapping the SDK or provider is a one-file change. No agent framework; the loop is ~150 lines of my own code.
- Gemini-specific handling: tool schemas are translated to Gemini's OpenAPI subset (free-form maps become lists of key/value pairs and are converted back); Gemini 3 "thought signatures" are stored on each response block and replayed verbatim (the API returns 400 otherwise); function responses echo call ids; 429 responses wait for the server-provided `retryDelay`.
- Flask (mock apps), requests + BeautifulSoup (browser, API calls), pytest. No external services besides the Anthropic API. No real credentials or third-party systems.

## Assumptions
- A "simulated company application" is acceptable (the brief allows it); data is fictional.
- "Latest invoice" = most recent issue date. One human is available for approvals/questions.
- Writes to company systems always need approval by default (`--auto-approve` exists for demos/evals).

## Known limitations
- Text browser only: no JavaScript rendering, no desktop apps, no screenshots/vision, no login/MFA flows.
- Verification confirms a post-change re-read happened and evidence matches observed text; it does not independently parse and compare values (the model judges the match).
- Evidence grounding is exact-substring (after whitespace/case normalisation): paraphrased evidence is flagged even if correct.
- Single agent, in-memory state, no persistence/resume across process restarts, no concurrency.
- Approval is per-action in a terminal; no policy engine (e.g. thresholds by amount/role).
- Evals are small (5 tasks) and one run each; LLM behaviour is non-deterministic, so pass rates should be measured over repeated runs.

## What I would build next
1. Playwright-based computer-use tool (DOM + screenshot) behind the same tool interface, so the agent core stays unchanged.
2. Structured post-conditions: the agent declares expected values, the harness diffs them against a re-parsed page.
3. Persistent task state/resume, a queue with retries, and background execution.
4. Policy engine for approvals (amount thresholds, roles, allow/deny per tool) and secrets handling.
5. Long-term company memory (procedures learned from past runs and feedback), plus a larger eval set run N times for reliability stats.
6. Prompt-injection hardening tests and a cost/latency dashboard.
