"""Tool layer. The agent core knows nothing about invoices or ERPs: it only sees
these generic tools. New capabilities = new entries in SPECS + a handler."""
from pathlib import Path

from .browser import Browser, BrowserError

SPECS = [
    {"name": "browser_open", "description": "Open a URL (absolute, or relative to the current page) in the text browser. "
     "Returns the page as text with [L<n>] link markers and <FORM F<n>> blocks.",
     "input_schema": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}},
    {"name": "browser_click", "description": "Follow link [L<n>] from the current page.",
     "input_schema": {"type": "object", "properties": {"link": {"type": "integer"}}, "required": ["link"]}},
    {"name": "browser_submit_form", "description": "Fill and submit form F<n> on the current page. Unspecified fields keep "
     "their defaults. POST submissions change company data and require human approval.",
     "input_schema": {"type": "object", "properties": {"form": {"type": "integer"},
                      "fields": {"type": "object", "additionalProperties": {"type": "string"}}},
                      "required": ["form", "fields"]}},
    {"name": "list_files", "description": "List files in the workspace folder (or a sub-folder).",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}},
    {"name": "read_file", "description": "Read a text file from the workspace.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write a text file inside the workspace (e.g. to save evidence).",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                      "required": ["path", "content"]}},
    {"name": "remember", "description": "Store a key fact (e.g. 'latest_invoice') in working memory. Memory is shown to "
     "you every turn even after old observations are trimmed from context.",
     "input_schema": {"type": "object", "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
                      "required": ["key", "value"]}},
    {"name": "ask_user", "description": "Ask the human a question when the request is ambiguous or you cannot proceed "
     "safely. Blocks until they answer.",
     "input_schema": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}},
    {"name": "finish", "description": "End the task. status='completed' is only accepted if the outcome was verified by "
     "re-reading the destination system after your last change, and you must quote evidence copied from pages you saw. "
     "Use 'blocked' (need human/permission) or 'failed' when you cannot complete it.",
     "input_schema": {"type": "object", "properties": {
         "status": {"type": "string", "enum": ["completed", "blocked", "failed"]},
         "summary": {"type": "string"},
         "evidence": {"type": "array", "items": {"type": "string"},
                      "description": "Short verbatim snippets from observed pages proving the outcome"}},
         "required": ["status", "summary"]}},
]


class Toolset:
    def __init__(self, browser: Browser, workspace="workspace"):
        self.browser = browser
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.memory: dict[str, str] = {}

    def specs(self):
        return SPECS

    def _safe(self, rel: str) -> Path:
        p = (self.workspace / (rel or ".")).resolve()
        if self.workspace not in p.parents and p != self.workspace:
            raise ValueError(f"Path '{rel}' escapes the workspace")
        return p

    # classification used by the harness (not by the model) ------------
    def mutation_preview(self, name: str, args: dict):
        """Return a human-readable description if this call changes external state."""
        if name == "browser_submit_form":
            method, action, data = self.browser.prepare_submit(args["form"], args.get("fields", {}))
            if method == "POST":
                return f"POST {action} with {data}"
        return None

    def is_observation(self, name: str) -> bool:
        return name in ("browser_open", "browser_click") or name == "browser_submit_form"

    # execution -------------------------------------------------------
    def call(self, name: str, args: dict):
        """Returns (text, ok, http_status|None)."""
        try:
            if name == "browser_open":
                p = self.browser.open(args["url"])
                return p.format(), p.ok, p.status
            if name == "browser_click":
                p = self.browser.click(int(args["link"]))
                return p.format(), p.ok, p.status
            if name == "browser_submit_form":
                p = self.browser.submit(int(args["form"]), args.get("fields", {}))
                return p.format(), p.ok, p.status
            if name == "list_files":
                d = self._safe(args.get("path", ""))
                return "\n".join(sorted(x.name + ("/" if x.is_dir() else "") for x in d.iterdir())) or "(empty)", True, None
            if name == "read_file":
                return self._safe(args["path"]).read_text()[:8000], True, None
            if name == "write_file":
                p = self._safe(args["path"])
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(args["content"])
                return f"Wrote {len(args['content'])} chars to {args['path']}", True, None
            if name == "remember":
                self.memory[args["key"]] = args["value"]
                return f"Remembered {args['key']}", True, None
        except (BrowserError, ValueError, OSError, KeyError) as e:
            return f"ERROR: {type(e).__name__}: {e}", False, None
        return f"ERROR: unknown tool '{name}'", False, None
