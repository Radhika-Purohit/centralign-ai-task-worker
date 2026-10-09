import json
import time
from pathlib import Path


class Trace:
    """Append-only JSONL log of everything the agent did: the audit trail."""

    def __init__(self, root="runs"):
        self.dir = Path(root) / time.strftime("%Y%m%d-%H%M%S")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.file = self.dir / "trace.jsonl"
        self.t0 = time.time()

    def log(self, kind: str, **data):
        rec = dict(t=round(time.time() - self.t0, 2), kind=kind, **data)
        with open(self.file, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def write_report(self, text: str):
        (self.dir / "report.md").write_text(text)
