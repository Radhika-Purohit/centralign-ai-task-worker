class ConsoleUI:
    C = dict(dim="\033[2m", b="\033[1m", y="\033[33m", g="\033[32m", r="\033[31m", c="\033[36m", x="\033[0m")

    def __init__(self, verbose=True):
        self.verbose = verbose

    def _p(self, color, s):
        print(f"{self.C[color]}{s}{self.C['x']}", flush=True)

    def thought(self, t):
        self._p("c", f"\n[agent] {t.strip()}")

    def action(self, name, args):
        self._p("b", f"  -> {name} {args}")

    def observation(self, text, ok):
        if self.verbose:
            lines = text.strip().split("\n")
            shown = "\n".join("     " + l for l in lines[:12]) + (f"\n     ... ({len(lines) - 12} more lines)" if len(lines) > 12 else "")
            self._p("dim" if ok else "r", shown)

    def approve(self, description):
        self._p("y", f"\n  APPROVAL REQUIRED: {description}")
        return input("  Approve? [y/N] ").strip().lower() in ("y", "yes")

    def ask(self, question):
        self._p("y", f"\n  AGENT ASKS: {question}")
        return input("  Your answer> ").strip()
