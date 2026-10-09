"""A small text browser the agent drives: real HTTP, numbered links, parsed forms.

Design notes
- Pages are rendered to compact text with [L<n>] link markers and <FORM F<n>> blocks,
  so the model can act by index instead of guessing selectors/URLs.
- Only allow-listed origins can be visited (basic safety boundary).
- GETs are idempotent, so transient failures (5xx / network) are retried here with
  backoff. POSTs are NEVER auto-retried (could double-submit); the agent decides.
"""
import re
import time
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

MAX_CHARS = 6000


class BrowserError(Exception):
    pass


@dataclass
class Page:
    status: int
    url: str
    text: str
    notes: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status < 400

    def format(self) -> str:
        head = f"HTTP {self.status} {self.url}"
        if self.notes:
            head += "\n(" + "; ".join(self.notes) + ")"
        return head + "\n\n" + self.text


class Browser:
    def __init__(self, allowed_origins, max_attempts: int = 2, backoff: float = 0.5):
        self.session = requests.Session()
        self.allowed = set(allowed_origins)
        self.max_attempts = max_attempts
        self.backoff = backoff
        self.url = None
        self.links: list[str] = []
        self.forms: list[dict] = []

    # ---- low level -------------------------------------------------
    def _check(self, url: str):
        p = urlparse(url)
        origin = f"{p.scheme}://{p.netloc}"
        if origin not in self.allowed:
            raise BrowserError(f"Blocked: {origin} is not an allowed origin ({sorted(self.allowed)})")

    def _request(self, method: str, url: str, **kw):
        self._check(url)
        attempts = self.max_attempts if method == "GET" else 1
        notes = []
        for i in range(1, attempts + 1):
            try:
                r = self.session.request(method, url, timeout=10, **kw)
            except requests.RequestException as e:
                if i == attempts:
                    raise BrowserError(f"Network error after {i} attempt(s): {e}")
                notes.append(f"attempt {i} network error, retried")
                time.sleep(self.backoff * i)
                continue
            if r.status_code >= 500 and i < attempts:
                notes.append(f"attempt {i} got HTTP {r.status_code}, retried automatically")
                time.sleep(self.backoff * i)
                continue
            if i > 1 and r.status_code < 500:
                notes.append(f"succeeded on attempt {i}")
            elif r.status_code >= 500 and attempts > 1:
                notes.append(f"gave up after {i} attempts")
            return r, notes

    def _render(self, r) -> Page:
        self.url = r.url
        soup = BeautifulSoup(r.text, "html.parser")
        for t in soup(["script", "style"]):
            t.decompose()
        title = soup.title.get_text(strip=True) if soup.title else ""

        self.links = []
        for a in soup.find_all("a", href=True):
            self.links.append(urljoin(r.url, a["href"]))
            a.replace_with(f"{a.get_text(' ', strip=True)} [L{len(self.links)}]")

        self.forms = []
        for form in soup.find_all("form"):
            fields, desc = [], []
            for el in form.find_all(["input", "select", "textarea"]):
                name, typ = el.get("name"), (el.get("type") or el.name).lower()
                if not name or typ in ("submit", "button"):
                    continue
                if el.name == "select":
                    opts = [o.get("value", o.get_text(strip=True)) for o in el.find_all("option")]
                    fields.append(dict(name=name, default=opts[0] if opts else "", options=opts))
                    desc.append(f"{name} (select: {opts})")
                else:
                    fields.append(dict(name=name, default=el.get("value", "") or "", options=None))
                    desc.append(f"{name} ({typ})")
            idx = len(self.forms) + 1
            method = (form.get("method") or "get").upper()
            action = urljoin(r.url, form.get("action") or r.url)
            self.forms.append(dict(method=method, action=action, fields=fields))
            form.replace_with(f"<FORM F{idx} {method} {urlparse(action).path} fields: {', '.join(desc)}>")

        for tr in soup.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            tr.replace_with(soup.new_string(" | ".join(cells) + "\n"))

        body = soup.body.get_text("\n") if soup.body else soup.get_text("\n")
        body = "\n".join(l for l in body.split("\n") if l.strip() not in ("", "|"))
        body = re.sub(r"\n\s*\n+", "\n", body).strip()
        text = f"Title: {title}\n{body}"
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS] + f"\n...[truncated, {len(text) - MAX_CHARS} more chars]"
        return Page(r.status_code, r.url, text)

    # ---- actions ---------------------------------------------------
    def open(self, url: str) -> Page:
        url = urljoin(self.url or "", url)
        r, notes = self._request("GET", url)
        page = self._render(r)
        page.notes = notes
        return page

    def click(self, n: int) -> Page:
        if not 1 <= n <= len(self.links):
            raise BrowserError(f"No link L{n} on the current page (it has {len(self.links)} links)")
        return self.open(self.links[n - 1])

    def prepare_submit(self, n: int, fields: dict):
        if not 1 <= n <= len(self.forms):
            raise BrowserError(f"No form F{n} on the current page (it has {len(self.forms)} forms)")
        form = self.forms[n - 1]
        known = {f["name"] for f in form["fields"]}
        unknown = set(fields) - known
        if unknown:
            raise BrowserError(f"Unknown field(s) {sorted(unknown)} for F{n}; valid: {sorted(known)}")
        data = {f["name"]: f["default"] for f in form["fields"]}
        data.update({k: str(v) for k, v in fields.items()})
        return form["method"], form["action"], data

    def submit(self, n: int, fields: dict) -> Page:
        method, action, data = self.prepare_submit(n, fields)
        if method == "GET":
            r, notes = self._request("GET", action, params=data)
        else:
            r, notes = self._request("POST", action, data=data)
        page = self._render(r)
        page.notes = notes
        return page
