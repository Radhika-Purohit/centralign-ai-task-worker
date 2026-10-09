import os
import re
import time
from dataclasses import dataclass, field

import requests


@dataclass
class LLMResponse:
    content: list            # list of {"type": "text"|"tool_use", ...} dicts
    stop_reason: str = ""
    usage: dict = field(default_factory=dict)


class LLMError(Exception):
    pass


class AnthropicLLM:
    """Calls the Anthropic Messages API over plain HTTPS (requests).

    Deliberately does not use the `anthropic` SDK: the SDK pulls in pydantic-core (a compiled
    extension) which some locked-down Windows machines block. Raw HTTP has the same capabilities
    for this use case and keeps the dependency list tiny. The agent loop only sees `complete()`,
    so swapping in the SDK or another provider is a one-file change.
    """

    def __init__(self, model: str | None = None, max_tokens: int = 2048):
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self.base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
        self.model = model or os.environ.get("WORKER_MODEL", "claude-sonnet-5-5")
        self.max_tokens = max_tokens
        self.http = requests.Session()

    def complete(self, system, messages, tools) -> LLMResponse:
        body = {"model": self.model, "max_tokens": self.max_tokens, "system": system, "messages": messages,
                "tools": tools, "tool_choice": {"type": "auto", "disable_parallel_tool_use": True}}
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        last = ""
        for attempt in range(1, 5):                      # retry rate limits / overload / network blips
            try:
                r = self.http.post(f"{self.base_url}/v1/messages", json=body, headers=headers, timeout=120)
            except requests.RequestException as e:
                last = f"network error: {e}"
            else:
                if r.status_code == 200:
                    return self._parse(r.json())
                last = f"HTTP {r.status_code}: {r.text[:500]}"
                if r.status_code not in (408, 409, 429, 500, 502, 503, 504, 529):
                    raise LLMError(last)                  # 400/401/403/404: retrying won't help
            time.sleep(2 ** attempt)
        raise LLMError(f"Anthropic API failed after retries. Last error: {last}")

    @staticmethod
    def _parse(data) -> LLMResponse:
        content = []
        for b in data.get("content", []):
            if b.get("type") == "text" and b.get("text", "").strip():
                content.append({"type": "text", "text": b["text"]})
            elif b.get("type") == "tool_use":
                content.append({"type": "tool_use", "id": b["id"], "name": b["name"], "input": b.get("input", {})})
        u = data.get("usage", {})
        return LLMResponse(content, data.get("stop_reason", ""), {"in": u.get("input_tokens", 0), "out": u.get("output_tokens", 0)})


class GeminiLLM:
    """Google Gemini via the REST generateContent API (plain HTTPS, no SDK).

    The agent speaks one internal format (Anthropic-style content blocks). This adapter translates:
      * tool schemas      -> Gemini functionDeclarations (OpenAPI subset: free-form maps become
                             lists of {key, value}; converted back when the model calls a tool)
      * messages          -> Gemini `contents` (assistant=model, tool results=functionResponse parts)
      * Gemini 3 "thought signatures": the raw response parts are stashed on each block
        (`_gemini_part`) and replayed verbatim, because Gemini 3.x returns HTTP 400 if they are dropped.
    """

    def __init__(self, model: str | None = None, max_tokens: int = 8192):
        self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
        self.base_url = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com").rstrip("/")
        self.model = model or os.environ.get("WORKER_MODEL", "gemini-3.5-flash")
        self.max_tokens = max_tokens
        self.http = requests.Session()
        self._n = 0
        self._maps: dict[str, set] = {}        # tool name -> params that were free-form maps

    # ---- schema translation -------------------------------------------------
    def _schema(self, s):
        out = {k: v for k, v in s.items() if k in ("type", "description", "enum", "required", "format")}
        if s.get("type") == "object":
            out["properties"] = {k: self._schema(v) for k, v in s.get("properties", {}).items()}
        elif s.get("type") == "array":
            out["items"] = self._schema(s["items"])
        return out

    def _declarations(self, tools):
        decls = []
        for t in tools:
            schema, props, maps = t["input_schema"], {}, set()
            for name, sub in schema.get("properties", {}).items():
                if sub.get("type") == "object" and "properties" not in sub:     # free-form map
                    maps.add(name)
                    props[name] = {"type": "array", "description": (sub.get("description", "") + " Give a list of {key, value} pairs.").strip(),
                                   "items": {"type": "object", "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
                                             "required": ["key", "value"]}}
                else:
                    props[name] = self._schema(sub)
            self._maps[t["name"]] = maps
            d = {"name": t["name"], "description": t["description"],
                 "parameters": {"type": "object", "properties": props}}
            if schema.get("required"):
                d["parameters"]["required"] = schema["required"]
            decls.append(d)
        return decls

    # ---- message translation ------------------------------------------------
    @staticmethod
    def _contents(messages):
        ids, contents = {}, []
        for m in messages:
            role = "user" if m["role"] == "user" else "model"
            if isinstance(m["content"], str):
                contents.append({"role": role, "parts": [{"text": m["content"]}]})
                continue
            parts = []
            for b in m["content"]:
                if b["type"] == "tool_use":
                    ids[b["id"]] = (b["name"], b.get("_gemini_id"))
                    parts.append(b.get("_gemini_part") or {"functionCall": {"name": b["name"], "args": b["input"]}})
                elif b["type"] == "text":
                    parts.append(b.get("_gemini_part") or {"text": b["text"]})
                elif b["type"] == "tool_result":
                    name, gid = ids.get(b["tool_use_id"], ("unknown", None))
                    fr = {"name": name, "response": {("error" if b.get("is_error") else "result"): b["content"]}}
                    if gid:
                        fr["id"] = gid
                    parts.append({"functionResponse": fr})
            if parts:
                contents.append({"role": role, "parts": parts})
        return contents

    # ---- API call -----------------------------------------------------------
    def complete(self, system, messages, tools) -> LLMResponse:
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": self._contents(messages),
                "tools": [{"functionDeclarations": self._declarations(tools)}],
                "toolConfig": {"functionCallingConfig": {"mode": "AUTO"}},
                "generationConfig": {"maxOutputTokens": self.max_tokens}}
        url = f"{self.base_url}/v1beta/models/{self.model}:generateContent"
        headers = {"x-goog-api-key": self.api_key, "content-type": "application/json"}
        last = ""
        for attempt in range(1, 7):
            wait = 2 ** attempt
            try:
                r = self.http.post(url, json=body, headers=headers, timeout=120)
            except requests.RequestException as e:
                last = f"network error: {e}"
            else:
                if r.status_code == 200:
                    return self._parse(r.json())
                last = f"HTTP {r.status_code}: {r.text[:600]}"
                if r.status_code not in (408, 429, 500, 502, 503, 504):
                    hint = ""
                    if r.status_code == 404:
                        hint = f"  (model '{self.model}' not found for your key - try --model gemini-2.5-flash)"
                    if r.status_code in (400, 403) and "API key" in r.text:
                        hint = "  (check GEMINI_API_KEY is set in THIS terminal and copied fully)"
                    raise LLMError(last + hint)
                m = re.search(r'"retryDelay":\s*"(\d+(?:\.\d+)?)s"', r.text)      # free tier: server tells us how long to wait
                if m:
                    wait = float(m.group(1)) + 1
            time.sleep(min(wait, 65))
        raise LLMError(f"Gemini API failed after retries (free-tier quota?). Last error: {last}")

    def _parse(self, data) -> LLMResponse:
        if not data.get("candidates"):
            raise LLMError(f"Gemini returned no candidates: {str(data.get('promptFeedback', data))[:300]}")
        cand = data["candidates"][0]
        content = []
        for p in (cand.get("content") or {}).get("parts", []):
            if p.get("thought"):
                continue
            if "functionCall" in p:
                fc = p["functionCall"]
                self._n += 1
                args = dict(fc.get("args") or {})
                for name in self._maps.get(fc["name"], ()):
                    if isinstance(args.get(name), list):
                        args[name] = {kv["key"]: str(kv.get("value", "")) for kv in args[name] if isinstance(kv, dict) and "key" in kv}
                content.append({"type": "tool_use", "id": fc.get("id") or f"call_{self._n}", "name": fc["name"],
                                "input": args, "_gemini_part": p, "_gemini_id": fc.get("id")})
            elif p.get("text", "").strip():
                content.append({"type": "text", "text": p["text"], "_gemini_part": p})
        u = data.get("usageMetadata", {})
        return LLMResponse(content, cand.get("finishReason", ""),
                           {"in": u.get("promptTokenCount", 0), "out": u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)})


def make_llm(provider: str | None = None, model: str | None = None):
    """Pick a backend: explicit arg > LLM_PROVIDER env > whichever API key is set."""
    has_gemini = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    has_claude = bool(os.environ.get("ANTHROPIC_API_KEY"))
    provider = provider or os.environ.get("LLM_PROVIDER") or ("anthropic" if has_claude else "gemini" if has_gemini else "")
    if provider == "gemini":
        if not has_gemini:
            raise SystemExit("GEMINI_API_KEY is not set in this terminal (see README).")
        return GeminiLLM(model)
    if provider == "anthropic":
        if not has_claude:
            raise SystemExit("ANTHROPIC_API_KEY is not set in this terminal (see README).")
        return AnthropicLLM(model)
    raise SystemExit("No API key found. Set GEMINI_API_KEY (or ANTHROPIC_API_KEY); see README.")
