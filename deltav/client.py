"""Delta V client SDK + connection profiles.

One place to hold how you reach the network — base URL(s) and API key —
so the CLI, a REPL, or your own Python code all connect the same way.
Supports several gateway URLs with automatic failover.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import httpx

CONFIG_PATH = Path.home() / ".deltav" / "client.json"


@dataclass
class Profile:
    base_urls: list[str] = field(default_factory=lambda: ["http://127.0.0.1:9000"])
    api_key: str = "deltav"
    model: str = "auto"

    def to_dict(self) -> dict:
        return {"base_urls": self.base_urls, "api_key": self.api_key, "model": self.model}


def load_profile(path: Path | None = None) -> Profile:
    path = path or CONFIG_PATH
    if path.exists():
        d = json.loads(path.read_text(encoding="utf-8"))
        urls = d.get("base_urls") or ([d["base_url"]] if d.get("base_url") else None)
        return Profile(base_urls=urls or ["http://127.0.0.1:9000"],
                       api_key=d.get("api_key", "deltav"), model=d.get("model", "auto"))
    return Profile()


def save_profile(profile: Profile, path: Path | None = None) -> Path:
    path = path or CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
    return path


class DeltaVClient:
    """Thin OpenAI/Anthropic-compatible client with multi-gateway failover."""

    def __init__(self, base_urls: list[str] | str | None = None, api_key: str = "deltav",
                 model: str = "auto", client: httpx.Client | None = None):
        if isinstance(base_urls, str):
            base_urls = [base_urls]
        self.base_urls = [u.rstrip("/") for u in (base_urls or ["http://127.0.0.1:9000"])]
        self.api_key = api_key
        self.model = model
        self.client = client or httpx.Client(timeout=600.0)

    @classmethod
    def from_profile(cls, profile: Profile | None = None, **kw) -> "DeltaVClient":
        p = profile or load_profile()
        return cls(base_urls=p.base_urls, api_key=p.api_key, model=p.model, **kw)

    @property
    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _request(self, method: str, path: str, **kw) -> httpx.Response:
        last: Exception | None = None
        for base in self.base_urls:
            try:
                resp = self.client.request(method, f"{base}{path}", headers=self._headers, **kw)
                if resp.status_code < 500:  # 4xx is a real answer, don't fail over
                    return resp
                last = httpx.HTTPStatusError("5xx", request=resp.request, response=resp)
            except httpx.HTTPError as exc:
                last = exc
        raise last or RuntimeError("no gateway reachable")

    # ------------------------------------------------------------- surfaces
    def health(self) -> dict:
        return self._request("GET", "/health").json()

    def models(self) -> list[dict]:
        return self._request("GET", "/v1/models").json().get("data", [])

    def chat(self, messages: list[dict], model: str | None = None,
             max_tokens: int = 512, **kw) -> dict:
        body = {"model": model or self.model, "messages": messages,
                "max_tokens": max_tokens, **kw}
        return self._request("POST", "/v1/chat/completions", json=body).json()

    def chat_stream(self, messages: list[dict], model: str | None = None,
                    max_tokens: int = 512, **kw):
        body = {"model": model or self.model, "messages": messages,
                "max_tokens": max_tokens, "stream": True, **kw}
        for base in self.base_urls:
            try:
                with self.client.stream("POST", f"{base}/v1/chat/completions",
                                        headers=self._headers, json=body) as resp:
                    if resp.status_code >= 500:
                        continue
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if line.startswith("data: ") and line != "data: [DONE]":
                            yield json.loads(line[6:])
                    return
            except httpx.HTTPError:
                continue
        raise RuntimeError("no gateway reachable for streaming")

    def agent(self, task: str, session_id: str = "", max_steps: int = 6, **kw) -> dict:
        body = {"task": task, "model": self.model, "max_steps": max_steps,
                "session_id": session_id, **kw}
        return self._request("POST", "/v1/agents/run", json=body).json()

    def swarm(self, task: str, n: int = 3, mode: str = "fanout",
              models: list[str] | None = None, **kw) -> dict:
        body = {"task": task, "n": n, "mode": mode, **kw}
        if models:
            body["models"] = models
        return self._request("POST", "/v1/swarm", json=body).json()

    def companion(self, message: str, history: list[dict] | None = None, **kw) -> dict:
        body = {"message": message, "model": self.model, **kw}
        if history:
            body["history"] = history
        return self._request("POST", "/v1/companion/chat", json=body).json()

    def companion_feedback(self, note: str) -> dict:
        return self._request("POST", "/v1/companion/feedback", json={"note": note}).json()

    def companion_memory(self) -> dict:
        return self._request("GET", "/v1/companion/memory").json()

    def embed(self, texts: list[str], model: str = "auto") -> list[list[float]]:
        body = {"input": texts, "model": model}
        data = self._request("POST", "/v1/embeddings", json=body).json()
        return [d["embedding"] for d in data.get("data", [])]

    def close(self) -> None:
        self.client.close()
