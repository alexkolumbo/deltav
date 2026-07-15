"""Agent memory: a session-scoped BM25 store, no external dependencies.

Embeddings would need an embedding model on every gateway; BM25 gives
useful recall today and the MemoryStore interface won't change when a
vector backend is added later.
"""
from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from pathlib import Path

_WORD_RE = re.compile(r"\w+", re.UNICODE)

K1 = 1.5
B = 0.75


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


class MemoryStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.items: list[dict] = []  # {"id", "session", "text", "meta", "ts"}
        if self.path is not None and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    self.items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    def add(self, session: str, text: str, meta: dict | None = None,
            vec: list[float] | None = None) -> dict:
        item = {
            "id": f"mem-{len(self.items) + 1}",
            "session": session,
            "text": text,
            "meta": meta or {},
            "ts": time.time(),
        }
        if vec is not None:
            item["vec"] = vec
        self.items.append(item)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
        return item

    def session_items(self, session: str) -> list[dict]:
        return [it for it in self.items if it["session"] == session]

    def search(self, session: str, query: str, k: int = 4) -> list[dict]:
        """BM25 over this session's memories; returns items with a score."""
        docs = self.session_items(session)
        if not docs:
            return []
        corpus = [_tokens(d["text"]) for d in docs]
        n = len(corpus)
        avg_len = sum(len(c) for c in corpus) / n
        df: Counter = Counter()
        for toks in corpus:
            df.update(set(toks))

        scored = []
        q_tokens = _tokens(query)
        for doc, toks in zip(docs, corpus):
            tf = Counter(toks)
            score = 0.0
            for term in q_tokens:
                if term not in tf:
                    continue
                idf = math.log((n - df[term] + 0.5) / (df[term] + 0.5) + 1.0)
                denom = tf[term] + K1 * (1 - B + B * len(toks) / avg_len)
                score += idf * tf[term] * (K1 + 1) / denom
            if score > 0:
                scored.append({**doc, "score": round(score, 4)})
        scored.sort(key=lambda d: -d["score"])
        return scored[:k]


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class VectorMemory(MemoryStore):
    """MemoryStore + optional network embedder.

    `embedder(texts) -> vectors` is an async callable — on the gateway it
    routes a PAID embedding job through the network itself. When no
    embedding node is live (or vectors are missing) everything degrades
    to BM25, so memory never breaks.
    """

    def __init__(self, path=None, embedder=None):
        super().__init__(path)
        self.embedder = embedder

    async def _embed(self, texts: list[str]) -> list[list[float]] | None:
        if self.embedder is None:
            return None
        try:
            return await self.embedder(texts)
        except Exception:
            return None

    async def aadd(self, session: str, text: str, meta: dict | None = None) -> dict:
        vecs = await self._embed([text])
        return self.add(session, text, meta, vec=vecs[0] if vecs else None)

    def vector_search(self, session: str, qvec: list[float], k: int = 4) -> list[dict]:
        scored = [
            {**it, "score": round(cosine(qvec, it["vec"]), 4)}
            for it in self.session_items(session)
            if it.get("vec")
        ]
        scored = [s for s in scored if s["score"] > 0]
        scored.sort(key=lambda d: -d["score"])
        return scored[:k]

    async def asearch(self, session: str, query: str, k: int = 4) -> list[dict]:
        vecs = await self._embed([query])
        if vecs:
            hits = self.vector_search(session, vecs[0], k)
            if hits:
                return hits
        return self.search(session, query, k)
