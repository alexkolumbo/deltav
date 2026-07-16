"""Per-user memory layer with a hard isolation boundary.

A UserMemory is bound to one identity at construction and every operation
is scoped to it — there is no method that takes another user's id. The
gateway builds a UserMemory from the *authenticated* identity, so a
caller cannot read or write anyone else's memory even by crafting the
request body. Facts, preferences and self-learned notes are kept per
user and recalled to shape future turns.
"""
from __future__ import annotations

from ..overlay.memory import VectorMemory

# Memory item kinds, in retrieval-priority order for the companion.
KIND_LEARNING = "learning"   # something the agent figured out / was told
KIND_FACT = "fact"           # a fact/preference the user shared
KIND_NOTE = "note"


class UserMemory:
    def __init__(self, store: VectorMemory, identity):
        self._store = store
        self._id = identity.user_id          # immutable isolation key
        self.address = identity.address

    # ------------------------------------------------------------- writes
    async def remember(self, text: str, kind: str = KIND_FACT, weight: float = 1.0) -> dict:
        return await self._store.aadd(self._id, text,
                                      meta={"kind": kind, "weight": weight})

    async def learn(self, text: str, weight: float = 2.0) -> dict:
        """Store a durable learning (self-improvement / feedback)."""
        return await self._store.aadd(self._id, text,
                                      meta={"kind": KIND_LEARNING, "weight": weight})

    # ------------------------------------------------------------- reads
    async def recall(self, query: str, k: int = 5) -> list[dict]:
        """Relevant memories for THIS user only, learnings weighted up."""
        hits = await self._store.asearch(self._id, query, k=k * 2)
        for h in hits:
            w = float(h.get("meta", {}).get("weight", 1.0))
            h["_rank"] = h.get("score", 0.0) * w
        hits.sort(key=lambda h: -h["_rank"])
        return hits[:k]

    def learnings(self, limit: int = 20) -> list[dict]:
        items = [it for it in self._store.session_items(self._id)
                 if it.get("meta", {}).get("kind") == KIND_LEARNING]
        return items[-limit:]

    def all_items(self) -> list[dict]:
        return self._store.session_items(self._id)

    def context_block(self, recalled: list[dict]) -> str:
        """Render learnings + recalled memories into a compact prompt block
        (kept short — small models drown in long context)."""
        lines = []
        learn = self.learnings(limit=5)
        if learn:
            lines.append("What you've learned about this user before:")
            lines += [f"- {it['text']}" for it in learn]
        fresh = [h for h in recalled if h.get("meta", {}).get("kind") != KIND_LEARNING]
        if fresh:
            lines.append("Relevant memory:")
            lines += [f"- {h['text']}" for h in fresh[:4]]
        return "\n".join(lines)
