"""User identity for the companion — the isolation boundary.

The identity is derived from *authentication*, never from a field the
caller controls. A funded `dvk_` API key IS an on-chain wallet, so its
address is a strong, self-owned identity. Without a key, a caller may
name an explicit local user id (for single-tenant setups), but they can
only ever name themselves — the gateway hands each request a memory
scoped to whatever this function returns, and nothing else.
"""
from __future__ import annotations

from dataclasses import dataclass

# Namespace prefix so companion data can never collide with other
# session_ids (agents, web/tg sessions) in the same memory store.
NS = "companion"


@dataclass(frozen=True)
class Identity:
    user_id: str          # the isolation key, e.g. "companion:dv1abc…"
    address: str          # on-chain wallet address if key-authenticated, else ""
    authenticated: bool   # True when derived from a dvk_ wallet key

    @property
    def namespace(self) -> str:
        return self.user_id


def resolve_identity(address: str = "", fallback_user: str = "") -> Identity:
    """Derive the isolation identity.

    `address` — the wallet address of a dvk_ key (authenticated); empty if
    the request had no key. `fallback_user` — an explicit local id for
    keyless setups. A key always wins; a keyless request with no explicit
    id becomes the shared "local" tenant (no cross-user data, one tenant)."""
    if address:
        return Identity(user_id=f"{NS}:{address}", address=address, authenticated=True)
    uid = (fallback_user or "local").strip() or "local"
    # keyless callers can only ever address a local, non-wallet namespace
    return Identity(user_id=f"{NS}:local:{uid}", address="", authenticated=False)
