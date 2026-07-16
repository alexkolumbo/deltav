"""Gateway API keys: each key IS an on-chain wallet.

A key is a custodial keypair held by the gateway. The consumer funds the
key's address with DVT (any wallet, `deltav send`); every request made
with the key is price-authorized by THAT keypair, so receipts charge the
consumer's account on-chain — the gateway wallet stays untouched.

The file stores sha256(api_key) -> record; the plaintext key is shown
exactly once at creation. Custodial by design (OpenAI-compatible
clients can't sign chain payloads) — the gateway operator is trusted,
exactly like any hosted API provider.
"""
from __future__ import annotations

import json
import os
import secrets
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..crypto import KeyPair, sha256_hex

KEY_PREFIX = "dvk_"

# Guard against unbounded minting: keys.json is rewritten whole on every
# create, so an unauthenticated mint flood is an O(n^2) disk-DoS.
MAX_KEYS = 100_000


class KeyLimitError(RuntimeError):
    """Raised when the key store is at capacity."""


@dataclass
class KeyRecord:
    key_hash: str
    seed: str
    address: str
    label: str = ""
    created: float = 0.0
    requests: int = 0
    tokens: int = 0
    spent_udvt: int = 0


class KeyStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.records: dict[str, KeyRecord] = {}
        if self.path is not None and self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.records = {h: KeyRecord(**r) for h, r in data.items()}

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.path.parent, 0o700)
        except OSError:
            pass
        self.path.write_text(
            json.dumps({h: asdict(r) for h, r in self.records.items()},
                       ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        # The file holds every custodial wallet's private seed — owner-only.
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def create(self, label: str = "") -> tuple[str, KeyRecord]:
        if len(self.records) >= MAX_KEYS:
            raise KeyLimitError("key store at capacity")
        api_key = KEY_PREFIX + secrets.token_hex(24)
        keypair = KeyPair.generate()
        record = KeyRecord(
            key_hash=sha256_hex(api_key.encode()),
            seed=keypair.seed_hex,
            address=keypair.address,
            label=label,
            created=time.time(),
        )
        self.records[record.key_hash] = record
        self._save()
        return api_key, record

    def resolve(self, api_key: str) -> KeyRecord | None:
        return self.records.get(sha256_hex(api_key.encode()))

    @staticmethod
    def keypair(record: KeyRecord) -> KeyPair:
        return KeyPair.from_seed_hex(record.seed)

    def record_usage(self, record: KeyRecord, tokens: int, spent_udvt: int) -> None:
        record.requests += 1
        record.tokens += tokens
        record.spent_udvt += spent_udvt
        self._save()
