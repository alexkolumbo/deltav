"""Blocks of the Delta V chain."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..crypto import KeyPair, address_from_public, canonical_json, sha256_hex, verify_signature
from .transaction import Tx

GENESIS_PROPOSER = "genesis"
ZERO_HASH = "0" * 64


def tx_root(txs: list[Tx]) -> str:
    return sha256_hex(canonical_json([tx.hash for tx in txs]))


@dataclass
class Block:
    height: int
    prev_hash: str
    timestamp: float
    proposer: str
    txs: list[Tx] = field(default_factory=list)
    state_root: str = ""
    # Liveness slot: 0 = primary proposer, s > 0 = fallback proposer that
    # stepped in after s * block_time of silence.
    slot: int = 0
    pubkey: str = ""
    signature: str = ""

    def header(self) -> dict:
        return {
            "height": self.height,
            "prev_hash": self.prev_hash,
            "timestamp": self.timestamp,
            "proposer": self.proposer,
            "slot": self.slot,
            "tx_root": tx_root(self.txs),
            "state_root": self.state_root,
        }

    @property
    def hash(self) -> str:
        return sha256_hex(canonical_json(self.header()))

    def sign(self, keypair: KeyPair) -> "Block":
        if keypair.address != self.proposer:
            raise ValueError("signer is not the proposer")
        self.pubkey = keypair.public_hex
        self.signature = keypair.sign(bytes.fromhex(self.hash))
        return self

    def verify_signature(self) -> bool:
        if self.proposer == GENESIS_PROPOSER:
            return self.height == 0
        if not self.pubkey or not self.signature:
            return False
        if address_from_public(self.pubkey) != self.proposer:
            return False
        return verify_signature(self.pubkey, bytes.fromhex(self.hash), self.signature)

    def to_dict(self) -> dict:
        return {
            "height": self.height,
            "prev_hash": self.prev_hash,
            "timestamp": self.timestamp,
            "proposer": self.proposer,
            "txs": [tx.to_dict() for tx in self.txs],
            "state_root": self.state_root,
            "slot": self.slot,
            "pubkey": self.pubkey,
            "signature": self.signature,
            "hash": self.hash,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Block":
        return cls(
            height=int(data["height"]),
            prev_hash=data["prev_hash"],
            timestamp=float(data["timestamp"]),
            proposer=data["proposer"],
            txs=[Tx.from_dict(t) for t in data.get("txs", [])],
            state_root=data.get("state_root", ""),
            slot=int(data.get("slot", 0)),
            pubkey=data.get("pubkey", ""),
            signature=data.get("signature", ""),
        )
