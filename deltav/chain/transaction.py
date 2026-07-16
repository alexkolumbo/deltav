"""Signed transactions of the Delta V chain."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..crypto import KeyPair, address_from_public, canonical_json, sha256_hex, verify_signature


class TxType(str, Enum):
    TRANSFER = "transfer"
    REGISTER_NODE = "register_node"
    ANNOUNCE_MODELS = "announce_models"
    STAKE = "stake"
    UNSTAKE = "unstake"
    INFERENCE_RECEIPT = "inference_receipt"
    SPOT_CHECK = "spot_check"
    # --- v2 reward mechanism ---
    AVAILABILITY_LEASE = "availability_lease"  # node announces it's online for N blocks
    DISPUTE = "dispute"                        # client flags a receipt for priority re-check


@dataclass
class Tx:
    type: str
    sender: str
    nonce: int
    payload: dict = field(default_factory=dict)
    fee: int = 0
    pubkey: str = ""
    signature: str = ""
    # Chain binding (v2): included in the signed bytes only when set, so a tx
    # signed for one chain can't be replayed on a sibling chain (audit C7).
    # Empty string reproduces v1 signed bytes exactly — alpha-3 stays valid.
    chain_id: str = ""

    def signing_payload(self) -> dict:
        base = {
            "type": self.type,
            "sender": self.sender,
            "nonce": self.nonce,
            "payload": self.payload,
            "fee": self.fee,
        }
        if self.chain_id:
            base["chain_id"] = self.chain_id
        return base

    def signing_bytes(self) -> bytes:
        return canonical_json(self.signing_payload())

    @property
    def hash(self) -> str:
        return sha256_hex(canonical_json(self.to_dict()))

    def sign(self, keypair: KeyPair) -> "Tx":
        if keypair.address != self.sender:
            raise ValueError("signer address does not match tx sender")
        self.pubkey = keypair.public_hex
        self.signature = keypair.sign(self.signing_bytes())
        return self

    def verify(self) -> bool:
        if not self.pubkey or not self.signature:
            return False
        if address_from_public(self.pubkey) != self.sender:
            return False
        return verify_signature(self.pubkey, self.signing_bytes(), self.signature)

    def to_dict(self) -> dict:
        out = {
            "type": self.type,
            "sender": self.sender,
            "nonce": self.nonce,
            "payload": self.payload,
            "fee": self.fee,
            "pubkey": self.pubkey,
            "signature": self.signature,
        }
        if self.chain_id:
            out["chain_id"] = self.chain_id
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "Tx":
        return cls(
            type=data["type"],
            sender=data["sender"],
            nonce=int(data["nonce"]),
            payload=dict(data.get("payload", {})),
            fee=int(data.get("fee", 0)),
            pubkey=data.get("pubkey", ""),
            signature=data.get("signature", ""),
            chain_id=data.get("chain_id", ""),
        )


def receipt_auth_bytes(request_hash: str, node: str, model: str, price_limit: int) -> bytes:
    """v1: what a requester signs to authorize paying `node` for one job."""
    return canonical_json(
        {"request_hash": request_hash, "node": node, "model": model, "price_limit": price_limit}
    )


def receipt_auth_bytes_v2(request_hash: str, node: str, model: str, price_limit: int,
                          auth_nonce: str, expiry: int, chain_id: str) -> bytes:
    """v2 payment authorization: a single-use voucher.

    Binding `auth_nonce` (settled exactly once), `expiry` (height) and
    `chain_id` means one signature authorizes exactly one settlement on one
    chain — a node can't re-bill the same authorization with a different
    output_hash (audit C1) or replay it elsewhere (audit C7)."""
    return canonical_json({
        "request_hash": request_hash, "node": node, "model": model,
        "price_limit": price_limit, "auth_nonce": auth_nonce,
        "expiry": expiry, "chain_id": chain_id,
    })
