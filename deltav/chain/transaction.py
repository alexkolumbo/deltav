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


@dataclass
class Tx:
    type: str
    sender: str
    nonce: int
    payload: dict = field(default_factory=dict)
    fee: int = 0
    pubkey: str = ""
    signature: str = ""

    def signing_payload(self) -> dict:
        return {
            "type": self.type,
            "sender": self.sender,
            "nonce": self.nonce,
            "payload": self.payload,
            "fee": self.fee,
        }

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
        return {
            "type": self.type,
            "sender": self.sender,
            "nonce": self.nonce,
            "payload": self.payload,
            "fee": self.fee,
            "pubkey": self.pubkey,
            "signature": self.signature,
        }

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
        )


def receipt_auth_bytes(request_hash: str, node: str, model: str, price_limit: int) -> bytes:
    """What a requester signs to authorize paying `node` for one inference job."""
    return canonical_json(
        {"request_hash": request_hash, "node": node, "model": model, "price_limit": price_limit}
    )
