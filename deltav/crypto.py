"""Ed25519 keys, addresses and canonical hashing for the Delta V chain."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

ADDRESS_PREFIX = "dv1"


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(obj: Any) -> bytes:
    """Deterministic JSON encoding used for every hash and signature."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def address_from_public(public_hex: str) -> str:
    return ADDRESS_PREFIX + sha256_hex(bytes.fromhex(public_hex))[:40]


class KeyPair:
    """An Ed25519 keypair identified on-chain by a dv1... address."""

    def __init__(self, private: Ed25519PrivateKey):
        self._private = private
        self._public = private.public_key()

    @classmethod
    def generate(cls) -> "KeyPair":
        return cls(Ed25519PrivateKey.generate())

    @classmethod
    def from_seed_hex(cls, seed_hex: str) -> "KeyPair":
        return cls(Ed25519PrivateKey.from_private_bytes(bytes.fromhex(seed_hex)))

    @property
    def seed_hex(self) -> str:
        raw = self._private.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        )
        return raw.hex()

    @property
    def public_hex(self) -> str:
        raw = self._public.public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
        return raw.hex()

    @property
    def address(self) -> str:
        return address_from_public(self.public_hex)

    def sign(self, message: bytes) -> str:
        return self._private.sign(message).hex()


def verify_signature(public_hex: str, message: bytes, signature_hex: str) -> bool:
    try:
        public = Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_hex))
        public.verify(bytes.fromhex(signature_hex), message)
        return True
    except (InvalidSignature, ValueError):
        return False
