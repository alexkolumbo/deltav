"""Wallet files: a keypair stored as JSON on disk."""
from __future__ import annotations

import json
from pathlib import Path

from .crypto import KeyPair

DEFAULT_DIR = Path.home() / ".deltav"


def wallet_path(name: str = "default") -> Path:
    return DEFAULT_DIR / f"{name}.wallet.json"


def save_wallet(keypair: KeyPair, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"address": keypair.address, "seed": keypair.seed_hex}, indent=2),
        encoding="utf-8",
    )
    return path


def load_wallet(path: str | Path) -> KeyPair:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return KeyPair.from_seed_hex(data["seed"])


def load_or_create(path: str | Path) -> KeyPair:
    path = Path(path)
    if path.exists():
        return load_wallet(path)
    keypair = KeyPair.generate()
    save_wallet(keypair, path)
    return keypair
