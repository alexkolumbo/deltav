"""Wallet files: a keypair stored as JSON on disk."""
from __future__ import annotations

import json
import os
from pathlib import Path

from .crypto import KeyPair

DEFAULT_DIR = Path.home() / ".deltav"


def wallet_path(name: str = "default") -> Path:
    return DEFAULT_DIR / f"{name}.wallet.json"


def _restrict(path: Path) -> None:
    """Best-effort owner-only perms on a file holding a private seed. A no-op
    for group/other on Windows (POSIX bits don't apply), effective on Linux."""
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def save_wallet(keypair: KeyPair, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    payload = json.dumps({"address": keypair.address, "seed": keypair.seed_hex}, indent=2)
    path.write_text(payload, encoding="utf-8")
    _restrict(path)
    return path


def load_wallet(path: str | Path) -> KeyPair:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return KeyPair.from_seed_hex(data["seed"])


def load_or_create(path: str | Path) -> KeyPair:
    """Load an existing wallet, or atomically create one. Uses O_CREAT|O_EXCL
    so two processes racing on the same path can't both generate and clobber
    each other's key."""
    path = Path(path)
    if path.exists():
        return load_wallet(path)
    keypair = KeyPair.generate()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    payload = json.dumps({"address": keypair.address, "seed": keypair.seed_hex}, indent=2)
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return load_wallet(path)  # lost the race — adopt the winner's wallet
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(payload)
    _restrict(path)
    return keypair
