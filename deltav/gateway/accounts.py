"""User accounts on top of the custodial key store — a real personal cabinet.

A key (dvk_) is great for SDKs but is a bare secret with no way back if lost and
no human identity. An **account** wraps one custodial wallet with a
username+password so a person can register from the web, log in, watch their
usage/balance, pick a default model, rotate their API key and reset a forgotten
password — without ever pasting a raw key.

Design:
* one account -> one `KeyStore` record (the custodial wallet + usage counters),
  so web (session) and SDK (dvk_ key) share ONE balance and ONE usage roll-up.
* password + recovery code are stored only as salted PBKDF2 hashes; the raw
  api-key and recovery code are shown exactly once.
* sessions are opaque random tokens with an expiry, resolvable to the account's
  wallet so a logged-in browser can do inference billed to itself.

No email dependency: a forgotten password is reset with the recovery code shown
at signup (regenerated on each successful reset).
"""
from __future__ import annotations

import hmac
import json
import os
import re
import secrets
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..crypto import KeyPair, sha256_hex
from .keys import KeyStore, KeyRecord

_PBKDF2_ROUNDS = 200_000
_SESSION_TTL = 30 * 24 * 3600.0        # 30 days
_USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{2,31}$")
MAX_ACCOUNTS = 100_000


class AccountError(RuntimeError):
    """Bad registration/login input — carries a user-safe message."""


def _hash(secret: str, salt: str) -> str:
    import hashlib
    return hashlib.pbkdf2_hmac("sha256", secret.encode(), bytes.fromhex(salt),
                               _PBKDF2_ROUNDS).hex()


def _verify(secret: str, salt: str, expected: str) -> bool:
    return hmac.compare_digest(_hash(secret, salt), expected)


@dataclass
class Account:
    username: str
    pw_hash: str
    pw_salt: str
    key_hash: str            # -> KeyStore record (custodial wallet + usage)
    address: str
    recovery_hash: str
    recovery_salt: str
    created: float = 0.0
    model_pref: str = "auto"


class AccountStore:
    def __init__(self, path: str | Path | None, keystore: KeyStore):
        self.path = Path(path) if path else None
        self.keys = keystore
        self.accounts: dict[str, Account] = {}
        self.sessions: dict[str, tuple[str, float]] = {}   # token -> (user, expiry)
        if self.path is not None and self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.accounts = {u: Account(**a) for u, a in data.get("accounts", {}).items()}
            self.sessions = {t: tuple(v) for t, v in data.get("sessions", {}).items()}

    # ------------------------------------------------------------- persistence
    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.path.parent, 0o700)
        except OSError:
            pass
        self._prune_sessions()
        self.path.write_text(json.dumps({
            "accounts": {u: asdict(a) for u, a in self.accounts.items()},
            "sessions": {t: list(v) for t, v in self.sessions.items()},
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        try:
            os.chmod(self.path, 0o600)   # holds password hashes + session tokens
        except OSError:
            pass

    def _prune_sessions(self) -> None:
        now = time.time()
        self.sessions = {t: v for t, v in self.sessions.items() if v[1] > now}

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _check_username(username: str) -> str:
        u = (username or "").strip().lower()
        if not _USERNAME_RE.match(u):
            raise AccountError("username: 3–32 chars, a–z 0–9 . _ - , starting alphanumeric")
        return u

    @staticmethod
    def _check_password(password: str) -> None:
        if len(password or "") < 8:
            raise AccountError("password must be at least 8 characters")

    def record_of(self, account: Account) -> KeyRecord | None:
        return self.keys.records.get(account.key_hash)

    # --------------------------------------------------------------- register
    def register(self, username: str, password: str) -> tuple[Account, str, str]:
        """Create an account + its custodial wallet. Returns
        (account, api_key, recovery_code) — the last two shown exactly once."""
        u = self._check_username(username)
        self._check_password(password)
        if u in self.accounts:
            raise AccountError("that username is taken")
        if len(self.accounts) >= MAX_ACCOUNTS:
            raise AccountError("registration is temporarily closed")
        api_key, record = self.keys.create(label=f"account:{u}")
        recovery_code = secrets.token_hex(10)
        pw_salt, rec_salt = secrets.token_hex(16), secrets.token_hex(16)
        acct = Account(
            username=u, pw_hash=_hash(password, pw_salt), pw_salt=pw_salt,
            key_hash=record.key_hash, address=record.address,
            recovery_hash=_hash(recovery_code, rec_salt), recovery_salt=rec_salt,
            created=time.time())
        self.accounts[u] = acct
        self._save()
        return acct, api_key, recovery_code

    # ------------------------------------------------------------------ login
    def login(self, username: str, password: str) -> str:
        u = (username or "").strip().lower()
        acct = self.accounts.get(u)
        if acct is None or not _verify(password, acct.pw_salt, acct.pw_hash):
            raise AccountError("wrong username or password")
        return self._new_session(u)

    def _new_session(self, username: str) -> str:
        token = "dvs_" + secrets.token_urlsafe(32)
        self.sessions[token] = (username, time.time() + _SESSION_TTL)
        self._save()
        return token

    def resolve_session(self, token: str) -> Account | None:
        entry = self.sessions.get(token)
        if entry is None:
            return None
        username, expiry = entry
        if expiry <= time.time():
            self.sessions.pop(token, None)
            return None
        return self.accounts.get(username)

    def logout(self, token: str) -> None:
        if self.sessions.pop(token, None) is not None:
            self._save()

    # -------------------------------------------------------------- password
    def change_password(self, username: str, old: str, new: str) -> None:
        acct = self.accounts.get((username or "").strip().lower())
        if acct is None or not _verify(old, acct.pw_salt, acct.pw_hash):
            raise AccountError("current password is wrong")
        self._check_password(new)
        acct.pw_salt = secrets.token_hex(16)
        acct.pw_hash = _hash(new, acct.pw_salt)
        self._save()

    def reset_password(self, username: str, recovery_code: str, new: str) -> str:
        """Reset with the recovery code; returns a fresh recovery code (the used
        one is now spent). Also revokes existing sessions."""
        acct = self.accounts.get((username or "").strip().lower())
        if acct is None or not _verify(recovery_code, acct.recovery_salt, acct.recovery_hash):
            raise AccountError("wrong username or recovery code")
        self._check_password(new)
        acct.pw_salt = secrets.token_hex(16)
        acct.pw_hash = _hash(new, acct.pw_salt)
        new_code = secrets.token_hex(10)
        acct.recovery_salt = secrets.token_hex(16)
        acct.recovery_hash = _hash(new_code, acct.recovery_salt)
        self.sessions = {t: v for t, v in self.sessions.items() if v[0] != acct.username}
        self._save()
        return new_code

    # ----------------------------------------------------------- key rotation
    def rotate_key(self, account: Account) -> str:
        """Issue a fresh api-key on the SAME wallet (same balance/usage), drop
        the old key hash. Returns the new raw key (shown once)."""
        old = self.keys.records.pop(account.key_hash, None)
        api_key = self.keys.KEY_PREFIX + secrets.token_hex(24) \
            if hasattr(self.keys, "KEY_PREFIX") else "dvk_" + secrets.token_hex(24)
        new_hash = sha256_hex(api_key.encode())
        if old is not None:
            old.key_hash = new_hash
            self.keys.records[new_hash] = old
        account.key_hash = new_hash
        self.keys._save()
        self._save()
        return api_key

    # -------------------------------------------------------------- settings
    def set_model(self, account: Account, model: str) -> None:
        account.model_pref = (model or "auto").strip() or "auto"
        self._save()
