"""User accounts / personal-cabinet store: registration, login, sessions,
password change + recovery reset, key rotation, persistence."""
import time

import pytest

from deltav.gateway.keys import KeyStore
from deltav.gateway.accounts import AccountStore, AccountError


def _store(tmp_path):
    ks = KeyStore(tmp_path / "keys.json")
    return AccountStore(tmp_path / "accounts.json", ks), ks


def test_register_creates_wallet_and_secrets(tmp_path):
    acc, ks = _store(tmp_path)
    a, api_key, recovery = acc.register("Alice", "hunter2pw")
    assert a.username == "alice"                       # normalised lower
    assert api_key.startswith("dvk_") and len(recovery) == 20
    assert a.address.startswith("dv1")
    assert ks.resolve(api_key).address == a.address    # key -> same wallet
    assert a.pw_hash and "hunter2pw" not in a.pw_hash   # never stored plaintext


def test_username_and_password_rules(tmp_path):
    acc, _ = _store(tmp_path)
    for bad in ("ab", "Has Space", "-lead", "a" * 40, "юзер"):
        with pytest.raises(AccountError):
            acc.register(bad, "longenough")
    with pytest.raises(AccountError):
        acc.register("bob", "short")                    # < 8 chars
    acc.register("bob", "longenough")
    with pytest.raises(AccountError):
        acc.register("BOB", "anotherpw1")               # duplicate (case-insensitive)


def test_login_session_and_inference_wallet(tmp_path):
    acc, _ = _store(tmp_path)
    a, _, _ = acc.register("carol", "password1")
    with pytest.raises(AccountError):
        acc.login("carol", "wrongpass")
    token = acc.login("carol", "password1")
    assert token.startswith("dvs_")
    resolved = acc.resolve_session(token)
    assert resolved.username == "carol"
    # the session maps to the account's custodial wallet (so web pays as them)
    assert acc.record_of(resolved).address == a.address


def test_session_expiry_and_logout(tmp_path):
    acc, _ = _store(tmp_path)
    acc.register("dave", "password1")
    token = acc.login("dave", "password1")
    acc.sessions[token] = ("dave", time.time() - 1)     # force-expire
    assert acc.resolve_session(token) is None
    token2 = acc.login("dave", "password1")
    acc.logout(token2)
    assert acc.resolve_session(token2) is None


def test_change_password(tmp_path):
    acc, _ = _store(tmp_path)
    acc.register("erin", "password1")
    with pytest.raises(AccountError):
        acc.change_password("erin", "wrong", "newpassword1")
    acc.change_password("erin", "password1", "newpassword1")
    with pytest.raises(AccountError):
        acc.login("erin", "password1")
    assert acc.login("erin", "newpassword1").startswith("dvs_")


def test_reset_password_with_recovery_code(tmp_path):
    acc, _ = _store(tmp_path)
    a, _, recovery = acc.register("frank", "password1")
    old_session = acc.login("frank", "password1")
    with pytest.raises(AccountError):
        acc.reset_password("frank", "badcode", "newpassword1")
    new_code = acc.reset_password("frank", recovery, "newpassword1")
    assert new_code and new_code != recovery            # fresh code issued
    assert acc.resolve_session(old_session) is None     # old sessions revoked
    assert acc.login("frank", "newpassword1").startswith("dvs_")
    with pytest.raises(AccountError):                    # spent code can't reuse
        acc.reset_password("frank", recovery, "another1pw")


def test_rotate_key_keeps_wallet_and_usage(tmp_path):
    acc, ks = _store(tmp_path)
    a, old_key, _ = acc.register("grace", "password1")
    rec = acc.record_of(a)
    ks.record_usage(rec, tokens=100, spent_udvt=900)
    new_key = acc.rotate_key(a)
    assert new_key != old_key
    assert ks.resolve(old_key) is None                  # old key revoked
    r2 = ks.resolve(new_key)
    assert r2.address == a.address and r2.tokens == 100  # same wallet + usage


def test_set_model_and_persistence(tmp_path):
    acc, ks = _store(tmp_path)
    a, _, _ = acc.register("heidi", "password1")
    acc.set_model(a, "auto")
    acc.set_model(a, "org/repo::m.gguf")
    token = acc.login("heidi", "password1")
    # reload from disk -> account + session survive a gateway restart
    acc2 = AccountStore(tmp_path / "accounts.json", KeyStore(tmp_path / "keys.json"))
    assert "heidi" in acc2.accounts
    assert acc2.accounts["heidi"].model_pref == "org/repo::m.gguf"
    assert acc2.resolve_session(token).username == "heidi"
