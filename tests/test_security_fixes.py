"""Regression tests for the security-audit hardening pass."""
from __future__ import annotations

import zipfile

import httpx
import pytest
from fastapi import FastAPI

from deltav.crypto import canonical_json, sha256_hex
from deltav.net.security import screen_url


# --------------------------------------------------------------- SSRF screen
async def test_screen_url_blocks_loopback_and_metadata():
    assert await screen_url("http://127.0.0.1:8085")            # loopback
    assert await screen_url("http://169.254.169.254/latest")    # cloud metadata
    assert await screen_url("http://[::1]:80")                  # ipv6 loopback
    assert await screen_url("http://0.0.0.0")                   # unspecified
    assert await screen_url("ftp://8.8.8.8")                    # non-http scheme


async def test_screen_url_allows_public_and_optionally_lan():
    assert await screen_url("http://8.8.8.8") == ""             # public ok
    # LAN is refused for the agent web-fetch (allow_private=False) ...
    assert await screen_url("http://10.0.0.223:9100")
    assert await screen_url("http://192.168.1.5:9100")
    # ... but allowed for node/peer reachability (LAN is legitimate there).
    assert await screen_url("http://10.0.0.223:9100", allow_private=True) == ""
    assert await screen_url("http://192.168.1.5:9100", allow_private=True) == ""


async def test_screen_url_port_restriction():
    assert await screen_url("http://8.8.8.8:22", allow_ports={80, 443})
    assert await screen_url("http://8.8.8.8:443", allow_ports={80, 443}) == ""


# ------------------------------------------------ spot-check job binding (C4)
# The reconstruction MUST byte-match how the serving handlers derive
# request_hash, or honest nodes get false-slashed. These pin that contract.
def test_request_hash_reconstruction_matches_handlers():
    from deltav.node.daemon import _request_hash_from_job

    chat = {"type": "chat", "prompt": "hi there", "model": "org/m::q.gguf",
            "max_tokens": 128, "temperature": 0.7, "seed": 3, "output_hash": "z"}
    assert _request_hash_from_job(chat) == sha256_hex(canonical_json(
        {"prompt": "hi there", "model": "org/m::q.gguf", "max_tokens": 128, "seed": 3}))

    stream = {"prompt": "hi there", "model": "org/m::q.gguf", "max_tokens": 128,
              "temperature": 0.7, "seed": 3, "output_hash": "z"}  # streaming: no "type"
    assert _request_hash_from_job(stream) == _request_hash_from_job(chat)

    embed = {"type": "embed", "texts": ["a", "b"], "model": "e", "output_hash": "z"}
    assert _request_hash_from_job(embed) == sha256_hex(canonical_json(
        {"texts": ["a", "b"], "model": "e"}))

    image = {"type": "image", "prompt": "p", "model": "d", "width": 512, "height": 640,
             "steps": 20, "seed": 1, "output_hash": "z"}
    assert _request_hash_from_job(image) == sha256_hex(canonical_json(
        {"prompt": "p", "model": "d", "w": 512, "h": 640, "steps": 20, "seed": 1}))


# ----------------------------------------------------------- calculator DoS
def test_calculator_rejects_power_bomb():
    from deltav.overlay.tools import calculate

    assert calculate("2 + 3 * 4") == "14"
    with pytest.raises(ValueError):
        calculate("9**9**9")
    with pytest.raises(ValueError):
        calculate("2**5000")


# ------------------------------------------------------------- mempool bound
def test_mempool_per_sender_cap(alice, bob):
    from deltav.chain import Mempool, Tx, TxType

    mp = Mempool()
    mp.MAX_PER_SENDER = 3
    made = 0
    for nonce in range(10):
        tx = Tx(type=TxType.TRANSFER.value, sender=alice.address, nonce=nonce,
                payload={"to": bob.address, "amount": 1}).sign(alice)
        if mp.add(tx):
            made += 1
    assert made == 3  # capped per sender
    # a different sender still gets in
    tx = Tx(type=TxType.TRANSFER.value, sender=bob.address, nonce=0,
            payload={"to": alice.address, "amount": 1}).sign(bob)
    assert mp.add(tx) is True


# ------------------------------------------------------------- wizard safety
def test_safe_under_rejects_traversal(tmp_path):
    from deltav.setup.wizard import _safe_under

    assert _safe_under(tmp_path, "model.gguf").parent == tmp_path.resolve()
    # directory components and traversal are stripped/rejected
    assert _safe_under(tmp_path, "../../etc/passwd").name == "passwd"
    assert _safe_under(tmp_path, "../../etc/passwd").parent == tmp_path.resolve()
    for bad in ("..", ".", "/"):
        with pytest.raises(ValueError):
            _safe_under(tmp_path, bad)


def test_safe_extract_rejects_zip_slip(tmp_path):
    from deltav.setup.wizard import _safe_extract

    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../escape.txt", "pwned")
    out = tmp_path / "out"
    out.mkdir()
    with zipfile.ZipFile(evil) as zf:
        with pytest.raises(ValueError):
            _safe_extract(zf, out)
    assert not (tmp_path / "escape.txt").exists()


# ------------------------------------------------------------- wallet safety
def test_wallet_load_or_create_is_idempotent(tmp_path):
    from deltav.wallet import load_or_create

    p = tmp_path / "w.json"
    a = load_or_create(p)
    b = load_or_create(p)          # second call must adopt the same key
    assert a.address == b.address


def test_key_store_capacity(tmp_path):
    from deltav.gateway.keys import KeyLimitError, KeyStore

    ks = KeyStore(tmp_path / "keys.json")
    ks_cap = 3
    from deltav.gateway import keys as keys_mod
    keys_mod.MAX_KEYS, orig = ks_cap, keys_mod.MAX_KEYS
    try:
        for _ in range(ks_cap):
            ks.create()
        with pytest.raises(KeyLimitError):
            ks.create()
    finally:
        keys_mod.MAX_KEYS = orig


# ---------------------------------------------------- light-client dup flag
def test_verify_charges_flags_replayed_authorization():
    from deltav.crypto import KeyPair
    from deltav.chain.transaction import receipt_auth_bytes, TxType
    from deltav.light import verify_charges

    payer = KeyPair.from_seed_hex("41" * 32)
    node = "dv1node"
    req_hash, model, price = "rh123", "m", 1000
    sig = payer.sign(receipt_auth_bytes(req_hash, node, model, price))

    def receipt(output_hash):
        return {"type": TxType.INFERENCE_RECEIPT.value, "sender": node, "payload": {
            "requester": payer.address, "requester_pubkey": payer.public_hex,
            "requester_sig": sig, "request_hash": req_hash, "output_hash": output_hash,
            "model": model, "price_limit": price, "tokens_in": 10, "tokens_out": 20}}

    # Same authorization, two different output_hashes → the second is a replay.
    blocks = [{"height": 1, "txs": [receipt("out1")]},
              {"height": 2, "txs": [receipt("out2")]}]
    res = verify_charges(blocks, payer.address)
    assert len(res.charges) == 2
    assert res.charges[0].authorized and not res.charges[0].duplicate
    assert res.charges[1].authorized and res.charges[1].duplicate
    assert res.all_authorized is False          # a replayed drain is not clean
    assert len(res.duplicate_charges) == 1
