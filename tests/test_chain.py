"""State machine rules: transfers, staking, node registry, receipts, slashing."""
import pytest

from deltav.chain.blockchain import build_genesis_state
from deltav.chain.state import StateError
from deltav.chain.transaction import Tx, TxType, receipt_auth_bytes
from deltav.config import DVT
from deltav.crypto import canonical_json, sha256_hex


def make_tx(kp, state, tx_type, payload, fee=0):
    tx = Tx(type=tx_type.value, sender=kp.address,
            nonce=state.account(kp.address).nonce, payload=payload, fee=fee)
    return tx.sign(kp)


def register_node(state, kp, models=None, endpoint="http://n:1"):
    tx = make_tx(kp, state, TxType.REGISTER_NODE, {
        "endpoint": endpoint,
        "hardware": {"vendor": "nvidia", "vram_mb": 12282},
        "models": models or ["m"],
    })
    state.apply_tx(tx, 1)


def make_receipt_tx(node_kp, requester_kp, state, *, tokens_in=10, tokens_out=20,
                    price_limit=None, model="m", request_hash=None, output_hash="out" * 10):
    request_hash = request_hash or sha256_hex(b"job1")
    price_limit = price_limit if price_limit is not None else 100_000
    auth = receipt_auth_bytes(request_hash, node_kp.address, model, price_limit)
    return make_tx(node_kp, state, TxType.INFERENCE_RECEIPT, {
        "requester": requester_kp.address,
        "requester_pubkey": requester_kp.public_hex,
        "requester_sig": requester_kp.sign(auth),
        "request_hash": request_hash,
        "output_hash": output_hash,
        "model": model,
        "seed": 0,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "price_limit": price_limit,
    })


def test_transfer(genesis, alice, bob):
    state = build_genesis_state(genesis)
    tx = make_tx(alice, state, TxType.TRANSFER, {"to": bob.address, "amount": 5 * DVT}, fee=100)
    state.apply_tx(tx, 1)
    assert state.account(bob.address).balance == 100_000 * DVT + 5 * DVT
    assert state.account(alice.address).balance == 100_000 * DVT - 5 * DVT - 100
    assert state.account(alice.address).nonce == 1


def test_transfer_insufficient(genesis, alice, bob):
    state = build_genesis_state(genesis)
    tx = make_tx(alice, state, TxType.TRANSFER, {"to": bob.address, "amount": 10**18})
    with pytest.raises(StateError, match="insufficient"):
        state.apply_tx(tx, 1)


def test_bad_nonce_rejected(genesis, alice, bob):
    state = build_genesis_state(genesis)
    tx = Tx(type=TxType.TRANSFER.value, sender=alice.address, nonce=5,
            payload={"to": bob.address, "amount": 1}).sign(alice)
    with pytest.raises(StateError, match="nonce"):
        state.apply_tx(tx, 1)


def test_tampered_signature_rejected(genesis, alice, bob):
    state = build_genesis_state(genesis)
    tx = make_tx(alice, state, TxType.TRANSFER, {"to": bob.address, "amount": 1 * DVT})
    tx.payload["amount"] = 50_000 * DVT  # tamper after signing
    with pytest.raises(StateError, match="signature"):
        state.apply_tx(tx, 1)


def test_stake_unstake(genesis, carol):
    state = build_genesis_state(genesis)
    state.apply_tx(make_tx(carol, state, TxType.STAKE, {"amount": 2_000 * DVT}), 1)
    assert state.account(carol.address).stake == 2_000 * DVT
    assert (carol.address, 2_000 * DVT) in state.validators()
    state.apply_tx(make_tx(carol, state, TxType.UNSTAKE, {"amount": 1_500 * DVT}), 2)
    assert state.account(carol.address).stake == 500 * DVT
    assert all(addr != carol.address for addr, _ in state.validators())


def test_register_and_announce(genesis, alice):
    state = build_genesis_state(genesis)
    register_node(state, alice, models=["model-a"])
    assert state.nodes[alice.address].models == ["model-a"]
    tx = make_tx(alice, state, TxType.ANNOUNCE_MODELS, {"models": ["model-b", "model-c"]})
    state.apply_tx(tx, 2)
    assert state.nodes[alice.address].models == ["model-b", "model-c"]


def test_receipt_pays_node(genesis, alice, bob):
    state = build_genesis_state(genesis)
    register_node(state, bob)
    tx = make_receipt_tx(bob, alice, state, tokens_in=10, tokens_out=20)
    state.apply_tx(tx, 2)
    price = 30 * state.params.price_per_token
    assert state.account(alice.address).balance == 100_000 * DVT - price
    assert state.account(bob.address).balance == 100_000 * DVT + price + state.params.inference_reward
    assert len(state.receipts) == 1
    node = state.nodes[bob.address]
    assert node.jobs_done == 1 and node.tokens_served == 30


def test_receipt_needs_requester_authorization(genesis, alice, bob, carol):
    state = build_genesis_state(genesis)
    register_node(state, bob)
    tx = make_receipt_tx(bob, alice, state)
    # bob tries to bill carol instead of alice using alice's signature
    tx.payload["requester"] = carol.address
    tx.payload["requester_pubkey"] = carol.public_hex
    tx.signature = ""  # re-sign the outer tx with the tampered payload
    tx.sign(bob)
    with pytest.raises(StateError, match="authorization|pubkey"):
        state.apply_tx(tx, 2)


def test_receipt_price_cap(genesis, alice, bob):
    state = build_genesis_state(genesis)
    register_node(state, bob)
    tx = make_receipt_tx(bob, alice, state, tokens_in=10_000, tokens_out=10_000, price_limit=100)
    with pytest.raises(StateError, match="price exceeds"):
        state.apply_tx(tx, 2)


def test_duplicate_receipt_rejected(genesis, alice, bob):
    state = build_genesis_state(genesis)
    register_node(state, bob)
    state.apply_tx(make_receipt_tx(bob, alice, state), 2)
    with pytest.raises(StateError, match="duplicate"):
        state.apply_tx(make_receipt_tx(bob, alice, state), 2)


def test_spot_check_ok_and_slash(genesis, alice, bob):
    state = build_genesis_state(genesis)
    register_node(state, bob)
    state.apply_tx(make_receipt_tx(bob, alice, state), 2)
    receipt_hash = next(iter(state.receipts))
    rep_before = state.nodes[bob.address].reputation
    stake_before = state.account(bob.address).stake

    # alice is a genesis validator — honest verdict
    ok_tx = make_tx(alice, state, TxType.SPOT_CHECK, {"receipt_hash": receipt_hash, "ok": True})
    state.apply_tx(ok_tx, 3)
    assert state.receipts[receipt_hash].checked and state.receipts[receipt_hash].check_ok
    assert state.nodes[bob.address].reputation > rep_before
    assert state.account(bob.address).stake == stake_before

    # second receipt gets a failing verdict -> slash
    state.apply_tx(make_receipt_tx(bob, alice, state, request_hash=sha256_hex(b"job2")), 3)
    bad_hash = next(h for h, r in state.receipts.items() if not r.checked)
    bad_tx = make_tx(alice, state, TxType.SPOT_CHECK, {"receipt_hash": bad_hash, "ok": False})
    state.apply_tx(bad_tx, 4)
    expected_slash = int(stake_before * state.params.slash_fraction)
    assert state.account(bob.address).stake == stake_before - expected_slash
    assert state.nodes[bob.address].reputation < rep_before


def test_spot_check_requires_stake(genesis, alice, bob, carol):
    state = build_genesis_state(genesis)
    register_node(state, bob)
    state.apply_tx(make_receipt_tx(bob, alice, state), 2)
    receipt_hash = next(iter(state.receipts))
    tx = make_tx(carol, state, TxType.SPOT_CHECK, {"receipt_hash": receipt_hash, "ok": False})
    with pytest.raises(StateError, match="validator stake"):
        state.apply_tx(tx, 3)


def test_node_cannot_check_itself(genesis, alice, bob):
    state = build_genesis_state(genesis)
    register_node(state, bob)
    state.apply_tx(make_receipt_tx(bob, alice, state), 2)
    receipt_hash = next(iter(state.receipts))
    tx = make_tx(bob, state, TxType.SPOT_CHECK, {"receipt_hash": receipt_hash, "ok": True})
    with pytest.raises(StateError, match="own receipt"):
        state.apply_tx(tx, 3)


def test_state_root_deterministic(genesis):
    s1 = build_genesis_state(genesis)
    s2 = build_genesis_state(genesis)
    assert s1.state_root() == s2.state_root()
    s2.account("dv1someone").balance += 1
    assert s1.state_root() != s2.state_root()
