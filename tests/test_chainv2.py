"""chain-v2 reward mechanism: escrow fee (settle-once) + epoch-deferred
emission + commitment-backed spot-check + dispute/clawback + availability
lease + determinism-from-backend + chain_id binding.

Every test also implicitly asserts v1 (version=1, alpha-3) is unchanged by
running the same primitives under version=1 where relevant.
"""
from __future__ import annotations

import pytest

from deltav.config import ChainParams, DVT
from deltav.crypto import KeyPair
from deltav.chain.state import State, NodeInfo, StateError
from deltav.chain.transaction import Tx, TxType, receipt_auth_bytes, receipt_auth_bytes_v2

CID = "deltav-v2-test"


def _params(**kw):
    base = dict(chain_id=CID, version=2, min_checkers=2, dispute_window=50,
               price_per_token=10, min_validator_stake=1_000 * DVT, pool_fee_bps=1000,
               inference_reward=DVT // 10, epoch_blocks=10, dev_fund="")
    base.update(kw)
    return ChainParams(**base)


def _keys():
    return (KeyPair.from_seed_hex("11" * 32), KeyPair.from_seed_hex("22" * 32),
            KeyPair.from_seed_hex("33" * 32), KeyPair.from_seed_hex("44" * 32))


def _state(params, client, node, v1, v2, backend="llamaserver", node_stake=0):
    st = State(params)
    st.account(client.address).balance = 1_000 * DVT
    st.account(v1.address).stake = 2_000 * DVT
    st.account(v2.address).stake = 2_000 * DVT
    if node_stake:
        st.account(node.address).stake = node_stake
    st.nodes[node.address] = NodeInfo(address=node.address, endpoint="http://n",
                                      hardware={"backend": backend})
    return st


def _tx(kp, ttype, payload, nonce, chain_id=CID):
    return Tx(type=ttype.value, sender=kp.address, nonce=nonce,
              payload=payload, chain_id=chain_id).sign(kp)


def _receipt_tx(node, client, nonce, auth_nonce, tokens=(20, 80), price_limit=1000,
                out="o", model="m", req="rh", chain_id=CID):
    auth = receipt_auth_bytes_v2(req, node.address, model, price_limit, auth_nonce, 0, chain_id)
    payload = dict(requester=client.address, requester_pubkey=client.public_hex,
                   requester_sig=client.sign(auth), request_hash=req, output_hash=out,
                   model=model, seed=0, tokens_in=tokens[0], tokens_out=tokens[1],
                   price_limit=price_limit, auth_nonce=auth_nonce, expiry=0)
    return _tx(node, TxType.INFERENCE_RECEIPT, payload, nonce, chain_id)


# ----------------------------------------------------- C1: settle exactly once
def test_v2_authorization_settles_once():
    client, node, v1, v2 = _keys()
    st = _state(_params(), client, node, v1, v2)
    st.apply_tx(_receipt_tx(node, client, 0, "an1", out="out1"), 1)
    assert st.nodes[node.address].pending_emission == DVT // 10   # emission deferred
    assert st.account(node.address).balance == 900               # fee settled (1000 - 10% pool)
    # Same auth_nonce, different output_hash → must be rejected (no re-billing).
    with pytest.raises(StateError, match="already settled"):
        st.apply_tx(_receipt_tx(node, client, 1, "an1", out="out2"), 2)


# --------------------------------------------------------- C6: no self-dealing
def test_v2_rejects_self_dealing_receipt():
    client, node, v1, v2 = _keys()
    st = _state(_params(), client, node, v1, v2)
    auth = receipt_auth_bytes_v2("rh", node.address, "m", 1000, "an", 0, CID)
    payload = dict(requester=node.address, requester_pubkey=node.public_hex,
                   requester_sig=node.sign(auth), request_hash="rh", output_hash="o",
                   model="m", seed=0, tokens_in=20, tokens_out=80, price_limit=1000,
                   auth_nonce="an", expiry=0)
    st.account(node.address).balance = 10 * DVT
    with pytest.raises(StateError, match="must differ"):
        st.apply_tx(_tx(node, TxType.INFERENCE_RECEIPT, payload, 0), 1)


# --------------------------------------------------- C7: chain-id replay guard
def test_v2_rejects_foreign_chain_id():
    client, node, v1, v2 = _keys()
    st = _state(_params(), client, node, v1, v2)
    with pytest.raises(StateError, match="not bound to this chain"):
        st.apply_tx(_tx(client, TxType.TRANSFER, {"to": node.address, "amount": 1},
                        0, chain_id="other-chain"), 1)


# ------------------------------------------ C5: determinism from node backend
def test_v2_determinism_taken_from_backend_not_payload():
    client, node, v1, v2 = _keys()
    st = _state(_params(), client, node, v1, v2, backend="llamaserver")
    rt = _receipt_tx(node, client, 0, "an")
    rt.payload["deterministic"] = False       # node lies in the payload
    rt.sign(node)
    st.apply_tx(rt, 1)
    r = next(iter(st.receipts.values()))
    assert r.deterministic is True            # llamaserver is a deterministic backend
    # a non-deterministic backend (e.g. an API relay) is respected
    st2 = _state(_params(), client, node, v1, v2, backend="groq")
    st2.apply_tx(_receipt_tx(node, client, 0, "an"), 1)
    assert next(iter(st2.receipts.values())).deterministic is False


# --------------------------------- C3: commitment + min_checkers before slash
def test_v2_slash_needs_min_checkers_and_commitment():
    client, node, v1, v2 = _keys()
    st = _state(_params(min_checkers=2), client, node, v1, v2, node_stake=2_000 * DVT)
    st.apply_tx(_receipt_tx(node, client, 0, "an"), 1)
    r = next(iter(st.receipts.values()))
    # a verdict with no commitment is rejected
    with pytest.raises(StateError, match="commitment"):
        st.apply_tx(_tx(v1, TxType.SPOT_CHECK, {"receipt_hash": r.receipt_hash, "ok": False}, 0), 2)
    stake0 = st.account(node.address).stake
    # one fail verdict alone must NOT slash (needs 2)
    st.apply_tx(_tx(v1, TxType.SPOT_CHECK,
                    {"receipt_hash": r.receipt_hash, "ok": False, "commitment": "c"}, 0), 3)
    assert st.account(node.address).stake == stake0 and not r.settled
    # same checker can't double-verdict
    with pytest.raises(StateError, match="already verdicted"):
        st.apply_tx(_tx(v1, TxType.SPOT_CHECK,
                        {"receipt_hash": r.receipt_hash, "ok": False, "commitment": "c"}, 1), 4)
    # a second independent fail crosses min_checkers → slash + clawback
    st.apply_tx(_tx(v2, TxType.SPOT_CHECK,
                    {"receipt_hash": r.receipt_hash, "ok": False, "commitment": "c"}, 0), 5)
    assert r.settled and r.check_ok is False
    assert st.account(node.address).stake < stake0


def test_v2_two_ok_checks_verify_receipt():
    client, node, v1, v2 = _keys()
    st = _state(_params(min_checkers=2), client, node, v1, v2)
    st.apply_tx(_receipt_tx(node, client, 0, "an"), 1)
    r = next(iter(st.receipts.values()))
    st.apply_tx(_tx(v1, TxType.SPOT_CHECK,
                    {"receipt_hash": r.receipt_hash, "ok": True, "commitment": "o"}, 0), 2)
    assert not r.checked
    st.apply_tx(_tx(v2, TxType.SPOT_CHECK,
                    {"receipt_hash": r.receipt_hash, "ok": True, "commitment": "o"}, 0), 3)
    assert r.checked and r.check_ok is True


# ------------------------------------------------ dispute → clawback + slash
def test_v2_dispute_makes_one_fail_slash_and_full_clawback():
    client, node, v1, v2 = _keys()
    st = _state(_params(), client, node, v1, v2, node_stake=2_000 * DVT)
    st.apply_tx(_receipt_tx(node, client, 0, "an"), 1)
    r = next(iter(st.receipts.values()))
    before = st.account(client.address).balance
    st.apply_tx(_tx(client, TxType.DISPUTE, {"receipt_hash": r.receipt_hash}, 0), 5)
    st.apply_tx(_tx(v1, TxType.SPOT_CHECK,
                    {"receipt_hash": r.receipt_hash, "ok": False, "commitment": "real"}, 0), 6)
    assert r.settled and r.check_ok is False
    assert st.account(client.address).balance - before == r.price_paid   # full refund
    assert st.nodes[node.address].pending_emission == 0                  # emission reversed


def test_v2_dispute_only_by_client_and_within_window():
    client, node, v1, v2 = _keys()
    st = _state(_params(dispute_window=5), client, node, v1, v2)
    st.apply_tx(_receipt_tx(node, client, 0, "an"), 1)
    r = next(iter(st.receipts.values()))
    with pytest.raises(StateError, match="only the paying client"):
        st.apply_tx(_tx(v1, TxType.DISPUTE, {"receipt_hash": r.receipt_hash}, 0), 2)
    st.height = 100
    with pytest.raises(StateError, match="window has closed"):
        st.apply_tx(_tx(client, TxType.DISPUTE, {"receipt_hash": r.receipt_hash}, 0), 100)


# ---------------------------------------------------------- availability lease
def test_v2_availability_lease():
    client, node, v1, v2 = _keys()
    # lease_until = height + blocks
    st2 = _state(_params(lease_blocks=40), client, node, v1, v2)
    st2.apply_tx(_tx(node, TxType.AVAILABILITY_LEASE, {"blocks": 40}, 0), 8)
    assert st2.nodes[node.address].lease_until == 48
    # lease length is capped at lease_blocks
    st3 = _state(_params(lease_blocks=40), client, node, v1, v2)
    st3.apply_tx(_tx(node, TxType.AVAILABILITY_LEASE, {"blocks": 9999}, 0), 0)
    assert st3.nodes[node.address].lease_until == 40


# --------------------------------------------- epoch settlement mints emission
def test_v2_emission_minted_at_epoch_only():
    client, node, v1, v2 = _keys()
    st = _state(_params(epoch_blocks=10), client, node, v1, v2)
    st.apply_tx(_receipt_tx(node, client, 0, "an"), 1)
    node_bal = st.account(node.address).balance
    assert st.nodes[node.address].pending_emission == DVT // 10
    # a non-epoch block does not mint
    st.apply_block_effects(proposer=v1.address, fees=0, missed=[], height=5)
    assert st.account(node.address).balance == node_bal + 0  # only proposer got block reward; node unchanged
    # the epoch block mints the deferred emission
    before = st.account(node.address).balance
    st.apply_block_effects(proposer=v1.address, fees=0, missed=[], height=10)
    assert st.account(node.address).balance - before >= DVT // 10
    assert st.nodes[node.address].pending_emission == 0


# ============================ v1 unchanged ============================
def test_v1_mints_emission_at_receipt_and_single_check_latches():
    client, node, v1, v2 = _keys()
    params = ChainParams(chain_id="alpha", version=1, price_per_token=10,
                         min_validator_stake=1_000 * DVT, pool_fee_bps=1000,
                         inference_reward=DVT // 10)
    st = State(params)
    st.account(client.address).balance = 1_000 * DVT
    st.account(v1.address).stake = 2_000 * DVT
    st.nodes[node.address] = NodeInfo(address=node.address, endpoint="http://n",
                                      hardware={"backend": "llamaserver"})
    # v1 receipt: no auth_nonce, v1 auth bytes, no chain_id
    auth = receipt_auth_bytes("rh", node.address, "m", 1000)
    payload = dict(requester=client.address, requester_pubkey=client.public_hex,
                   requester_sig=client.sign(auth), request_hash="rh", output_hash="o",
                   model="m", seed=0, tokens_in=20, tokens_out=80, price_limit=1000)
    st.apply_tx(Tx(type=TxType.INFERENCE_RECEIPT.value, sender=node.address, nonce=0,
                   payload=payload).sign(node), 1)
    # v1 mints emission immediately (no pending)
    assert st.account(node.address).balance == 900 + DVT // 10
    assert st.nodes[node.address].pending_emission == 0
    # v1 single check latches immediately (no commitment/min_checkers)
    r = next(iter(st.receipts.values()))
    st.apply_tx(Tx(type=TxType.SPOT_CHECK.value, sender=v1.address, nonce=0,
                   payload={"receipt_hash": r.receipt_hash, "ok": True}).sign(v1), 2)
    assert r.checked and r.check_ok is True
    # And critically: the v1 state root must NOT carry the chain-v2 node/receipt
    # fields, or a fresh node re-validating from genesis hits 'state_root
    # mismatch' at the first register/receipt block and can never sync.
    ndict = st.to_dict()["nodes"][node.address]
    assert "pending_emission" not in ndict and "lease_until" not in ndict
    rdict = st.to_dict()["receipts"][r.receipt_hash]
    for k in ("settle_key", "emission", "ok_checks", "fail_checks",
              "checked_by", "disputed", "dispute_deadline", "settled"):
        assert k not in rdict, f"v2 field {k} leaked into the v1 receipt root"
    # non-v2 fields still serialize as before (reputation moved after the check)
    assert "reputation" in ndict and rdict["deterministic"] is True


def test_state_root_backward_compat_omits_default_v2_fields():
    """Direct guard on the serializer: v2 fields vanish at their default and
    reappear once set, keeping historical (version=1) roots byte-stable while
    staying deterministic for version=2."""
    from deltav.chain.state import Receipt

    st = State(_params(version=1))
    a = "dv1" + "0" * 40
    st.nodes[a] = NodeInfo(address=a, endpoint="http://n")          # defaults
    st.nodes["dv1" + "1" * 40] = NodeInfo(address="dv1" + "1" * 40, endpoint="http://m",
                                          pending_emission=7, lease_until=42)  # set
    default_node = st.to_dict()["nodes"][a]
    set_node = st.to_dict()["nodes"]["dv1" + "1" * 40]
    assert "pending_emission" not in default_node and "lease_until" not in default_node
    assert set_node["pending_emission"] == 7 and set_node["lease_until"] == 42

    st.receipts["h"] = Receipt(receipt_hash="h", node=a, requester="r", model="m",
                               request_hash="rh", output_hash="o", seed=0,
                               tokens_in=1, tokens_out=1, price_paid=1, height=1)
    rd = st.to_dict()["receipts"]["h"]
    assert not ({"settle_key", "emission", "ok_checks", "fail_checks", "checked_by",
                 "disputed", "dispute_deadline", "settled"} & set(rd))


def test_account_ro_read_does_not_pollute_state_root():
    """A read of a not-yet-funded account (a status endpoint, the node's own
    nonce during registration, mempool pruning) must NOT insert it. account()
    auto-creates; on a node whose address isn't in genesis that adds a phantom
    empty account to the LIVE chain state mid-sync, so validate_block sees an
    extra account and every block fails 'state_root mismatch' — the node can
    never sync. account_ro() is the read-only path that avoids it."""
    st = State(_params(version=1))
    root0 = st.state_root()
    ghost = "dv1" + "e" * 40
    acc = st.account_ro(ghost)
    assert acc.balance == 0 and acc.nonce == 0
    assert ghost not in st.accounts and st.state_root() == root0   # untouched
    # the mutating accessor (only for apply_tx) DOES insert and shifts the root
    st.account(ghost)
    assert ghost in st.accounts and st.state_root() != root0
