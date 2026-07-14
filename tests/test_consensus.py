"""Consensus: proposer selection, block validation, fork choice."""
import pytest

from deltav.chain.blockchain import Blockchain
from deltav.chain.consensus import ConsensusError, expected_proposer
from deltav.chain.transaction import Tx, TxType
from deltav.config import DVT


def signed_transfer(kp, chain, to, amount):
    tx = Tx(type=TxType.TRANSFER.value, sender=kp.address,
            nonce=chain.state.account(kp.address).nonce,
            payload={"to": to, "amount": amount})
    return tx.sign(kp)


def test_proposer_is_deterministic_and_staked(genesis, alice, bob, carol):
    chain = Blockchain(genesis)
    p1 = expected_proposer(chain.state, 1, chain.head.hash)
    p2 = expected_proposer(chain.state, 1, chain.head.hash)
    assert p1 == p2
    assert p1 in (alice.address, bob.address)  # genesis validators only
    assert p1 != carol.address


def test_block_lifecycle(genesis, alice, bob, carol):
    chain = Blockchain(genesis)
    proposer_kp = {alice.address: alice, bob.address: bob}[chain.next_proposer()]
    tx = signed_transfer(carol, chain, alice.address, 7 * DVT)
    block = chain.build_block(proposer_kp, [tx], timestamp=1000.0)
    chain.add_block(block)
    assert chain.height == 1
    expected = 100_000 * DVT + 7 * DVT
    if proposer_kp.address == alice.address:
        expected += genesis.params.block_reward
    assert chain.state.account(alice.address).balance == expected
    assert chain.state.account(carol.address).balance == 100_000 * DVT - 7 * DVT


def test_wrong_proposer_rejected(genesis, alice, bob, carol):
    chain = Blockchain(genesis)
    expected = chain.next_proposer()
    wrong_kp = alice if expected != alice.address else bob
    block = chain.build_block(wrong_kp, [], timestamp=1000.0)
    with pytest.raises(ConsensusError, match="proposer"):
        chain.add_block(block)


def test_invalid_tx_dropped_at_build(genesis, alice, bob, carol):
    chain = Blockchain(genesis)
    proposer_kp = {alice.address: alice, bob.address: bob}[chain.next_proposer()]
    bad = signed_transfer(carol, chain, alice.address, 10**18)  # can't afford
    good = signed_transfer(carol, chain, alice.address, 1 * DVT)
    block = chain.build_block(proposer_kp, [bad, good], timestamp=1.0)
    assert len(block.txs) == 1
    chain.add_block(block)


def test_tampered_state_root_rejected(genesis, alice, bob):
    chain = Blockchain(genesis)
    proposer_kp = {alice.address: alice, bob.address: bob}[chain.next_proposer()]
    block = chain.build_block(proposer_kp, [], timestamp=1.0)
    block.state_root = "ff" * 32
    block.sign(proposer_kp)
    with pytest.raises(ConsensusError, match="state_root"):
        chain.add_block(block)


def _extend(chain, keys, n, t0=100.0):
    for i in range(n):
        kp = keys[chain.next_proposer()]
        block = chain.build_block(kp, [], timestamp=t0 + i)
        chain.add_block(block)


def test_fork_choice_longest_wins(genesis, alice, bob):
    keys = {alice.address: alice, bob.address: bob}
    ours = Blockchain(genesis)
    theirs = Blockchain(genesis)
    _extend(ours, keys, 2)
    _extend(theirs, keys, 5, t0=200.0)
    assert ours.replace([b.to_dict() for b in theirs.blocks])
    assert ours.height == 5
    # shorter chain is never adopted
    short = Blockchain(genesis)
    _extend(short, keys, 1, t0=300.0)
    assert not ours.replace([b.to_dict() for b in short.blocks])
    assert ours.height == 5


def test_fork_with_invalid_block_rejected(genesis, alice, bob):
    keys = {alice.address: alice, bob.address: bob}
    ours = Blockchain(genesis)
    theirs = Blockchain(genesis)
    _extend(theirs, keys, 4)
    dicts = [b.to_dict() for b in theirs.blocks]
    dicts[2]["txs"] = [{"bogus": True}]
    assert not ours.replace(dicts)
    assert ours.height == 0
