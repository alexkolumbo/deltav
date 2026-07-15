"""Phase 2: liveness slots + jailing, unbonding, persistence,
incremental sync, fuzzy verification, peer discovery."""
import asyncio

import httpx
import pytest

from deltav.chain.blockchain import Blockchain, build_genesis_state
from deltav.chain.consensus import ConsensusError, expected_proposer
from deltav.chain.transaction import Tx, TxType
from deltav.compute.base import DeviceInfo
from deltav.config import DVT, ChainParams, Genesis
from deltav.crypto import KeyPair
from deltav.node import NodeConfig, NodeDaemon
from deltav.node.verify import spot_check_verdict

from conftest import MultiTransport


def keys_map(*kps):
    return {kp.address: kp for kp in kps}


def produce(chain, keys, *, exclude=None, timestamp=100.0):
    """Produce the next block with the lowest-slot proposer not in `exclude`."""
    for slot in range(chain.state.params.max_slots):
        proposer = chain.next_proposer(slot=slot)
        if proposer is None:
            raise AssertionError("no validators")
        if exclude and proposer in exclude:
            continue
        block = chain.build_block(keys[proposer], [], timestamp, slot=slot)
        chain.add_block(block)
        return block
    raise AssertionError("no usable slot found")


# ------------------------------------------------------------- liveness

def test_fallback_slot_accepted_and_miss_recorded(genesis, alice, bob):
    keys = keys_map(alice, bob)
    chain = Blockchain(genesis)
    # skip the primary proposer until a height where slot0 != slot1 proposer
    for _ in range(30):
        primary = chain.next_proposer(slot=0)
        backup = chain.next_proposer(slot=1)
        if primary != backup:
            break
        produce(chain, keys)
    else:
        pytest.skip("validators never disagreed across slots")
    block = chain.build_block(keys[backup], [], 100.0, slot=1)
    chain.add_block(block)
    assert chain.head.slot == 1
    assert chain.state.account(primary).misses == 1
    # proposing later resets the miss counter
    for _ in range(30):
        if chain.next_proposer(slot=0) == primary:
            produce(chain, keys)
            break
        produce(chain, keys)
    assert chain.state.account(primary).misses == 0


def test_wrong_slot_proposer_rejected(genesis, alice, bob):
    keys = keys_map(alice, bob)
    chain = Blockchain(genesis)
    for _ in range(30):
        primary = chain.next_proposer(slot=0)
        backup = chain.next_proposer(slot=1)
        if primary != backup:
            break
        produce(chain, keys)
    else:
        raise AssertionError("validators never disagreed across slots")
    # backup signs a block claiming slot 0
    block = chain.build_block(keys[backup], [], 100.0, slot=0)
    with pytest.raises(ConsensusError, match="wrong proposer"):
        chain.add_block(block)


def test_repeated_misses_jail_validator(alice, bob):
    params = ChainParams(block_time=0.05, jail_after_misses=2, jail_blocks=5)
    genesis = Genesis(
        params=params,
        alloc={alice.address: 100_000 * DVT, bob.address: 100_000 * DVT},
        stakes={alice.address: 10_000 * DVT, bob.address: 10_000 * DVT},
    )
    keys = keys_map(alice, bob)
    chain = Blockchain(genesis)
    victim = alice.address

    # keep producing while never letting the victim propose
    for _ in range(200):
        if chain.state.account(victim).jailed_until > chain.height:
            break
        produce(chain, keys, exclude={victim})
    else:
        raise AssertionError("victim never got jailed")

    jailed_until = chain.state.account(victim).jailed_until
    assert all(addr != victim for addr, _ in chain.state.validators())

    # after the jail period the validator is eligible again
    while chain.height < jailed_until:
        produce(chain, keys, exclude={victim})
    assert any(addr == victim for addr, _ in chain.state.validators())


def test_sibling_lower_slot_wins(genesis, alice, bob):
    keys = keys_map(alice, bob)
    chain = Blockchain(genesis)
    for _ in range(30):
        primary = chain.next_proposer(slot=0)
        backup = chain.next_proposer(slot=1)
        if primary != backup:
            break
        produce(chain, keys)
    else:
        raise AssertionError("validators never disagreed across slots")

    slot0_block = chain.build_block(keys[primary], [], 100.0, slot=0)
    slot1_block = chain.build_block(keys[backup], [], 101.0, slot=1)

    # the fallback block lands first...
    chain.add_block(slot1_block)
    assert chain.head.slot == 1
    # ...but the primary's block deterministically takes over
    assert chain.replace_sibling(slot0_block)
    assert chain.head.slot == 0 and chain.head.proposer == primary
    # no bogus miss is recorded for the primary on the winning branch
    assert chain.state.account(primary).misses == 0
    # and the reverse never happens: a worse sibling cannot displace the head
    assert not chain.replace_sibling(slot1_block)


# ------------------------------------------------------------ unbonding

def make_tx(kp, state, tx_type, payload):
    tx = Tx(type=tx_type.value, sender=kp.address,
            nonce=state.account(kp.address).nonce, payload=payload)
    return tx.sign(kp)


def test_unstake_goes_through_unbonding(genesis, alice, bob, carol):
    keys = keys_map(alice, bob)
    chain = Blockchain(genesis)
    unbonding_blocks = genesis.params.unbonding_blocks

    tx = make_tx(carol, chain.state, TxType.STAKE, {"amount": 5_000 * DVT})
    chain.add_block(chain.build_block(keys[chain.next_proposer()], [tx], 1.0))
    balance_after_stake = chain.state.account(carol.address).balance

    tx = make_tx(carol, chain.state, TxType.UNSTAKE, {"amount": 5_000 * DVT})
    chain.add_block(chain.build_block(keys[chain.next_proposer()], [tx], 2.0))
    acc = chain.state.account(carol.address)
    assert acc.stake == 0
    assert acc.balance == balance_after_stake          # NOT released yet
    assert sum(u["amount"] for u in acc.unbonding) == 5_000 * DVT
    release = acc.unbonding[0]["release_height"]
    assert release == chain.height + unbonding_blocks

    while chain.height < release:
        produce(chain, keys)
    acc = chain.state.account(carol.address)
    assert not acc.unbonding
    assert acc.balance == balance_after_stake + 5_000 * DVT


def test_slash_reaches_unbonding_funds(genesis, alice, bob):
    state = build_genesis_state(genesis)
    # bob unstakes everything, then gets caught lying before release
    state.apply_tx(make_tx(bob, state, TxType.UNSTAKE, {"amount": 10_000 * DVT}), 1)
    assert state.account(bob.address).stake == 0
    supply_before = state.supply
    burned = state.slash(bob.address, 0.10)
    assert burned == 1_000 * DVT                      # 10% of the unbonding total
    assert sum(u["amount"] for u in state.account(bob.address).unbonding) == 9_000 * DVT
    assert state.supply == supply_before - burned


# ------------------------------------------------------ fuzzy verification

def test_exact_verdict_for_deterministic_backends():
    assert spot_check_verdict(
        receipt_deterministic=True, checker_deterministic=True,
        receipt_output_hash="abc", recomputed_output_hash="abc",
        receipt_tokens_out=100, recomputed_tokens_out=100)
    assert not spot_check_verdict(
        receipt_deterministic=True, checker_deterministic=True,
        receipt_output_hash="abc", recomputed_output_hash="def",
        receipt_tokens_out=100, recomputed_tokens_out=100)


def test_fuzzy_verdict_for_gpu_backends():
    # hashes differ (non-deterministic sampling) but token counts agree -> ok
    assert spot_check_verdict(
        receipt_deterministic=False, checker_deterministic=True,
        receipt_output_hash="abc", recomputed_output_hash="def",
        receipt_tokens_out=100, recomputed_tokens_out=110)
    # claimed 500 tokens, re-run produces 40 -> billing fraud, fail
    assert not spot_check_verdict(
        receipt_deterministic=False, checker_deterministic=True,
        receipt_output_hash="abc", recomputed_output_hash="def",
        receipt_tokens_out=500, recomputed_tokens_out=40)


# ------------------------------------------------------------ persistence

def test_chain_survives_restart(genesis, alice, bob, tmp_path):
    keys = keys_map(alice, bob)
    path = tmp_path / "blocks.jsonl"
    chain = Blockchain(genesis, path=path)
    for _ in range(4):
        produce(chain, keys)
    root = chain.state.state_root()

    reopened = Blockchain(genesis, path=path)
    assert reopened.height == 4
    assert reopened.state.state_root() == root


def test_corrupt_tail_is_truncated(genesis, alice, bob, tmp_path):
    keys = keys_map(alice, bob)
    path = tmp_path / "blocks.jsonl"
    chain = Blockchain(genesis, path=path)
    for _ in range(3):
        produce(chain, keys)
    with path.open("a", encoding="utf-8") as fh:
        fh.write("{ not valid json\n")
    reopened = Blockchain(genesis, path=path)
    assert reopened.height == 3


# ------------------------------------------------------- incremental sync

def test_extend_appends_only_the_tail(genesis, alice, bob):
    keys = keys_map(alice, bob)
    long_chain = Blockchain(genesis)
    for _ in range(6):
        produce(long_chain, keys)
    short_chain = Blockchain(genesis)
    for block in long_chain.blocks[1:3]:
        short_chain.add_block(block)
    appended = short_chain.extend(long_chain.blocks_from(3))
    assert appended == 4
    assert short_chain.head.hash == long_chain.head.hash


def test_extend_rejects_fork_tail(genesis, alice, bob):
    keys = keys_map(alice, bob)
    ours = Blockchain(genesis)
    theirs = Blockchain(genesis)
    produce(ours, keys, timestamp=1.0)
    produce(theirs, keys, timestamp=2.0)   # different timestamp -> different hash
    produce(theirs, keys, timestamp=3.0)
    assert ours.extend(theirs.blocks_from(2)) == 0   # prev_hash mismatch
    assert ours.height == 1


# ---------------------------------------------------------- peer discovery

MODEL = "bartowski/Llama-3.2-3B-Instruct-GGUF::Llama-3.2-3B-Instruct-Q4_K_M.gguf"


async def test_peer_discovery_via_exchange(params):
    """C only seeds A; B is learned through peer exchange / chain registry."""
    node_keys = [KeyPair.from_seed_hex(f"{i}{i}" * 32) for i in (4, 5, 6)]
    genesis = Genesis(
        params=params,
        alloc={kp.address: 100_000 * DVT for kp in node_keys},
        stakes={kp.address: 10_000 * DVT for kp in node_keys},
    )
    urls = [f"http://127.0.0.1:{9201 + i}" for i in range(3)]
    seeds = [[urls[1]], [urls[0]], [urls[0]]]  # A<->B; C knows only A
    transport = MultiTransport()
    daemons = []
    for i, kp in enumerate(node_keys):
        cfg = NodeConfig(
            port=9201 + i, endpoint=urls[i], peers=seeds[i],
            backend="mock", models=[MODEL],
            device=DeviceInfo(vendor="nvidia", name="test", vram_mb=12282),
        )
        daemon = NodeDaemon(kp, genesis, cfg, client=httpx.AsyncClient(transport=transport))
        transport.add(urls[i], daemon.app)
        daemons.append(daemon)
    for d in daemons:
        await d.start()
    try:
        deadline = asyncio.get_event_loop().time() + 8.0
        while asyncio.get_event_loop().time() < deadline:
            if urls[1] in daemons[2].peers:
                break
            await asyncio.sleep(0.05)
        assert urls[1] in daemons[2].peers, "C never discovered B"
        assert urls[2] in daemons[1].peers, "B never discovered C"
    finally:
        for d in daemons:
            await d.stop()


# ------------------------------------------------------- liveness, e2e

async def test_chain_survives_dead_validator(params):
    """With one of two validators dead, fallback slots keep blocks coming."""
    node_keys = [KeyPair.from_seed_hex("77" * 32), KeyPair.from_seed_hex("88" * 32)]
    genesis = Genesis(
        params=params,
        alloc={kp.address: 100_000 * DVT for kp in node_keys},
        stakes={kp.address: 10_000 * DVT for kp in node_keys},
    )
    urls = ["http://127.0.0.1:9301", "http://127.0.0.1:9302"]
    transport = MultiTransport()
    daemons = []
    for i, kp in enumerate(node_keys):
        cfg = NodeConfig(port=9301 + i, endpoint=urls[i], peers=[urls[1 - i]],
                         backend="mock",
                         device=DeviceInfo(vendor="nvidia", name="test", vram_mb=12282))
        daemon = NodeDaemon(kp, genesis, cfg, client=httpx.AsyncClient(transport=transport))
        transport.add(urls[i], daemon.app)
        daemons.append(daemon)
    for d in daemons:
        await d.start()
    try:
        deadline = asyncio.get_event_loop().time() + 5.0
        while asyncio.get_event_loop().time() < deadline and daemons[1].chain.height < 3:
            await asyncio.sleep(0.05)
        assert daemons[1].chain.height >= 3

        await daemons[0].stop()  # validator 0 dies
        survivor = daemons[1]
        start_height = survivor.chain.height
        deadline = asyncio.get_event_loop().time() + 8.0
        while asyncio.get_event_loop().time() < deadline:
            if survivor.chain.height >= start_height + 4:
                break
            await asyncio.sleep(0.05)
        assert survivor.chain.height >= start_height + 4, \
            "chain stalled after a validator died"
        # Every block after the death was either produced by the survivor at
        # its own slot 0, or via a fallback slot — and fallback use must have
        # charged the dead proposer with misses. (RANDAO makes the schedule
        # random, so the dead validator may legitimately never be drawn.)
        dead = node_keys[0].address
        tail = survivor.chain.blocks[start_height + 1:]
        assert all(b.proposer == node_keys[1].address for b in tail)
        used_fallback = any(b.slot > 0 for b in tail)
        dead_acc = survivor.chain.state.account(dead)
        if used_fallback:
            assert dead_acc.misses > 0 or dead_acc.jailed_until > 0
    finally:
        for d in daemons:
            await d.stop()
