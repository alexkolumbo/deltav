"""Tokenomics: cost-anchored pricing + chain pool distribution."""
import pytest

from deltav.chain.blockchain import Blockchain, build_genesis_state
from deltav.chain.transaction import TxType
from deltav.config import DVT, ChainParams, Genesis
from deltav.crypto import sha256_hex
from deltav.economics import (
    REFERENCE_TPS,
    REFERENCE_WATTS,
    kwh_per_million_tokens,
    price_report,
    suggested_price_udvt,
)

from test_chain import make_receipt_tx, make_tx, register_node
from test_phase2 import keys_map, produce


# ------------------------------------------------------------- economics

def test_energy_math_reference_node():
    # 150 W / 30 tok/s = 5 J/token -> 1.389 kWh per 1M tokens
    assert kwh_per_million_tokens(150, 30) == pytest.approx(1.389, abs=0.001)


def test_reference_peg_matches_default_network_price():
    """The whole point: the default price (10 udvt/token) must cover the
    reference node's electricity + 50% at the reference peg."""
    assert suggested_price_udvt(REFERENCE_WATTS, REFERENCE_TPS) == 10


def test_price_report_composition():
    r = price_report(watts=150, tokens_per_sec=30)
    assert r.cost_usd_per_million == pytest.approx(0.2153, abs=0.001)
    assert r.price_usd_per_million == pytest.approx(0.3229, abs=0.001)
    assert r.suggested_price_udvt == 10


def test_cheap_electricity_lowers_asking_price():
    # a node in a cheap-power region can undercut the network default
    cheap = suggested_price_udvt(150, 30, electricity_usd_kwh=0.05)
    assert cheap < 10
    # a fast node (better tok/s per watt) also prices lower
    fast = suggested_price_udvt(150, 90)
    assert fast <= 4


def test_our_6600m_profile():
    """RX 6600M node: ~130 W system, ~30 tok/s -> profitable at default price."""
    r = price_report(watts=130, tokens_per_sec=30)
    assert r.suggested_price_udvt <= 10  # default network price covers it


# ------------------------------------------------------------ chain pool

@pytest.fixture
def pool_genesis(alice, bob, carol):
    params = ChainParams(block_time=0.05, epoch_blocks=4, dev_fund=carol.address)
    return Genesis(
        params=params,
        alloc={alice.address: 100_000 * DVT, bob.address: 100_000 * DVT},
        stakes={alice.address: 10_000 * DVT, bob.address: 10_000 * DVT},
    )


def test_pool_accrues_and_distributes(pool_genesis, alice, bob, carol):
    keys = keys_map(alice, bob)
    chain = Blockchain(pool_genesis)

    # bob's node serves a job -> 10% of the payment lands in the pool
    register = make_tx(bob, chain.state, TxType.REGISTER_NODE, {"endpoint": "http://n:1"})
    chain.add_block(chain.build_block(keys[chain.next_proposer()], [register], 1.0))
    rtx = make_receipt_tx(bob, alice, chain.state, tokens_in=100, tokens_out=100)
    chain.add_block(chain.build_block(keys[chain.next_proposer()], [rtx], 2.0))

    price = 200 * chain.state.params.price_per_token
    expected_pool = price * chain.state.params.pool_fee_bps // 10_000
    assert chain.state.pool == expected_pool
    assert chain.state.nodes[bob.address].epoch_tokens == 200

    dev_before = chain.state.account(carol.address).balance
    bob_before = chain.state.account(bob.address).balance

    # cross the epoch boundary (epoch_blocks=4)
    while chain.height % 4 != 0:
        produce(chain, keys, timestamp=10.0 + chain.height)

    dev_cut = expected_pool * 3000 // 10_000
    node_share = expected_pool - dev_cut  # bob is the only worker
    assert chain.state.account(carol.address).balance == dev_before + dev_cut
    assert chain.state.account(bob.address).balance >= bob_before + node_share
    assert chain.state.pool == 0
    assert chain.state.nodes[bob.address].epoch_tokens == 0  # counters reset


def test_pool_split_pro_rata_between_workers(genesis, alice, bob, carol):
    state = build_genesis_state(genesis)
    state.params.epoch_blocks = 10
    state.params.dev_fund = ""
    register_node(state, bob, endpoint="http://b:1")
    register_node(state, carol, endpoint="http://c:1")
    # bob serves 3x the tokens carol does
    state.apply_tx(make_receipt_tx(bob, alice, state, tokens_in=100, tokens_out=200,
                                   request_hash=sha256_hex(b"j1")), 2)
    state.apply_tx(make_receipt_tx(carol, alice, state, tokens_in=50, tokens_out=50,
                                   request_hash=sha256_hex(b"j2")), 2)
    pool = state.pool
    assert pool > 0
    bob_before = state.account(bob.address).balance
    carol_before = state.account(carol.address).balance

    state.apply_block_effects("dv1someproposer", 0, [], 10)  # epoch boundary

    bob_gain = state.account(bob.address).balance - bob_before
    carol_gain = state.account(carol.address).balance - carol_before
    assert bob_gain == pool * 300 // 400
    assert carol_gain == pool * 100 // 400
    assert state.pool == pool - bob_gain - carol_gain  # only integer dust


def test_no_dev_fund_means_everything_to_workers(genesis, alice, bob):
    state = build_genesis_state(genesis)
    state.params.epoch_blocks = 5
    state.params.dev_fund = ""
    register_node(state, bob)
    state.apply_tx(make_receipt_tx(bob, alice, state, tokens_in=10, tokens_out=10), 2)
    pool = state.pool
    before = state.account(bob.address).balance
    state.apply_block_effects("dv1p", 0, [], 5)
    assert state.account(bob.address).balance == before + pool
    assert state.pool == 0


def test_pool_survives_epoch_with_no_workers(genesis, alice, bob, carol):
    """Fees accrued but every worker's counters are empty (e.g. receipts
    landed in a previous epoch) — the worker share must stay in the pool."""
    state = build_genesis_state(genesis)
    state.params.epoch_blocks = 5
    state.params.dev_fund = carol.address
    register_node(state, bob)
    state.apply_tx(make_receipt_tx(bob, alice, state, tokens_in=10, tokens_out=10), 2)
    state.apply_block_effects("dv1p", 0, [], 5)   # epoch 1: distributes
    # simulate accrual without work in epoch 2
    state.pool += 1000
    state.apply_block_effects("dv1p", 0, [], 10)  # epoch 2: no workers
    dev_cut = 1000 * 3000 // 10_000
    assert state.pool == 1000 - dev_cut  # worker share carried forward