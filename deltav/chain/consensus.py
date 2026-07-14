"""Proof-of-stake consensus: deterministic stake-weighted proposer selection.

Every honest node derives the same proposer for height H from
sha256(prev_hash : H) mapped onto the cumulative stake of the validator
set — no communication needed, forks resolve by longest-valid-chain.
"""
from __future__ import annotations

import hashlib

from .block import Block, GENESIS_PROPOSER
from .state import State, StateError


class ConsensusError(Exception):
    pass


def expected_proposer(state: State, height: int, prev_hash: str) -> str | None:
    validators = state.validators()
    total = sum(stake for _, stake in validators)
    if total <= 0:
        return None
    seed_bytes = hashlib.sha256(f"{prev_hash}:{height}".encode()).digest()
    ticket = int.from_bytes(seed_bytes, "big") % total
    acc = 0
    for address, stake in validators:
        acc += stake
        if ticket < acc:
            return address
    return validators[-1][0]


def validate_block(state: State, prev: Block, block: Block) -> State:
    """Validate `block` against `state` (the state after `prev`).

    Returns the new state on success, raises ConsensusError otherwise.
    """
    if block.height != prev.height + 1:
        raise ConsensusError(f"bad height {block.height}, head is {prev.height}")
    if block.prev_hash != prev.hash:
        raise ConsensusError("prev_hash mismatch")
    if len(block.txs) > state.params.max_txs_per_block:
        raise ConsensusError("too many txs")
    if not block.verify_signature():
        raise ConsensusError("bad block signature")

    proposer = expected_proposer(state, block.height, prev.hash)
    if proposer is None:
        raise ConsensusError("no validators — chain cannot progress")
    if block.proposer != proposer:
        raise ConsensusError(f"wrong proposer {block.proposer}, expected {proposer}")

    new_state = state.clone()
    fees = 0
    for tx in block.txs:
        try:
            new_state.apply_tx(tx, block.height)
        except StateError as exc:
            raise ConsensusError(f"invalid tx {tx.hash[:12]}: {exc}") from exc
        fees += tx.fee
    new_state.apply_block_rewards(block.proposer, fees)
    new_state.height = block.height

    if block.state_root != new_state.state_root():
        raise ConsensusError("state_root mismatch")
    return new_state
