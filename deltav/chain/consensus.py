"""Proof-of-stake consensus with liveness fallback slots.

Every honest node derives the proposer for (height, slot) from
sha256(prev_hash : height : slot) mapped onto the cumulative stake of
the validator set — no communication needed. Slot 0 is the primary
proposer; if it stays silent, slot 1's proposer may step in after one
extra block_time, slot 2 after two, and so on up to max_slots. Skipped
proposers accumulate misses and get jailed. Forks resolve by
longest-valid-chain.
"""
from __future__ import annotations

import hashlib

from .block import Block, GENESIS_PROPOSER
from .state import State, StateError


class ConsensusError(Exception):
    pass


def expected_proposer(state: State, height: int, prev_hash: str, slot: int = 0) -> str | None:
    validators = state.validators()
    total = sum(stake for _, stake in validators)
    if total <= 0:
        return None
    seed_bytes = hashlib.sha256(f"{prev_hash}:{height}:{slot}".encode()).digest()
    ticket = int.from_bytes(seed_bytes, "big") % total
    acc = 0
    for address, stake in validators:
        acc += stake
        if ticket < acc:
            return address
    return validators[-1][0]


def missed_proposers(state: State, height: int, prev_hash: str, slot: int) -> list[str]:
    """Proposers of slots 0..slot-1 who stayed silent this height."""
    return [
        p for s in range(slot)
        if (p := expected_proposer(state, height, prev_hash, s)) is not None
    ]


def validate_block(state: State, prev: Block, block: Block) -> State:
    """Validate `block` against `state` (the state after `prev`).

    Returns the new state on success, raises ConsensusError otherwise.
    """
    if block.height != prev.height + 1:
        raise ConsensusError(f"bad height {block.height}, head is {prev.height}")
    if block.prev_hash != prev.hash:
        raise ConsensusError("prev_hash mismatch")
    if not 0 <= block.slot < state.params.max_slots:
        raise ConsensusError(f"slot {block.slot} out of range")
    if len(block.txs) > state.params.max_txs_per_block:
        raise ConsensusError("too many txs")
    if not block.verify_signature():
        raise ConsensusError("bad block signature")

    proposer = expected_proposer(state, block.height, prev.hash, block.slot)
    if proposer is None:
        raise ConsensusError("no validators — chain cannot progress")
    if block.proposer != proposer:
        raise ConsensusError(
            f"wrong proposer {block.proposer} for slot {block.slot}, expected {proposer}"
        )

    new_state = state.clone()
    fees = 0
    for tx in block.txs:
        try:
            new_state.apply_tx(tx, block.height)
        except StateError as exc:
            raise ConsensusError(f"invalid tx {tx.hash[:12]}: {exc}") from exc
        fees += tx.fee
    missed = missed_proposers(state, block.height, prev.hash, block.slot)
    new_state.apply_block_effects(block.proposer, fees, missed, block.height)
    new_state.height = block.height

    if block.state_root != new_state.state_root():
        raise ConsensusError("state_root mismatch")
    return new_state
