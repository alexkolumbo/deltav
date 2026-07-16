"""Chain container + mempool + on-disk persistence."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from ..config import Genesis
from ..crypto import KeyPair
from .block import Block, GENESIS_PROPOSER, ZERO_HASH
from .consensus import (
    ConsensusError,
    expected_proposer,
    missed_proposers,
    randao_message,
    validate_block,
)
from .state import State, StateError
from .transaction import Tx

log = logging.getLogger("deltav.chain")


def build_genesis_state(genesis: Genesis) -> State:
    state = State(genesis.params)
    for address, balance in sorted(genesis.alloc.items()):
        state.account(address).balance = int(balance)
        state.supply += int(balance)
    for address, stake in sorted(genesis.stakes.items()):
        state.account(address).stake = int(stake)
        state.supply += int(stake)
    return state


def build_genesis_block(genesis: Genesis) -> Block:
    state = build_genesis_state(genesis)
    return Block(
        height=0,
        prev_hash=ZERO_HASH,
        timestamp=genesis.timestamp,
        proposer=GENESIS_PROPOSER,
        txs=[],
        state_root=state.state_root(),
    )


class Blockchain:
    # Keep a full state copy every N blocks so fork validation restarts
    # from the nearest checkpoint instead of genesis.
    SNAPSHOT_INTERVAL = 32
    MAX_SNAPSHOTS = 8

    def __init__(self, genesis: Genesis, path: str | Path | None = None):
        self.genesis = genesis
        self.path = Path(path) if path else None
        self.blocks: list[Block] = [build_genesis_block(genesis)]
        self.state: State = build_genesis_state(genesis)
        # State as of blocks[-2] — makes sibling reorgs O(1 block).
        self._prev_state: State | None = None
        self._snapshots: dict[int, State] = {}
        self.metrics = {"sibling_fast": 0, "replace_base_height": None}
        # True when on-disk replay stopped before the file's end — the
        # persisted tail is corrupt/forked, so the node must NOT produce on
        # this partial prefix (that bakes a fork); it re-syncs from peers.
        self.restore_incomplete = False
        if self.path is not None:
            if self.path.exists():
                self._load_from_disk()
            else:
                # Persist the genesis block up front so the on-disk chain is
                # self-contained from height 0 — readers (light clients,
                # forensics) shouldn't have to know the genesis out of band.
                self._append_disk(self.blocks[0])

    @property
    def head(self) -> Block:
        return self.blocks[-1]

    @property
    def height(self) -> int:
        return self.head.height

    def next_proposer(self, slot: int = 0) -> str | None:
        return expected_proposer(self.state, self.height + 1, self.head.hash, slot)

    # --------------------------------------------------------- persistence
    def _load_from_disk(self) -> None:
        """Replay blocks.jsonl. If replay stops before the file ends, the
        tail is corrupt/forked: we keep the valid prefix but flag the chain
        `restore_incomplete` so the daemon re-syncs from peers instead of
        producing on top of the prefix (which would fork the network)."""
        if not self.path.exists():
            return
        total = 0
        loaded = 0
        broke = False
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    block = Block.from_dict(json.loads(line))
                    if block.height == 0:
                        if block.hash != self.blocks[0].hash:
                            log.warning("stored chain has a different genesis — ignoring file")
                            return
                        continue
                    self._commit(block)
                    loaded += 1
                except (ConsensusError, ValueError, KeyError) as exc:
                    log.error("REPLAY BROKE at height %s: %s — persisted tail is corrupt; "
                              "will re-sync from peers, not produce on the partial prefix",
                              self.height + 1, exc)
                    broke = True
                    break
        if loaded:
            log.info("restored chain to height %s from %s", self.height, self.path)
        # incomplete only if we stopped mid-file (broke) — a clean full read
        # that simply ended is complete.
        self.restore_incomplete = broke
        if not broke:
            self._rewrite_disk()

    def _append_disk(self, block: Block) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(block.to_dict(), sort_keys=True) + "\n")

    def _rewrite_disk(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for block in self.blocks:
                fh.write(json.dumps(block.to_dict(), sort_keys=True) + "\n")
        tmp.replace(self.path)

    # ------------------------------------------------------------ snapshots
    def _take_snapshot(self, height: int, state: State) -> None:
        if height > 0 and height % self.SNAPSHOT_INTERVAL == 0:
            self._snapshots[height] = state.clone()
            for stale in sorted(self._snapshots)[: -self.MAX_SNAPSHOTS]:
                del self._snapshots[stale]

    def _state_at(self, height: int) -> State:
        """Reconstruct the state after our own block at `height`, starting
        from the nearest checkpoint at or below it."""
        if height == self.height:
            return self.state.clone()
        if height == self.height - 1 and self._prev_state is not None:
            return self._prev_state.clone()
        base = max((h for h in self._snapshots if h <= height), default=0)
        state = self._snapshots[base].clone() if base else build_genesis_state(self.genesis)
        prev = self.blocks[base]
        for block in self.blocks[base + 1 : height + 1]:
            state = validate_block(state, prev, block)
            prev = block
        return state

    # -------------------------------------------------------------- growth
    def _commit(self, block: Block) -> None:
        new_state = validate_block(self.state, self.head, block)
        self._prev_state = self.state
        self.blocks.append(block)
        self.state = new_state
        self._take_snapshot(block.height, new_state)

    def add_block(self, block: Block) -> None:
        self._commit(block)
        self._append_disk(block)

    def extend(self, block_dicts: list[dict]) -> int:
        """Incremental sync: append consecutive blocks on top of our head.

        Returns how many were appended; stops at the first that doesn't
        fit (caller falls back to a full replace on a fork).
        """
        appended = 0
        for data in block_dicts:
            try:
                block = Block.from_dict(data)
            except (KeyError, ValueError):
                break
            if block.height != self.height + 1:
                continue
            try:
                self.add_block(block)
                appended += 1
            except ConsensusError:
                break
        return appended

    def build_block(self, keypair: KeyPair, txs: list[Tx], timestamp: float, slot: int = 0) -> Block:
        """Assemble, apply-check and sign the next block; txs that don't apply are dropped."""
        trial = self.state.clone()
        included: list[Tx] = []
        fees = 0
        height = self.height + 1
        for tx in txs:
            if len(included) >= self.state.params.max_txs_per_block:
                break
            try:
                trial.apply_tx(tx, height)
            except StateError:
                continue
            included.append(tx)
            fees += tx.fee
        reveal = keypair.sign(randao_message(self.genesis.params.chain_id, height))
        missed = missed_proposers(self.state, height, self.head.hash, slot)
        trial.apply_block_effects(keypair.address, fees, missed, height, reveal)
        trial.height = height
        block = Block(
            height=height,
            prev_hash=self.head.hash,
            timestamp=timestamp,
            proposer=keypair.address,
            txs=included,
            state_root=trial.state_root(),
            slot=slot,
            randao_reveal=reveal,
        )
        return block.sign(keypair)

    def replace_sibling(self, block: Block) -> bool:
        """Deterministic tie-break at equal height: a competing head with a
        LOWER slot wins. Without this, timing jitter lets fallback blocks
        beat the primary proposer's block and honest validators collect
        bogus misses (and eventually get jailed for being merely busy).

        O(1 block): validates against the retained pre-head state."""
        if block.height != self.height or self.height == 0:
            return False
        head = self.head
        if block.prev_hash != head.prev_hash:
            return False
        if (block.slot, block.hash) >= (head.slot, head.hash):
            return False
        try:
            prev_state = self._prev_state.clone() if self._prev_state is not None \
                else self._state_at(self.height - 1)
            new_state = validate_block(prev_state, self.blocks[-2], block)
        except ConsensusError:
            return False
        self.blocks[-1] = block
        self.state = new_state
        if block.height in self._snapshots:
            self._snapshots[block.height] = new_state.clone()
        self.metrics["sibling_fast"] += 1
        self._rewrite_disk()
        return True

    def replace(self, block_dicts: list[dict]) -> bool:
        """Adopt a longer valid chain, re-validating only past the common
        ancestor (from the nearest state checkpoint, not genesis)."""
        if len(block_dicts) <= len(self.blocks):
            return False
        try:
            incoming = [Block.from_dict(b) for b in block_dicts]
        except (KeyError, ValueError):
            return False
        if not incoming or incoming[0].hash != self.blocks[0].hash:
            return False

        # When our persisted prefix is suspect, don't trust our snapshots to
        # short-circuit — the common-ancestor state must be rebuilt from
        # genesis so a forked prefix can't poison the adopted chain.
        if self.restore_incomplete:
            common = 1  # only genesis is trusted
        else:
            common = 0
            for ours, theirs in zip(self.blocks, incoming):
                if ours.hash != theirs.hash:
                    break
                common += 1
        ancestor = common - 1  # >= 0, genesis always shared

        try:
            state = self._state_at(ancestor) if not self.restore_incomplete \
                else build_genesis_state(self.genesis)
            prev_block = incoming[ancestor]
            prev_state: State | None = None
            snapshots: dict[int, State] = {
                h: s for h, s in self._snapshots.items()
                if h <= ancestor and not self.restore_incomplete
            }
            for block in incoming[ancestor + 1 :]:
                prev_state = state
                state = validate_block(state, prev_block, block)
                prev_block = block
                if block.height % self.SNAPSHOT_INTERVAL == 0:
                    snapshots[block.height] = state.clone()
        except ConsensusError:
            return False

        self.blocks = incoming
        self.state = state
        self._prev_state = prev_state
        self._snapshots = dict(sorted(snapshots.items())[-self.MAX_SNAPSHOTS:])
        self.metrics["replace_base_height"] = max(
            (h for h in self._snapshots if h <= ancestor), default=0)
        self.restore_incomplete = False  # a full valid chain healed us
        self._rewrite_disk()
        return True

    def blocks_from(self, start: int, count: int = 500) -> list[dict]:
        return [b.to_dict() for b in self.blocks[start : start + count]]


class Mempool:
    # Local admission bounds (not consensus): stop a flood of validly-signed
    # but never-applying txs (e.g. throwaway keys with future nonces) from
    # growing memory without limit.
    MAX_TXS = 20_000
    MAX_PER_SENDER = 256

    def __init__(self) -> None:
        self.txs: dict[str, Tx] = {}

    def add(self, tx: Tx) -> bool:
        """Accept a well-signed tx; full validity is checked at block build time."""
        if not tx.verify():
            return False
        h = tx.hash
        if h in self.txs:
            return False
        if len(self.txs) >= self.MAX_TXS:
            return False
        if sum(1 for t in self.txs.values() if t.sender == tx.sender) >= self.MAX_PER_SENDER:
            return False
        self.txs[h] = tx
        return True

    def collect(self) -> list[Tx]:
        """Deterministic order: by (sender, nonce) so sequential nonces apply cleanly."""
        return sorted(self.txs.values(), key=lambda t: (t.sender, t.nonce, t.hash))

    def prune(self, state) -> None:
        """Drop consumed-nonce txs and txs that can no longer ever apply.

        A tx whose nonce is current but whose body is invalid (e.g. a
        duplicate spot check) would otherwise block every later nonce of
        the same sender forever.
        """
        stale = [
            h for h, tx in self.txs.items()
            if tx.nonce < state.account(tx.sender).nonce
        ]
        for h in stale:
            del self.txs[h]

        trial = state.clone()
        for tx in self.collect():
            try:
                trial.apply_tx(tx, state.height + 1)
            except StateError:
                if tx.nonce <= trial.account(tx.sender).nonce:
                    self.txs.pop(tx.hash, None)

    def __len__(self) -> int:
        return len(self.txs)
