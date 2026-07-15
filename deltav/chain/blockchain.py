"""Chain container + mempool + on-disk persistence."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from ..config import Genesis
from ..crypto import KeyPair
from .block import Block, GENESIS_PROPOSER, ZERO_HASH
from .consensus import ConsensusError, expected_proposer, missed_proposers, validate_block
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
    def __init__(self, genesis: Genesis, path: str | Path | None = None):
        self.genesis = genesis
        self.path = Path(path) if path else None
        self.blocks: list[Block] = [build_genesis_block(genesis)]
        self.state: State = build_genesis_state(genesis)
        if self.path is not None:
            self._load_from_disk()

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
        """Replay blocks.jsonl; a corrupt/forked tail is truncated, not fatal."""
        if not self.path.exists():
            return
        loaded = 0
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
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
                    log.warning("stopping replay at height %s: %s", self.height + 1, exc)
                    break
        if loaded:
            log.info("restored chain to height %s from %s", self.height, self.path)
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

    # -------------------------------------------------------------- growth
    def _commit(self, block: Block) -> None:
        new_state = validate_block(self.state, self.head, block)
        self.blocks.append(block)
        self.state = new_state

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
        missed = missed_proposers(self.state, height, self.head.hash, slot)
        trial.apply_block_effects(keypair.address, fees, missed, height)
        trial.height = height
        block = Block(
            height=height,
            prev_hash=self.head.hash,
            timestamp=timestamp,
            proposer=keypair.address,
            txs=included,
            state_root=trial.state_root(),
            slot=slot,
        )
        return block.sign(keypair)

    def replace(self, block_dicts: list[dict]) -> bool:
        """Adopt a longer valid chain (full re-validation from genesis)."""
        if len(block_dicts) <= len(self.blocks):
            return False
        try:
            candidate = Blockchain(self.genesis)
            incoming = [Block.from_dict(b) for b in block_dicts]
            if incoming[0].hash != candidate.blocks[0].hash:
                return False
            for block in incoming[1:]:
                candidate._commit(block)
        except (ConsensusError, KeyError, ValueError):
            return False
        if candidate.height > self.height:
            self.blocks = candidate.blocks
            self.state = candidate.state
            self._rewrite_disk()
            return True
        return False

    def blocks_from(self, start: int, count: int = 500) -> list[dict]:
        return [b.to_dict() for b in self.blocks[start : start + count]]


class Mempool:
    def __init__(self) -> None:
        self.txs: dict[str, Tx] = {}

    def add(self, tx: Tx) -> bool:
        """Accept a well-signed tx; full validity is checked at block build time."""
        if not tx.verify():
            return False
        h = tx.hash
        if h in self.txs:
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
