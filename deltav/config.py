"""Chain parameters and genesis definition."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# 1 DVT (Delta V Token) = 1_000_000 udvt. All on-chain amounts are int udvt.
DVT = 1_000_000


@dataclass
class ChainParams:
    chain_id: str = "deltav-local-1"
    block_time: float = 2.0
    # An account must stake at least this much to propose blocks / spot-check.
    min_validator_stake: int = 1_000 * DVT
    block_reward: int = 2 * DVT
    # Emission paid to a node per verified-able inference receipt.
    inference_reward: int = DVT // 10
    # Emission paid to a validator per submitted spot check.
    check_reward: int = DVT // 50
    # What the requester pays the node, per generated+prompt token.
    price_per_token: int = 10
    # Fraction of receipts validators are expected to re-execute.
    spot_check_rate: float = 0.25
    # Fraction of a node's stake burned when a spot check fails.
    slash_fraction: float = 0.05
    max_txs_per_block: int = 100
    # Liveness: fallback proposer slots per height. Slot s may produce
    # once (s+1) * block_time has elapsed since the previous block.
    max_slots: int = 8
    # A validator that misses its slot this many times gets jailed
    # (excluded from proposing/checking) for jail_blocks.
    jail_after_misses: int = 3
    jail_blocks: int = 50
    # Unstaked funds stay slashable for this many blocks before release.
    unbonding_blocks: int = 20
    # Tokenomics: this share of every receipt payment (basis points)
    # accrues to the chain pool instead of the serving node directly.
    pool_fee_bps: int = 1000
    # Every epoch the pool distributes: dev_share_bps to the dev fund,
    # the rest to nodes pro-rata to tokens served during the epoch.
    dev_fund: str = ""
    dev_share_bps: int = 3000
    epoch_blocks: int = 600

    # --- reward-mechanism version (v2 = the escrow/epoch/verification model) ---
    # version 1 = the original optimistic model (alpha-3, unchanged).
    # version 2 = fee settled once per session (one-time auth nonce), emission
    #             deferred to epoch and gated on the receipt not failing a
    #             verification, slashing needs a reproducible commitment +
    #             min_checkers, plus availability leases and client disputes.
    version: int = 1
    # Backends whose (model, temp=0, seed) output is bit-reproducible — the
    # receipt's `deterministic` is taken from the node's REGISTERED backend,
    # not a payload flag the payee controls (audit C5).
    deterministic_backends: list = field(default_factory=lambda: [
        "mock", "llamacpp", "llamaserver", "asic"])
    # A client can flag a receipt for a priority re-check within this window.
    dispute_window: int = 50
    # Independent commitment-backed FAIL verdicts (or one dispute-confirmed
    # fail) required before a node is slashed — no single validator can slash.
    min_checkers: int = 2
    # A node must hold a live availability lease to be routed to; lease length.
    lease_blocks: int = 40
    # Refund the requester this fraction of a confirmed-bad job (clawback).
    clawback_bps: int = 10_000


@dataclass
class Genesis:
    params: ChainParams = field(default_factory=ChainParams)
    alloc: dict[str, int] = field(default_factory=dict)
    # Initial validator stakes — PoS needs at least one staked account at
    # height 1, otherwise no one can propose the block that would carry
    # the first STAKE transaction.
    stakes: dict[str, int] = field(default_factory=dict)
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            "params": asdict(self.params),
            "alloc": dict(self.alloc),
            "stakes": dict(self.stakes),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Genesis":
        return cls(
            params=ChainParams(**data.get("params", {})),
            alloc={k: int(v) for k, v in data.get("alloc", {}).items()},
            stakes={k: int(v) for k, v in data.get("stakes", {}).items()},
            timestamp=float(data.get("timestamp", 0.0)),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Genesis":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
