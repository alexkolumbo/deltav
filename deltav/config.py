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
