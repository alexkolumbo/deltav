"""Node scoring for smart routing.

Pure function so routing decisions are unit-testable. Higher is better.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import DVT


@dataclass
class NodeView:
    """What the router knows about one node (chain registry + live health)."""
    address: str
    endpoint: str
    vram_mb: int
    models: list[str]
    reputation: float
    stake: int
    last_seen: int
    active: bool = True
    load: float = 0.0        # 0..1, from live /health polls
    alive: bool = True        # answered the last health poll
    price_per_token: int = 0  # udvt; 0 = network default

W_MODEL_READY = 3.0   # node already announced/loaded the model — no cold start
W_REPUTATION = 2.0
W_STAKE = 1.0
W_LOAD = 1.5
W_PRICE = 1.0         # cheaper-than-default earns up to 2x this weight
W_FRESHNESS = 0.5
STAKE_SATURATION = 10_000 * DVT


def score_node(node: NodeView, model_ref: str, chain_height: int,
               default_price: int = 10) -> float:
    if not node.active or not node.alive:
        return float("-inf")
    ready = 1.0 if model_ref in node.models else 0.0
    stake_norm = min(1.0, node.stake / STAKE_SATURATION)
    staleness = max(0, chain_height - node.last_seen)
    freshness = 1.0 / (1.0 + staleness / 10.0)
    asking = node.price_per_token or default_price
    cheapness = min(2.0, default_price / max(1, asking))
    return (
        W_MODEL_READY * ready
        + W_REPUTATION * node.reputation
        + W_STAKE * stake_norm
        + W_LOAD * (1.0 - min(1.0, node.load))
        + W_PRICE * cheapness
        + W_FRESHNESS * freshness
    )
