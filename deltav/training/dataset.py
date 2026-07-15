"""Turn on-chain receipts + spot-check verdicts into a training dataset.

Pure, testable: given the chain state (receipts) and the jobs a node
retained (prompt/output), yield labelled samples. Verified-good samples
are supervised-fine-tuning fuel; slashed ones are negatives. This is what
a future trainer consumes — it needs no GPU to assemble.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ReceiptSample:
    request_hash: str
    model: str
    prompt: str
    output_hash: str
    tokens: int
    height: int
    # Reward from the chain's own verification:
    #   +1 verified, -1 slashed, 0 unchecked.
    reward: float
    output: str | None = None  # filled when the producing node's job is available


def reward_of(receipt) -> float:
    if not receipt.checked:
        return 0.0
    return 1.0 if receipt.check_ok else -1.0


def iter_training_samples(receipts, jobs: dict | None = None, min_reward: float = 1.0):
    """Yield ReceiptSample for on-chain receipts whose reward >= min_reward.

    `receipts` — iterable of chain Receipt objects; `jobs` — optional
    {request_hash: {prompt, ...}} from a node to attach the actual text."""
    jobs = jobs or {}
    for r in receipts:
        reward = reward_of(r)
        if reward < min_reward:
            continue
        job = jobs.get(r.request_hash, {})
        yield ReceiptSample(
            request_hash=r.request_hash,
            model=r.model,
            prompt=job.get("prompt", ""),
            output_hash=r.output_hash,
            tokens=r.tokens_in + r.tokens_out,
            height=r.height,
            reward=reward,
            output=job.get("output"),
        )
