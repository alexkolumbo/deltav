"""Groundwork for a network-native model and its reinforcement training.

This is scaffolding — interfaces and a design, not a trainer. Full RLHF /
distributed fine-tuning is a heavy-machine job; the point here is that the
network already produces the two things training needs, so the path is
clear:

  * a DATASET — every INFERENCE_RECEIPT is a (prompt, output) pair already
    committed on-chain, with the serving model and token counts;
  * a REWARD signal — every SPOT_CHECK is an independent verdict on that
    output (ok / slashed), i.e. a cheap, decentralized correctness label.

The `TrainingCoordinator` interface below defines how a powerful node
would (in a later phase) pull verified samples from the chain, run a
fine-tune / RL step, and publish an updated checkpoint the router can
serve as a first-class network model.
"""
from .dataset import ReceiptSample, iter_training_samples
from .coordinator import TrainingCoordinator, TrainingRound

__all__ = [
    "ReceiptSample",
    "iter_training_samples",
    "TrainingCoordinator",
    "TrainingRound",
]
