"""TrainingCoordinator — interface for a network-native model's training.

Deliberately an ABC with a runnable dry-run, not a real trainer. A future
powerful node implements `fine_tune_step` (LoRA/RL on real hardware); the
orchestration around it — pull verified samples, train, checkpoint,
announce — is defined here so the rest of the system (router, catalog,
receipts) already fits.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .dataset import ReceiptSample


@dataclass
class TrainingRound:
    round_id: int
    samples: int
    positive: int
    negative: int
    base_model: str
    checkpoint: str = ""      # ref/path of the produced checkpoint
    notes: list[str] = field(default_factory=list)


class TrainingCoordinator(ABC):
    """Drives one round: collect verified samples -> train -> publish."""

    def __init__(self, base_model: str, min_samples: int = 64):
        self.base_model = base_model
        self.min_samples = min_samples

    def collect(self, samples: list[ReceiptSample]) -> tuple[list, list]:
        """Split into positives (verified) and negatives (slashed)."""
        pos = [s for s in samples if s.reward > 0 and s.prompt and s.output]
        neg = [s for s in samples if s.reward < 0]
        return pos, neg

    @abstractmethod
    def fine_tune_step(self, positives: list[ReceiptSample],
                       negatives: list[ReceiptSample]) -> str:
        """Run one fine-tune / RL step; return the checkpoint ref. Heavy —
        implemented only on training-capable nodes."""

    def run_round(self, round_id: int, samples: list[ReceiptSample]) -> TrainingRound:
        pos, neg = self.collect(samples)
        report = TrainingRound(round_id=round_id, samples=len(samples),
                               positive=len(pos), negative=len(neg),
                               base_model=self.base_model)
        if len(pos) < self.min_samples:
            report.notes.append(
                f"not enough verified samples ({len(pos)}/{self.min_samples}) — skipping")
            return report
        report.checkpoint = self.fine_tune_step(pos, neg)
        report.notes.append(f"trained on {len(pos)} verified samples")
        return report


class DryRunCoordinator(TrainingCoordinator):
    """No-op coordinator: proves the pipeline shape without a GPU."""

    def fine_tune_step(self, positives, negatives) -> str:
        return f"{self.base_model}+ft-{len(positives)}samples"
