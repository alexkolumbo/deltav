"""Spot-check verdict logic, factored out of the daemon for testability.

Exact mode: both the receipt's backend and the checker's backend claim
bit-reproducible output -> the recomputed output hash must match exactly.

Fuzzy mode: either side is non-deterministic (GPU sampling, Groq LPUs)
-> hashes cannot match by construction, so we verify the *shape* of the
claimed work instead: the re-executed job must produce a similar number
of tokens. Catches the classic fraud (billing for tokens never
generated) without punishing honest non-determinism.
"""
from __future__ import annotations

FUZZY_ABS_TOLERANCE = 4
FUZZY_REL_TOLERANCE = 0.25


def spot_check_verdict(
    *,
    receipt_deterministic: bool,
    checker_deterministic: bool,
    receipt_output_hash: str,
    recomputed_output_hash: str,
    receipt_tokens_out: int,
    recomputed_tokens_out: int,
) -> bool:
    if receipt_deterministic and checker_deterministic:
        return recomputed_output_hash == receipt_output_hash
    tolerance = max(FUZZY_ABS_TOLERANCE, int(receipt_tokens_out * FUZZY_REL_TOLERANCE))
    return abs(recomputed_tokens_out - receipt_tokens_out) <= tolerance
