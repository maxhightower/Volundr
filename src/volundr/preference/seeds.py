"""Variation seeding for build step 4 ("generate variations near approved").

Integer seeds have no semantic neighbourhood — seed N+1 is unrelated to seed N.
The standard way to stay *near* an approved image is the sub-seed technique:
keep the approved `seed` fixed and blend in a random `variation_seed` by a small
`variation_strength`. Strength 0 reproduces the approved image; higher strength
drifts further away.
"""

from __future__ import annotations

import random

from volundr.models import GenerationParams, GenerationResult

_SEED_MAX = 2**32 - 1


def variation_params(
    approved: GenerationResult,
    n: int,
    strength: float = 0.2,
    rng: random.Random | None = None,
) -> list[GenerationParams]:
    """Return `n` param sets that are variations of an approved generation.

    Each keeps the approved base `seed` and gets a fresh random `variation_seed`
    at the given `variation_strength`.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")

    rng = rng or random.Random()
    base = approved.params
    out: list[GenerationParams] = []
    used: set[int] = {base.seed}
    while len(out) < n:
        vs = rng.randint(0, _SEED_MAX)
        if vs in used:
            continue
        used.add(vs)
        from dataclasses import replace

        out.append(replace(base, variation_seed=vs, variation_strength=strength))
    return out
