import random

import pytest

from volundr.models import GenerationParams, GenerationResult
from volundr.preference.seeds import variation_params


def approved(seed: int = 42) -> GenerationResult:
    return GenerationResult(
        params=GenerationParams(prompt="dragon", seed=seed), image_path="/a.png"
    )


def test_variations_keep_base_seed_and_set_variation_seed():
    out = variation_params(approved(42), n=5, strength=0.2, rng=random.Random(0))
    assert len(out) == 5
    for p in out:
        assert p.seed == 42
        assert p.variation_seed is not None
        assert p.variation_strength == 0.2


def test_variation_seeds_are_unique_and_exclude_base():
    out = variation_params(approved(42), n=10, rng=random.Random(1))
    vseeds = [p.variation_seed for p in out]
    assert len(set(vseeds)) == len(vseeds)
    assert 42 not in vseeds


def test_deterministic_with_seeded_rng():
    a = variation_params(approved(), n=3, rng=random.Random(123))
    b = variation_params(approved(), n=3, rng=random.Random(123))
    assert [p.variation_seed for p in a] == [p.variation_seed for p in b]


def test_zero_variations():
    assert variation_params(approved(), n=0) == []


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_strength_bounds(bad):
    with pytest.raises(ValueError):
        variation_params(approved(), n=1, strength=bad)


def test_negative_n_rejected():
    with pytest.raises(ValueError):
        variation_params(approved(), n=-1)
