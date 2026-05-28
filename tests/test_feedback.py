import pytest

from volundr.models import (
    ControlNetSpec,
    Feedback,
    FeedbackKind,
    GenerationParams,
    GenerationResult,
    IPAdapterSpec,
    LoraSpec,
    RegionPrompt,
)
from volundr.preference.feedback import FeedbackStore


def make_result(seed: int, **kw) -> GenerationResult:
    params = GenerationParams(prompt="dragon", seed=seed, **kw)
    return GenerationResult(params=params, image_path=f"/out/{seed}.png")


def test_record_requires_known_generation():
    store = FeedbackStore()
    with pytest.raises(KeyError):
        store.record(Feedback(generation_id="missing", kind=FeedbackKind.APPROVE))


def test_approved_and_denied_pools():
    store = FeedbackStore()
    a = store.add_generation(make_result(1))
    b = store.add_generation(make_result(2))
    c = store.add_generation(make_result(3))
    store.record(Feedback(a.id, FeedbackKind.APPROVE))
    store.record(Feedback(b.id, FeedbackKind.DENY))
    store.record(Feedback(c.id, FeedbackKind.APPROVE))

    assert {g.seed for g in store.approved()} == {1, 3}
    assert {g.seed for g in store.denied()} == {2}


def test_approved_dedupes_and_preserves_order():
    store = FeedbackStore()
    a = store.add_generation(make_result(1))
    store.record(Feedback(a.id, FeedbackKind.APPROVE))
    store.record(Feedback(a.id, FeedbackKind.APPROVE, strength=2.0))
    assert [g.seed for g in store.approved()] == [1]


def test_roundtrip_persistence(tmp_path):
    store = FeedbackStore()
    r = make_result(
        7,
        denoising_start=0.3,
        loras=(LoraSpec("dragon-style", 0.8),),
        controlnets=(ControlNetSpec("scribble", "/sketch.png", weight=0.6),),
        regions=(
            RegionPrompt(
                "/mask.png",
                positive="wing",
                ip_adapter=IPAdapterSpec(image="/patch.png", weight=0.5, method="style"),
            ),
        ),
    )
    store.add_generation(r)
    store.record(Feedback(r.id, FeedbackKind.LEAN_TOWARD, strength=1.5, tokens=("scales",)))

    path = tmp_path / "store.json"
    store.save(path)
    loaded = FeedbackStore.load(path)

    g = loaded.generation(r.id)
    assert g.seed == 7
    assert g.params.denoising_start == 0.3
    assert g.params.loras == (LoraSpec("dragon-style", 0.8),)
    assert g.params.controlnets[0].weight == 0.6
    assert g.params.regions[0].positive == "wing"
    assert g.params.regions[0].ip_adapter == IPAdapterSpec(
        image="/patch.png", weight=0.5, method="style"
    )
    fb = loaded.feedback_for(r.id)[0]
    assert fb.kind is FeedbackKind.LEAN_TOWARD
    assert fb.tokens == ("scales",)
    assert fb.strength == 1.5
