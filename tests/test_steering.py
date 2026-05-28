from volundr.models import Feedback, FeedbackKind, GenerationParams, GenerationResult
from volundr.preference.feedback import FeedbackStore
from volundr.preference.steering import SteeringState


def result(seed):
    return GenerationResult(
        params=GenerationParams(prompt="dragon", seed=seed), image_path=f"/{seed}.png"
    )


def build_store():
    store = FeedbackStore()
    a, b, c = (store.add_generation(result(s)) for s in (1, 2, 3))
    store.record(Feedback(a.id, FeedbackKind.APPROVE))
    store.record(Feedback(b.id, FeedbackKind.DENY))
    store.record(Feedback(c.id, FeedbackKind.LEAN_TOWARD, tokens=("scales",)))
    return store


def test_steering_collects_seed_pools():
    state = SteeringState.from_store(build_store(), base_positive=("dragon",))
    assert state.approved_seeds == [1]
    assert state.denied_seeds == {2}


def test_steering_applies_lean_to_params():
    state = SteeringState.from_store(build_store(), base_positive=("dragon", "scales"))
    out = state.apply_to(GenerationParams())
    assert out.prompt == "dragon, (scales:1.25)"
    assert out.negative_prompt == ""
