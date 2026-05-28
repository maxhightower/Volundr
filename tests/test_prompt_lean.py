from volundr.models import Feedback, FeedbackKind
from volundr.preference.prompt_lean import PromptLean, render_weighted


def lean(tokens, kind, strength=1.0):
    return Feedback(generation_id="g", kind=kind, strength=strength, tokens=tokens)


def test_render_bare_token_at_weight_one():
    assert render_weighted([("dragon", 1.0)]) == "dragon"


def test_render_weighted_syntax():
    assert render_weighted([("scales", 1.25), ("wings", 1.0)]) == "(scales:1.25), wings"


def test_lean_toward_raises_weight_in_positive():
    pl = PromptLean(base_positive=("dragon", "scales"))
    pl.apply([lean(("scales",), FeedbackKind.LEAN_TOWARD)])
    pos = dict(pl.positive())
    assert pos["dragon"] == 1.0
    assert pos["scales"] == 1.25  # 1 + 1.0 * 0.25
    assert pl.negative() == []


def test_lean_away_moves_token_to_negative():
    pl = PromptLean(base_positive=("dragon", "blurry"))
    pl.apply([lean(("blurry",), FeedbackKind.LEAN_AWAY, strength=2.0)])
    pos = dict(pl.positive())
    neg = dict(pl.negative())
    assert "blurry" not in pos
    assert neg["blurry"] == 1.5  # 1 + 2.0 * 0.25, clamped at max_weight


def test_accumulation_across_feedback():
    pl = PromptLean(base_positive=("scales",))
    pl.apply(
        [
            lean(("scales",), FeedbackKind.LEAN_TOWARD),
            lean(("scales",), FeedbackKind.LEAN_TOWARD),
            lean(("scales",), FeedbackKind.LEAN_AWAY),
        ]
    )
    # net +1 -> weight 1.25, stays positive
    assert dict(pl.positive())["scales"] == 1.25


def test_net_negative_crosses_into_negative():
    pl = PromptLean(base_positive=("scales",))
    pl.apply(
        [
            lean(("scales",), FeedbackKind.LEAN_TOWARD),
            lean(("scales",), FeedbackKind.LEAN_AWAY),
            lean(("scales",), FeedbackKind.LEAN_AWAY),
        ]
    )
    assert "scales" not in dict(pl.positive())
    assert "scales" in dict(pl.negative())


def test_approve_deny_ignored_by_prompt_lean():
    pl = PromptLean(base_positive=("dragon",))
    pl.apply([lean(("dragon",), FeedbackKind.APPROVE)])
    assert dict(pl.positive())["dragon"] == 1.0


def test_base_negative_stays_negative():
    pl = PromptLean(base_positive=("dragon",), base_negative=("lowres",))
    assert "lowres" in dict(pl.negative())
    assert "lowres" not in dict(pl.positive())


def test_render_full_prompt():
    pl = PromptLean(base_positive=("dragon", "scales"))
    pl.apply([lean(("scales",), FeedbackKind.LEAN_TOWARD)])
    positive, negative = pl.render()
    assert positive == "dragon, (scales:1.25)"
    assert negative == ""
