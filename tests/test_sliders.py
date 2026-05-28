import pytest

from volundr.models import Feedback, FeedbackKind, LoraSpec
from volundr.preference.sliders import (
    GuidanceEdit,
    Slider,
    SliderBank,
    apply_sega_guidance,
)


def lean(label, kind, strength=1.0):
    return Feedback(generation_id="g", kind=kind, strength=strength, tokens=(label,))


def test_slider_clamps_value():
    assert Slider("x", value=5.0, max_value=1.0).clamped_value() == 1.0
    assert Slider("x", value=-5.0, min_value=-1.0).clamped_value() == -1.0


def test_slider_nudged_clamps_and_is_immutable():
    s = Slider("x", value=0.95, max_value=1.0)
    n = s.nudged(0.1)
    assert n.value == 1.0
    assert s.value == 0.95  # frozen / unchanged


def test_bank_lean_toward_and_away():
    bank = SliderBank([Slider("spikiness"), Slider("glow")], step=0.1)
    bank.apply(
        [
            lean("spikiness", FeedbackKind.LEAN_TOWARD),
            lean("spikiness", FeedbackKind.LEAN_TOWARD),
            lean("glow", FeedbackKind.LEAN_AWAY),
        ]
    )
    vals = bank.values()
    assert vals["spikiness"] == pytest.approx(0.2)
    assert vals["glow"] == pytest.approx(-0.1)


def test_bank_strength_scales_nudge():
    bank = SliderBank([Slider("spikiness")], step=0.1)
    bank.apply([lean("spikiness", FeedbackKind.LEAN_TOWARD, strength=2.0)])
    assert bank.values()["spikiness"] == pytest.approx(0.2)


def test_bank_ignores_approve_deny_and_unknown_labels():
    bank = SliderBank([Slider("spikiness")])
    bank.apply(
        [
            lean("spikiness", FeedbackKind.APPROVE),
            lean("unknown", FeedbackKind.LEAN_TOWARD),
        ]
    )
    assert bank.values()["spikiness"] == 0.0


def test_active_loras_uses_signed_weight_and_skips_zero():
    bank = SliderBank(
        [
            Slider("spikiness", value=0.4, lora="spikiness_slider"),
            Slider("glow", value=-0.3, lora="glow_slider"),
            Slider("idle", value=0.0, lora="idle_slider"),  # zero -> skipped
            Slider("textonly", value=0.5, concept_prompt="metallic"),  # no lora -> skipped
        ]
    )
    loras = bank.active_loras()
    assert LoraSpec("spikiness_slider", 0.4) in loras
    assert LoraSpec("glow_slider", -0.3) in loras  # negative weight allowed
    assert len(loras) == 2


def test_guidance_axes_selects_concept_sliders():
    bank = SliderBank(
        [
            Slider("metallic", value=0.5, concept_prompt="metallic sheen"),
            Slider("quiet", value=0.0, concept_prompt="muted"),  # zero -> skipped
            Slider("loraonly", value=0.5, lora="x"),  # no concept -> skipped
        ]
    )
    axes = bank.guidance_axes()
    assert [a.label for a in axes] == ["metallic"]


def test_apply_sega_guidance_linear_combination_with_floats():
    # Using floats as 1-element "tensors": result = base + sum(scale*gate*dir)
    base = 1.0
    edits = [
        GuidanceEdit(direction=2.0, scale=0.5),            # +1.0
        GuidanceEdit(direction=4.0, scale=-0.25),          # -1.0
        GuidanceEdit(direction=10.0, scale=1.0, gate=0.0),  # gated off -> 0
    ]
    assert apply_sega_guidance(base, edits) == pytest.approx(1.0)


def test_apply_sega_guidance_no_edits_is_identity():
    assert apply_sega_guidance(3.14, []) == 3.14
