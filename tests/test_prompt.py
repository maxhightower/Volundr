import random

from volundr.models import LoraSpec
from volundr.prompt import StylePreset, expand, expand_all, extract_loras


# --- wildcards / dynamic prompts -----------------------------------------
def test_expand_picks_one_variant_deterministically():
    out = expand("a {red|blue|green} dragon", rng=random.Random(0))
    assert out in {"a red dragon", "a blue dragon", "a green dragon"}
    # same seed -> same result
    assert out == expand("a {red|blue|green} dragon", rng=random.Random(0))


def test_expand_resolves_nested_variants():
    out = expand("{a|{b|c}}", rng=random.Random(1))
    assert out in {"a", "b", "c"}


def test_expand_resolves_wildcards():
    wc = {"color": ["crimson", "azure"]}
    out = expand("a __color__ scale", rng=random.Random(2), wildcards=wc)
    assert out in {"a crimson scale", "a azure scale"}


def test_expand_all_combinations():
    assert set(expand_all("{a|b} {1|2}")) == {"a 1", "a 2", "b 1", "b 2"}


def test_expand_all_with_wildcards_and_dedup():
    wc = {"x": ["p", "p", "q"]}  # duplicate option
    assert expand_all("__x__", wc) == ["p", "q"]


def test_expand_cleanup_collapses_empty_choice():
    # empty option should not leave dangling spaces/commas
    assert expand("scales{|, glowing}", rng=random.Random(0)) in {"scales", "scales, glowing"}


# --- style presets --------------------------------------------------------
def test_style_placeholder_substitution():
    s = StylePreset("ink", positive="line art, {prompt}, monochrome", negative="color")
    pos, neg = s.apply("a dragon", "blurry")
    assert pos == "line art, a dragon, monochrome"
    assert neg == "blurry, color"


def test_style_appends_when_no_placeholder():
    s = StylePreset("vivid", positive="vivid colors")
    pos, neg = s.apply("a dragon")
    assert pos == "a dragon, vivid colors"
    assert neg == ""


# --- inline lora tags -----------------------------------------------------
def test_extract_loras_with_and_without_weight():
    cleaned, loras = extract_loras("a dragon <lora:dragonstyle:0.8> in flight <lora:detail>")
    assert cleaned == "a dragon in flight"
    assert loras == (LoraSpec("dragonstyle", 0.8), LoraSpec("detail", 1.0))


def test_extract_loras_negative_weight():
    _, loras = extract_loras("<lora:slider:-0.5>")
    assert loras == (LoraSpec("slider", -0.5),)


def test_extract_loras_none_present():
    cleaned, loras = extract_loras("just a prompt")
    assert cleaned == "just a prompt"
    assert loras == ()
