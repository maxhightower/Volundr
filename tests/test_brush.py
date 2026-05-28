import pytest

from volundr.brush import Brush, apply_brush
from volundr.models import GenerationParams


def test_capture_sets_core_fields():
    b = Brush.capture("ink-hatch", patch="/ref/crop.png", text="dense ink crosshatch")
    assert b.name == "ink-hatch"
    assert b.patch == "/ref/crop.png"
    assert b.text == "dense ink crosshatch"
    assert b.ip_method == "style"  # default best for technique transfer


def test_region_carries_patch_as_ip_adapter_and_text():
    b = Brush.capture("glow", patch="/p.png", text="soft rim light", ip_weight=0.5)
    region = b.region("/mask.png")
    assert region.mask == "/mask.png"
    assert region.positive == "soft rim light"
    assert region.ip_adapter is not None
    assert region.ip_adapter.image == "/p.png"
    assert region.ip_adapter.weight == 0.5
    assert region.ip_adapter.method == "style"


def test_apply_brush_appends_region_and_sets_denoise():
    params = GenerationParams(prompt="a dragon")
    b = Brush.capture("scales", patch="/scales.png", text="metallic scales", denoise_strength=0.7)
    out = apply_brush(params, b, mask="/m.png")

    assert len(out.regions) == 1
    assert out.regions[0].ip_adapter.image == "/scales.png"
    assert out.denoising_start == pytest.approx(0.3)  # 1 - 0.7
    # no structure_image supplied -> no controlnet added
    assert out.controlnets == ()


def test_apply_brush_adds_structure_controlnet_when_image_given():
    params = GenerationParams()
    b = Brush.capture("hatch", patch="/p.png", structure_control="lineart", structure_weight=0.8)
    out = apply_brush(params, b, mask="/m.png", structure_image="/target_lineart.png")

    assert len(out.controlnets) == 1
    cn = out.controlnets[0]
    assert cn.model == "lineart"
    assert cn.image == "/target_lineart.png"
    assert cn.weight == 0.8


def test_no_structure_control_skips_controlnet_even_with_image():
    params = GenerationParams()
    b = Brush.capture("free", patch="/p.png", structure_control=None)
    out = apply_brush(params, b, mask="/m.png", structure_image="/ignored.png")
    assert out.controlnets == ()


def test_structure_raises_without_control_model():
    b = Brush.capture("free", patch="/p.png", structure_control=None)
    with pytest.raises(ValueError):
        b.structure("/img.png")


def test_apply_brush_preserves_existing_regions_and_controlnets():
    from volundr.models import ControlNetSpec, RegionPrompt

    params = GenerationParams(
        regions=(RegionPrompt("/existing_mask.png", positive="background"),),
        controlnets=(ControlNetSpec("scribble", "/sketch.png"),),
    )
    b = Brush.capture("tex", patch="/p.png")
    out = apply_brush(params, b, mask="/m.png", structure_image="/s.png")
    assert len(out.regions) == 2
    assert len(out.controlnets) == 2  # original scribble + brush lineart
