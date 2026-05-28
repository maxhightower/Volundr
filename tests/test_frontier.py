from volundr.models import ControlNetSpec, GenerationParams, LoraSpec
from volundr.pixelart import PixelArtSpec, PixelSet
from volundr.pixelart.frontier import (
    DIRS_4,
    DIRS_8,
    AnimationRequest,
    AnimationSpec,
    Direction,
    PoseFrame,
    TurnaroundRequest,
)


# --- directional rotation -------------------------------------------------
def test_turnaround_one_params_per_direction_with_identity():
    req = TurnaroundRequest(
        reference="/hero.png",
        base=GenerationParams(prompt="a knight"),
        directions=DIRS_4,
    )
    out = req.per_direction_params()
    assert [d for d, _ in out] == list(DIRS_4)
    for d, params in out:
        # every view shares the identity IP-Adapter
        assert len(params.ip_adapters) == 1
        assert params.ip_adapters[0].image == "/hero.png"
        # facing phrase appended
        assert f"facing {d.facing_phrase}" in params.prompt
        assert params.prompt.startswith("a knight")


def test_turnaround_8way_and_pose_controls():
    req = TurnaroundRequest(
        reference="/hero.png",
        base=GenerationParams(prompt="a mage"),
        directions=DIRS_8,
        pose_controls={Direction.N: "/pose_n.png"},
    )
    out = dict(req.per_direction_params())
    assert len(out) == 8
    # the direction with a pose control gets an extra controlnet; others don't
    assert len(out[Direction.N].controlnets) == 1
    assert out[Direction.N].controlnets[0].image == "/pose_n.png"
    assert len(out[Direction.S].controlnets) == 0


def test_turnaround_applies_pixel_set_lock():
    ps = PixelSet("retro", PixelArtSpec(64, 64), base_seed=7,
                  loras=(LoraSpec("pixel-art-xl", 1.0),), style="pixel art")
    req = TurnaroundRequest(
        reference="/hero.png",
        base=GenerationParams(prompt="a knight", seed=0),
        directions=(Direction.S,),
        pixel_set=ps,
    )
    _, params = req.per_direction_params()[0]
    assert params.seed == 7
    assert any(l.name == "pixel-art-xl" for l in params.loras)
    assert "pixel art" in params.prompt
    assert "facing south" in params.prompt


# --- animation ------------------------------------------------------------
def test_animation_per_frame_params_ordered_with_pose_and_identity():
    req = AnimationRequest(
        reference="/hero.png",
        base=GenerationParams(prompt="a knight"),
        frames=(
            PoseFrame("/p2.png", index=2),
            PoseFrame("/p0.png", index=0),
            PoseFrame("/p1.png", index=1),
        ),
        spec=AnimationSpec(action="walk", fps=8),
    )
    params = req.per_frame_params()
    assert len(params) == 3
    # frames sorted by index -> control images in 0,1,2 order
    images = [p.controlnets[-1].image for p in params]
    assert images == ["/p0.png", "/p1.png", "/p2.png"]
    for p in params:
        assert len(p.ip_adapters) == 1  # identity held across frames


def test_animation_sheet_packs_frames():
    req = AnimationRequest(
        reference="/hero.png",
        base=GenerationParams(),
        frames=tuple(PoseFrame(f"/p{i}.png", index=i) for i in range(4)),
        spec=AnimationSpec(action="run"),
    )
    atlas = req.sheet(32, 32, columns=4)
    assert atlas.width == 128 and atlas.height == 32
    assert not atlas.overlaps()
    assert set(atlas.to_dict()["frames"]) == {"run_0", "run_1", "run_2", "run_3"}


def test_animation_locked_set_applied_to_frames():
    ps = PixelSet("retro", PixelArtSpec(48, 48), base_seed=99, style="pixel art")
    req = AnimationRequest(
        reference="/hero.png",
        base=GenerationParams(prompt="a goblin", seed=0),
        frames=(PoseFrame("/p0.png", index=0),),
        pixel_set=ps,
    )
    p = req.per_frame_params()[0]
    assert p.seed == 99
    assert "pixel art" in p.prompt
