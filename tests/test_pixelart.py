import pytest

from volundr.models import GenerationParams, LoraSpec
from volundr.pixelart import (
    Atlas,
    Dither,
    PixelArtSpec,
    PixelSet,
    load_palette_gpl,
    load_palette_hex,
    pack_grid,
    pack_shelf,
    parse_hex,
    tilemap_from_grid,
    wang_index,
)


# --- spec / palette -------------------------------------------------------
def test_parse_hex_long_short_and_bad():
    assert parse_hex("#1A2B3C") == (26, 43, 60)
    assert parse_hex("fff") == (255, 255, 255)
    with pytest.raises(ValueError):
        parse_hex("xyz")


def test_load_palette_hex_skips_comments():
    text = "; lospec palette\n1a1c2c\n#5d275d\n\nnot-a-color\nef7d57\n"
    assert load_palette_hex(text) == ("#1a1c2c", "#5d275d", "#ef7d57")


def test_load_palette_gpl():
    text = "GIMP Palette\nName: Test\n#\n 26  43  60  dark\n255 255 255 white\n"
    assert load_palette_gpl(text) == ("#1a2b3c", "#ffffff")


def test_effective_colors_palette_overrides_max():
    spec = PixelArtSpec(64, 64, palette=("#000000", "#ffffff"), max_colors=16)
    assert spec.effective_colors() == 2
    assert PixelArtSpec(64, 64, max_colors=8).effective_colors() == 8


def test_integer_scale_alignment():
    spec = PixelArtSpec(64, 64)
    assert spec.integer_scale_from(1024, 1024) == (16, 16)
    assert spec.integer_scale_from(1000, 1024) is None  # misaligned grid


# --- pixel set (consistency lock) ----------------------------------------
def test_pixelset_lock_from_params_captures_seed_and_loras():
    spec = PixelArtSpec(64, 64)
    src = GenerationParams(prompt="a knight", seed=123, loras=(LoraSpec("pixel-art-xl", 1.0),))
    ps = PixelSet.lock("retro", spec, params=src, style="pixel art, 16-bit")
    assert ps.base_seed == 123
    assert ps.loras == (LoraSpec("pixel-art-xl", 1.0),)


def test_pixelset_apply_merges_loras_pins_seed_and_style():
    spec = PixelArtSpec(64, 64)
    ps = PixelSet(
        "retro", spec, base_seed=123,
        loras=(LoraSpec("pixel-art-xl", 1.0),), style="pixel art",
    )
    out = ps.apply(GenerationParams(prompt="a goblin", seed=0, loras=(LoraSpec("extra", 0.5),)))
    assert out.seed == 123
    assert out.prompt == "a goblin, pixel art"
    assert set(l.name for l in out.loras) == {"pixel-art-xl", "extra"}


def test_pixelset_locked_lora_overrides_same_name():
    spec = PixelArtSpec(64, 64)
    ps = PixelSet("r", spec, loras=(LoraSpec("shared", 1.0),))
    out = ps.apply(GenerationParams(loras=(LoraSpec("shared", 0.2),)))
    assert out.loras == (LoraSpec("shared", 1.0),)  # locked weight wins


# --- sheet packing --------------------------------------------------------
def test_pack_grid_layout_and_no_overlap():
    atlas = pack_grid(16, 16, count=4, columns=2)
    assert atlas.width == 32 and atlas.height == 32
    assert not atlas.overlaps()
    coords = {(f.x, f.y) for f in atlas.frames}
    assert coords == {(0, 0), (16, 0), (0, 16), (16, 16)}


def test_pack_grid_padding_and_default_columns():
    atlas = pack_grid(16, 16, count=9, padding=1)
    assert len(atlas.frames) == 9
    assert not atlas.overlaps()  # default near-square (3 cols) with padding


def test_pack_grid_names_validation():
    with pytest.raises(ValueError):
        pack_grid(16, 16, count=2, names=["only_one"])


def test_pack_grid_atlas_to_dict():
    atlas = pack_grid(8, 8, count=1, names=["idle"])
    assert atlas.to_dict() == {"size": {"w": 8, "h": 8}, "frames": {"idle": {"x": 0, "y": 0, "w": 8, "h": 8}}}


def test_pack_shelf_wraps_rows():
    cells = [("a", 30, 10), ("b", 30, 20), ("c", 30, 10)]  # max_width 64 -> a,b share row; c wraps
    atlas = pack_shelf(cells, max_width=64)
    assert not atlas.overlaps()
    by_name = {f.name: f for f in atlas.frames}
    assert by_name["a"].y == 0 and by_name["b"].y == 0
    assert by_name["c"].y == 20  # below the tallest (b=20) cell on row 0


# --- auto-tiling ----------------------------------------------------------
def test_wang_index_bitmask():
    assert wang_index(False, False, False, False) == 0
    assert wang_index(True, False, False, False) == 1   # N
    assert wang_index(False, True, False, False) == 2   # E
    assert wang_index(True, True, True, True) == 15      # all


def test_tilemap_from_grid_edges_and_empty():
    grid = [
        [True, True],
        [True, False],
    ]
    tm = tilemap_from_grid(grid)
    # top-left: E neighbour (1,0)=True, S neighbour (1,0)... compute:
    # (0,0): N=oob F, E=(0,1)T, S=(1,0)T, W=oob F -> E|S = 2|4 = 6
    assert tm[0][0] == 6
    # (0,1): N=F, E=oob F, S=(1,1)F, W=(0,0)T -> W = 8
    assert tm[0][1] == 8
    assert tm[1][1] == -1  # empty cell


def test_tilemap_rejects_ragged():
    with pytest.raises(ValueError):
        tilemap_from_grid([[True], [True, False]])
