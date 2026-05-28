"""Pixel-art core: spec, locked set, sprite-sheet packing, auto-tiling.

The GPU-independent half of Völundr's pixel-art support. The actual
quantize/dither/downscale and any diffusion run as ported custom nodes on the
GPU box; everything here — the post-pass *spec*, the consistency-locking set,
sheet layout, and tile rules — is pure logic and fully tested.
"""

from volundr.pixelart.sheet import Atlas, pack_grid, pack_shelf
from volundr.pixelart.pixelset import PixelSet
from volundr.pixelart.spec import (
    Dither,
    PixelArtSpec,
    load_palette_gpl,
    load_palette_hex,
    parse_hex,
)
from volundr.pixelart.tiles import WANG_4BIT_LAYOUT, tilemap_from_grid, wang_index

__all__ = [
    "Atlas",
    "pack_grid",
    "pack_shelf",
    "PixelSet",
    "Dither",
    "PixelArtSpec",
    "load_palette_gpl",
    "load_palette_hex",
    "parse_hex",
    "WANG_4BIT_LAYOUT",
    "tilemap_from_grid",
    "wang_index",
]
