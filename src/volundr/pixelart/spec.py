"""The pixel-art post-pass spec + palette handling.

Raw SDXL output is pixel-art-*styled*, not true pixel art, so it always gets a
post pass: downscale-to-grid -> palette quantization -> optional dither. This
module is the *configuration* of that pass (the pass itself runs on the GPU box /
as a ported node). A locked PixelArtSpec applied across a whole sprite set is the
single most important lever for consistency.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Dither(str, Enum):
    NONE = "none"
    FLOYD_STEINBERG = "floyd_steinberg"
    BAYER2 = "bayer2"
    BAYER4 = "bayer4"
    BAYER8 = "bayer8"


def parse_hex(color: str) -> tuple[int, int, int]:
    """Parse '#rrggbb' or 'rrggbb' (also short '#rgb') to an (r, g, b) tuple."""
    c = color.strip().lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    if len(c) != 6:
        raise ValueError(f"bad hex color: {color!r}")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def load_palette_hex(text: str) -> tuple[str, ...]:
    """Parse a Lospec '.hex' palette (one hex code per line, with or without '#').

    Lines that aren't valid hex colors (comments, blanks) are skipped.
    """
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith((";", "//")):
            continue
        try:
            parse_hex(line)
        except ValueError:
            continue  # comment or junk, not a color
        out.append("#" + line.lstrip("#").lower())
    return tuple(out)


def load_palette_gpl(text: str) -> tuple[str, ...]:
    """Parse a GIMP '.gpl' palette ('R G B  name' rows; header/comments skipped)."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.lower().startswith(("gimp palette", "name:", "columns:")):
            continue
        parts = line.split()
        if len(parts) >= 3 and all(p.isdigit() for p in parts[:3]):
            r, g, b = (int(p) for p in parts[:3])
            out.append("#{:02x}{:02x}{:02x}".format(r, g, b))
    return tuple(out)


@dataclass(frozen=True)
class PixelArtSpec:
    """Target grid + palette + dither for the post pass.

    `palette` (locked hex colors) takes precedence; when empty, the pass reduces
    to `max_colors` via k-means. `downscale` selects how source pixels collapse
    onto the target grid.
    """

    target_width: int
    target_height: int
    palette: tuple[str, ...] = ()
    max_colors: int = 16
    dither: Dither = Dither.NONE
    downscale: str = "nearest"  # "nearest" | "dominant" | "center"

    def effective_colors(self) -> int:
        return len(self.palette) if self.palette else self.max_colors

    def integer_scale_from(self, src_w: int, src_h: int) -> tuple[int, int] | None:
        """Return integer (x_scale, y_scale) if the source divides cleanly onto the
        target grid, else None — a None signals grid misalignment (blurry pixels),
        so the caller should adjust the render size."""
        if self.target_width <= 0 or self.target_height <= 0:
            raise ValueError("target dims must be positive")
        if src_w % self.target_width or src_h % self.target_height:
            return None
        return src_w // self.target_width, src_h // self.target_height
