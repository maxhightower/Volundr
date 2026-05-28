"""Sprite-sheet / texture-atlas packing (pure layout geometry).

`pack_grid` lays out uniform frames (animation cycles, directional sets) in a
fixed grid; `pack_shelf` packs variably-sized cells (tilesets, mixed assets) with
a simple shelf/row algorithm. Both emit an `Atlas` with a JSON-serializable
`to_dict` so the GPU box / engine can blit and an importer can read frame rects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Frame:
    name: str
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class Atlas:
    width: int
    height: int
    frames: tuple[Frame, ...] = ()

    def to_dict(self) -> dict:
        return {
            "size": {"w": self.width, "h": self.height},
            "frames": {
                f.name: {"x": f.x, "y": f.y, "w": f.w, "h": f.h} for f in self.frames
            },
        }

    def overlaps(self) -> bool:
        boxes = [(f.x, f.y, f.x + f.w, f.y + f.h) for f in self.frames]
        for i in range(len(boxes)):
            ax0, ay0, ax1, ay1 = boxes[i]
            for j in range(i + 1, len(boxes)):
                bx0, by0, bx1, by1 = boxes[j]
                if ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1:
                    return True
        return False


def pack_grid(
    frame_w: int,
    frame_h: int,
    count: int,
    columns: int | None = None,
    padding: int = 0,
    names: list[str] | None = None,
) -> Atlas:
    """Pack `count` uniform frames into a grid.

    Defaults to a near-square column count. `names` (if given) labels frames in
    order; otherwise they're "0", "1", ...
    """
    if count <= 0:
        return Atlas(width=0, height=0)
    if columns is None:
        columns = math.ceil(math.sqrt(count))
    columns = max(1, columns)
    rows = math.ceil(count / columns)

    if names is not None and len(names) != count:
        raise ValueError("names length must equal count")

    frames = []
    for i in range(count):
        col, row = i % columns, i // columns
        x = padding + col * (frame_w + padding)
        y = padding + row * (frame_h + padding)
        frames.append(Frame(names[i] if names else str(i), x, y, frame_w, frame_h))

    width = padding + columns * (frame_w + padding)
    height = padding + rows * (frame_h + padding)
    return Atlas(width=width, height=height, frames=tuple(frames))


def pack_shelf(cells: list[tuple[str, int, int]], max_width: int, padding: int = 0) -> Atlas:
    """Shelf-pack variably-sized `(name, w, h)` cells within `max_width`.

    Rows ("shelves") grow to the tallest cell placed on them; a new shelf starts
    when the next cell would exceed `max_width`.
    """
    x = padding
    y = padding
    shelf_h = 0
    used_w = 0
    frames = []
    for name, w, h in cells:
        if x > padding and x + w + padding > max_width:
            # wrap to next shelf
            y += shelf_h + padding
            x = padding
            shelf_h = 0
        frames.append(Frame(name, x, y, w, h))
        x += w + padding
        shelf_h = max(shelf_h, h)
        used_w = max(used_w, x)
    height = y + shelf_h + padding if frames else 0
    return Atlas(width=used_w if frames else 0, height=height, frames=tuple(frames))
