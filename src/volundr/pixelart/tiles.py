"""Auto-tiling — 4-bit edge Wang tiles.

For a boolean grid of "filled" cells, each filled cell picks one of 16 tile
variants from the bitmask of which orthogonal neighbours are also filled. This is
the classic 16-tile auto-tiling rule (e.g. for terrain edges/borders). A 47-tile
"blob" variant (which also considers diagonals) is a documented future extension;
16-tile is the clean, universally-supported baseline.

Bit order (LSB->MSB): North=1, East=2, South=4, West=8.
`WANG_4BIT_LAYOUT` maps each 0..15 index to its (col, row) slot in a 4x4 sheet,
so the artist knows which variant to draw where.
"""

from __future__ import annotations

NORTH, EAST, SOUTH, WEST = 1, 2, 4, 8

# Canonical 4x4 sheet layout: index i sits at (i % 4, i // 4).
WANG_4BIT_LAYOUT: dict[int, tuple[int, int]] = {i: (i % 4, i // 4) for i in range(16)}


def wang_index(north: bool, east: bool, south: bool, west: bool) -> int:
    """Tile index 0..15 from the four orthogonal neighbours' filled state."""
    return (
        (NORTH if north else 0)
        | (EAST if east else 0)
        | (SOUTH if south else 0)
        | (WEST if west else 0)
    )


def tilemap_from_grid(grid: list[list[bool]]) -> list[list[int]]:
    """Map a boolean fill grid to per-cell tile indices.

    Filled cells get their Wang index from filled orthogonal neighbours
    (out-of-bounds counts as empty); empty cells map to -1.
    """
    h = len(grid)
    w = len(grid[0]) if h else 0
    if any(len(row) != w for row in grid):
        raise ValueError("grid must be rectangular")

    def filled(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w and grid[r][c]

    out: list[list[int]] = []
    for r in range(h):
        row: list[int] = []
        for c in range(w):
            if not grid[r][c]:
                row.append(-1)
            else:
                row.append(
                    wang_index(
                        north=filled(r - 1, c),
                        east=filled(r, c + 1),
                        south=filled(r + 1, c),
                        west=filled(r, c - 1),
                    )
                )
        out.append(row)
    return out
