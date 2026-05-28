"""Directional rotation (4/8-way turnarounds) — frontier scaffolding.

Approach (see docs/PIXEL_ART_FRONTIERS.md): hold character identity fixed with a
whole-image IP-Adapter reference, vary the facing per direction via a directional
prompt suffix and (optionally) a per-direction pose/silhouette ControlNet, and run
every view through the same locked PixelSet so the set shares one palette/grid.

This builder produces one GenerationParams per requested direction. It does NOT
achieve true multi-view consistency on its own — that's the open research gap the
plan addresses with reference models / fine-tuning. What it guarantees is that
every view shares identity conditioning, the locked look, and aligned framing.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum

from volundr.models import ControlNetSpec, GenerationParams, IPAdapterSpec
from volundr.pixelart.pixelset import PixelSet


class Direction(str, Enum):
    S = "south"
    SW = "south_west"
    W = "west"
    NW = "north_west"
    N = "north"
    NE = "north_east"
    E = "east"
    SE = "south_east"

    @property
    def facing_phrase(self) -> str:
        return self.value.replace("_", "-")


# Top-down/RPG convention: 4-way is the cardinal subset, 8-way adds diagonals.
DIRS_4: tuple[Direction, ...] = (Direction.S, Direction.W, Direction.N, Direction.E)
DIRS_8: tuple[Direction, ...] = (
    Direction.S, Direction.SW, Direction.W, Direction.NW,
    Direction.N, Direction.NE, Direction.E, Direction.SE,
)


@dataclass
class TurnaroundRequest:
    reference: str  # identity reference image -> whole-image IP-Adapter
    base: GenerationParams
    directions: tuple[Direction, ...] = DIRS_4
    ip_weight: float = 0.7
    ip_model: str = "ip_adapter_sdxl"
    # Optional per-direction control image (e.g. a directional pose/silhouette).
    pose_controls: dict[Direction, str] = field(default_factory=dict)
    control_model: str = "openpose"
    control_weight: float = 0.7
    pixel_set: PixelSet | None = None

    def per_direction_params(self) -> list[tuple[Direction, GenerationParams]]:
        # Lock the house style once; each view then adds its facing + identity.
        base = self.pixel_set.apply(self.base) if self.pixel_set else self.base
        identity = IPAdapterSpec(
            image=self.reference, weight=self.ip_weight, model=self.ip_model, method="full"
        )
        out: list[tuple[Direction, GenerationParams]] = []
        for d in self.directions:
            prompt = ", ".join(p for p in (base.prompt, f"facing {d.facing_phrase}") if p)
            controlnets = base.controlnets
            if d in self.pose_controls:
                controlnets = controlnets + (
                    ControlNetSpec(
                        model=self.control_model,
                        image=self.pose_controls[d],
                        weight=self.control_weight,
                    ),
                )
            params = replace(
                base,
                prompt=prompt,
                ip_adapters=base.ip_adapters + (identity,),
                controlnets=controlnets,
            )
            out.append((d, params))
        return out
