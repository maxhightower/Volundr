"""Scaffolding for the two hard frontiers: directional rotation + animation.

These encode the *approach* — identity IP-Adapter + per-direction/per-frame
control + a locked PixelSet for consistency + sheet packing — as GPU-independent,
testable request builders. Final visual quality needs the running fork on a GPU
(and, for top quality, purpose-trained models). See docs/PIXEL_ART_FRONTIERS.md
for the full plan, milestones, fallbacks, and evaluation.
"""

from volundr.pixelart.frontier.animation import (
    AnimationRequest,
    AnimationSpec,
    PoseFrame,
)
from volundr.pixelart.frontier.rotation import (
    DIRS_4,
    DIRS_8,
    Direction,
    TurnaroundRequest,
)

__all__ = [
    "AnimationRequest",
    "AnimationSpec",
    "PoseFrame",
    "DIRS_4",
    "DIRS_8",
    "Direction",
    "TurnaroundRequest",
]
