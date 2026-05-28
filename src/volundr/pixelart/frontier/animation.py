"""Skeleton/pose-driven sprite animation — frontier scaffolding.

Approach (see docs/PIXEL_ART_FRONTIERS.md): drive each frame with a pose control
image from a pose sequence, hold identity with a whole-image IP-Adapter, run every
frame through the SAME locked PixelSet (one palette/grid → no per-frame flicker),
then pack the frames into a sprite sheet.

This builder produces one GenerationParams per pose frame plus the sheet layout.
It does not solve temporal coherence on its own — that's the frontier the plan
addresses (locked-palette quantization, keyframe propagation, video-diffusion
options). What it guarantees: shared identity, a single locked look across all
frames, and a correct sheet atlas.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from volundr.models import ControlNetSpec, GenerationParams, IPAdapterSpec
from volundr.pixelart.pixelset import PixelSet
from volundr.pixelart.sheet import Atlas, pack_grid


@dataclass(frozen=True)
class PoseFrame:
    """One frame's pose/control image (e.g. an OpenPose render) and its order."""

    control_image: str
    index: int


@dataclass(frozen=True)
class AnimationSpec:
    action: str = "walk"  # "walk" | "run" | "attack" | ...
    fps: int = 8
    loop: bool = True


@dataclass
class AnimationRequest:
    reference: str  # identity reference -> whole-image IP-Adapter
    base: GenerationParams
    frames: tuple[PoseFrame, ...]
    spec: AnimationSpec = field(default_factory=AnimationSpec)
    control_model: str = "openpose"
    control_weight: float = 0.8
    ip_weight: float = 0.7
    ip_model: str = "ip_adapter_sdxl"
    pixel_set: PixelSet | None = None

    def _ordered(self) -> list[PoseFrame]:
        return sorted(self.frames, key=lambda f: f.index)

    def per_frame_params(self) -> list[GenerationParams]:
        base = self.pixel_set.apply(self.base) if self.pixel_set else self.base
        identity = IPAdapterSpec(
            image=self.reference, weight=self.ip_weight, model=self.ip_model, method="full"
        )
        out = []
        for frame in self._ordered():
            pose = ControlNetSpec(
                model=self.control_model,
                image=frame.control_image,
                weight=self.control_weight,
            )
            out.append(
                replace(
                    base,
                    ip_adapters=base.ip_adapters + (identity,),
                    controlnets=base.controlnets + (pose,),
                )
            )
        return out

    def sheet(self, frame_w: int, frame_h: int, columns: int | None = None) -> Atlas:
        names = [f"{self.spec.action}_{f.index}" for f in self._ordered()]
        return pack_grid(frame_w, frame_h, len(self.frames), columns=columns, names=names)
