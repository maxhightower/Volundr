"""Generative paintbrush (v2 feature) — capture a style, paint it onto a region.

Feasibility verdict: yes, with no per-brush training. A "brush" is a captured
reference patch (used as an IP-Adapter image prompt) plus a text string. Painting
it onto a mask becomes an InvokeAI regional-guidance layer carrying that
IP-Adapter + text, run as a masked inpaint denoise.

The crucial design point: the motivating brushes (shading, crosshatch, highlight,
wallpaper texture) are STYLE/TEXTURE transfer, not object insertion — so this uses
style injection (IP-Adapter, "style" method) anchored by a ControlNet derived from
the *target* canvas, so existing forms get re-rendered in the new technique rather
than replaced. (Paint-by-Example / AnyDoor are deliberately NOT used — they paste
objects.) Honest caveat: fine high-frequency techniques like crosshatch transfer
unreliably because CLIP-image style is low-frequency; lean on the structure
ControlNet + the text string and set expectations to "hatch-like".

This module builds the conditioning (GPU-independent, testable). Capturing the
patch (cropping the reference image) and deriving the structure image (running a
lineart/scribble preprocessor on the target) happen in the UI / on the GPU box;
here they're passed in as handles.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from volundr.models import (
    ControlNetSpec,
    GenerationParams,
    IPAdapterSpec,
    RegionPrompt,
)


@dataclass(frozen=True)
class Brush:
    """A reusable captured style/technique.

    `patch` is a handle to the cropped reference region (the IP-Adapter image).
    `structure_control` names the ControlNet to derive from the *target* so forms
    survive ("lineart"/"scribble"/"depth", or None to let the brush reshape
    freely). `denoise_strength` is how much the painted region is re-rendered
    (1.0 = ignore what's there, ~0.6-0.8 = restyle existing forms).
    """

    name: str
    patch: str
    text: str = ""
    ip_weight: float = 0.6
    ip_method: str = "style"
    structure_control: str | None = "lineart"
    structure_weight: float = 0.7
    denoise_strength: float = 0.7
    seamless: bool = False  # tileable generation, for wallpaper/material brushes

    @classmethod
    def capture(cls, name: str, patch: str, text: str = "", **kwargs) -> "Brush":
        """Capture a brush from a selected reference region + description string."""
        return cls(name=name, patch=patch, text=text, **kwargs)

    def region(self, mask: str) -> RegionPrompt:
        """The regional-guidance layer for a stroke on `mask`."""
        return RegionPrompt(
            mask=mask,
            positive=self.text,
            ip_adapter=IPAdapterSpec(
                image=self.patch, weight=self.ip_weight, method=self.ip_method
            ),
        )

    def structure(self, structure_image: str) -> ControlNetSpec:
        """The target-derived structure ControlNet (call only if structure_control set)."""
        if self.structure_control is None:
            raise ValueError(f"brush {self.name!r} has no structure_control")
        return ControlNetSpec(
            model=self.structure_control,
            image=structure_image,
            weight=self.structure_weight,
        )


def apply_brush(
    params: GenerationParams,
    brush: Brush,
    mask: str,
    structure_image: str | None = None,
) -> GenerationParams:
    """Apply a brush stroke to `params`, returning updated params.

    Appends the brush's region (text + IP-Adapter) and, when a `structure_image`
    is supplied and the brush requests structure locking, a target-derived
    ControlNet. Sets `denoising_start` from the brush's denoise strength so the
    masked region is re-rendered rather than regenerated from scratch.
    """
    regions = params.regions + (brush.region(mask),)

    controlnets = params.controlnets
    if brush.structure_control is not None and structure_image is not None:
        controlnets = controlnets + (brush.structure(structure_image),)

    return replace(
        params,
        regions=regions,
        controlnets=controlnets,
        denoising_start=1.0 - brush.denoise_strength,
    )
