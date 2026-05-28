"""Völundr SEGA-style slider guidance — InvokeAI denoise extension (Tier B).

THIS FILE IS FORK-SIDE CODE, not part of the importable `volundr` package. It is
installed into the Völundr InvokeAI fork (it imports `invokeai.backend...`, which
is only present there) and is intentionally excluded from this repo's test suite.
The pure, testable math it relies on lives in
`volundr.preference.sliders.apply_sega_guidance` (unit-tested).

WHAT IT DOES
------------
Implements inference-time "lean" sliders defined by a text concept (no per-axis
training) by hooking InvokeAI's modular denoise loop. Research (read against
InvokeAI source, v2026-05-27) confirmed the relevant hooks exist:
`ExtensionCallbackType.PRE_UNET` and `.POST_COMBINE_NOISE_PREDS`.

Per the architecture finding, extension *registration* is hardcoded in
`DenoiseLatents`, so ship this via a custom denoise node that instantiates an
`ExtensionsManager` and adds this extension (rather than patching core).

APPROACH (SEGA / Concept-Sliders-guidance)
------------------------------------------
For each active concept axis e with signed scale s_e:
  1. PRE_UNET: append the axis's concept conditioning to the batch so the UNet
     also predicts noise for `c_e` in the same forward pass (keeps it to one
     batched pass rather than N sequential passes — the 24GB-friendly path).
  2. POST_COMBINE_NOISE_PREDS: pull eps(c_e) and eps(uncond), form the direction
     `d_e = eps(c_e) - eps(uncond)`, optionally gate it (warmup steps + a
     magnitude/percentile salience mask, à la SEGA), and add s_e * gate * d_e to
     the already-CFG-combined prediction via `apply_sega_guidance`.

INTEGRATION TODOs (must be wired/validated against the live DenoiseContext on the
GPU box — field names below are indicative, confirm against the fork source):
  * how the batch conditioning is represented and appended in PRE_UNET
  * how to slice per-concept predictions back out in POST_COMBINE_NOISE_PREDS
  * the exact attribute holding the combined prediction on the context object
  * SEGA salience masking (quantile threshold on |d_e|) using torch
"""

from __future__ import annotations

from dataclasses import dataclass

# Fork-side imports (present only inside the InvokeAI fork environment):
from invokeai.backend.stable_diffusion.extensions.base import (  # type: ignore
    ExtensionBase,
    callback,
)
from invokeai.backend.stable_diffusion.extension_callback_type import (  # type: ignore
    ExtensionCallbackType,
)

from volundr.preference.sliders import GuidanceEdit, apply_sega_guidance


@dataclass
class ConceptAxis:
    label: str
    concept_prompt: str
    scale: float          # signed slider value (lean toward +, lean away -)
    warmup_steps: int = 5  # SEGA: hold off applying until the layout settles


class VolundrSegaExtension(ExtensionBase):
    """Adds text-defined lean sliders to the denoise loop (no per-axis training)."""

    def __init__(self, axes: list[ConceptAxis]) -> None:
        super().__init__()
        self.axes = [a for a in axes if a.scale != 0.0]
        self._step = 0

    @callback(ExtensionCallbackType.PRE_UNET)
    def stage_concept_conditioning(self, ctx) -> None:  # noqa: ANN001 (DenoiseContext)
        # TODO(GPU box): append each axis's concept conditioning to ctx's batch so
        # the UNet predicts eps(c_e) alongside cond/uncond in one forward pass.
        # Encode `axis.concept_prompt` with the same compel path the prompt nodes use.
        ...

    @callback(ExtensionCallbackType.POST_COMBINE_NOISE_PREDS)
    def steer(self, ctx) -> None:  # noqa: ANN001 (DenoiseContext)
        edits: list[GuidanceEdit] = []
        for i, axis in enumerate(self.axes):
            gate = 0.0 if self._step < axis.warmup_steps else 1.0
            # TODO(GPU box): slice eps(c_e) and eps(uncond) out of the batched
            # prediction for this axis; apply SEGA salience masking to the direction.
            direction = self._direction_for(ctx, i)  # eps(c_e) - eps(uncond)
            edits.append(GuidanceEdit(direction=direction, scale=axis.scale, gate=gate))

        # The one line that is fully specified and unit-tested:
        ctx.noise_pred = apply_sega_guidance(ctx.noise_pred, edits)  # TODO: confirm attr
        self._step += 1

    def _direction_for(self, ctx, axis_index: int):  # noqa: ANN001
        raise NotImplementedError(
            "wire against the live DenoiseContext on the GPU box (see module TODOs)"
        )
