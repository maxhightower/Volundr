"""PixelSet — the locked "house style" every sprite and frame inherits.

This is the consistency engine and Völundr's structural edge over tools that buy
consistency through training. When the user approves a result "into the set", we
freeze the levers that make a sprite collection look uniform: the post-pass spec
(palette + grid + dither), the base seed, the pixel-art LoRA stack, and a style
prompt fragment. Applying the set to a fresh request biases generation toward the
same look; applying the same `spec` to every frame's post pass is what kills the
per-frame palette flicker that plagues open animation pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from volundr.models import GenerationParams, LoraSpec
from volundr.pixelart.spec import PixelArtSpec


@dataclass(frozen=True)
class PixelSet:
    name: str
    spec: PixelArtSpec
    base_seed: int | None = None
    loras: tuple[LoraSpec, ...] = ()
    style: str = ""  # locked style fragment appended to prompts

    @classmethod
    def lock(
        cls,
        name: str,
        spec: PixelArtSpec,
        params: GenerationParams | None = None,
        style: str = "",
    ) -> "PixelSet":
        """Freeze a set, optionally seeding seed/LoRAs from an approved generation."""
        if params is None:
            return cls(name=name, spec=spec, style=style)
        return cls(
            name=name,
            spec=spec,
            base_seed=params.seed,
            loras=params.loras,
            style=style,
        )

    def apply(self, params: GenerationParams) -> GenerationParams:
        """Bias a fresh request toward the locked look.

        Merges the locked LoRAs (locked weights win on name collisions), appends
        the style fragment, and pins the seed when one is locked. The `spec` is
        applied separately at the post-processing stage.
        """
        merged: dict[str, LoraSpec] = {l.name: l for l in params.loras}
        for l in self.loras:
            merged[l.name] = l  # locked LoRA overrides an incoming one of same name

        prompt = params.prompt
        if self.style and self.style not in prompt:
            prompt = ", ".join(p for p in (prompt, self.style) if p)

        seed = self.base_seed if self.base_seed is not None else params.seed

        return replace(params, prompt=prompt, loras=tuple(merged.values()), seed=seed)
