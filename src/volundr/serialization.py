"""(De)serialization for GenerationParams, shared by the feedback store and the
reproducible-metadata feature so the round-trip logic lives in exactly one place.

`params_from_dict` is tolerant (missing keys fall back to dataclass defaults) so
it can parse partial or older metadata embedded in an external PNG.
"""

from __future__ import annotations

from dataclasses import asdict

from volundr.models import (
    ControlNetSpec,
    GenerationParams,
    IPAdapterSpec,
    LoraSpec,
    RegionPrompt,
)


def params_to_dict(params: GenerationParams) -> dict:
    return asdict(params)


def region_from_dict(x: dict) -> RegionPrompt:
    ip = x.get("ip_adapter")
    return RegionPrompt(
        mask=x["mask"],
        positive=x.get("positive", ""),
        negative=x.get("negative", ""),
        ip_adapter=IPAdapterSpec(**ip) if ip else None,
    )


def params_from_dict(d: dict) -> GenerationParams:
    defaults = GenerationParams()
    return GenerationParams(
        prompt=d.get("prompt", defaults.prompt),
        negative_prompt=d.get("negative_prompt", defaults.negative_prompt),
        model=d.get("model", defaults.model),
        seed=d.get("seed", defaults.seed),
        steps=d.get("steps", defaults.steps),
        cfg_scale=d.get("cfg_scale", defaults.cfg_scale),
        width=d.get("width", defaults.width),
        height=d.get("height", defaults.height),
        denoising_start=d.get("denoising_start", defaults.denoising_start),
        loras=tuple(LoraSpec(**x) for x in d.get("loras", [])),
        controlnets=tuple(ControlNetSpec(**x) for x in d.get("controlnets", [])),
        regions=tuple(region_from_dict(x) for x in d.get("regions", [])),
        ip_adapters=tuple(IPAdapterSpec(**x) for x in d.get("ip_adapters", [])),
        variation_seed=d.get("variation_seed", defaults.variation_seed),
        variation_strength=d.get("variation_strength", defaults.variation_strength),
    )
