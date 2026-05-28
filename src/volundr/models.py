"""Core data model shared across the generation and preference layers.

These types are engine-agnostic: they describe *what* to generate and *how the
user reacted*, independent of how InvokeAI's graph happens to be wired. The
`invoke` layer translates `GenerationParams` into an InvokeAI graph; the
`preference` layer consumes `GenerationResult` + `Feedback`.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum


@dataclass(frozen=True)
class LoraSpec:
    """A LoRA loaded from disk, applied on top of the base model."""

    name: str
    weight: float = 1.0


@dataclass(frozen=True)
class ControlNetSpec:
    """Sketch conditioning. `image` is a path/handle to the user's scribble."""

    model: str  # e.g. "scribble" / "lineart"
    image: str
    # How strongly the sketch constrains the output (the parameter step 2 verifies).
    weight: float = 1.0
    begin_step_percent: float = 0.0
    end_step_percent: float = 1.0


@dataclass(frozen=True)
class IPAdapterSpec:
    """An image prompt (IP-Adapter) — carries a reference patch into a region.

    This is the substrate for the generative paintbrush: a captured patch becomes
    the `image`. `method` selects the injection style — "style" (InstantStyle-ish
    style-only blocks, best for shading/texture/technique brushes), "full", or
    "composition".
    """

    image: str  # path/handle to the reference patch
    weight: float = 0.6
    model: str = "ip_adapter_sdxl"
    method: str = "style"  # "full" | "style" | "composition"
    begin_step_percent: float = 0.0
    end_step_percent: float = 1.0


@dataclass(frozen=True)
class RegionPrompt:
    """A spatial label tied to a user-drawn mask (InvokeAI regional guidance).

    A region may carry an `ip_adapter` image reference in addition to text — this
    is how a paintbrush stroke applies a captured style patch to its mask.
    """

    mask: str  # path/handle to the mask image
    positive: str = ""
    negative: str = ""
    ip_adapter: "IPAdapterSpec | None" = None


@dataclass(frozen=True)
class GenerationParams:
    """A fully-specified generation request.

    `variation_seed` / `variation_strength` follow the standard sub-seed
    technique (as in A1111 / InvokeAI): hold `seed` fixed and blend in
    `variation_seed` noise by `variation_strength` to get images *near* an
    approved one. This is how "generate variations" (build step 4) stays close
    to an approved result rather than re-rolling from scratch.
    """

    prompt: str = ""
    negative_prompt: str = ""
    model: str = "stable-diffusion-xl-base-1.0"
    seed: int = 0
    steps: int = 30
    cfg_scale: float = 7.0
    width: int = 1024
    height: int = 1024
    # Fraction of the schedule to skip before denoising (inpaint/img2img). 0.0 =
    # generate from pure noise; >0 keeps more of the existing latent (a brush
    # stroke with denoise_strength s sets this to 1 - s).
    denoising_start: float = 0.0
    loras: tuple[LoraSpec, ...] = ()
    controlnets: tuple[ControlNetSpec, ...] = ()
    regions: tuple[RegionPrompt, ...] = ()
    # Whole-image image prompts (vs. per-region ones on RegionPrompt). Used for
    # character-identity conditioning in turnarounds / animation frames.
    ip_adapters: tuple[IPAdapterSpec, ...] = ()
    variation_seed: int | None = None
    variation_strength: float = 0.0

    def with_seed(self, seed: int) -> "GenerationParams":
        return replace(self, seed=seed)


class FeedbackKind(str, Enum):
    APPROVE = "approve"
    DENY = "deny"
    LEAN_TOWARD = "lean_toward"
    LEAN_AWAY = "lean_away"
    RATE = "rate"  # star rating; the star count is carried in Feedback.strength


@dataclass(frozen=True)
class GenerationResult:
    """The outcome of one generation: the params used plus where the image landed."""

    params: GenerationParams
    image_path: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)

    @property
    def seed(self) -> int:
        return self.params.seed


@dataclass(frozen=True)
class Feedback:
    """One user reaction to a generation.

    `tokens` records which descriptive tokens the lean applies to (used by the
    prompt-level lean in build step 5). `strength` scales lean intensity and is
    ignored for approve/deny.
    """

    generation_id: str
    kind: FeedbackKind
    strength: float = 1.0
    tokens: tuple[str, ...] = ()
    created_at: float = field(default_factory=time.time)
