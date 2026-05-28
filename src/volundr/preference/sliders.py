"""Concept Sliders — build step 6, the differentiated "lean" feature.

Background
----------
MCDPO (the original plan) has no public code and requires fine-tuning SDXL with
a custom objective before its sliders exist. Concept Sliders (Gandikota et al.,
https://sliders.baulab.info/, repo https://github.com/rohitgandikota/sliders)
get us labeled, bidirectional lean axes far more cheaply, and split into two
integration tiers that share this module + the preference loop.

Tier A — signed-weight LoRA sliders (recommended first; ships on stock InvokeAI)
    A Concept Slider is a low-rank (LoRA) adaptor trained offline so that scaling
    it by +alpha enhances an attribute and -alpha suppresses it. Training is fast
    on 24GB (rank ~4, minutes per axis) using the rohitgandikota/sliders repo with
    a positive/negative prompt pair (e.g. "intricate scales" vs "smooth skin").
    INFERENCE NEEDS NO NEW DIFFUSION CODE: a slider is just a LoRA loaded at a
    *signed, continuously adjustable* weight. InvokeAI's `sdxl_lora_loader.weight`
    is a float multiplier on the LoRA delta and accepts negatives, so the slider
    value maps straight onto it (see `SliderBank.active_loras`). The UI exposes a
    labeled slider from `min_value`..`max_value`; "lean toward/away" nudges it.

Tier B — SEGA-style inference-time guidance (no per-axis training)
    Semantic Guidance (SEGA, Brack et al., https://arxiv.org/abs/2301.12247) and
    the original Concept Sliders "guidance" variant steer by adding concept
    *directions* to the noise prediction at each step, with a per-axis scale — i.e.
    sliders with zero training, defined by a text concept on the fly. The direction
    for axis e is `eps(x, c_e) - eps(x, c_uncond)` (extra UNet passes); the combined
    prediction is `cfg_pred + sum_e gate_e * scale_e * direction_e`. That sum is
    `apply_sega_guidance` below (pure, testable). It belongs at InvokeAI's
    `post_combine_noise_preds` denoise hook — see
    `invokeai_extension/volundr_sega.py` for the (fork-side) ExtensionBase wrapper.
    Cost: each active axis adds UNet forward passes per step, so cap concurrent
    Tier-B axes on 24GB.

Both tiers feed the same `SliderBank`, so the UI and preference loop don't care
which one backs a given axis.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Sequence, TypeVar

from volundr.models import Feedback, FeedbackKind, LoraSpec


@dataclass(frozen=True)
class Slider:
    """A labeled, bidirectional control axis.

    `lora` set  -> Tier A (signed-weight LoRA).
    `concept_prompt` set -> Tier B (SEGA guidance direction from this text).
    A slider may set both; `active_loras` uses `lora`, the guidance hook uses
    `concept_prompt`.
    """

    label: str
    value: float = 0.0
    min_value: float = -1.0
    max_value: float = 1.0
    lora: str | None = None
    concept_prompt: str | None = None

    def clamped_value(self) -> float:
        return max(self.min_value, min(self.max_value, self.value))

    def nudged(self, delta: float) -> "Slider":
        return replace(self, value=max(self.min_value, min(self.max_value, self.value + delta)))


class SliderBank:
    """Holds the active sliders and folds lean feedback into their values.

    Lean feedback targets a slider by naming its label in `Feedback.tokens`
    (LEAN_TOWARD raises the value, LEAN_AWAY lowers it). This is the dimensional
    sibling of `PromptLean`: PromptLean leans on prompt *tokens*, SliderBank leans
    on labeled *axes*.
    """

    def __init__(self, sliders: Iterable[Slider] = (), step: float = 0.1) -> None:
        self.step = step
        self._by_label: dict[str, Slider] = {s.label: s for s in sliders}

    def __contains__(self, label: str) -> bool:
        return label in self._by_label

    def get(self, label: str) -> Slider:
        return self._by_label[label]

    def values(self) -> dict[str, float]:
        return {label: s.clamped_value() for label, s in self._by_label.items()}

    def apply(self, feedback: Iterable[Feedback]) -> "SliderBank":
        for f in feedback:
            if f.kind is FeedbackKind.LEAN_TOWARD:
                sign = 1.0
            elif f.kind is FeedbackKind.LEAN_AWAY:
                sign = -1.0
            else:
                continue
            for label in f.tokens:
                slider = self._by_label.get(label)
                if slider is not None:
                    self._by_label[label] = slider.nudged(sign * self.step * f.strength)
        return self

    def active_loras(self) -> tuple[LoraSpec, ...]:
        """Tier A: sliders with a LoRA and a non-zero value, as signed-weight LoRAs."""
        out = []
        for s in self._by_label.values():
            v = s.clamped_value()
            if s.lora is not None and v != 0.0:
                out.append(LoraSpec(name=s.lora, weight=v))
        return tuple(out)

    def guidance_axes(self) -> tuple[Slider, ...]:
        """Tier B: sliders driven by a concept prompt with a non-zero value."""
        return tuple(
            s
            for s in self._by_label.values()
            if s.concept_prompt is not None and s.clamped_value() != 0.0
        )


# --- Tier B core: SEGA-style guidance combination ------------------------
T = TypeVar("T")  # an array-like supporting + and scalar *  (numpy/torch/float)


@dataclass
class GuidanceEdit:
    """One axis's contribution to the noise prediction at the current step.

    `direction` is `eps(concept) - eps(uncond)` (same shape as the prediction).
    `scale` is the signed slider value; `gate` in [0, 1] models SEGA warmup /
    salience masking applied upstream (1.0 = fully active).
    """

    direction: T
    scale: float
    gate: float = 1.0


def apply_sega_guidance(cfg_pred: T, edits: Sequence[GuidanceEdit]) -> T:
    """Add the per-axis guidance directions onto the CFG noise prediction.

    `final = cfg_pred + sum_e gate_e * scale_e * direction_e`

    Deliberately array-library-agnostic: works on numpy arrays, torch tensors, or
    plain floats (used in tests) — anything supporting `+` and scalar `*`.
    """
    result = cfg_pred
    for e in edits:
        result = result + e.direction * (e.scale * e.gate)
    return result
