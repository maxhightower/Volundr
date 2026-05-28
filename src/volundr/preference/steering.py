"""Accumulated steering signal derived from all feedback so far.

This is the foundation the build order grows into: today it aggregates the
*prompt-level* lean (step 5) and the approved/denied seed pools (step 4). Step 7
(latent direction vector) and v2 (LoRA fine-tune on approvals) will extend this
same object with additional signal, but the seam — "turn accumulated feedback
into a bias on the next GenerationParams" — stays here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from volundr.models import GenerationParams
from volundr.preference.feedback import FeedbackStore
from volundr.preference.prompt_lean import PromptLean


@dataclass
class SteeringState:
    lean: PromptLean
    approved_seeds: list[int] = field(default_factory=list)
    denied_seeds: set[int] = field(default_factory=set)

    @classmethod
    def from_store(
        cls,
        store: FeedbackStore,
        base_positive: tuple[str, ...] = (),
        base_negative: tuple[str, ...] = (),
    ) -> "SteeringState":
        lean = PromptLean(base_positive=base_positive, base_negative=base_negative)
        lean.apply(store.all_feedback)
        return cls(
            lean=lean,
            approved_seeds=[g.seed for g in store.approved()],
            denied_seeds={g.seed for g in store.denied()},
        )

    def apply_to(self, params: GenerationParams) -> GenerationParams:
        """Bias a fresh request with the accumulated signal.

        Overrides prompt/negative with the leaned versions. Leaves seed choice to
        the caller (see `seeds.variation_params`); `denied_seeds` is exposed so
        callers can skip seeds the user already rejected.
        """
        positive, negative = self.lean.render()
        return replace(params, prompt=positive, negative_prompt=negative)
