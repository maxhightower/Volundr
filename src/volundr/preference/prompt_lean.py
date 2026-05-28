"""Prompt-level lean — build step 5.

The simple version of the "lean" mechanic, shipping before the latent
(Concept Sliders) version: a thumbs-up raises the weight on a generation's
descriptive tokens; a thumbs-down pushes them toward the negative prompt.

Each token accumulates a net score (toward strengths minus away strengths).
Net-positive tokens render into the positive prompt with an increased weight;
net-negative tokens cross into the negative prompt. This keeps the lean
*continuous and accumulating* rather than a one-shot re-roll.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable

from volundr.models import Feedback, FeedbackKind


@dataclass
class PromptLean:
    base_positive: tuple[str, ...] = ()
    base_negative: tuple[str, ...] = ()
    scale: float = 0.25
    max_weight: float = 1.5

    def __post_init__(self) -> None:
        # token -> net score; seed base tokens at neutral so order is stable.
        self._score: "OrderedDict[str, float]" = OrderedDict()
        for t in self.base_positive:
            self._score.setdefault(t, 0.0)
        # Base negatives carry an implicit negative bias so they stay negative.
        self._base_negative = set(self.base_negative)
        for t in self.base_negative:
            self._score.setdefault(t, 0.0)

    def apply(self, feedback: Iterable[Feedback]) -> "PromptLean":
        for f in feedback:
            if f.kind is FeedbackKind.LEAN_TOWARD:
                sign = 1.0
            elif f.kind is FeedbackKind.LEAN_AWAY:
                sign = -1.0
            else:
                continue  # approve/deny are handled by seeds/steering, not prompt
            for token in f.tokens:
                self._score[token] = self._score.get(token, 0.0) + sign * f.strength
        return self

    def _weight(self, score: float) -> float:
        w = 1.0 + abs(score) * self.scale
        return min(w, self.max_weight)

    def positive(self) -> list[tuple[str, float]]:
        out = []
        for token, score in self._score.items():
            in_base_neg = token in self._base_negative
            if score > 0 or (score == 0 and not in_base_neg):
                out.append((token, self._weight(score)))
        return out

    def negative(self) -> list[tuple[str, float]]:
        out = []
        for token, score in self._score.items():
            in_base_neg = token in self._base_negative
            if score < 0 or (score == 0 and in_base_neg):
                out.append((token, self._weight(score)))
        return out

    def render(self) -> tuple[str, str]:
        return render_weighted(self.positive()), render_weighted(self.negative())


def render_weighted(tokens: list[tuple[str, float]]) -> str:
    """Render (token, weight) pairs to A1111/compel-style attention syntax.

    Weight ~1.0 renders as a bare token; otherwise `(token:1.20)`.
    """
    parts = []
    for token, weight in tokens:
        if abs(weight - 1.0) < 1e-6:
            parts.append(token)
        else:
            parts.append(f"({token}:{weight:.2f})")
    return ", ".join(parts)
