"""Style presets — named prompt templates (A1111 / Fooocus convention).

A preset wraps the user's prompt: if the template contains `{prompt}` it's
substituted in place, otherwise the style text is appended. Negative text merges
into the negative prompt.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StylePreset:
    name: str
    positive: str = "{prompt}"
    negative: str = ""

    def apply(self, prompt: str, negative_prompt: str = "") -> tuple[str, str]:
        if "{prompt}" in self.positive:
            pos = self.positive.replace("{prompt}", prompt)
        else:
            pos = ", ".join(p for p in (prompt, self.positive) if p)
        neg = ", ".join(p for p in (negative_prompt, self.negative) if p)
        return pos, neg
