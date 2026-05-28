"""Inline LoRA tags — A1111-style `<lora:name:weight>` parsed out of a prompt.

Lets users stack LoRAs from the prompt box. Returns the cleaned prompt plus the
extracted `LoraSpec`s (weight defaults to 1.0 when omitted), ready to merge into
`GenerationParams.loras`.
"""

from __future__ import annotations

import re

from volundr.models import LoraSpec

_LORA_TAG = re.compile(r"<lora:([^:>]+)(?::(-?\d*\.?\d+))?>")


def extract_loras(prompt: str) -> tuple[str, tuple[LoraSpec, ...]]:
    loras: list[LoraSpec] = []

    def _take(m: "re.Match[str]") -> str:
        weight = float(m.group(2)) if m.group(2) is not None else 1.0
        loras.append(LoraSpec(name=m.group(1), weight=weight))
        return ""

    cleaned = _LORA_TAG.sub(_take, prompt)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+,", ",", cleaned).strip().strip(",").strip()
    return cleaned, tuple(loras)
