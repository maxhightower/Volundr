"""Records generations and the user's reactions, with JSON persistence.

Persistence matters even for single-user local work: a session's accumulated
approve/deny history is what later steps (latent direction in step 7, optional
fine-tune in v2) train on.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from volundr.models import (
    ControlNetSpec,
    Feedback,
    FeedbackKind,
    GenerationParams,
    GenerationResult,
    LoraSpec,
    RegionPrompt,
)


class FeedbackStore:
    """In-memory store of generations + feedback, serializable to a JSON file."""

    def __init__(self) -> None:
        self._generations: dict[str, GenerationResult] = {}
        self._feedback: list[Feedback] = []

    # --- recording -------------------------------------------------------
    def add_generation(self, result: GenerationResult) -> GenerationResult:
        self._generations[result.id] = result
        return result

    def record(self, feedback: Feedback) -> Feedback:
        if feedback.generation_id not in self._generations:
            raise KeyError(f"unknown generation_id: {feedback.generation_id}")
        self._feedback.append(feedback)
        return feedback

    # --- queries ---------------------------------------------------------
    def generation(self, gen_id: str) -> GenerationResult:
        return self._generations[gen_id]

    def feedback_for(self, gen_id: str) -> list[Feedback]:
        return [f for f in self._feedback if f.generation_id == gen_id]

    def _ids_with(self, kind: FeedbackKind) -> list[str]:
        # Preserve chronological order, de-duplicated.
        seen: dict[str, None] = {}
        for f in self._feedback:
            if f.kind is kind:
                seen.setdefault(f.generation_id, None)
        return list(seen)

    def approved(self) -> list[GenerationResult]:
        return [self._generations[i] for i in self._ids_with(FeedbackKind.APPROVE)]

    def denied(self) -> list[GenerationResult]:
        return [self._generations[i] for i in self._ids_with(FeedbackKind.DENY)]

    @property
    def all_feedback(self) -> list[Feedback]:
        return list(self._feedback)

    # --- persistence -----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "generations": [_result_to_dict(r) for r in self._generations.values()],
            "feedback": [_feedback_to_dict(f) for f in self._feedback],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, data: dict) -> "FeedbackStore":
        store = cls()
        for r in data.get("generations", []):
            store._generations[r["id"]] = _result_from_dict(r)
        for f in data.get("feedback", []):
            store._feedback.append(_feedback_from_dict(f))
        return store

    @classmethod
    def load(cls, path: str | Path) -> "FeedbackStore":
        return cls.from_dict(json.loads(Path(path).read_text()))


# --- (de)serialization helpers -------------------------------------------
def _params_from_dict(d: dict) -> GenerationParams:
    return GenerationParams(
        prompt=d["prompt"],
        negative_prompt=d["negative_prompt"],
        model=d["model"],
        seed=d["seed"],
        steps=d["steps"],
        cfg_scale=d["cfg_scale"],
        width=d["width"],
        height=d["height"],
        loras=tuple(LoraSpec(**x) for x in d["loras"]),
        controlnets=tuple(ControlNetSpec(**x) for x in d["controlnets"]),
        regions=tuple(RegionPrompt(**x) for x in d["regions"]),
        variation_seed=d["variation_seed"],
        variation_strength=d["variation_strength"],
    )


def _result_to_dict(r: GenerationResult) -> dict:
    return {
        "id": r.id,
        "created_at": r.created_at,
        "image_path": r.image_path,
        "params": asdict(r.params),
    }


def _result_from_dict(d: dict) -> GenerationResult:
    return GenerationResult(
        params=_params_from_dict(d["params"]),
        image_path=d["image_path"],
        id=d["id"],
        created_at=d["created_at"],
    )


def _feedback_to_dict(f: Feedback) -> dict:
    return {
        "generation_id": f.generation_id,
        "kind": f.kind.value,
        "strength": f.strength,
        "tokens": list(f.tokens),
        "created_at": f.created_at,
    }


def _feedback_from_dict(d: dict) -> Feedback:
    return Feedback(
        generation_id=d["generation_id"],
        kind=FeedbackKind(d["kind"]),
        strength=d["strength"],
        tokens=tuple(d["tokens"]),
        created_at=d["created_at"],
    )
