"""Records generations and the user's reactions, with JSON persistence.

Persistence matters even for single-user local work: a session's accumulated
approve/deny history is what later steps (latent direction in step 7, optional
fine-tune in v2) train on.
"""

from __future__ import annotations

import json
from pathlib import Path

from volundr.models import Feedback, FeedbackKind, GenerationResult
from volundr.serialization import params_from_dict, params_to_dict


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

    def ratings(self) -> dict[str, float]:
        """Latest star rating per generation (gallery ratings double as preference signal)."""
        out: dict[str, float] = {}
        for f in self._feedback:
            if f.kind is FeedbackKind.RATE:
                out[f.generation_id] = f.strength  # later overwrites earlier
        return out

    def top_rated(self, min_stars: float = 4.0) -> list[GenerationResult]:
        return [
            self._generations[gid]
            for gid, stars in self.ratings().items()
            if stars >= min_stars
        ]

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
def _result_to_dict(r: GenerationResult) -> dict:
    return {
        "id": r.id,
        "created_at": r.created_at,
        "image_path": r.image_path,
        "params": params_to_dict(r.params),
    }


def _result_from_dict(d: dict) -> GenerationResult:
    return GenerationResult(
        params=params_from_dict(d["params"]),
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
