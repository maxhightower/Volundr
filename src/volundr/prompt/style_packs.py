"""Import community style packs into StylePreset objects.

Supports the two formats users already have:

  * Fooocus / ComfyUI JSON: a list of {"name", "prompt", "negative_prompt"}
    objects, where "prompt" uses the `{prompt}` placeholder convention.
  * Automatic1111 CSV: rows of `name,prompt,negative_prompt` (with a header).

Both map directly onto StylePreset (its `apply()` already substitutes
`{prompt}` or appends when absent), so loading is pure parsing + light
boundary validation. stdlib only — no new dependencies.

StyleLibrary merges packs from multiple sources under namespaced keys
("<source>/<name>") so identically named presets from different packs coexist.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from volundr.prompt.styles import StylePreset


def _coerce(name: object, prompt: object, negative: object) -> StylePreset | None:
    """Build a StylePreset from raw fields, or None if the entry is unusable."""
    if not isinstance(name, str) or not name.strip():
        return None
    if not isinstance(prompt, str) or not prompt.strip():
        return None
    neg = negative if isinstance(negative, str) else ""
    # Default StylePreset.positive is "{prompt}"; preserve that if prompt empty.
    return StylePreset(name=name.strip(), positive=prompt, negative=neg)


def load_fooocus_json(path: str | Path) -> list[StylePreset]:
    """Parse a Fooocus/ComfyUI styles JSON file into StylePresets.

    Malformed or incomplete entries (missing name/prompt) are skipped.
    Raises ValueError if the file is not a JSON list.
    """
    # utf-8-sig tolerates a leading BOM, common in community files.
    text = Path(path).read_text(encoding="utf-8-sig")
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of style objects")
    presets: list[StylePreset] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        preset = _coerce(
            entry.get("name"),
            entry.get("prompt"),
            entry.get("negative_prompt"),
        )
        if preset is not None:
            presets.append(preset)
    return presets


def load_a1111_csv(path: str | Path) -> list[StylePreset]:
    """Parse an Automatic1111 styles.csv into StylePresets.

    Expects columns name, prompt, negative_prompt. Tolerates a leading BOM,
    extra columns, and a literal repeated header row. Rows missing a name or
    prompt are skipped.
    """
    text = Path(path).read_text(encoding="utf-8-sig")
    reader = csv.DictReader(text.splitlines())
    presets: list[StylePreset] = []
    for row in reader:
        name = row.get("name")
        # Guard against a duplicated header row showing up as data.
        if name == "name" and row.get("prompt") == "prompt":
            continue
        preset = _coerce(name, row.get("prompt"), row.get("negative_prompt"))
        if preset is not None:
            presets.append(preset)
    return presets


class StyleLibrary:
    """A name-indexed collection of StylePresets merged from multiple packs.

    Keys are namespaced as "<source>/<name>" so presets with the same name from
    different packs do not clobber each other. The StylePreset's own `.name` is
    left untouched (only the lookup key is namespaced), so generation output is
    unaffected by where a preset came from.
    """

    def __init__(self) -> None:
        self._presets: dict[str, StylePreset] = {}

    def add(self, presets: list[StylePreset], source: str) -> None:
        """Add presets under the given source namespace.

        Within a single call, a later duplicate "<source>/<name>" key wins.
        """
        if not source:
            raise ValueError("source must be a non-empty namespace string")
        for preset in presets:
            self._presets[f"{source}/{preset.name}"] = preset

    def load_dir(self, directory: str | Path, source: str | None = None) -> None:
        """Load every *.json (Fooocus) and *.csv (A1111) file in a directory.

        When `source` is None, each file's stem becomes its source namespace.
        """
        for file in sorted(Path(directory).iterdir()):
            if file.suffix.lower() == ".json":
                presets = load_fooocus_json(file)
            elif file.suffix.lower() == ".csv":
                presets = load_a1111_csv(file)
            else:
                continue
            self.add(presets, source=source or file.stem)

    def get(self, key: str) -> StylePreset:
        """Return the preset for a namespaced "<source>/<name>" key."""
        return self._presets[key]

    def names(self) -> list[str]:
        """Sorted list of namespaced keys."""
        return sorted(self._presets)

    def __len__(self) -> int:
        return len(self._presets)

    def __contains__(self, key: object) -> bool:
        return key in self._presets
