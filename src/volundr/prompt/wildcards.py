"""Dynamic prompts — `{a|b|c}` variants and `__wildcard__` file references.

Mirrors the sd-dynamic-prompts syntax users expect:
  * `{red|blue|green}` picks one option (nestable: `{a|{b|c}}`)
  * `__color__` is replaced by an option from the `wildcards` mapping for "color"

`expand` resolves randomly (seedable for reproducibility); `expand_all` returns
every combination (combinatorial).
"""

from __future__ import annotations

import random
import re

# Innermost variant group (no braces inside) so nesting resolves bottom-up.
_VARIANT = re.compile(r"\{([^{}]*)\}")
_WILDCARD = re.compile(r"__([A-Za-z0-9_\-/]+)__")


def _cleanup(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+,", ",", text)
    return text.strip().strip(",").strip()


def expand(
    template: str,
    rng: random.Random | None = None,
    wildcards: dict[str, list[str]] | None = None,
    max_iters: int = 1000,
) -> str:
    """Resolve one random concrete prompt from `template`."""
    rng = rng or random.Random()
    wildcards = wildcards or {}
    text = template
    for _ in range(max_iters):
        m = _WILDCARD.search(text)
        if m:
            options = wildcards.get(m.group(1), [""])
            text = text[: m.start()] + rng.choice(options) + text[m.end() :]
            continue
        m = _VARIANT.search(text)
        if m:
            text = text[: m.start()] + rng.choice(m.group(1).split("|")) + text[m.end() :]
            continue
        break
    return _cleanup(text)


def expand_all(
    template: str,
    wildcards: dict[str, list[str]] | None = None,
) -> list[str]:
    """Return every combination the template can produce (order-preserving, de-duped)."""
    wildcards = wildcards or {}
    mw = _WILDCARD.search(template)
    mv = _VARIANT.search(template)
    if not mw and not mv:
        return [_cleanup(template)]

    # Resolve whichever token appears first.
    if mw and (not mv or mw.start() <= mv.start()):
        options, span = wildcards.get(mw.group(1), [""]), mw.span()
    else:
        options, span = mv.group(1).split("|"), mv.span()

    out: list[str] = []
    for opt in options:
        sub = template[: span[0]] + opt + template[span[1] :]
        out.extend(expand_all(sub, wildcards))
    seen: dict[str, None] = {}
    for x in out:
        seen.setdefault(x, None)
    return list(seen)
