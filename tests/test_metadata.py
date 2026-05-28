import json
import struct
import zlib

import pytest

from volundr.metadata import (
    build_metadata,
    embed_in_png,
    extract_from_png,
    parse_metadata,
)
from volundr.models import GenerationParams, LoraSpec

_PNG_SIG = b"\x89PNG\r\n\x1a\n"


def _chunk(typ, data):
    return struct.pack(">I", len(data)) + typ + data + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF)


def minimal_png():
    return (
        _PNG_SIG
        + _chunk(b"IHDR", b"\x00" * 13)
        + _chunk(b"IDAT", zlib.compress(b"\x00"))
        + _chunk(b"IEND", b"")
    )


def test_build_and_parse_roundtrip():
    params = GenerationParams(prompt="a dragon", seed=99, loras=(LoraSpec("style", 0.7),))
    meta = build_metadata(params, extra={"app": "volundr"})
    assert meta["generator"] == "volundr"
    assert meta["app"] == "volundr"
    assert parse_metadata(meta) == params


def test_embed_and_extract_from_png():
    params = GenerationParams(prompt="a dragon", seed=7)
    meta = build_metadata(params)
    png = embed_in_png(minimal_png(), meta)

    assert png.startswith(_PNG_SIG)
    assert png.rfind(b"IEND") > png.rfind(b"tEXt")  # IEND stays last
    extracted = extract_from_png(png)
    # JSON normalizes tuples -> lists; the round-trip to params is the real check.
    assert extracted == json.loads(json.dumps(meta))
    assert parse_metadata(extracted) == params


def test_extract_returns_none_when_absent():
    assert extract_from_png(minimal_png()) is None


def test_embed_rejects_non_png():
    with pytest.raises(ValueError):
        embed_in_png(b"not a png", {})
