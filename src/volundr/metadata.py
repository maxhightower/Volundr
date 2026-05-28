"""Reproducible workflow metadata embedded in saved PNGs.

A strong community expectation (every UI ships it): a generated image should
carry the parameters that made it, so it can be dropped back in to reproduce or
remix. This embeds a JSON payload in a PNG `tEXt` chunk using only the stdlib
(no Pillow), and reads it back.
"""

from __future__ import annotations

import json
import struct
import zlib

from volundr.models import GenerationParams
from volundr.serialization import params_from_dict, params_to_dict

__version__ = "0.1.0"
_PNG_SIG = b"\x89PNG\r\n\x1a\n"
_KEYWORD = "volundr"


def build_metadata(params: GenerationParams, extra: dict | None = None) -> dict:
    meta = {"generator": "volundr", "version": __version__, "params": params_to_dict(params)}
    if extra:
        meta.update(extra)
    return meta


def parse_metadata(meta: dict) -> GenerationParams:
    return params_from_dict(meta.get("params", {}))


def _chunk(typ: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + typ
        + data
        + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF)
    )


def embed_in_png(png: bytes, meta: dict, keyword: str = _KEYWORD) -> bytes:
    """Return `png` with `meta` added as a tEXt chunk (inserted before IEND)."""
    if png[:8] != _PNG_SIG:
        raise ValueError("not a PNG")
    # json defaults to ascii-escaped output, which is valid latin-1 for tEXt.
    payload = keyword.encode("latin-1") + b"\x00" + json.dumps(meta).encode("latin-1")
    chunk = _chunk(b"tEXt", payload)
    iend_len_pos = png.rfind(b"IEND") - 4  # start of IEND's 4-byte length field
    if iend_len_pos < 8:
        raise ValueError("malformed PNG: no IEND")
    return png[:iend_len_pos] + chunk + png[iend_len_pos:]


def extract_from_png(png: bytes, keyword: str = _KEYWORD) -> dict | None:
    """Return the embedded metadata dict, or None if absent."""
    if png[:8] != _PNG_SIG:
        raise ValueError("not a PNG")
    pos = 8
    while pos + 8 <= len(png):
        length = struct.unpack(">I", png[pos : pos + 4])[0]
        typ = png[pos + 4 : pos + 8]
        data = png[pos + 8 : pos + 8 + length]
        if typ == b"tEXt":
            kw, _, txt = data.partition(b"\x00")
            if kw.decode("latin-1") == keyword:
                return json.loads(txt.decode("latin-1"))
        if typ == b"IEND":
            break
        pos += 12 + length
    return None
