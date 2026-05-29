import json

import pytest

from volundr.prompt import StyleLibrary, load_a1111_csv, load_fooocus_json


# --- Fooocus JSON ---------------------------------------------------------
def test_load_fooocus_json_maps_fields(tmp_path):
    data = [
        {"name": "SAI Anime", "prompt": "anime artwork, {prompt}", "negative_prompt": "photo"},
        {"name": "Bare", "prompt": "cinematic"},  # no negative_prompt
    ]
    f = tmp_path / "sdxl_styles_sai.json"
    f.write_text(json.dumps(data), encoding="utf-8")

    presets = load_fooocus_json(f)
    assert len(presets) == 2
    sai = presets[0]
    assert sai.name == "SAI Anime"
    pos, neg = sai.apply("a dragon", "blurry")
    assert pos == "anime artwork, a dragon"
    assert neg == "blurry, photo"
    assert presets[1].negative == ""


def test_load_fooocus_json_tolerates_bom(tmp_path):
    f = tmp_path / "styles.json"
    f.write_bytes(b"\xef\xbb\xbf" + json.dumps([{"name": "X", "prompt": "p"}]).encode("utf-8"))
    assert [p.name for p in load_fooocus_json(f)] == ["X"]


def test_load_fooocus_json_skips_malformed_entries(tmp_path):
    data = [
        {"name": "Good", "prompt": "p"},
        {"name": "", "prompt": "p"},        # empty name -> skip
        {"name": "NoPrompt"},                # missing prompt -> skip
        "not-an-object",                     # wrong type -> skip
    ]
    f = tmp_path / "styles.json"
    f.write_text(json.dumps(data), encoding="utf-8")
    assert [p.name for p in load_fooocus_json(f)] == ["Good"]


def test_load_fooocus_json_rejects_non_list(tmp_path):
    f = tmp_path / "bad.json"
    f.write_text(json.dumps({"name": "X"}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_fooocus_json(f)


# --- A1111 CSV ------------------------------------------------------------
def test_load_a1111_csv_maps_fields(tmp_path):
    f = tmp_path / "styles.csv"
    f.write_text(
        "name,prompt,negative_prompt\n"
        "Ink,\"line art, {prompt}\",color\n"
        "Plain,vivid colors,\n",
        encoding="utf-8",
    )
    presets = load_a1111_csv(f)
    assert [p.name for p in presets] == ["Ink", "Plain"]
    pos, neg = presets[0].apply("a dragon")
    assert pos == "line art, a dragon"
    assert neg == "color"
    # No placeholder -> appended
    assert presets[1].apply("a dragon")[0] == "a dragon, vivid colors"


def test_load_a1111_csv_skips_repeated_header_and_blank_rows(tmp_path):
    f = tmp_path / "styles.csv"
    f.write_text(
        "name,prompt,negative_prompt\n"
        "name,prompt,negative_prompt\n"   # duplicated header row
        "Good,p,\n"
        ",,\n",                            # blank/empty name -> skip
        encoding="utf-8",
    )
    assert [p.name for p in load_a1111_csv(f)] == ["Good"]


# --- StyleLibrary ---------------------------------------------------------
def test_library_namespaces_and_keeps_colliding_names(tmp_path):
    sai = tmp_path / "sai.json"
    sai.write_text(json.dumps([{"name": "Anime", "prompt": "sai {prompt}"}]), encoding="utf-8")
    mk = tmp_path / "mk.json"
    mk.write_text(json.dumps([{"name": "Anime", "prompt": "mk {prompt}"}]), encoding="utf-8")

    lib = StyleLibrary()
    lib.add(load_fooocus_json(sai), source="sai")
    lib.add(load_fooocus_json(mk), source="mk")

    assert len(lib) == 2
    assert lib.names() == ["mk/Anime", "sai/Anime"]
    assert lib.get("sai/Anime").apply("x")[0] == "sai x"
    assert lib.get("mk/Anime").apply("x")[0] == "mk x"
    assert "sai/Anime" in lib


def test_library_add_requires_source(tmp_path):
    lib = StyleLibrary()
    with pytest.raises(ValueError):
        lib.add([], source="")


def test_library_load_dir_uses_file_stem_as_source(tmp_path):
    (tmp_path / "fooocus_pack.json").write_text(
        json.dumps([{"name": "A", "prompt": "p"}]), encoding="utf-8"
    )
    (tmp_path / "a1111_pack.csv").write_text(
        "name,prompt,negative_prompt\nB,q,\n", encoding="utf-8"
    )
    (tmp_path / "ignore.txt").write_text("not a pack", encoding="utf-8")

    lib = StyleLibrary()
    lib.load_dir(tmp_path)
    assert lib.names() == ["a1111_pack/B", "fooocus_pack/A"]


def test_library_load_dir_explicit_source_overrides_stem(tmp_path):
    (tmp_path / "pack.json").write_text(
        json.dumps([{"name": "A", "prompt": "p"}]), encoding="utf-8"
    )
    lib = StyleLibrary()
    lib.load_dir(tmp_path, source="custom")
    assert lib.names() == ["custom/A"]
