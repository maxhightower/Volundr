import pytest

from volundr.invoke.graph import build_sdxl_graph
from volundr.models import ControlNetSpec, GenerationParams, LoraSpec, RegionPrompt


def edge_pairs(graph):
    return {
        (e["source"]["node_id"], e["source"]["field"],
         e["destination"]["node_id"], e["destination"]["field"])
        for e in graph["edges"]
    }


def test_minimal_text2img_graph_structure():
    g = build_sdxl_graph(GenerationParams(prompt="a dragon", seed=5, steps=20))
    nodes = g["nodes"]
    assert {"model", "pos", "neg", "noise", "denoise", "l2i"} <= set(nodes)
    assert nodes["noise"]["seed"] == 5
    assert nodes["denoise"]["steps"] == 20
    pairs = edge_pairs(g)
    assert ("pos", "conditioning", "denoise", "positive_conditioning") in pairs
    assert ("neg", "conditioning", "denoise", "negative_conditioning") in pairs
    assert ("denoise", "latents", "l2i", "latents") in pairs


def test_lora_chain_threads_unet_and_clip():
    g = build_sdxl_graph(
        GenerationParams(loras=(LoraSpec("a"), LoraSpec("b", 0.5)))
    )
    assert g["nodes"]["lora_1"]["weight"] == 0.5
    pairs = edge_pairs(g)
    # model -> lora_0 -> lora_1, then lora_1 feeds denoise unet
    assert ("model", "unet", "lora_0", "unet") in pairs
    assert ("lora_0", "unet", "lora_1", "unet") in pairs
    assert ("lora_1", "unet", "denoise", "unet") in pairs


def test_single_controlnet_wired_directly():
    g = build_sdxl_graph(
        GenerationParams(controlnets=(ControlNetSpec("scribble", "/s.png", weight=0.7),))
    )
    assert g["nodes"]["controlnet_0"]["control_weight"] == 0.7
    assert ("controlnet_0", "control", "denoise", "control") in edge_pairs(g)


def test_multiple_controlnets_use_collect():
    g = build_sdxl_graph(
        GenerationParams(
            controlnets=(
                ControlNetSpec("scribble", "/a.png"),
                ControlNetSpec("lineart", "/b.png"),
            )
        )
    )
    assert "control_collect" in g["nodes"]
    pairs = edge_pairs(g)
    assert ("control_collect", "collection", "denoise", "control") in pairs


def test_regions_not_yet_supported():
    with pytest.raises(NotImplementedError):
        build_sdxl_graph(GenerationParams(regions=(RegionPrompt("/m.png", "wing"),)))


def test_variation_blending_not_yet_supported():
    with pytest.raises(NotImplementedError):
        build_sdxl_graph(GenerationParams(variation_seed=99))
