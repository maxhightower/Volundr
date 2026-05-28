"""Best-effort builder for an InvokeAI SDXL generation graph.

SCOPE / HONESTY NOTE
--------------------
InvokeAI's node schema is version-specific and this builder has NOT been run
against a live instance (no GPU in the scaffolding environment). It produces a
*structurally* coherent text2img graph (model -> LoRA chain -> SDXL compel
prompts -> noise -> denoise -> latents-to-image) for the parameters we're
confident map cleanly. Validate node type names and field names against the
target instance's `/openapi.json` before trusting runtime behaviour.

Features that need intricate, version-specific wiring are intentionally NOT
faked here — the builder raises `NotImplementedError` rather than silently
dropping them, so they get wired (and validated) on the GPU box:
  * regional guidance / control layers (`params.regions`)
  * sub-seed variation blending (`params.variation_seed`)
"""

from __future__ import annotations

from volundr.models import GenerationParams


def build_sdxl_graph(params: GenerationParams) -> dict:
    if params.regions:
        raise NotImplementedError(
            "regional guidance wiring is deferred to live InvokeAI validation; "
            "build it as control layers on the GPU box (build step 3)"
        )
    if params.variation_seed is not None:
        raise NotImplementedError(
            "sub-seed variation blending must be wired against the live noise "
            "node and validated (build step 4)"
        )

    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    def add_edge(src_node: str, src_field: str, dst_node: str, dst_field: str) -> None:
        edges.append(
            {
                "source": {"node_id": src_node, "field": src_field},
                "destination": {"node_id": dst_node, "field": dst_field},
            }
        )

    nodes["model"] = {
        "id": "model",
        "type": "sdxl_model_loader",
        "model": {"model_name": params.model, "base_model": "sdxl"},
    }

    # LoRA chain: thread unet/clip/clip2 through each loader in order.
    unet_src, clip_src, clip2_src = "model", "model", "model"
    for i, lora in enumerate(params.loras):
        node_id = f"lora_{i}"
        nodes[node_id] = {
            "id": node_id,
            "type": "sdxl_lora_loader",
            "lora": {"model_name": lora.name, "base_model": "sdxl"},
            "weight": lora.weight,
        }
        add_edge(unet_src, "unet", node_id, "unet")
        add_edge(clip_src, "clip", node_id, "clip")
        add_edge(clip2_src, "clip2", node_id, "clip2")
        unet_src = clip_src = clip2_src = node_id

    nodes["pos"] = {"id": "pos", "type": "sdxl_compel_prompt", "prompt": params.prompt}
    nodes["neg"] = {
        "id": "neg",
        "type": "sdxl_compel_prompt",
        "prompt": params.negative_prompt,
    }
    for n in ("pos", "neg"):
        add_edge(clip_src, "clip", n, "clip")
        add_edge(clip2_src, "clip2", n, "clip2")

    nodes["noise"] = {
        "id": "noise",
        "type": "noise",
        "seed": params.seed,
        "width": params.width,
        "height": params.height,
    }

    denoise = {
        "id": "denoise",
        "type": "denoise_latents",
        "steps": params.steps,
        "cfg_scale": params.cfg_scale,
        "denoising_start": 0.0,
        "denoising_end": 1.0,
    }
    nodes["denoise"] = denoise
    add_edge(unet_src, "unet", "denoise", "unet")
    add_edge("pos", "conditioning", "denoise", "positive_conditioning")
    add_edge("neg", "conditioning", "denoise", "negative_conditioning")
    add_edge("noise", "noise", "denoise", "noise")

    # ControlNet (scribble/lineart) — the sketch conditioning of build step 2.
    if params.controlnets:
        control_ids = []
        for i, cn in enumerate(params.controlnets):
            cid = f"controlnet_{i}"
            nodes[cid] = {
                "id": cid,
                "type": "controlnet",
                "control_model": {"model_name": cn.model, "base_model": "sdxl"},
                "control_weight": cn.weight,
                "begin_step_percent": cn.begin_step_percent,
                "end_step_percent": cn.end_step_percent,
                "image": {"image_name": cn.image},
            }
            control_ids.append(cid)
        if len(control_ids) == 1:
            add_edge(control_ids[0], "control", "denoise", "control")
        else:
            nodes["control_collect"] = {"id": "control_collect", "type": "collect"}
            for cid in control_ids:
                add_edge(cid, "control", "control_collect", "item")
            add_edge("control_collect", "collection", "denoise", "control")

    nodes["l2i"] = {"id": "l2i", "type": "l2i", "fp32": False}
    add_edge("model", "vae", "l2i", "vae")
    add_edge("denoise", "latents", "l2i", "latents")

    return {"id": "volundr_sdxl", "nodes": nodes, "edges": edges}
