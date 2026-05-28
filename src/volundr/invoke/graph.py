"""Builder for an InvokeAI SDXL generation graph.

Node type strings and field names below were validated by reading InvokeAI's
invocation source (invokeai/app/invocations/*.py, version dated 2026-05-27, the
same line a fresh fork tracks): `sdxl_model_loader`, `sdxl_lora_loader`,
`sdxl_compel_prompt`, `noise`, `denoise_latents`, `controlnet`, `l2i`, `collect`.

Model-bearing fields (main model, LoRA, ControlNet model) are typed
`ModelIdentifierField` — `{key, hash, name, base, type}` — whose values come from
the *running install's model DB*, not from a human label. The caller supplies a
`resolve` callable (see `InvokeAIClient.model_resolver`) that maps a friendly
name to that identifier; the builder embeds whatever it returns. This is the one
real correction over the first draft, which used the obsolete
`{model_name, base_model}` shape.

Features needing intricate, version-specific wiring are intentionally NOT faked
— the builder raises `NotImplementedError` rather than silently dropping them, so
they get wired and validated on the GPU box:
  * regional guidance / control layers (`params.regions`)
  * sub-seed variation blending (`params.variation_seed`)
"""

from __future__ import annotations

from typing import Callable

from volundr.models import GenerationParams

# A resolver maps a friendly model name to an InvokeAI ModelIdentifierField dict.
ModelResolver = Callable[[str], dict]


def build_sdxl_graph(params: GenerationParams, resolve: ModelResolver) -> dict:
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
        "model": resolve(params.model),
    }

    # LoRA chain: thread unet/clip/clip2 through each loader in order.
    unet_src, clip_src, clip2_src = "model", "model", "model"
    for i, lora in enumerate(params.loras):
        node_id = f"lora_{i}"
        nodes[node_id] = {
            "id": node_id,
            "type": "sdxl_lora_loader",
            "lora": resolve(lora.name),
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
        "denoising_start": params.denoising_start,
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
                "control_model": resolve(cn.model),
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
