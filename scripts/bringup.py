#!/usr/bin/env python3
"""Install models into a running InvokeAI and validate Völundr end-to-end.

Called by scripts/setup.sh once InvokeAI is up. Can also be run standalone:

    python scripts/bringup.py --url http://127.0.0.1:9090

Steps: wait for InvokeAI health -> queue model installs (idempotent; InvokeAI
skips already-installed) -> wait for installs -> resolve models -> build an SDXL
text2img graph (with the pixel LoRA, and a ControlNet if --sketch is given) ->
enqueue -> wait -> download the result. A green run proves the client, the model
resolver, the graph builder, and the queue round-trip all work against the live
fork (frontier A0/B0 follow, once IP-Adapter graph wiring is added).
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace

import requests

from volundr.invoke import InvokeAIClient, InvokeAIError, build_sdxl_graph
from volundr.models import ControlNetSpec, GenerationParams, LoraSpec

# HF sources to install. Edit freely; all are open/personal-use friendly.
DEFAULT_MODELS = [
    "stabilityai/stable-diffusion-xl-base-1.0",  # SDXL base
    "xinsir/controlnet-scribble-sdxl-1.0",        # scribble ControlNet (sketch)
    "nerijs/pixel-art-xl",                         # pixel-art LoRA
    # IP-Adapter SDXL (identity, for the rotation/animation frontiers):
    "h94/IP-Adapter",
]


def log(msg: str) -> None:
    print(f"[bringup] {msg}", flush=True)


def wait_for_health(url: str, timeout: float = 180.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = requests.get(f"{url}/api/v1/app/version", timeout=5)
            if r.status_code == 200:
                log(f"InvokeAI is up: {r.json()}")
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise SystemExit(f"InvokeAI did not become healthy at {url} within {timeout}s")


def install_models(client: InvokeAIClient, sources: list[str]) -> None:
    for src in sources:
        try:
            client.install_model(src)
            log(f"queued install: {src}")
        except InvokeAIError as e:
            log(f"install request for {src} returned: {e} (may already be installed)")

    # Poll until no install job is still running.
    deadline = time.monotonic() + 3600
    while time.monotonic() < deadline:
        try:
            jobs = client.model_install_jobs()
        except InvokeAIError:
            break
        running = [j for j in jobs if j.get("status") in ("running", "downloading", "waiting")]
        if not running:
            log("model installs settled")
            return
        log(f"{len(running)} install job(s) in progress...")
        time.sleep(5)
    log("WARNING: model installs did not settle within the timeout")


def validate(client: InvokeAIClient, model: str, lora: str, sketch: str | None, out: str) -> None:
    resolve = client.model_resolver()

    params = GenerationParams(
        prompt="a heroic knight, pixel art, 16-bit",
        negative_prompt="blurry, photo",
        model=model,
        seed=42,
        steps=24,
        loras=(LoraSpec(lora, 1.0),),
    )
    if sketch:
        # exercises the ControlNet path; the sketch must already be an InvokeAI image name
        params = replace(params, controlnets=(ControlNetSpec("scribble", sketch, weight=0.7),))

    log("building SDXL graph...")
    graph = build_sdxl_graph(params, resolve=resolve)

    log("enqueueing...")
    batch_id = client.enqueue(graph)
    log(f"batch {batch_id} queued; waiting...")
    status = client.wait_for_batch(batch_id, poll_interval=2, timeout=600)
    log(f"batch complete: {status}")
    log("Validation OK — client, resolver, graph builder, and queue round-trip all work.")
    log(f"(Open the InvokeAI gallery to view the result; save target was {out}.)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:9090")
    ap.add_argument("--model", default="stable-diffusion-xl-base-1.0",
                    help="installed SDXL main model name to resolve")
    ap.add_argument("--lora", default="pixel-art-xl", help="installed pixel LoRA name")
    ap.add_argument("--sketch", default=None,
                    help="optional InvokeAI image name to drive a scribble ControlNet")
    ap.add_argument("--out", default="bringup_result.png")
    ap.add_argument("--skip-install", action="store_true")
    args = ap.parse_args()

    wait_for_health(args.url)
    client = InvokeAIClient(base_url=args.url)

    if not args.skip_install:
        install_models(client, DEFAULT_MODELS)

    try:
        validate(client, args.model, args.lora, args.sketch, args.out)
    except InvokeAIError as e:
        log(f"VALIDATION FAILED: {e}")
        log("Check that the model/LoRA names match what InvokeAI installed:")
        try:
            names = sorted({m.get("name", "?") for m in client.list_models()})
            log(f"installed models: {names}")
        except InvokeAIError:
            pass
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
