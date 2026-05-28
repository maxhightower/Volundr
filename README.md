# Völundr

A sketch-conditioned image generation tool with a **preference-steering feedback loop**.
You draw on a canvas, label regions, and generate options you can approve, deny, lean
toward, or lean away from. Subsequent generations are biased by that accumulated feedback.

Named for Völundr, the Norse smith-god — a forge for custom imagery. The motivating use
case is dragon imagery in specific styles, but the tool is **not** hardcoded to dragons;
it loads arbitrary LoRAs on top of a base diffusion model.

The **preference loop** is the core value. The canvas and generation plumbing are means
to that end.

---

## Status

Verification phase complete; architecture decisions locked (below). The GPU-independent core
is built and tested (`src/volundr/`, 36 passing tests). The next hands-on step requires a GPU
and runs on the user's hardware (see "Next step").

### Repo layout / what's built

```
src/volundr/
  models.py              # engine-agnostic types: GenerationParams, Feedback, results
  invoke/
    client.py            # graph-agnostic InvokeAI queue client (enqueue/poll/fetch image)
    graph.py             # SDXL graph builder (text2img + LoRA + ControlNet), schema-validated
                         #   against InvokeAI invocation source (v2026-05-27)
  preference/            # ← the core value
    feedback.py          # FeedbackStore: record + JSON persistence of approve/deny/lean
    seeds.py             # variation_params: sub-seed variations near an approved image (step 4)
    prompt_lean.py       # PromptLean: token-weight lean / push-to-negative (step 5)
    steering.py          # SteeringState: accumulate all feedback into a bias on next params
tests/                   # pure-logic + mocked-HTTP coverage (no GPU needed)
```

**Built & tested here (GPU-independent):** the preference loop (feedback store + persistence,
variation seeding, prompt-level lean, accumulated steering) and the InvokeAI REST client.

**Schema-validated against InvokeAI source:** `graph.py`'s node type strings and field names
were checked against the InvokeAI invocation source (the version a fresh fork tracks). The one
correction over the first draft: model/LoRA/ControlNet fields are `ModelIdentifierField`s
(`{key, hash, name, base, type}`) resolved from the running install's model DB at runtime via
`InvokeAIClient.model_resolver()` — not human-readable name strings.

**Deliberately deferred to the GPU box (not faked):** `graph.py` raises `NotImplementedError`
for regional-guidance wiring and sub-seed variation blending (both need the live install to
validate). Concept Sliders (step 6) is training-based and unwritten. The InvokeAI **frontend
fork** is created (under the user's GitHub account); its backend custom-node + canvas patches
are pending. This environment's GitHub access is scoped to `maxhightower/volundr` only.

Run tests: `pip install -e . && pytest`

## Locked decisions

| Area | Decision | Notes |
|------|----------|-------|
| Generation engine | **InvokeAI end-to-end** | Reverses the original "ComfyUI engine" plan. InvokeAI's own backend covers SDXL, scribble/lineart ControlNet, built-in regional guidance, and arbitrary-LoRA loading, and exposes a `post_combine_noise_preds` denoise hook where custom preference steering belongs. One system instead of two. |
| Base model | **SDXL 1.0** | Mature ecosystem, fits 24GB. Flux is a v2 option. |
| Sketch conditioning | **ControlNet** scribble/lineart | First-class in InvokeAI. |
| Regional prompting | InvokeAI **Regional Guidance Layers** | Per-mask positive/negative prompts, built in (≥ v4.2). |
| "Lean" algorithm | **Concept Sliders** | Replaces MCDPO (see Finding 1). Small LoRA slider adaptors, ~minutes to train each on 24GB, give labeled lean-toward / lean-away axes at inference. |
| Frontend | **Fork InvokeAI** (React) | No frontend plugin API exists, so custom approve/deny/lean UI requires a fork. Backend orchestration can be a clean custom node (no fork). |
| Deployment | Local **or** rented cloud GPU | Portable; no fixed-GPU or multi-GPU assumptions. Cost/security-conscious. |
| Scope | **Personal use, single-user, local-first** | Makes GPL contamination a non-issue. |

## Verification findings (May 2026)

### Finding 1 — The original "lean sliders without retraining" premise was wrong
- **MCDPO** ("Multi-reward Conditional DPO", arXiv 2512.10237, Dec 2025) has **no public code**, and its
  dimensional sliders only work *after* fine-tuning SDXL with the MCDPO objective. You cannot bolt
  sliders onto stock SDXL. The "no retraining at inference" claim applies only to an already-fine-tuned
  checkpoint.
- **D3PO** (arXiv 2311.13231) is pure alignment fine-tuning with **no dimensional sliders**, SD1.5-oriented.
- **Conclusion:** neither gives a zero-training lean slider on stock SDXL. → Pivoted to **Concept Sliders**
  (published, has code) for build step 6. This only affects step 6; steps 1–5 are unaffected.

### Finding 2 — InvokeAI backend is capable enough to drop ComfyUI
SDXL, scribble/lineart ControlNet, regional guidance, arbitrary LoRA all first-class. Custom CFG /
preference steering belongs at the `post_combine_noise_preds` extension callback in its denoise loop.
Extension registration is currently hardcoded, so steering ships as a **custom denoise node** (or a
small core patch), not via auto-discovery.

### Finding 3 — Frontend fork required
InvokeAI has a clean backend custom-node system (orchestration = no fork) but **no frontend plugin API**.
Custom approve/deny/lean UI ⇒ fork the React frontend.

### Finding 4 — Licensing (low-stakes for personal use)
InvokeAI Apache-2.0 · ComfyUI GPL-3 (unused now) · SDXL 1.0 OpenRAIL++ (commercial-OK, no revenue cap)
· diffusers/transformers Apache-2.0. Research-code licensing was the only watch-item and is moot
(MCDPO has no code; reimplement from papers regardless).

#### Sources
- InvokeAI: https://github.com/invoke-ai/InvokeAI · custom nodes https://invoke-ai.github.io/InvokeAI/contributing/INVOCATIONS/ · modular backend PR https://github.com/invoke-ai/InvokeAI/pull/6606 · regional guidance https://support.invoke.ai/support/solutions/articles/151000165024-regional-guidance-layers
- MCDPO: https://arxiv.org/abs/2512.10237
- D3PO: https://arxiv.org/abs/2311.13231 · code https://github.com/yk7333/d3po
- Concept Sliders: https://sliders.baulab.info/ · https://github.com/rohitgandikota/sliders
- SDXL license: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md

## Revised build order (do not skip steps)

1. **Stand up InvokeAI** locally with SDXL + a test LoRA. Drive its HTTP/Socket.IO API from Python and
   get images back. *(Replaces the old ComfyUI step.)*
2. **Add Scribble/Lineart ControlNet.** Feed a static sketch file; confirm output respects the sketch and
   that a conditioning-strength parameter visibly controls adherence.
3. **Fork InvokeAI**; run locally. Confirm the existing canvas + masking (regional guidance) + LoRA-loading
   flow works end-to-end before adding anything.
4. **Minimal feedback UI** (v1): approve (lock seed + save), deny (discard), generate-variations (new seeds
   near approved). Must work cleanly before anything more ambitious.
5. **Prompt-level lean:** thumbs-up raises weights on a generation's descriptive tokens; thumbs-down pushes
   them into the negative prompt. Ships before the latent version.
6. **Concept Sliders steering** (the differentiated feature): train small LoRA slider adaptors for
   user-labeled lean axes; expose as inference-time sliders wired into the denoise loop. Hardest/longest step.
7. **Session preference accumulation** (stretch): log approve/deny pairs to derive a latent direction vector
   biasing subsequent generations. Periodic LoRA fine-tune on approvals is **v2**, not v1.

**Do not attempt steps 5–7 until 1–4 are solid.**

## Non-goals for v1
No multi-user/collab · no video/3D/animation · no in-app model-training UI (LoRAs trained externally via
kohya_ss / ai-toolkit, loaded from disk) · no cloud SaaS · no mobile · dragons are not a special case.

## Next step

Build step 1 needs a GPU (local 4090/3090 or rented cloud) and cannot run in a GPU-less CI/cloud container.
On a GPU box: install InvokeAI, download SDXL 1.0 + a test LoRA (a public Civitai/HF LoRA stands in until
project LoRAs are supplied), and confirm a Python script can submit a graph over the API and retrieve an image.
