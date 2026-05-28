# Pixel-Art Frontiers — Plan for Directional Rotation & Skeleton Animation

The two hard PixelLab-class features. PixelLab solves these with **purpose-trained
proprietary models**; we don't have those, so the plan is to get as far as
possible with open, composable methods and to lean on the thing PixelLab lacks —
Völundr's **preference loop + locked PixelSet** — to win back consistency.

The scaffolding (`volundr.pixelart.frontier.{rotation,animation}`) already builds
the GPU-independent request structure. This doc is the runtime/quality plan.

## The shared principle: consistency is structural, not magic

Both frontiers are really one problem — **keep one character/look identical across
many renders** — attacked on three axes:

1. **Identity** — a whole-image IP-Adapter from a reference render (`ip_adapters`
   on `GenerationParams`). Optionally reference-only attention or a per-character
   LoRA for stronger lock.
2. **Layout/pose** — a per-view/per-frame ControlNet (pose, silhouette, or depth)
   so the *configuration* changes while identity holds.
3. **Look** — the locked `PixelSet` (palette + grid + dither + seed + LoRA stack)
   applied to generation *and* to every frame's post pass. This is what kills the
   per-frame palette flicker that sinks naive open pipelines.

Shared infrastructure to build once (serves both frontiers):

- **Pose/skeleton library** — canonical OpenPose/silhouette sequences for walk,
  run, attack, idle, and 4/8 facing directions, authored once (the
  Sprite-Sheet-Diffusion paper shows estimators fail on sprite anatomy, so we
  *supply* poses rather than detect them).
- **Identity conditioning node** — IP-Adapter (+ optional reference-only) wired
  into the InvokeAI denoise graph on the GPU box (the `ip_adapters` graph wiring
  currently raises `NotImplementedError`).
- **Pixel post pass** — ported `PixelArtSpec` quantize/dither/downscale node, run
  with the *same* locked palette for every output.

---

## Frontier A — Directional rotation (4/8-way turnarounds)

**Goal:** from one approved character, produce consistent S/W/N/E (+ diagonals)
views suitable for top-down / isometric games.

**Difficulty:** ⭐⭐⭐⭐ — true multi-view identity is an open problem; expect
"recognizably the same character," not pixel-faithful turnarounds, without
dedicated models.

### Milestones
- **A0 — Baseline (compose existing parts).** `TurnaroundRequest.per_direction_params`
  + identity IP-Adapter + directional prompt suffix ("facing west") + locked
  PixelSet. Generate 4 views, eyeball identity drift. *(Scaffolding done; needs
  IP-Adapter graph wiring.)*
- **A1 — Pose/silhouette control.** Add per-direction silhouette/pose ControlNet
  (`pose_controls`) so facing is enforced by control, not just prompt. Build the
  4/8 directional silhouette set.
- **A2 — Identity hardening.** Compare IP-Adapter weight sweeps, reference-only,
  and a quick per-character LoRA (B-LoRA / 1-image LoRA) for the strongest lock;
  pick the cheapest that holds identity across all views.
- **A3 — Multi-view consistency.** Evaluate purpose-built character-consistency
  methods (multi-view diffusion, character turnaround LoRAs/models) as an optional
  upgrade. Flag licensing per model.
- **A4 — Set integration.** "Approve a turnaround into the set" freezes the
  reference + palette so later characters can reuse framing; pack views via
  `pack_grid`.

### Fallback ladder (use the lowest rung that looks acceptable)
1. Prompt-only facing (A0) → 2. + directional silhouette ControlNet (A1) →
3. + per-character LoRA (A2) → 4. dedicated multi-view model (A3) →
5. assisted: generate front + back, **derive side views by mirroring**, artist
   touch-up (pixel art tolerates manual fixes well).

### Evaluation
- Identity: CLIP/DINO cosine similarity between views (target > ~0.85) + manual.
- Facing correctness: does a pose classifier / manual check confirm direction?
- Set consistency: all views share the locked palette exactly (assertable).

---

## Frontier B — Skeleton/pose-driven animation (walk/run/attack)

**Goal:** short looping sprite cycles (8–16 frames) of an approved character.

**Difficulty:** ⭐⭐⭐⭐ — temporal coherence at low res + flicker are the walls.

### Milestones
- **B0 — Frame-by-frame baseline.** `AnimationRequest.per_frame_params`: per-frame
  pose ControlNet + identity IP-Adapter + locked PixelSet, then `sheet()` packs
  them. *(Scaffolding done.)* Render a walk cycle from a canned pose sequence.
- **B1 — Flicker control.** Quantize **every frame against the one locked palette**
  (PixelSet.spec) and apply pixelization *after* any temporal smoothing. Measure
  frame-to-frame palette delta (should be ~0).
- **B2 — Temporal coherence.** Evaluate two paths and pick per-action:
  - *Frame-by-frame + strong control* (editable, sharper, our default) — add
    EbSynth-style propagation from one clean keyframe to reduce jitter.
  - *Video diffusion (AnimateDiff + pixel LoRA)* — smoother motion, but jitters
    detail; better for ambient loops than precise cycles.
- **B3 — Pose authoring.** Ship the canonical pose library (walk/run/attack/idle);
  let users nudge skeletons. No auto-pose-estimation (fails on sprites).
- **B4 — Export.** Sprite sheet (`pack_grid`) + JSON atlas + frame timing from
  `AnimationSpec.fps`; optional GIF/APNG preview on the GPU box.

### Fallback ladder
1. Frame-by-frame, prompt + pose (B0) → 2. + locked-palette quantize (B1) →
3. + keyframe propagation (B2 frame path) → 4. AnimateDiff for motion that
   doesn't need frame-precision → 5. assisted: generate keyframes only,
   interpolate/clean manually.

### Evaluation
- Temporal: warp/flicker metric across frames; locked-palette delta == 0.
- Identity: same as A, across frames.
- Motion plausibility: manual + loop seamlessness (last→first frame).

---

## How the preference loop accelerates both

- **Approve-into-set** freezes palette/grid/seed/LoRAs → the dominant consistency
  lever, for free.
- **Lean axes** over palette / pixel-density / dither / identity-strength let the
  user dial the house style; approvals accumulate it.
- **Ratings** rank candidate views/frames so the best identity-preserving ones are
  promoted automatically.

## Risks / open questions
- IP-Adapter identity may be too weak for clean turnarounds → may force a per-
  character LoRA step (small training, fits 24GB) — acceptable but adds a step.
- Pose estimators unusable on stylized sprites → we author poses (infra cost).
- AnimateDiff + pixel LoRA interaction at low res is unproven for crisp cycles →
  treat as optional, not the default path.
- Licensing: vet every model/LoRA individually; for personal use the open stack
  (nerijs OpenRAIL, open ComfyUI nodes) is clean. Retro Diffusion models are
  EULA-locked and cannot be run locally regardless.
