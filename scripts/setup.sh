#!/usr/bin/env bash
#
# Völundr one-shot GPU-box bring-up.
#
# Run this once, at home, from inside your checkout of the Völundr repo:
#
#     bash scripts/setup.sh
#
# It will: check prerequisites -> clone/update your InvokeAI fork -> make a venv
# -> install InvokeAI (editable, from the fork) + Völundr -> launch InvokeAI ->
# install models -> run an end-to-end validation generation.
#
# It is idempotent: re-running updates the fork, reuses the venv, and skips
# already-installed models.
#
# ── HONESTY NOTE ──────────────────────────────────────────────────────────────
# This script was written WITHOUT a GPU/InvokeAI to test against. The Völundr
# parts (venv, pip install -e ., bringup.py) are solid; the InvokeAI-specific
# bits (install command, launch command, model-install API) follow documented
# v4/v5 behaviour but may need a one-line tweak for your exact fork. Every such
# spot is a CONFIG variable below or fails loudly with guidance — nothing is
# silently assumed. Read the CONFIG block before the first run.
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── CONFIG (override via env, e.g. `FORK_URL=... bash scripts/setup.sh`) ────────
FORK_URL="${FORK_URL:-https://github.com/maxhightower/InvokeAI.git}"
FORK_BRANCH="${FORK_BRANCH:-main}"
INVOKE_DIR="${INVOKE_DIR:-$HOME/volundr/InvokeAI-fork}"   # where the fork is cloned
INVOKE_ROOT="${INVOKE_ROOT:-$HOME/volundr/invokeai-root}" # InvokeAI data/models dir
VENV_DIR="${VENV_DIR:-$HOME/volundr/.venv}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-9090}"
# Install command for InvokeAI from the fork checkout. Most forks: editable install.
INVOKE_INSTALL_CMD="${INVOKE_INSTALL_CMD:-pip install -e .}"
# CUDA torch wheel. Plain PyPI `torch` is CPU-only, so we install the GPU build
# explicitly from the pytorch index BEFORE InvokeAI (whose deps then see it as
# satisfied). The index tag must match torch's available CUDA builds — torch 2.7.x
# ships cu118/cu126/cu128, NOT cu124. cu128 needs a recent driver (CUDA >= 12.8).
TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu128}"
TORCH_PKGS="${TORCH_PKGS:-torch torchvision}"
# Launch command for the InvokeAI web server. Newer InvokeAI (v5/v6) takes host/port
# from invokeai.yaml, NOT CLI flags, so we don't pass --host/--port here.
INVOKE_LAUNCH_CMD="${INVOKE_LAUNCH_CMD:-invokeai-web}"
# Python interpreter to build the venv with. Leave empty to auto-detect a real
# Python (avoids the broken Microsoft Store `python3` stub on Windows).
PYTHON="${PYTHON:-}"
# Low-VRAM mode: write a conservative invokeai.yaml tuned for small (<=8GB) GPUs.
LOW_VRAM="${LOW_VRAM:-1}"
# ────────────────────────────────────────────────────────────────────────────────

VOLUNDR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
URL="http://${HOST}:${PORT}"
log() { printf '\n\033[1;36m[setup]\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31m[setup:ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

# ── 1. Prerequisites ───────────────────────────────────────────────────────────
log "Checking prerequisites..."
command -v git >/dev/null   || die "git not found"

# Resolve a *working* Python. On Windows the bare `python3`/`python` on PATH are
# often Microsoft Store stubs that print a "not found" message and exit non-zero,
# so we test each candidate by actually running it. The `py` launcher is the most
# reliable way to find a real install.
if [ -z "$PYTHON" ]; then
  if command -v py >/dev/null 2>&1 && py -3.11 --version >/dev/null 2>&1; then
    PYTHON="$(py -3.11 -c 'import sys; print(sys.executable)')"
  elif command -v py >/dev/null 2>&1 && py -3 --version >/dev/null 2>&1; then
    PYTHON="$(py -3 -c 'import sys; print(sys.executable)')"
  elif python3 --version >/dev/null 2>&1; then PYTHON=python3
  elif python --version >/dev/null 2>&1; then PYTHON=python
  fi
fi
[ -n "$PYTHON" ] && "$PYTHON" --version >/dev/null 2>&1 \
  || die "No working Python found. Install Python 3.11/3.12, or set PYTHON=/path/to/python.exe"
PYV=$("$PYTHON" -c 'import sys; print("%d.%d"%sys.version_info[:2])')
log "python ${PYV} ($PYTHON)"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
else
  log "WARNING: nvidia-smi not found — generation will be very slow or fail without a GPU."
fi

# ── 2. Clone / update the InvokeAI fork ──────────────────────────────────────────
mkdir -p "$(dirname "$INVOKE_DIR")"
if [ -d "$INVOKE_DIR/.git" ]; then
  log "Updating existing fork at $INVOKE_DIR"
  git -C "$INVOKE_DIR" fetch origin "$FORK_BRANCH" && git -C "$INVOKE_DIR" checkout "$FORK_BRANCH" && git -C "$INVOKE_DIR" pull --ff-only origin "$FORK_BRANCH"
else
  log "Cloning fork $FORK_URL -> $INVOKE_DIR"
  git clone --branch "$FORK_BRANCH" "$FORK_URL" "$INVOKE_DIR"
fi

# ── 3. Virtualenv ────────────────────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
  log "Creating venv at $VENV_DIR"
  "$PYTHON" -m venv "$VENV_DIR"
fi
# Windows venvs put the activator under Scripts/, POSIX ones under bin/.
if [ -f "$VENV_DIR/Scripts/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/Scripts/activate"
elif [ -f "$VENV_DIR/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
else
  die "venv activator not found under $VENV_DIR (Scripts/ or bin/)"
fi
python -m pip install --quiet --upgrade pip

# ── 4. Install InvokeAI (from fork) + Völundr ────────────────────────────────────
# Install the CUDA torch build first so InvokeAI's dependency resolution finds it
# already satisfied and doesn't pull the CPU-only wheel from PyPI.
log "Installing CUDA torch ($TORCH_PKGS) from $TORCH_CUDA_INDEX ..."
pip install $TORCH_PKGS --index-url "$TORCH_CUDA_INDEX" || die \
  "CUDA torch install failed. Pick a TORCH_CUDA_INDEX matching your driver (cu118/cu126/cu128)."
python -c "import torch; assert torch.cuda.is_available(), 'torch installed but CUDA not available'; print('torch', torch.__version__, 'CUDA OK:', torch.cuda.get_device_name(0))" \
  || die "torch is installed but CUDA is not available — wrong wheel for this driver/GPU."

log "Installing InvokeAI from the fork (can take a while)..."
( cd "$INVOKE_DIR" && eval "$INVOKE_INSTALL_CMD" ) || die \
  "InvokeAI install failed. Adjust INVOKE_INSTALL_CMD for your fork (see its README)."

log "Installing Völundr (this repo) into the same venv..."
( cd "$VOLUNDR_DIR" && pip install -e . )

# Make the Völundr package importable by the fork-side SEGA extension too.
log "Völundr installed; the fork-side denoise extension (invokeai_extension/) is a"
log "manual step — it needs a custom denoise node (see invokeai_extension/volundr_sega.py)."

export INVOKEAI_ROOT="$INVOKE_ROOT"
mkdir -p "$INVOKE_ROOT"

# ── 4b. Low-VRAM config (for small GPUs, e.g. 8GB) ───────────────────────────────
# SDXL nominally wants ~12-24GB; on an 8GB card it only fits with fp16 + partial
# model loading + a small VRAM cache + CPU offload. We write a conservative
# invokeai.yaml only if one doesn't already exist (so we never clobber yours).
# Key names follow InvokeAI v4/v5; unknown keys may need a tweak for your fork.
if [ "$LOW_VRAM" = "1" ] && [ ! -f "$INVOKE_ROOT/invokeai.yaml" ]; then
  log "Writing low-VRAM invokeai.yaml (8GB-class GPU) -> $INVOKE_ROOT/invokeai.yaml"
  cat > "$INVOKE_ROOT/invokeai.yaml" <<YAML
# Völundr low-VRAM defaults for ~8GB GPUs. Adjust if your fork's schema differs.
schema_version: 4.0.3
host: $HOST
port: $PORT
device: cuda
precision: float16
# Keep little resident in VRAM; stream weights as needed.
enable_partial_loading: true
vram: 0.25
lazy_offload: true
# Modest host RAM cache for model weights (GB).
ram: 6
YAML
fi

# ── 5. Launch InvokeAI (skip if already serving) ─────────────────────────────────
if curl -fsS "${URL}/api/v1/app/version" >/dev/null 2>&1; then
  log "InvokeAI already serving at $URL — reusing it."
else
  log "Launching InvokeAI web server (host/port from invokeai.yaml; logs -> $INVOKE_ROOT/invokeai-web.log)..."
  nohup "$INVOKE_LAUNCH_CMD" >"$INVOKE_ROOT/invokeai-web.log" 2>&1 &
  echo $! > "$INVOKE_ROOT/invokeai-web.pid"
  log "Started (pid $(cat "$INVOKE_ROOT/invokeai-web.pid")). Waiting for health..."
fi

# ── 6 & 7. Install models + validate end-to-end ──────────────────────────────────
log "Running model install + end-to-end validation..."
python "$VOLUNDR_DIR/scripts/bringup.py" --url "$URL" \
  || die "Bring-up validation failed — see the messages above and $INVOKE_ROOT/invokeai-web.log"

log "DONE. InvokeAI is running at $URL and a validation image was generated."
log "Next: wire ip_adapters/region graph nodes in the fork, then run frontier A0/B0."
