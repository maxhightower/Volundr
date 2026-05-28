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
# Launch command for the InvokeAI web server.
INVOKE_LAUNCH_CMD="${INVOKE_LAUNCH_CMD:-invokeai-web}"
# ────────────────────────────────────────────────────────────────────────────────

VOLUNDR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
URL="http://${HOST}:${PORT}"
log() { printf '\n\033[1;36m[setup]\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31m[setup:ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

# ── 1. Prerequisites ───────────────────────────────────────────────────────────
log "Checking prerequisites..."
command -v git >/dev/null   || die "git not found"
command -v python3 >/dev/null || die "python3 not found"
PYV=$(python3 -c 'import sys; print("%d.%d"%sys.version_info[:2])')
log "python ${PYV}"
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
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --quiet --upgrade pip

# ── 4. Install InvokeAI (from fork) + Völundr ────────────────────────────────────
log "Installing InvokeAI from the fork (this pulls torch + CUDA wheels; can take a while)..."
( cd "$INVOKE_DIR" && eval "$INVOKE_INSTALL_CMD" ) || die \
  "InvokeAI install failed. Adjust INVOKE_INSTALL_CMD for your fork (see its README)."

log "Installing Völundr (this repo) into the same venv..."
( cd "$VOLUNDR_DIR" && pip install -e . )

# Make the Völundr package importable by the fork-side SEGA extension too.
log "Völundr installed; the fork-side denoise extension (invokeai_extension/) is a"
log "manual step — it needs a custom denoise node (see invokeai_extension/volundr_sega.py)."

export INVOKEAI_ROOT="$INVOKE_ROOT"
mkdir -p "$INVOKE_ROOT"

# ── 5. Launch InvokeAI (skip if already serving) ─────────────────────────────────
if curl -fsS "${URL}/api/v1/app/version" >/dev/null 2>&1; then
  log "InvokeAI already serving at $URL — reusing it."
else
  log "Launching InvokeAI web server (logs -> $INVOKE_ROOT/invokeai-web.log)..."
  nohup "$INVOKE_LAUNCH_CMD" --host "$HOST" --port "$PORT" >"$INVOKE_ROOT/invokeai-web.log" 2>&1 &
  echo $! > "$INVOKE_ROOT/invokeai-web.pid"
  log "Started (pid $(cat "$INVOKE_ROOT/invokeai-web.pid")). Waiting for health..."
fi

# ── 6 & 7. Install models + validate end-to-end ──────────────────────────────────
log "Running model install + end-to-end validation..."
python "$VOLUNDR_DIR/scripts/bringup.py" --url "$URL" \
  || die "Bring-up validation failed — see the messages above and $INVOKE_ROOT/invokeai-web.log"

log "DONE. InvokeAI is running at $URL and a validation image was generated."
log "Next: wire ip_adapters/region graph nodes in the fork, then run frontier A0/B0."
