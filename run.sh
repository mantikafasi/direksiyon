#!/usr/bin/env bash
# Start the steering bot tuned for a 6 GB RTX 3060 laptop GPU.
#
# Measured on that card (Llama 3.1 8B Instruct, NF4 + double quant, bf16 compute):
#   weights 4.33 GiB of 5.72 GiB visible; peak grows ~0.21 GiB per 1k tokens of context
#   ctx 3072 -> peak 5.09 GiB (0.4 GiB spare); ctx 4096 -> peak 5.30 GiB (no spare)
#   decode ~26 tok/s, entirely on the GPU
#
# These tuning values are set here, not in .env (the token and other settings stay in .env).
# To try something else for one run, pass it from the shell: CONTEXT_TOKENS=2048 ./run.sh
set -euo pipefail
cd "$(dirname "$0")"

# The embedding table (~1 GiB) is only looked up, never multiplied, so keep it in RAM.
# That is what lets every decoder layer stay on the GPU with room left for context.
: "${EMBED_ON_CPU:=true}"
: "${CPU_OFFLOAD:=false}"           # only used when EMBED_ON_CPU=false

# Context budget. 3072 leaves ~0.4 GiB of headroom; 4096 runs but sits at the limit.
: "${CONTEXT_TOKENS:=3072}"
: "${MAX_NEW_TOKENS:=256}"
: "${PROBE_MAX_NEW_TOKENS:=160}"
: "${HISTORY_TURNS:=8}"
: "${MAX_QUEUE:=4}"                 # one generation at a time; the rest wait
export EMBED_ON_CPU CPU_OFFLOAD CONTEXT_TOKENS MAX_NEW_TOKENS PROBE_MAX_NEW_TOKENS HISTORY_TURNS MAX_QUEUE

# Avoids allocator fragmentation when running a few hundred MB from the limit.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
# The model is cached locally, so skip the Hub round-trip at startup.
# Unset this (or run with HF_HUB_OFFLINE=0) for the first download.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

if command -v nvidia-smi >/dev/null 2>&1; then
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 | tr -d ' ')
  if [[ "${used:-0}" -gt 600 ]]; then
    echo "warning: ${used} MiB of VRAM is already in use by other processes; context may OOM." >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >&2 || true
  fi
fi

echo "run.sh: context=${CONTEXT_TOKENS} history=${HISTORY_TURNS} max_new=${MAX_NEW_TOKENS} embed_on_cpu=${EMBED_ON_CPU}" >&2

PYTHON=.venv/bin/python
[[ -x "$PYTHON" ]] || PYTHON=python3
exec "$PYTHON" -m llmsteer.bot "$@"
