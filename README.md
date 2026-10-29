# Discord Llama neuron-steering bot

This bot runs `meta-llama/Llama-3.1-8B-Instruct` locally with 4-bit NF4 weights and applies Neuron Persona-style pushes to the residual stream.

## Security first

The Discord token belongs only in `.env`, which is gitignored; never commit it. If a token ever leaks (pasted in chat, a screenshot, a commit), reset it in the Discord Developer Portal and put the new one in `.env`.

## 6 GB GPU setup

The intended target is a 6 GB RTX 3060 Mobile:

- 4-bit NF4 model loading
- `EMBED_ON_CPU=true`: the ~1 GiB token-embedding table stays in system RAM (it is only looked up), so all 32 layers and the LM head fit on the GPU with no per-token CPU streaming
- `CONTEXT_TOKENS=3072`, one generation at a time
- `CPU_OFFLOAD`/`GPU_MEMORY_GIB` apply only if you set `EMBED_ON_CPU=false`

Measured on the RTX 3060 Laptop (5.72 GiB visible): weights 4.33 GiB, about 0.21 GiB more per 1k tokens of context, ~26 tok/s decode. A 2.9k-token prompt with steering and 256 new tokens peaks at 5.06 GiB; 4096 context runs but leaves no headroom. When the history outgrows `CONTEXT_TOKENS` the oldest turns are dropped, never the newest message. The bot logs VRAM use once it is ready.

The bot stores one FP32 sidecar vector per selected neuron and adds it explicitly after the selected decoder layer. This avoids relying on the quantized `down_proj` column for the injected direction. Normal model computation remains 4-bit.

The Llama checkpoint is gated. Accept the model license on Hugging Face and authenticate with `hf auth login` or `HF_TOKEN` before the first start.

## Install

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[cuda]'
```

Install the CUDA build of PyTorch appropriate for the laptop. The package's `cuda` extra installs bitsandbytes.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Use the CUDA wheel matching the installed NVIDIA driver; the command above is an example, not a universal requirement.

## Configure

```bash
cp .env.example .env
# edit .env and put the replacement token there
```

Set `HF_TOKEN` separately if the Hugging Face client needs it. Do not put either secret in source control.

In the Discord Developer Portal, enable **Message Content Intent** under Bot → Privileged Gateway Intents for mention replies. `GUILD_ID` is optional. The bot uses prefix commands only and uses it to clear stale slash commands from that server on startup.

Start (tuned for the 6 GB card):

```bash
./run.sh
```

`run.sh` sets the memory and context values itself, so they override the same keys in `.env`. Pass one from the shell to change it for a single run, e.g. `CONTEXT_TOKENS=2048 ./run.sh`. `python -m llmsteer.bot` also works and uses `.env` alone.

## Commands

The bot uses regular prefix commands. The default prefixes are `.` and `!`:

```text
.ask hello
.steer L17N12250 +2.0
.steer Layer 0 · Neuron 1467 +2
.steer L17N12250 -0.75, L8N1788 +1.0
.steer_show
.steer_clear
.probe L17N12250 +1.25
.reset
.status
```

Give `.steer` only a coordinate to open a button panel:

```text
.steer Layer 0 · Neuron 1467
.steer L0N1467
```

The `+` and `-` buttons add to or subtract from that neuron's strength (shared by the whole channel, capped by `ALPHA_MAX`, and the neuron drops out when it reaches 0). **Test** runs the `.probe` generation with the current steering, and **Remove** takes the neuron out. Only the person who opened the panel can use it, and it goes inactive after 10 minutes. Set the button increments with `STEER_STEPS` (default `0.05,0.1,0.25,0.5,0.8`).

`.probe` uses the site's style: `Who are you?`, greedy decoding, and up to 160 new tokens. `.ask` uses sampling for better conversational behavior.

Leading steering also works:

```text
.ask L17N12250 +2.0 | explain quantum tunneling
```

A mention works as a message reply:

```text
@Bot L17N12250 +2.0 explain quantum tunneling
```

Layers and neurons are zero-based. Llama 3.1 has layers `0..31` and MLP neurons `0..14335`.

Changing channel steering clears that channel's history by default. This prevents old assistant turns from mixing incompatible personas. Set `CLEAR_HISTORY_ON_STEER=false` only if that behavior is intentional.

## Steering equation

For a selected layer `L` and neuron `n`, the residual hook adds:

```text
sign(alpha) * |alpha| * r_L * W_down[:, n] / ||W_down[:, n]||
```

The original checkpoint column is loaded lazily from safetensors. `r_L` is measured from the canonical `Who are you?` prompt and cached in memory; `R_MODE=per_prompt` recomputes it for every request at roughly twice the inference cost.

`INJECT_MODE=residual` is recommended. `INJECT_MODE=activation` is available for comparison and lets the quantized `down_proj` provide the write direction itself.

## Verification

```bash
pytest -q
```

The CPU test suite uses a tiny random Llama model to verify parser behavior, direction extraction, hook cleanup, residual delta magnitude, and equivalence between residual and activation injection. It does not prove 6 GB CUDA throughput; test the actual laptop with the real checkpoint before exposing the bot to a server.

## Fidelity to the Neuron Persona site

The recipe is the site's: `h_L += alpha * r_L * W_down[:, n] / ||W_down[:, n]||` on the output of layer `L`, with `r_L` the median residual norm after layer `L` on the canonical `Who are you?` prompt (first token excluded). `scripts/verify_site.py` replays the site's featured cases.

Expect to see the site's personas only approximately:

- The site published greedy replies to `Who are you?` from the full-precision model, one prompt only. Other prompts are not covered, and its own page says a neuron can act differently there.
- The coherent window is narrow. Around the site's strength the reply goes from the normal assistant, to a persona, to garbage within roughly 10-15%. That is why the panel has fine `+/-0.05` and `+/-0.1` steps: nudge until the persona lands.
- On the full-precision model, exact reproduction of the site's reply is about 3 in 12 featured cases and much lower for neurons in layers 0-5; later layers reproduce more often. The site notes that about a third of replies under strong pushes change when only the batch layout changes.
- 4-bit NF4 shifts results further (the unsteered `Who are you?` reply already differs from the site's), so near-threshold personas may need a slightly different strength than the site's.

Applying the push at every position (the implemented behavior) scored at least as well as pushing only the last prompt token, only generated tokens, or from the user turn on, and measuring `r_L` on the layer's input scored lower.
