"""Replay the Neuron Persona site's featured Llama cases and compare replies.

Stop the bot first (the model needs the GPU), then:

    HF_HUB_OFFLINE=1 .venv/bin/python scripts/verify_site.py

Each case is steered up/down on "Who are you?" with greedy decoding, the way the site
produced it. The site ran the full-precision model and we run 4-bit NF4, so word-for-word
equality is not guaranteed; the persona should still show up.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request

os.environ.setdefault("DISCORD_TOKEN", "unused")

from llmsteer.config import load_config  # noqa: E402
from llmsteer.model import CANONICAL_PROMPT, SteeringModel  # noqa: E402
from llmsteer.steering import NeuronSteer  # noqa: E402

SUMMARY = "https://neuronpersonas.com/data/llama/summary.json"


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


def persona_shown(persona: str, reply: str) -> bool:
    words = [w for w in norm(persona).split() if len(w) > 3 or w.isdigit()]
    return bool(words) and all(w in norm(reply) for w in words)


def main() -> None:
    request = urllib.request.Request(SUMMARY, headers={"User-Agent": "llmsteer-verify"})
    featured = json.load(urllib.request.urlopen(request))["featured"]
    model = SteeringModel(load_config())
    messages = [{"role": "user", "content": CANONICAL_PROMPT}]

    print(f"baseline: {model.generate(messages, [], max_new_tokens=60, greedy=True)!r}\n")
    exact = shown = 0
    for layer, neuron, sign, persona, strength, site_reply in featured:
        steer = NeuronSteer(layer, neuron, sign * strength)
        ours = model.generate(messages, [steer], max_new_tokens=60, greedy=True)
        same = norm(ours) == norm(site_reply)
        has_persona = persona_shown(persona, ours)
        exact += same
        shown += has_persona
        print(f"{steer}  [{persona}]  exact={same}  persona={has_persona}")
        print(f"   site: {site_reply!r}\n   ours: {ours[:140]!r}")
    print(f"\nexact {exact}/{len(featured)}, persona visible {shown}/{len(featured)}")


if __name__ == "__main__":
    main()
