"""Parsing and validation of neuron steering specs.

The Discord-facing syntax mirrors the Neuron Persona Library's coordinates:

    L17N12250 +2.0        push layer 17 neuron 12250 up at 2.0x
    L17N12250 -0.75       push the same neuron down at 0.75x
    L17N12250 up 1.25     keyword form
    L3N11154 +1.25, L8N1788 -0.75     several at once

The strength is the site's alpha: the push is added to the residual stream with
norm ``alpha * r_L``, where ``r_L`` is the median residual norm after layer L.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

LAYER_NEURON = r"L\s*(\d+)\s*N\s*(\d+)"
NATURAL_LAYER_NEURON = r"Layer\s*(\d+)\s*[·:,-]?\s*Neuron\s*(\d+)"
_NUMBER = r"[0-9]*\.?[0-9]+"

_KEYWORD = re.compile(rf"^{LAYER_NEURON}\s*(up|down)\s*({_NUMBER})$", re.IGNORECASE)
_SIGNED = re.compile(rf"^{LAYER_NEURON}(?:\s*([+-])\s*({_NUMBER})|\s+({_NUMBER}))$", re.IGNORECASE)
_NATURAL_KEYWORD = re.compile(
    rf"^{NATURAL_LAYER_NEURON}\s*(up|down)\s*({_NUMBER})$", re.IGNORECASE
)
_NATURAL_SIGNED = re.compile(
    rf"^{NATURAL_LAYER_NEURON}(?:\s*([+-])\s*({_NUMBER})|\s+({_NUMBER}))$",
    re.IGNORECASE,
)
_ID_ONLY = re.compile(rf"^{LAYER_NEURON}(?!\d)", re.IGNORECASE)
_NATURAL_ID_ONLY = re.compile(rf"^{NATURAL_LAYER_NEURON}(?!\d)", re.IGNORECASE)
_SPLIT = re.compile(r"[,;]")
_LEAD_SEP = re.compile(r"^[\s,;|]+")


class SpecError(ValueError):
    """A steering spec that cannot be parsed or does not fit the model."""


@dataclass(frozen=True)
class NeuronSteer:
    """One neuron pushed one way, at one strength."""

    layer: int
    neuron: int
    alpha: float

    @property
    def direction(self) -> str:
        return "up" if self.alpha >= 0 else "down"

    def __str__(self) -> str:
        sign = "+" if self.alpha >= 0 else "-"
        return f"L{self.layer}N{self.neuron} {sign}{abs(self.alpha):g}"

    def coerce(self, n_layers: int, n_neurons: int, alpha_max: float) -> NeuronSteer:
        """Reject coordinates outside the model, and strengths outside the tested range."""
        if not 0 <= self.layer < n_layers:
            raise SpecError(
                f"layer {self.layer} out of range: model has {n_layers} layers (0-{n_layers - 1})"
            )
        if not 0 <= self.neuron < n_neurons:
            raise SpecError(
                f"neuron {self.neuron} out of range: layers have {n_neurons} neurons (0-{n_neurons - 1})"
            )
        if self.alpha == 0:
            raise SpecError("strength 0 does nothing; use a non-zero strength")
        if abs(self.alpha) > alpha_max:
            raise SpecError(
                f"strength {abs(self.alpha):g} exceeds the allowed maximum {alpha_max:g}"
            )
        return self


def _parse_one(text: str) -> NeuronSteer | None:
    """Parse a single spec, or return None if the text is not one."""
    for pattern in (_KEYWORD, _NATURAL_KEYWORD):
        match = pattern.match(text)
        if match:
            layer, neuron, word, number = match.groups()
            alpha = float(number) * (-1.0 if word.lower() == "down" else 1.0)
            return NeuronSteer(int(layer), int(neuron), alpha)

    for pattern in (_SIGNED, _NATURAL_SIGNED):
        match = pattern.match(text)
        if match:
            layer, neuron, sign, signed_number, unsigned_number = match.groups()
            number = signed_number or unsigned_number
            alpha = float(number) * (-1.0 if sign == "-" else 1.0)
            return NeuronSteer(int(layer), int(neuron), alpha)

    return None


def parse_spec(text: str, *, default_strength: float | None = None) -> list[NeuronSteer]:
    """Parse one or more steering specs.

    Raises SpecError with the offending fragment when a piece does not parse.
    """
    if text is None:
        raise SpecError("empty steering spec")
    raw = text.strip()
    if not raw:
        raise SpecError("empty steering spec")

    out: list[NeuronSteer] = []
    for piece in _SPLIT.split(raw):
        piece = piece.strip()
        if not piece:
            continue
        steer = _parse_one(piece)
        if steer is None:
            # several specs separated only by whitespace: "L4N1338 -0.2 L24N396 -0.6"
            run, leftover = split_leading_specs(piece)
            if run and not leftover:
                out.extend(run)
                continue
            # bare coordinate plus a separately supplied strength
            if default_strength is not None:
                ident = _ID_ONLY.match(piece)
                if ident:
                    layer, neuron = ident.groups()
                    out.append(NeuronSteer(int(layer), int(neuron), float(default_strength)))
                    continue
            raise SpecError(
                f"cannot read {piece!r}; use L<layer>N<neuron> <strength>, "
                "e.g. L17N12250 +2.0 or L17N12250 down 0.75"
            )
        out.append(steer)

    if not out:
        raise SpecError("empty steering spec")
    return merge(out)


def merge(steers: list[NeuronSteer]) -> list[NeuronSteer]:
    """Combine repeats of the same neuron by adding their strengths.

    The injection is linear, so two pushes on one neuron are one push.
    """
    total: dict[tuple[int, int], float] = {}
    order: list[tuple[int, int]] = []
    for steer in steers:
        key = (steer.layer, steer.neuron)
        if key not in total:
            order.append(key)
            total[key] = 0.0
        total[key] += steer.alpha
    merged = [NeuronSteer(layer, neuron, total[(layer, neuron)]) for layer, neuron in order]
    return [steer for steer in merged if steer.alpha != 0.0]


def parse_bare_coordinate(text: str) -> tuple[int, int] | None:
    """Return ``(layer, neuron)`` when the text is only a coordinate, with no strength.

    ``"Layer 0 · Neuron 1467"`` -> ``(0, 1467)``; ``"L0N1467 +2"`` -> ``None``.
    """
    stripped = (text or "").strip()
    for pattern in (_ID_ONLY, _NATURAL_ID_ONLY):
        match = pattern.match(stripped)
        if match and match.end() == len(stripped):
            layer, neuron = match.groups()
            return int(layer), int(neuron)
    return None


def nudge(steers: list[NeuronSteer], layer: int, neuron: int, delta: float) -> list[NeuronSteer]:
    """Add ``delta`` to one neuron's strength, dropping it if it lands on zero."""
    out: list[NeuronSteer] = []
    found = False
    for steer in steers:
        if (steer.layer, steer.neuron) != (layer, neuron):
            out.append(steer)
            continue
        found = True
        alpha = round(steer.alpha + delta, 6)  # keep 0.1 + 0.2 - 0.3 from leaving dust
        if alpha:
            out.append(NeuronSteer(layer, neuron, alpha))
    if not found and round(delta, 6):
        out.append(NeuronSteer(layer, neuron, round(delta, 6)))
    return out


def split_leading_specs(text: str) -> tuple[list[NeuronSteer], str]:
    """Peel steering specs off the front of a chat message.

    ``"L17N12250 +2.0 hello"`` -> ``([L17N12250 +2.0], "hello")``.
    Text that merely starts with an L-number is left alone.
    """
    steers: list[NeuronSteer] = []
    rest = text
    while True:
        stripped = _LEAD_SEP.sub("", rest, count=1)
        if not stripped:
            break
        ident = _ID_ONLY.match(stripped) or _NATURAL_ID_ONLY.match(stripped)
        if not ident:
            rest = stripped
            break

        tail = stripped[ident.end() :]
        value = re.match(rf"\s*(?:(up|down)\s*)?([+-]?)\s*({_NUMBER})", tail, re.IGNORECASE)
        if not value:
            rest = stripped
            break
        candidate_end = ident.end() + value.end()
        if not _is_boundary(stripped, candidate_end):
            rest = stripped
            break
        steer = _parse_one(stripped[:candidate_end])
        if steer is None:
            rest = stripped
            break
        steers.append(steer)
        rest = stripped[candidate_end:]

    if not steers:
        return [], text
    rest = _LEAD_SEP.sub("", rest, count=1)
    return merge(steers), rest


def _is_boundary(text: str, index: int) -> bool:
    if index >= len(text):
        return True
    return text[index] in " \t\r\n,;"


def random_steer(
    n_layers: int,
    n_neurons: int,
    *,
    low: float = 0.7,
    high: float = 1.0,
    rng: random.Random | None = None,
) -> NeuronSteer:
    """Pick a random neuron with a strength in ``[-high, -low]`` or ``[low, high]``."""
    rng = rng or random
    magnitude = round(rng.uniform(low, high), 2)
    sign = rng.choice((-1.0, 1.0))
    return NeuronSteer(rng.randrange(n_layers), rng.randrange(n_neurons), sign * magnitude)


def format_plan(steers: list[NeuronSteer]) -> str:
    return ", ".join(str(steer) for steer in steers) if steers else "none"
