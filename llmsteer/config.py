"""Environment-backed configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


DEFAULT_STEER_STEPS = (0.05, 0.1, 0.25, 0.5, 0.8)


def _steps(name: str) -> tuple[float, ...]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return DEFAULT_STEER_STEPS
    steps = tuple(float(piece) for piece in raw.split(",") if piece.strip())
    if not steps or len(steps) > 5 or any(step <= 0 for step in steps):
        raise ValueError(f"{name} must be 1-5 positive comma-separated numbers, e.g. 0.1,0.5,0.8")
    return steps


@dataclass(frozen=True)
class Config:
    discord_token: str
    guild_id: int | None
    model_id: str
    model_path: str | None
    load_4bit: bool
    inject_mode: str
    r_mode: str
    max_new_tokens: int
    probe_max_new_tokens: int
    history_turns: int
    context_tokens: int
    temperature: float
    top_p: float
    repetition_penalty: float
    alpha_max: float
    max_queue: int
    gpu_memory_gib: float
    cpu_offload: bool
    clear_history_on_steer: bool
    steer_steps: tuple[float, ...] = DEFAULT_STEER_STEPS
    embed_on_cpu: bool = True


def load_config() -> Config:
    """Load ``.env`` if python-dotenv is installed, then read environment values."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    token = os.getenv("DISCORD_TOKEN", "").strip()
    if not token:
        raise RuntimeError("DISCORD_TOKEN is required; put the replacement token in .env")

    guild_raw = os.getenv("GUILD_ID", "").strip()
    guild_id = int(guild_raw) if guild_raw else None
    inject_mode = os.getenv("INJECT_MODE", "residual").strip().lower()
    r_mode = os.getenv("R_MODE", "canonical").strip().lower()
    if inject_mode not in {"residual", "activation"}:
        raise ValueError("INJECT_MODE must be residual or activation")
    if r_mode not in {"canonical", "per_prompt"}:
        raise ValueError("R_MODE must be canonical or per_prompt")

    return Config(
        discord_token=token,
        guild_id=guild_id,
        model_id=os.getenv("MODEL_ID", DEFAULT_MODEL_ID),
        model_path=os.getenv("MODEL_PATH") or None,
        load_4bit=_bool("LOAD_4BIT", True),
        inject_mode=inject_mode,
        r_mode=r_mode,
        max_new_tokens=_int("MAX_NEW_TOKENS", 256),
        probe_max_new_tokens=_int("PROBE_MAX_NEW_TOKENS", 160),
        history_turns=_int("HISTORY_TURNS", 6),
        context_tokens=_int("CONTEXT_TOKENS", 1024),
        temperature=_float("TEMPERATURE", 0.7),
        top_p=_float("TOP_P", 0.9),
        repetition_penalty=_float("REPETITION_PENALTY", 1.1),
        alpha_max=_float("ALPHA_MAX", 2.5),
        max_queue=_int("MAX_QUEUE", 4),
        gpu_memory_gib=_float("GPU_MEMORY_GIB", 5.2),
        cpu_offload=_bool("CPU_OFFLOAD", True),
        clear_history_on_steer=_bool("CLEAR_HISTORY_ON_STEER", True),
        steer_steps=_steps("STEER_STEPS"),
        embed_on_cpu=_bool("EMBED_ON_CPU", True),
    )
