"""Neuron-level steering for Llama 3.1 8B, driven from Discord."""

from .steering import NeuronSteer, SpecError, parse_spec, split_leading_specs
from .config import Config, load_config

__all__ = [
    "NeuronSteer",
    "SpecError",
    "parse_spec",
    "split_leading_specs",
    "Config",
    "load_config",
]

__version__ = "0.1.0"
