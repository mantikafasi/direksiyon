"""Lazy extraction of original-precision Llama down-projection columns."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


class DirectionStore:
    """Read ``W_down[:, neuron]`` from the same checkpoint used by the model.

    The quantized model is loaded from the BF16 checkpoint, so this sidecar vector
    represents the original steering direction while keeping the full model 4-bit.
    """

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self._weight_map: dict[str, str] = {}
        self._handles: dict[Path, Any] = {}
        self._memory: dict[tuple[int, int], torch.Tensor] = {}
        self._load_index()

    def _load_index(self) -> None:
        index_path = self.model_dir / "model.safetensors.index.json"
        if index_path.exists():
            data = json.loads(index_path.read_text())
            self._weight_map = data["weight_map"]
            return

        # A single-file checkpoint is common for small test models.
        single = self.model_dir / "model.safetensors"
        if single.exists():
            self._weight_map = {"__single__": single.name}
            return
        raise FileNotFoundError(f"no safetensors index or model.safetensors in {self.model_dir}")

    def _key(self, layer: int) -> str:
        return f"model.layers.{layer}.mlp.down_proj.weight"

    def _shard(self, layer: int) -> Path:
        key = self._key(layer)
        filename = self._weight_map.get(key)
        if filename is None:
            filename = self._weight_map.get("__single__")
        if filename is None:
            raise KeyError(f"checkpoint has no {key}")
        return self.model_dir / filename

    def _handle(self, path: Path) -> Any:
        handle = self._handles.get(path)
        if handle is None:
            handle = safe_open(str(path), framework="pt", device="cpu")
            self._handles[path] = handle
        return handle

    def column(self, layer: int, neuron: int) -> torch.Tensor:
        """Return one CPU FP32 direction vector, cached by layer and neuron."""
        key = (layer, neuron)
        if key in self._memory:
            return self._memory[key]

        handle = self._handle(self._shard(layer))
        tensor_key = self._key(layer)
        # get_tensor is intentionally used instead of relying on get_slice: it is
        # supported across safetensors versions and only costs one 117 MB tensor
        # transiently on the 8B model.
        matrix = handle.get_tensor(tensor_key)
        if matrix.ndim != 2 or not 0 <= neuron < matrix.shape[1]:
            raise IndexError(f"neuron {neuron} is outside down_proj shape {tuple(matrix.shape)}")
        vector = matrix[:, neuron].detach().to(dtype=torch.float32).contiguous()
        del matrix
        self._memory[key] = vector
        return vector

    def close(self) -> None:
        for handle in self._handles.values():
            close = getattr(handle, "__exit__", None)
            if close:
                close(None, None, None)
        self._handles.clear()
