"""4-bit Llama loading and residual-stream neuron steering."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Mapping

import torch
from torch import nn

from .config import Config
from .directions import DirectionStore
from .steering import NeuronSteer


# The prompt the Neuron Persona site measures r_L on and steers.
CANONICAL_PROMPT = "Who are you?"


class CpuEmbedding(nn.Module):
    """Token-embedding lookup that stays in system RAM.

    The table is ~1 GiB for Llama 3.1 8B but only a few rows are read per step, so
    only the looked-up rows travel to the GPU. The table is a buffer, not a parameter,
    so ``model.device`` keeps reporting the GPU.
    """

    def __init__(self, weight: torch.Tensor, device: torch.device | str):
        super().__init__()
        self.register_buffer("weight", weight.detach().to("cpu").contiguous(), persistent=False)
        self.target = torch.device(device)
        self.embedding_dim = weight.shape[1]

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        rows = nn.functional.embedding(input_ids.to("cpu"), self.weight)
        return rows.to(self.target, non_blocking=True)


class SteeringModel:
    """A Transformers causal LM with explicit FP32/BF16 steering directions."""

    def __init__(self, config: Config):
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if config.load_4bit and not torch.cuda.is_available():
            raise RuntimeError(
                "4-bit Llama mode requires CUDA; no NVIDIA device is visible. "
                "Run this bot on the RTX 3060 host."
            )
        self.config = config
        self.model_dir = Path(
            config.model_path
            or snapshot_download(
                config.model_id,
                allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt"],
            )
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        embed_on_cpu = config.load_4bit and config.embed_on_cpu
        kwargs: dict[str, object] = {
            "device_map": (
                {
                    "model.embed_tokens": "cpu",
                    "model.layers": 0,
                    "model.norm": 0,
                    "model.rotary_emb": 0,
                    "lm_head": 0,
                }
                if embed_on_cpu
                else "auto" if config.load_4bit and config.cpu_offload else {"": 0} if config.load_4bit else "auto"
            ),
            "low_cpu_mem_usage": True,
            "dtype": torch.bfloat16 if config.load_4bit else torch.float32,
        }
        if config.load_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                llm_int8_enable_fp32_cpu_offload=config.cpu_offload or embed_on_cpu,
            )
            if config.cpu_offload and not embed_on_cpu:
                kwargs["max_memory"] = {0: f"{config.gpu_memory_gib:g}GiB", "cpu": "24GiB"}
        elif torch.cuda.is_available():
            kwargs["device_map"] = {"": 0}
            kwargs["dtype"] = torch.bfloat16

        self.model = AutoModelForCausalLM.from_pretrained(self.model_dir, **kwargs)
        if embed_on_cpu:
            self._move_embedding_to_cpu()
        self.model.eval()
        self.layers = self.model.model.layers
        self.n_layers = int(self.model.config.num_hidden_layers)
        self.n_neurons = int(self.model.config.intermediate_size)
        self.hidden_size = int(self.model.config.hidden_size)
        self.device = next(self.model.parameters()).device
        self.directions = DirectionStore(self.model_dir)
        self.r_l: dict[int, float] = {}

    def _move_embedding_to_cpu(self) -> None:
        """Replace accelerate's offloaded embedding with a plain CPU lookup.

        accelerate would copy the whole table to the GPU on every forward pass.
        """
        from accelerate.hooks import remove_hook_from_module

        inner = self.model.model
        embed = inner.embed_tokens
        hook = getattr(embed, "_hf_hook", None)
        weights_map = getattr(hook, "weights_map", None)
        weight = weights_map["weight"] if weights_map is not None else embed.weight
        if weight.device.type == "meta":
            raise RuntimeError("could not recover the embedding weights from accelerate")
        remove_hook_from_module(embed)
        inner.embed_tokens = CpuEmbedding(weight, torch.device("cuda", 0))

    def vram_summary(self) -> str:
        if not torch.cuda.is_available():
            return "no CUDA device"
        free, total = torch.cuda.mem_get_info()
        gib = 2**30
        return (
            f"VRAM: {torch.cuda.memory_allocated() / gib:.2f} GiB allocated, "
            f"{torch.cuda.max_memory_allocated() / gib:.2f} GiB peak, "
            f"{free / gib:.2f} of {total / gib:.2f} GiB free"
        )

    @classmethod
    def from_parts(cls, model, directions, *, r_l: Mapping[int, float] | None = None):
        """Construct a runtime around a model and fake direction store for tests."""
        obj = cls.__new__(cls)
        obj.config = None
        obj.model_dir = None
        obj.tokenizer = None
        obj.model = model
        obj.model.eval()
        obj.layers = model.model.layers
        obj.n_layers = int(model.config.num_hidden_layers)
        obj.n_neurons = int(model.config.intermediate_size)
        obj.hidden_size = int(model.config.hidden_size)
        obj.device = next(model.parameters()).device
        obj.directions = directions
        obj.r_l = dict(r_l or {})
        return obj

    def validate(self, steers: Iterable[NeuronSteer], alpha_max: float = 2.5) -> list[NeuronSteer]:
        return [s.coerce(self.n_layers, self.n_neurons, alpha_max) for s in steers]

    def render_messages(self, messages: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        limit = self.config.context_tokens if self.config else 2048
        messages = list(messages)
        while True:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            encoded = self.tokenizer(text, return_tensors="pt")
            if encoded["input_ids"].shape[1] <= limit or len(messages) <= 1:
                break
            # Over budget: forget the oldest exchange rather than cutting off the newest message.
            messages = messages[2:] if len(messages) > 2 else messages[-1:]
        if encoded["input_ids"].shape[1] > limit:
            encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=limit)
        return encoded

    def _measure_r(self, inputs: dict[str, torch.Tensor], layers: Iterable[int]) -> dict[int, float]:
        """Median residual norm after each wanted layer, for the first token excluded.

        Runs the decoder stack only and reads just the wanted layers through hooks. Going
        through the LM head with ``output_hidden_states`` would materialize logits for every
        position plus every layer's states, which does not fit next to the weights on 6 GB.
        """
        measured: dict[int, float] = {}
        handles = []

        def capture(layer: int):
            def hook(_module, _args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                measured[layer] = float(hidden[0, 1:].float().norm(dim=-1).median().item())

            return hook

        try:
            for layer in set(layers):
                handles.append(self.layers[layer].register_forward_hook(capture(layer)))
            with torch.inference_mode():
                self.model.model(
                    **{key: value.to(self.device) for key, value in inputs.items()},
                    use_cache=False,
                )
        finally:
            for handle in handles:
                handle.remove()
        return measured

    def ensure_r(self, inputs: dict[str, torch.Tensor], layers: Iterable[int]) -> dict[int, float]:
        layers = set(layers)
        missing = layers - self.r_l.keys()
        if missing:
            self.r_l.update(self._measure_r(inputs, missing))
        return {layer: self.r_l[layer] for layer in layers}

    @staticmethod
    def _add_to_layer_output(output, delta: torch.Tensor):
        if torch.is_tensor(output):
            return output + delta.to(device=output.device, dtype=output.dtype)
        if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
            first = output[0] + delta.to(device=output[0].device, dtype=output[0].dtype)
            return (first, *output[1:])
        raise TypeError(f"unsupported decoder-layer output type: {type(output)!r}")

    @contextmanager
    def steering_hooks(self, steers: Iterable[NeuronSteer], r_l: Mapping[int, float]):
        """Install hooks for one generation and always remove them afterward."""
        steers = list(steers)
        handles = []
        grouped: dict[int, list[NeuronSteer]] = {}
        for steer in steers:
            grouped.setdefault(steer.layer, []).append(steer)

        try:
            for layer, entries in grouped.items():
                vectors = [(entry, self.directions.column(layer, entry.neuron)) for entry in entries]
                if self.config is None or self.config.inject_mode == "residual":
                    delta = torch.zeros(self.hidden_size, dtype=torch.float32)
                    for entry, vector in vectors:
                        norm = vector.norm().clamp_min(1e-8)
                        delta += entry.alpha * r_l[layer] * vector / norm
                    delta = delta.to(self.device)

                    def residual_hook(_module, _args, output, delta=delta):
                        return self._add_to_layer_output(output, delta)

                    handles.append(self.layers[layer].register_forward_hook(residual_hook))
                else:
                    shifts = {
                        entry.neuron: entry.alpha * r_l[layer] / vector.norm().clamp_min(1e-8)
                        for entry, vector in vectors
                    }

                    def activation_hook(_module, args, shifts=shifts):
                        activation = args[0].clone()
                        for neuron, shift in shifts.items():
                            activation[..., neuron] += shift
                        return (activation, *args[1:])

                    handles.append(
                        self.layers[layer].mlp.down_proj.register_forward_pre_hook(activation_hook)
                    )
            yield
        finally:
            for handle in handles:
                handle.remove()

    def generate(
        self,
        messages: list[dict[str, str]],
        steers: Iterable[NeuronSteer] = (),
        *,
        max_new_tokens: int | None = None,
        greedy: bool = False,
    ) -> str:
        """Generate one answer with the requested neuron pushes."""
        steers = self.validate(list(steers), self.config.alpha_max if self.config else 2.5)
        inputs = self.render_messages(messages)
        layers = {steer.layer for steer in steers}
        r_l: dict[int, float] = {}
        if layers:
            if self.config is None:
                r_l = self.ensure_r(inputs, layers)
            elif self.config.r_mode == "per_prompt":
                r_l = self._measure_r(inputs, layers)
            else:
                # canonical: r_L comes from "Who are you?", whatever is being asked now
                canonical = self.render_messages([{"role": "user", "content": CANONICAL_PROMPT}])
                r_l = self.ensure_r(canonical, layers)

        generation: dict[str, object] = {
            "max_new_tokens": max_new_tokens or (self.config.max_new_tokens if self.config else 64),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        if greedy:
            generation["do_sample"] = False
        else:
            generation.update(
                {
                    "do_sample": True,
                    "temperature": self.config.temperature if self.config else 0.7,
                    "top_p": self.config.top_p if self.config else 0.9,
                    "repetition_penalty": self.config.repetition_penalty if self.config else 1.1,
                }
            )

        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode(), self.steering_hooks(steers, r_l):
            output = self.model.generate(**inputs, **generation)
        prompt_len = inputs["input_ids"].shape[1]
        return self.tokenizer.decode(
            output[0, prompt_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

    def describe(self) -> str:
        quant = "4-bit NF4" if self.config and self.config.load_4bit else "full precision"
        return f"{self.config.model_id if self.config else self.model.config.name_or_path} ({quant}, {self.n_layers} layers × {self.n_neurons} neurons)"

    def close(self) -> None:
        self.directions.close()
