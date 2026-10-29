import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaForCausalLM

from llmsteer.directions import DirectionStore
from llmsteer.model import SteeringModel
from llmsteer.steering import NeuronSteer


class FakeDirections:
    def __init__(self, vector):
        self.vector = vector.float()

    def column(self, layer, neuron):
        return self.vector


def _capture_target(model, layer_index, fn):
    captured = {}

    def capture(_module, _args, output):
        captured["hidden"] = output[0].detach().clone() if isinstance(output, tuple) else output.detach().clone()
        return output

    handle = model.model.layers[layer_index].register_forward_hook(capture)
    try:
        fn()
    finally:
        handle.remove()
    return captured["hidden"]


def _tiny_model():
    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    return LlamaForCausalLM(config).eval()


def test_direction_store_reads_down_projection_column():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        expected = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        save_file({"model.layers.1.mlp.down_proj.weight": expected}, str(root / "model.safetensors"))
        store = DirectionStore(root)
        assert torch.equal(store.column(1, 2), expected[:, 2])
        store.close()


def test_residual_and_activation_hooks_apply_the_same_linear_push():
    baseline_model = _tiny_model()
    residual_model = _tiny_model()
    activation_model = _tiny_model()
    activation_model.load_state_dict(baseline_model.state_dict())
    residual_model.load_state_dict(baseline_model.state_dict())

    layer = 1
    neuron = 7
    vector = baseline_model.model.layers[layer].mlp.down_proj.weight[:, neuron].detach().float()
    directions = FakeDirections(vector)
    r_l = {layer: 1.75}
    steer = [NeuronSteer(layer, neuron, 1.25)]
    input_ids = torch.tensor([[3, 5, 9, 11]], dtype=torch.long)

    baseline_runtime = SteeringModel.from_parts(baseline_model, directions, r_l=r_l)
    residual_runtime = SteeringModel.from_parts(residual_model, directions, r_l=r_l)
    activation_runtime = SteeringModel.from_parts(activation_model, directions, r_l=r_l)
    baseline_hidden = _capture_target(
        baseline_model,
        layer,
        lambda: baseline_model(input_ids=input_ids, use_cache=False),
    )
    residual_hidden = _run_with_hooks(residual_runtime, input_ids, steer, layer)
    activation_hidden = _run_with_hooks(activation_runtime, input_ids, steer, layer)

    expected_delta = 1.25 * 1.75 * vector / vector.norm()
    actual_delta = residual_hidden - baseline_hidden
    assert torch.allclose(actual_delta, expected_delta.view(1, 1, -1), atol=2e-5, rtol=2e-5)
    assert torch.allclose(activation_hidden - baseline_hidden, actual_delta, atol=2e-5, rtol=2e-5)


def _run_with_hooks(runtime, input_ids, steer, layer):
    with runtime.steering_hooks(steer, runtime.r_l):
        return _capture_target(
            runtime.model,
            layer,
            lambda: runtime.model(input_ids=input_ids, use_cache=False),
        )




def test_cpu_embedding_matches_embedding_and_keeps_table_off_device():
    from torch import nn

    from llmsteer.model import CpuEmbedding

    torch.manual_seed(0)
    reference = nn.Embedding(50, 8)
    cpu = CpuEmbedding(reference.weight, "cpu")
    ids = torch.tensor([[1, 5, 49]])
    assert torch.equal(cpu(ids), reference(ids))
    assert not list(cpu.parameters())  # a buffer, so model.device is decided by the real layers


def test_render_messages_drops_oldest_turns_before_truncating():
    class Tok:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return " ".join(m["content"] for m in messages)

        def __call__(self, text, return_tensors=None, truncation=False, max_length=None):
            ids = list(range(len(text.split())))
            if truncation:
                ids = ids[:max_length]
            return {"input_ids": torch.tensor([ids])}

    obj = SteeringModel.__new__(SteeringModel)
    obj.config = type("C", (), {"context_tokens": 6})()
    obj.tokenizer = Tok()
    history = [
        {"role": "user", "content": "a b c"},
        {"role": "assistant", "content": "d e f"},
        {"role": "user", "content": "newest question"},
    ]
    assert obj.render_messages(history)["input_ids"].shape[1] == 2
