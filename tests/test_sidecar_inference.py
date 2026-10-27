"""Llama/Nanbeige sidecar inference checks (FlashAttention dependency required)."""
import pytest
import torch
import torch.distributed as dist
from transformers import LlamaConfig

from fastdms.engine.dms import load_dms_metadata
from fastdms.models.llama import LlamaForCausalLM
from fastdms.models.sidecar import attach_sidecar
from fastdms.utils.context import set_context, reset_context, get_context
from training.sidecar import LinearSidecar, export_sidecar


def test_nanbeige_gqa_sidecar_preserves_queries(tmp_path, monkeypatch):
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    config = LlamaConfig(vocab_size=32, hidden_size=56, intermediate_size=64,
                         num_hidden_layers=1, num_attention_heads=14,
                         num_key_value_heads=2, head_dim=4, max_position_embeddings=32,
                         rope_theta=120000000)
    model = LlamaForCausalLM(config)
    # Inference weights normally come from checkpoint loading.
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.1)
    sidecar = LinearSidecar(1, 2, 56)
    with torch.no_grad():
        sidecar.bias[0] = torch.tensor([1.0, -1.0])
    export_sidecar(sidecar, config, tmp_path, artifact_fingerprint="a" * 64,
                   data_manifest_sha256="b" * 64, trainer_commit="c" * 40,
                   seed=1, window_size=2, target_cr=2, summary={})
    metadata = load_dms_metadata(tmp_path)
    attach_sidecar(model, metadata)
    attn = model.model.layers[0].self_attn
    attn.attn.layer_id = 0
    # Capture Q after rotary application instead of invoking a GPU kernel.
    captured = []
    def attention(q, k, v, **kwargs):
        captured.append(q.clone())
        return q
    monkeypatch.setattr(attn.attn, "forward", attention)
    hidden = torch.randn(3, 56)
    positions = torch.arange(3)
    try:
        set_context(is_prefill=True)
        attn(positions, hidden)
        set_context(is_prefill=True, dms_capture_enabled=True,
                    dms_layer_evictions={}, dms_alpha_scale=1.0, dms_alpha_offset=0.0)
        attn(positions, hidden)
        torch.testing.assert_close(captured[0], captured[1], rtol=0, atol=0)
        assert get_context().dms_layer_evictions[0].tolist() == [[True, False]] * 3
    finally:
        reset_context()

    # Tensor-parallel ranks must receive their own contiguous KV-head slice.
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_rank", lambda: 1)
    parallel = LlamaForCausalLM(config)
    attach_sidecar(parallel, metadata)
    assert parallel.model.layers[0].self_attn.sidecar_bias.tolist() == [-1.0]

    path = tmp_path / "sidecar.safetensors"
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="SHA-256"):
        load_dms_metadata(tmp_path)
