import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from training.sidecar import LinearSidecar, eviction_labels, export_sidecar, future_attention_mass
from training.train_sidecar import train_batch


def tiny_model():
    config = LlamaConfig(vocab_size=32, hidden_size=56, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=14,
                         num_key_value_heads=2, head_dim=4)
    config._attn_implementation = "sdpa"
    return LlamaForCausalLM(config).eval().requires_grad_(False)


def test_labels_window_and_ties():
    labels, eligible = eviction_labels(torch.ones(1, 10, 2), 2, 2)
    assert eligible == 7
    assert labels[0, :3].eq(1).all()
    assert labels[0, 3:].eq(0).all()


def test_future_mass_gqa_reference():
    torch.manual_seed(3)
    q = torch.randn(1, 14, 10, 4)
    k = torch.randn(1, 2, 10, 4)
    actual = future_attention_mass(q, k, 2, 3)
    reference = torch.zeros(1, 10, 2)
    for t in range(10):
        for h in range(14):
            p = (q[0, h, t] @ k[0, h // 7, :t + 1].T / 2).softmax(-1)
            for s in range(max(0, t - 2)):
                reference[0, s, h // 7] += p[s]
    torch.testing.assert_close(actual, reference)


def test_frozen_training_and_accumulation():
    torch.manual_seed(1)
    base = tiny_model()
    heads = LinearSidecar(2, 2, 56)
    ids = torch.randint(0, 32, (2, 10))
    original = {k: v.clone() for k, v in base.state_dict().items()}
    loss = train_batch(base, heads, ids, window_size=2, target_cr=2)
    assert torch.isfinite(loss)
    grads = [p.grad.clone() for p in heads.parameters()]
    assert grads[0].abs().sum() > 0
    heads.zero_grad()
    for row in ids:
        train_batch(base, heads, row[None], window_size=2, target_cr=2, loss_scale=0.5)
    for p, grad in zip(heads.parameters(), grads):
        torch.testing.assert_close(p.grad, grad)
    assert all(p.grad is None for p in base.parameters())
    for k, v in base.state_dict().items():
        assert torch.equal(v, original[k])
    assert all(not layer.self_attn._forward_pre_hooks for layer in base.model.layers)


def test_export_roundtrip(tmp_path):
    base = tiny_model()
    heads = LinearSidecar(2, 2, 56)
    with torch.no_grad():
        heads.weight.normal_()
    metadata = export_sidecar(heads, base.config, tmp_path, artifact_fingerprint="a" * 64,
                             data_manifest_sha256="b" * 64, trainer_commit="c" * 40,
                             seed=42, window_size=2, target_cr=2, summary={})
    restored = LinearSidecar.load(tmp_path / "sidecar.safetensors")
    torch.testing.assert_close(restored.weight, heads.weight.bfloat16().float())
    assert metadata["zero_borrowed_query_channel"] is False


def test_hipengine_loader_and_projection(tmp_path):
    # Optional cross-repository contract test against the actual consumer.
    import os
    root = os.environ.get("HIPENGINE_ROOT")
    if not root:
        pytest.skip("set HIPENGINE_ROOT to test the actual hipEngine consumer")
    sys.path.insert(0, root)
    from hipengine.kvcache.dms import load_dms_retrofit_config
    from hipengine.kvcache.dms_sidecar import load_external_dms_sidecar
    from hipengine.kvcache.dms_labels import future_attention_mass_cpu
    test_export_roundtrip(tmp_path)
    config = load_dms_retrofit_config(tmp_path, expected_physical_layer_ids=[0, 1])
    source = load_external_dms_sidecar(config)
    model = LinearSidecar.load(tmp_path / "sidecar.safetensors")
    hidden = torch.randn(5, 56).bfloat16().float()
    actual, decisions = source.project(hidden.numpy(), physical_layer_id=1)
    expected = model(hidden, 1).detach().numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-5)
    np.testing.assert_array_equal(decisions, expected > 0)
    q, k = torch.randn(1, 14, 10, 4), torch.randn(1, 2, 10, 4)
    reference = future_attention_mass_cpu(q[0].transpose(0, 1).numpy(),
                                          k[0].transpose(0, 1).numpy(), window_size=2)
    np.testing.assert_allclose(future_attention_mass(q, k, 2)[0].numpy(), reference,
                               atol=1e-6, rtol=1e-5)
