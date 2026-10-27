"""External linear DMS heads and hipEngine-compatible future-attention labels."""
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import save_file, load_file


INPUT_STAGE = "post_attn_rmsnorm_pre_q_projection"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class LinearSidecar(nn.Module):
    def __init__(self, layers, kv_heads, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(layers, kv_heads, hidden_size))
        self.bias = nn.Parameter(torch.zeros(layers, kv_heads))

    def forward(self, hidden, layer):
        return torch.nn.functional.linear(hidden.float(), self.weight[layer], self.bias[layer])

    @classmethod
    def load(cls, path):
        tensors = load_file(str(path))
        model = cls(*tensors["weight"].shape)
        model.load_state_dict(tensors)
        return model


@torch.no_grad()
def future_attention_mass(query, key, window_size, query_chunk=128, query_stride=1):
    """Post-RoPE Q/K [batch, heads, tokens, dim]; sum future use per KV head.

    A key is eligible only when query_position - key_position > window_size,
    matching hipEngine. Chunk queries to bound temporary attention memory.
    """
    if window_size < 0 or query_chunk <= 0 or query_stride <= 0:
        raise ValueError("window_size must be non-negative and query_chunk positive")
    batch, heads, tokens, dim = query.shape
    kv_heads = key.shape[1]
    if heads % kv_heads or key.shape != (batch, kv_heads, tokens, dim):
        raise ValueError("Q/K must have matching geometry and divisible GQA heads")
    group = heads // kv_heads
    mass = torch.zeros(batch, kv_heads, tokens, device=query.device, dtype=torch.float32)
    keys = torch.arange(tokens, device=query.device)
    # One KV head at a time avoids expanding all GQA keys.
    for head in range(kv_heads):
        k = key[:, head].float()
        for start in range(0, tokens, query_chunk):
            end = min(tokens, start + query_chunk)
            positions_selected = torch.arange(start, end, device=query.device)[::query_stride]
            q = query[:, head * group:(head + 1) * group, positions_selected].float()
            scores = torch.matmul(q, k[:, None].transpose(-1, -2)) / math.sqrt(dim)
            positions = positions_selected[:, None]
            scores.masked_fill_(keys[None, :] > positions, float("-inf"))
            probs = scores.softmax(-1)
            probs *= (positions - keys[None, :] > window_size)
            mass[:, head] += probs.sum(dim=(1, 2))
    return mass.transpose(1, 2)


@torch.no_grad()
def eviction_labels(mass, window_size, target_cr):
    """Evict the least-used eligible history; ties prefer earlier positions."""
    tokens = mass.shape[1]
    if window_size < 0 or target_cr < 1:
        raise ValueError("window_size must be non-negative and target_cr at least one")
    eligible = max(0, tokens - window_size - 1)
    count = eligible - math.ceil(eligible / target_cr)
    labels = torch.zeros_like(mass)
    order = mass[:, :eligible].argsort(dim=1, stable=True)
    labels.scatter_(1, order[:, :count], 1)
    return labels, eligible


def export_sidecar(model, config, output, *, artifact_fingerprint, data_manifest_sha256,
                   trainer_commit, seed, window_size, target_cr, summary):
    """Write BF16 tensors and schema-v2 metadata without changing base weights."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tensors = {name: value.detach().cpu().to(torch.bfloat16).contiguous()
               for name, value in model.state_dict().items()}
    if not all(torch.isfinite(value).all() for value in tensors.values()):
        raise ValueError("sidecar tensors must be finite")
    sidecar = output / "sidecar.safetensors"
    save_file(tensors, str(sidecar))
    layers, heads, hidden = tensors["weight"].shape
    metadata = dict(
        schema_version=2, artifact_fingerprint=artifact_fingerprint,
        model_family=config.model_type, decision_source="external_linear_sidecar_v1",
        physical_layer_ids=list(range(layers)), num_layers=layers,
        num_q_heads=config.num_attention_heads, num_kv_heads=heads,
        head_dim=getattr(config, "head_dim", None) or hidden // config.num_attention_heads,
        hidden_size=hidden, input_stage=INPUT_STAGE, prefill_selection_mode="threshold",
        window_size=window_size, target_compression_ratio=target_cr,
        alpha_scale=1.0, alpha_offset=0.0, borrowed_query_channel=None,
        zero_borrowed_query_channel=False, corrected_mask=False,
        trained_checkpoint=True, evidence_source="training_summary.json",
        sidecar=dict(path=sidecar.name, format="safetensors", dtype="bfloat16",
                     weight_tensor="weight", bias_tensor="bias",
                     weight_shape=list(tensors["weight"].shape),
                     bias_shape=list(tensors["bias"].shape), sha256=sha256_file(sidecar)),
        training=dict(method="future_attention_distillation_v1",
                      data_manifest_sha256=data_manifest_sha256,
                      trainer_commit=trainer_commit, fastdms_reference_commit=trainer_commit,
                      seed=seed),
    )
    for name, payload in (("dms_metadata.json", metadata), ("training_summary.json", summary)):
        (output / name).write_text(json.dumps(payload, indent=2) + "\n")
    return metadata
