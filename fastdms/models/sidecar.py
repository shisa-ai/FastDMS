"""Attach schema-v2 external decision heads to the Llama inference path."""
import json
import torch
import torch.distributed as dist
from safetensors.torch import load_file


def exact_budget_decisions(scores, cu_seqlens, window, target_cr):
    """Rank each sequence/head independently, protecting age <= window."""
    import math
    decisions = torch.zeros_like(scores, dtype=torch.bool)
    boundaries = cu_seqlens.tolist()
    for start, end in zip(boundaries, boundaries[1:]):
        eligible = max(0, end - start - window - 1)
        count = eligible - math.ceil(eligible / target_cr)
        if count:
            order = scores[start:start + eligible].argsort(dim=0, descending=True, stable=True)
            decisions[start:start + eligible].scatter_(0, order[:count], True)
    return decisions


def attach_sidecar(model, metadata):
    from fastdms.models.llama import LlamaForCausalLM
    if not isinstance(model, LlamaForCausalLM):
        raise ValueError("external DMS sidecars require the Llama inference implementation")
    with open(metadata.source_path, encoding="utf-8") as handle:
        raw = json.load(handle)
    raw = raw.get("dms", raw.get("config", raw))
    layers = model.model.layers
    if raw.get("prefill_selection_mode", "threshold") not in {"threshold", "exact_budget"}:
        raise ValueError("unsupported external sidecar prefill selection mode")
    if (raw.get("input_stage") != "post_attn_rmsnorm_pre_q_projection"
            or raw.get("physical_layer_ids") != list(range(len(layers)))
            or raw.get("corrected_mask", False) or raw.get("borrowed_query_channel") is not None):
        raise ValueError("sidecar input stage, query semantics or physical layer map mismatch")
    tensors = load_file(metadata.sidecar_path, device="cpu")
    spec = raw["sidecar"]
    weight, bias = tensors[spec["weight_tensor"]], tensors[spec["bias_tensor"]]
    attn = layers[0].self_attn
    input_size = (attn.qkv_proj.weight_fp8_t.shape[0] if attn.qkv_proj._fp8_weight_enabled
                  else attn.qkv_proj.weight.shape[1])
    expected = (len(layers), attn.total_num_kv_heads, input_size)
    if (raw.get("num_layers") != len(layers)
            or raw.get("num_kv_heads") != attn.total_num_kv_heads
            or raw.get("hidden_size") != expected[2]
            or spec.get("format") != "safetensors" or spec.get("dtype") != "bfloat16"):
        raise ValueError("sidecar metadata geometry or format does not match the model")
    if spec.get("weight_shape") != list(expected) or spec.get("bias_shape") != list(expected[:2]):
        raise ValueError("sidecar declared tensor shapes do not match the model")
    if (tuple(weight.shape) != expected or tuple(bias.shape) != expected[:2]
            or weight.dtype != torch.bfloat16 or bias.dtype != torch.bfloat16
            or not torch.isfinite(weight).all() or not torch.isfinite(bias).all()
            or raw["num_q_heads"] != attn.total_num_heads or raw["head_dim"] != attn.head_dim):
        raise ValueError("sidecar tensor geometry, dtype or values do not match the model")
    rank = dist.get_rank()
    start = rank * attn.num_kv_heads
    for index, layer in enumerate(layers):
        attention = layer.self_attn
        device = attention.qkv_proj.weight.device
        # FP32 accumulation matches hipEngine's external projection contract.
        attention.sidecar_selection_mode = raw.get("prefill_selection_mode", "threshold")
        attention.sidecar_target_cr = int(raw["target_compression_ratio"])
        if attention.sidecar_target_cr < 1:
            raise ValueError("sidecar target compression ratio must be positive")
        attention.register_buffer("sidecar_weight", weight[index, start:start + attn.num_kv_heads].to(device=device, dtype=torch.float32))
        attention.register_buffer("sidecar_bias", bias[index, start:start + attn.num_kv_heads].to(device=device, dtype=torch.float32))
