import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    import json
    config_path = os.path.join(path, "config.json")
    with open(config_path) as handle:
        quantization = json.load(handle).get("quantization_config")
    if quantization:
        return load_channel_fp8_model(model, path, quantization)
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))


def load_channel_fp8_model(model, path, quantization):
    """Load compressed-tensors channel FP8 directly, preserving saved quantization."""
    from fastdms.layers.linear import LinearBase, RowParallelLinear
    groups = quantization.get("config_groups", {})
    if (quantization.get("quant_method") != "compressed-tensors"
            or quantization.get("format") != "float-quantized" or len(groups) != 1):
        raise ValueError("supported quantization is compressed-tensors channel FP8")
    group = next(iter(groups.values()))
    weights = group.get("weights", {})
    activations = group.get("input_activations", {})
    if (weights.get("strategy") != "channel" or weights.get("num_bits") != 8
            or weights.get("type") != "float" or not weights.get("symmetric")
            or activations.get("strategy") != "token" or not activations.get("dynamic")
            or activations.get("num_bits") != 8 or activations.get("type") != "float"):
        raise ValueError("FP8 requires symmetric channel weights and dynamic token activations")
    mapping = getattr(model, "packed_modules_mapping", {})
    modules = dict(model.named_modules())
    pending = {}
    seen = set()
    for file in sorted(glob(os.path.join(path, "*.safetensors"))):
        with safe_open(file, "pt", "cpu") as handle:
            for name in handle.keys():
                if name.endswith(".weight_scale"):
                    continue
                tensor = handle.get_tensor(name)
                if tensor.dtype != torch.float8_e4m3fn:
                    param = model.get_parameter(name)
                    getattr(param, "weight_loader", default_weight_loader)(param, tensor)
                    continue
                if not name.endswith(".weight"):
                    raise ValueError(f"unexpected FP8 tensor {name}")
                source_module = name.removesuffix(".weight")
                scale_name = source_module + ".weight_scale"
                if scale_name not in handle.keys():
                    raise ValueError(f"missing colocated scale for {name}")
                scale = handle.get_tensor(scale_name)
                if scale.shape != (tensor.shape[0], 1) or not torch.isfinite(scale).all() or not (scale > 0).all():
                    raise ValueError(f"invalid channel scales for {name}")
                leaf = source_module.rsplit(".", 1)[-1]
                packed, shard = mapping.get(leaf, (leaf, None))
                destination = source_module.rsplit(".", 1)[0] + "." + packed
                module = modules[destination]
                if not isinstance(module, LinearBase):
                    raise ValueError(f"FP8 checkpoint target is not a supported linear: {destination}")
                if destination not in pending:
                    pending[destination] = (
                        nn.Parameter(torch.empty_like(module.weight, dtype=torch.float8_e4m3fn), requires_grad=False),
                        nn.Parameter(torch.empty((module.weight.shape[0], 1), device=module.weight.device, dtype=torch.float32), requires_grad=False),
                        set(),
                    )
                target, scales, parts = pending[destination]
                if shard in parts:
                    raise ValueError(f"duplicate FP8 shard for {name}")
                parts.add(shard)
                if shard is None:
                    module.weight_loader(target, tensor)
                    if isinstance(module, RowParallelLinear):
                        scales.data.copy_(scale)
                    else:
                        module.weight_loader(scales, scale)
                else:
                    module.weight_loader(target, tensor, shard)
                    module.weight_loader(scales, scale, shard)
                seen.add(destination)
    for name, (weight, scales, parts) in pending.items():
        module = modules[name]
        expected = {"q", "k", "v"} if name.endswith(".qkv_proj") else ({0, 1} if name.endswith(".gate_up_proj") else {None})
        if parts != expected:
            raise ValueError(f"incomplete FP8 checkpoint shards for {name}: {parts}")
        module._set_fp8_buffer("weight_fp8_t", weight.detach().t())
        module._set_fp8_buffer("weight_scale_inv", scales.detach().t().contiguous())
        module._fp8_module_name = name
        module._fp8_weight_enabled = True
        del module._parameters["weight"]
        module.register_buffer("weight", torch.empty(0, device=weight.device, dtype=weight.dtype), persistent=False)
    if not seen:
        raise ValueError("FP8 checkpoint contains no FP8 linear weights")
