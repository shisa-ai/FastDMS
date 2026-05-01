from __future__ import annotations

from transformers import PretrainedConfig
from torch import nn

from fastdms.models.llama import LlamaForCausalLM
from fastdms.models.qwen3 import Qwen3ForCausalLM


_MODEL_TYPES: dict[str, type[nn.Module]] = {
    "llama": LlamaForCausalLM,
    "qwen3": Qwen3ForCausalLM,
}

_ARCHITECTURES: dict[str, type[nn.Module]] = {
    "LlamaForCausalLM": LlamaForCausalLM,
    "Qwen3ForCausalLM": Qwen3ForCausalLM,
}


def get_model_class(config: PretrainedConfig) -> type[nn.Module]:
    model_type = getattr(config, "model_type", None)
    if model_type in _MODEL_TYPES:
        return _MODEL_TYPES[model_type]

    for architecture in getattr(config, "architectures", []) or []:
        if architecture in _ARCHITECTURES:
            return _ARCHITECTURES[architecture]

    architectures = ", ".join(getattr(config, "architectures", []) or [])
    raise ValueError(
        f"unsupported model type {model_type!r}"
        + (f" with architectures [{architectures}]" if architectures else "")
    )


def build_model(config: PretrainedConfig) -> nn.Module:
    return get_model_class(config)(config)
