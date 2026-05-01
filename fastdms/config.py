import os
from dataclasses import dataclass
import torch
from transformers import AutoConfig


_KV_CACHE_DTYPES = {
    "auto": None,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
if hasattr(torch, "float8_e4m3fn"):
    _KV_CACHE_DTYPES["float8_e4m3fn"] = torch.float8_e4m3fn
if hasattr(torch, "float8_e5m2"):
    _KV_CACHE_DTYPES["float8_e5m2"] = torch.float8_e5m2


def _resolve_dtype_value(value) -> torch.dtype | None:
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        return _KV_CACHE_DTYPES.get(value.removeprefix("torch."))
    return None


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    kv_cache_dtype: str = "auto"
    num_kvcache_blocks: int = -1
    compact_kv_enabled: bool = False
    compact_kv_capacity_tokens: int = 0
    compact_kv_retention_mode: str = "all"
    compact_kv_retention_stride: int = 1
    compact_kv_retention_recent_tokens: int = 0
    dms_metadata_path: str | None = None
    dms_window_size: int | None = None
    dms_alpha_scale: float | None = None
    dms_alpha_offset: float | None = None
    num_page_override: int | None = None
    token_pool_tokens: int | None = None
    compact_kv_budget_bytes: int | None = None
    compact_kv_target_live_tokens_per_seq: int | None = None
    compact_kv_layer_major_metadata: bool = False

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.kv_cache_dtype in _KV_CACHE_DTYPES, (
            f"kv_cache_dtype must be one of {sorted(_KV_CACHE_DTYPES)}, got {self.kv_cache_dtype!r}"
        )
        assert self.compact_kv_retention_mode in ("all", "dms_stride", "dms")
        assert self.compact_kv_retention_stride > 0
        assert self.compact_kv_retention_recent_tokens >= 0
        if self.dms_window_size is not None:
            assert self.dms_window_size >= 0
        if self.dms_alpha_scale is not None:
            assert self.dms_alpha_scale != 0.0
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        model_dtype = _resolve_dtype_value(self.hf_config.dtype)
        kv_dtype = self.resolve_kv_cache_dtype()
        if (
            model_dtype is not None
            and kv_dtype != model_dtype
            and self.kv_cache_dtype not in ("float8_e4m3fn", "float8_e5m2")
        ):
            raise ValueError(
                "mixed KV cache dtype is not supported yet: "
                f"kv_cache_dtype={self.kv_cache_dtype!r} resolves to {kv_dtype}, "
                f"but model dtype is {model_dtype}. Only FP8 storage is currently allowed."
            )

    def resolve_kv_cache_dtype(self) -> torch.dtype:
        dtype = _KV_CACHE_DTYPES[self.kv_cache_dtype]
        if dtype is not None:
            return dtype
        dtype = _resolve_dtype_value(self.hf_config.dtype)
        if dtype is not None:
            return dtype
        raise ValueError(f"cannot resolve kv_cache_dtype='auto' from hf_config dtype {self.hf_config.dtype!r}")
