import os
import re

import torch
import triton
import triton.language as tl
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from fastdms.utils.profiler import get_profiler


_FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)


def fp8_weights_enabled() -> bool:
    return os.environ.get("NANOVLLM_FP8_WEIGHTS", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def fp8_weight_scope() -> str:
    return os.environ.get("NANOVLLM_FP8_WEIGHT_SCOPE", "mlp").strip().lower()


def fp8_keep_bf16_weights() -> bool:
    return os.environ.get("NANOVLLM_FP8_KEEP_BF16_WEIGHTS", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _global_keep_bf16_explicit() -> bool:
    raw = os.environ.get("NANOVLLM_FP8_KEEP_BF16_WEIGHTS")
    return raw is not None and raw.strip() != ""


def fp8_keep_bf16_gate_up_weights() -> bool:
    if _global_keep_bf16_explicit():
        default = fp8_keep_bf16_weights()
    else:
        default = False
    return _env_bool("NANOVLLM_FP8_KEEP_BF16_GATE_UP", default)


def fp8_keep_bf16_down_weights() -> bool:
    if _global_keep_bf16_explicit():
        default = fp8_keep_bf16_weights()
    else:
        default = False
    return _env_bool("NANOVLLM_FP8_KEEP_BF16_DOWN", default)


def fp8_keep_bf16_attention_weights() -> bool:
    if _global_keep_bf16_explicit():
        default = fp8_keep_bf16_weights()
    else:
        default = False
    return _env_bool("NANOVLLM_FP8_KEEP_BF16_ATTENTION", default)


def fp8_min_rows() -> int:
    return max(1, int(os.environ.get("NANOVLLM_FP8_MIN_ROWS", "2")))


def _fp8_min_rows_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return max(1, int(raw))


def fp8_gate_up_min_rows() -> int:
    return _fp8_min_rows_env("NANOVLLM_FP8_GATE_UP_MIN_ROWS", fp8_min_rows())


def fp8_down_min_rows() -> int:
    return _fp8_min_rows_env("NANOVLLM_FP8_DOWN_MIN_ROWS", fp8_min_rows())


def fp8_down_row1_triton_mode() -> str:
    raw = os.environ.get("NANOVLLM_FP8_DOWN_ROW1_TRITON", "quantized").strip().lower()
    if raw in {"", "0", "false", "no", "off"}:
        return "off"
    if raw in {"1", "true", "yes", "on", "quant", "quantized", "fp8"}:
        return "quantized"
    if raw == "raw_unsafe":
        return "raw"
    raise ValueError("NANOVLLM_FP8_DOWN_ROW1_TRITON must be off, quantized, or raw_unsafe")


def fp8_down_row1_triton_enabled() -> bool:
    return fp8_down_row1_triton_mode() != "off"


def fp8_gate_up_row1_triton_mode() -> str:
    raw = os.environ.get("NANOVLLM_FP8_GATE_UP_ROW1_TRITON", "quantized").strip().lower()
    if raw in {"", "0", "false", "no", "off"}:
        return "off"
    if raw in {"1", "true", "yes", "on", "quant", "quantized", "fp8"}:
        return "quantized"
    if raw == "raw_unsafe":
        return "raw"
    raise ValueError("NANOVLLM_FP8_GATE_UP_ROW1_TRITON must be off, quantized, or raw_unsafe")


def fp8_gate_up_row1_triton_enabled() -> bool:
    return fp8_gate_up_row1_triton_mode() != "off"


def fp8_attention_row1_shadow_enabled() -> bool:
    return _env_bool("NANOVLLM_FP8_ATTENTION_ROW1_SHADOW", True)


def fp8_attention_full_enabled() -> bool:
    return _env_bool("NANOVLLM_FP8_ATTENTION_FULL", True)


def fp8_attention_row1_triton_mode() -> str:
    default = "quantized" if (fp8_attention_row1_shadow_enabled() or fp8_attention_full_enabled()) else "0"
    raw = os.environ.get("NANOVLLM_FP8_ATTENTION_ROW1_TRITON", default).strip().lower()
    if raw in {"", "0", "false", "no", "off"}:
        return "off"
    if raw in {"1", "true", "yes", "on", "quant", "quantized", "fp8"}:
        return "quantized"
    if raw == "raw_unsafe":
        return "raw"
    raise ValueError("NANOVLLM_FP8_ATTENTION_ROW1_TRITON must be off, quantized, or raw_unsafe")


def fp8_attention_row1_triton_enabled() -> bool:
    return fp8_attention_row1_triton_mode() != "off"


def fp8_row1_triton_mode() -> str:
    raw = os.environ.get("NANOVLLM_FP8_ROW1_TRITON", "0").strip().lower()
    if raw in {"", "0", "false", "no", "off"}:
        return "off"
    if raw in {"1", "true", "yes", "on", "quant", "quantized", "fp8"}:
        return "quantized"
    if raw == "raw_unsafe":
        return "raw"
    raise ValueError("NANOVLLM_FP8_ROW1_TRITON must be off, quantized, or raw_unsafe")


def fp8_row1_triton_enabled() -> bool:
    return fp8_row1_triton_mode() != "off"


def fp8_row1_matvec_impl() -> str:
    raw = os.environ.get("NANOVLLM_FP8_ROW1_MATVEC_IMPL", "dot").strip().lower()
    if raw in {"", "scalar", "serial"}:
        return "scalar"
    if raw in {"dot", "tc", "tensorcore", "tensor_core"}:
        return "dot"
    raise ValueError("NANOVLLM_FP8_ROW1_MATVEC_IMPL must be scalar or dot")


def _fp8_row1_matvec_env_int(name: str, default: int, allowed: set[int]) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    value = int(raw)
    if value not in allowed:
        allowed_text = ", ".join(str(v) for v in sorted(allowed))
        raise ValueError(f"{name} must be one of {allowed_text}; got {value}")
    return value


def fp8_row1_matvec_dot_block_k() -> int:
    return _fp8_row1_matvec_env_int("NANOVLLM_FP8_ROW1_MATVEC_DOT_BLOCK_K", 256, {64, 128, 256})


def fp8_row1_matvec_dot_block_n() -> int:
    return _fp8_row1_matvec_env_int("NANOVLLM_FP8_ROW1_MATVEC_DOT_BLOCK_N", 32, {16, 32, 64, 128})


def fp8_small_m_triton_enabled() -> bool:
    return _env_bool("NANOVLLM_FP8_SMALL_M_TRITON", True)


def fp8_small_m_triton_max_rows() -> int:
    return _fp8_row1_matvec_env_int("NANOVLLM_FP8_SMALL_M_TRITON_MAX_ROWS", 8, {2, 4, 8, 16})


def fp8_small_m_triton_block_n() -> int:
    return _fp8_row1_matvec_env_int("NANOVLLM_FP8_SMALL_M_TRITON_BLOCK_N", 32, {16, 32, 64, 128})


def int4_row1_weights_enabled() -> bool:
    return _env_bool("NANOVLLM_INT4_ROW1_WEIGHTS", False)


def int4_row1_scope() -> str:
    return os.environ.get("NANOVLLM_INT4_ROW1_SCOPE", "all").strip().lower()


def _int4_row1_scope_parts() -> list[str]:
    scope = int4_row1_scope()
    parts = [part.strip() for part in scope.replace(",", "+").split("+") if part.strip()]
    return parts or ["none"]


def _module_layer_id(module_name: str | None) -> int | None:
    if not module_name:
        return None
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", module_name)
    if match is None:
        return None
    return int(match.group(1))


def _int4_row1_layer_filter_allows(filter_text: str | None, module_name: str | None) -> bool:
    if filter_text is None or filter_text == "":
        return True
    value = filter_text.strip().lower()
    if value in {"all", "*"}:
        return True
    layer_id = _module_layer_id(module_name)
    if layer_id is None:
        return False
    for item in value.split("|"):
        item = item.strip()
        if not item:
            continue
        if item == "even":
            if layer_id % 2 == 0:
                return True
            continue
        if item == "odd":
            if layer_id % 2 == 1:
                return True
            continue
        if "-" in item:
            lo_text, hi_text = item.split("-", 1)
            lo = int(lo_text)
            hi = int(hi_text)
            if lo <= layer_id <= hi:
                return True
            continue
        if layer_id == int(item):
            return True
    return False


def _split_int4_row1_scope_filter(scope: str) -> tuple[str, str | None]:
    if "@" not in scope:
        return scope, None
    base, layer_filter = scope.split("@", 1)
    if not base or not layer_filter:
        raise ValueError(
            "NANOVLLM_INT4_ROW1_SCOPE layer filters must use scope@layers, "
            f"got {scope!r}"
        )
    return base, layer_filter


def _int4_row1_module_in_single_scope(scope: str, module_name: str | None) -> bool:
    scope, layer_filter = _split_int4_row1_scope_filter(scope)
    if not _int4_row1_layer_filter_allows(layer_filter, module_name):
        return False
    name = module_name or ""
    if scope in {"all", "transformer"}:
        return True
    if scope in {"lm_head", "head"}:
        return False
    if scope in {"mlp_lm_head", "mlp+lm_head", "mlp_head", "mlp+head"}:
        return ".mlp." in name
    if scope in {"mlp", "ffn"}:
        return ".mlp." in name
    if scope in {"attention", "attn"}:
        return ".self_attn." in name
    if scope == "qkv":
        return name.endswith(".qkv_proj")
    if scope in {"attn_out", "o_proj"}:
        return name.endswith(".o_proj")
    if scope in {"mlp_gate_up", "gate_up"}:
        return name.endswith(".gate_up_proj")
    if scope in {"mlp_down", "down"}:
        return name.endswith(".down_proj")
    if scope in {"none", "off", "0", "false"}:
        return False
    raise ValueError(
        "NANOVLLM_INT4_ROW1_SCOPE must be one or more of all, transformer, mlp, attention, "
        "qkv, attn_out, mlp_gate_up, mlp_down, lm_head, mlp_lm_head, none; "
        f"got {scope!r}"
    )


def int4_row1_module_in_scope(module_name: str | None) -> bool:
    return any(_int4_row1_module_in_single_scope(scope, module_name) for scope in _int4_row1_scope_parts())


def int4_row1_lm_head_enabled() -> bool:
    if not int4_row1_weights_enabled():
        return False
    for scope in _int4_row1_scope_parts():
        scope, layer_filter = _split_int4_row1_scope_filter(scope)
        if layer_filter is not None:
            continue
        if scope in {"all", "lm_head", "head", "mlp_lm_head", "mlp_head"}:
            return True
        if scope in {
            "transformer",
            "mlp",
            "ffn",
            "attention",
            "attn",
            "qkv",
            "attn_out",
            "o_proj",
            "mlp_gate_up",
            "gate_up",
            "mlp_down",
            "down",
            "none",
            "off",
            "0",
            "false",
        }:
            continue
        raise ValueError(
            "NANOVLLM_INT4_ROW1_SCOPE must be one or more of all, transformer, mlp, attention, "
            "qkv, attn_out, mlp_gate_up, mlp_down, lm_head, mlp_lm_head, none; "
            f"got {scope!r}"
        )
    return False


def int4_row1_group_size() -> int:
    return _fp8_row1_matvec_env_int("NANOVLLM_INT4_ROW1_GROUP_SIZE", 256, {32, 64, 128, 256})


def int4_row1_quant_mode() -> str:
    raw = os.environ.get("NANOVLLM_INT4_ROW1_QUANT_MODE", "symmetric").strip().lower()
    if raw in {"", "sym", "symmetric"}:
        return "symmetric"
    if raw in {"asym", "asymmetric", "affine"}:
        return "asymmetric"
    raise ValueError("NANOVLLM_INT4_ROW1_QUANT_MODE must be symmetric or asymmetric")


def int4_row1_inner_k_tiles() -> int:
    return _fp8_row1_matvec_env_int("NANOVLLM_INT4_ROW1_INNER_K_TILES", 2, {2, 4, 8})


def int4_row1_max_rows() -> int:
    return _fp8_row1_matvec_env_int("NANOVLLM_INT4_ROW1_MAX_ROWS", 1, {1, 2, 4, 8, 16})


def fp8_module_min_rows(module_name: str | None = None) -> int:
    if module_name is None:
        return fp8_min_rows()
    if module_name.endswith(".gate_up_proj"):
        return fp8_gate_up_min_rows()
    if module_name.endswith(".down_proj"):
        return fp8_down_min_rows()
    return fp8_min_rows()


def fp8_module_keep_bf16_weight(module_name: str | None = None) -> bool:
    if module_name is None:
        return fp8_keep_bf16_weights()
    if ".self_attn." in module_name:
        return fp8_keep_bf16_attention_weights()
    if module_name.endswith(".gate_up_proj"):
        return fp8_keep_bf16_gate_up_weights()
    if module_name.endswith(".down_proj"):
        return fp8_keep_bf16_down_weights()
    return fp8_keep_bf16_weights()


def _amax_to_scale(amax: torch.Tensor) -> torch.Tensor:
    one = torch.ones((), device=amax.device, dtype=torch.float32)
    return torch.where(amax > 0, _FP8_MAX / amax, one).to(torch.float32)


def _as_col_major_b(mat: torch.Tensor) -> torch.Tensor:
    # torch._scaled_mm's FP8 path expects B in column-major layout.
    if mat.stride(0) == 1:
        return mat.contiguous() if not mat.is_contiguous() and mat.stride(0) != 1 else mat
    return mat.t().contiguous().t()


def _can_use_int4_row1_weight(weight: torch.Tensor, group_size: int, inner_k_tiles: int) -> bool:
    if not hasattr(torch, "_convert_weight_to_int4pack") or not hasattr(torch, "_weight_int4pack_mm"):
        return False
    if weight.dim() != 2:
        return False
    out_features = int(weight.size(0))
    in_features = int(weight.size(1))
    return (
        weight.is_cuda
        and out_features % 8 == 0
        and in_features % 2 == 0
        and in_features % group_size == 0
        and in_features % (inner_k_tiles * 16) == 0
    )


def _quantize_weight_to_int4pack(
    weight: torch.Tensor,
    *,
    group_size: int | None = None,
    inner_k_tiles: int | None = None,
    quant_mode: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int] | None:
    if group_size is None:
        group_size = int4_row1_group_size()
    if inner_k_tiles is None:
        inner_k_tiles = int4_row1_inner_k_tiles()
    if quant_mode is None:
        quant_mode = int4_row1_quant_mode()
    quant_mode = quant_mode.strip().lower()
    if not _can_use_int4_row1_weight(weight, group_size, inner_k_tiles):
        return None
    out_features = int(weight.size(0))
    in_features = int(weight.size(1))
    groups = in_features // group_size
    grouped = weight.reshape(out_features, groups, group_size).to(torch.float32)
    one = torch.ones((), device=weight.device, dtype=torch.float32)
    if quant_mode in {"sym", "symmetric"}:
        amax = grouped.abs().amax(dim=2)
        scale = torch.where(amax > 0, amax / 7.0, one)
        zero = torch.zeros_like(scale)
        q_signed = torch.round(grouped / scale.unsqueeze(2)).clamp(-8, 7).to(torch.int16)
        q = (q_signed + 8).to(torch.uint8).reshape(out_features, in_features)
    elif quant_mode in {"asym", "asymmetric", "affine"}:
        w_min = grouped.amin(dim=2)
        w_max = grouped.amax(dim=2)
        span = w_max - w_min
        nonzero = span > 0
        scale = torch.where(nonzero, span / 15.0, one)
        # torch._weight_int4pack_mm interprets scale_zeros[..., 1] as the
        # real-valued additive term in (q - 8) * scale + zero.
        zero = torch.where(nonzero, w_min + 8.0 * scale, w_min)
        q = torch.round((grouped - zero.unsqueeze(2)) / scale.unsqueeze(2) + 8.0)
        q = q.clamp(0, 15).to(torch.uint8).reshape(out_features, in_features)
    else:
        raise ValueError("int4 quant_mode must be symmetric or asymmetric")
    packed_nibbles = ((q[:, 0::2] << 4) | q[:, 1::2]).contiguous()
    weight_int4pack = torch._convert_weight_to_int4pack(packed_nibbles, inner_k_tiles)
    scale_zeros = torch.empty((groups, out_features, 2), device=weight.device, dtype=torch.bfloat16)
    scale_zeros[:, :, 0] = scale.t().to(torch.bfloat16)
    scale_zeros[:, :, 1] = zero.t().to(torch.bfloat16)
    return weight_int4pack, scale_zeros.contiguous(), group_size


def _int4_row1_matvec(
    x: torch.Tensor,
    *,
    weight_int4pack: torch.Tensor,
    weight_int4_scale_zeros: torch.Tensor,
    q_group_size: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    in_features = int(weight_int4_scale_zeros.size(0)) * int(q_group_size)
    out_features = int(weight_int4_scale_zeros.size(1))
    x_2d = x.reshape(-1, in_features).contiguous()
    if x_2d.dtype is not torch.bfloat16:
        raise RuntimeError("CUDA int4 row1/small-M matmul currently requires BF16 activations")
    y_2d = torch._weight_int4pack_mm(x_2d, weight_int4pack, int(q_group_size), weight_int4_scale_zeros)
    if bias is not None:
        y_2d = y_2d + bias.to(dtype=y_2d.dtype)
    return y_2d.reshape(*x.shape[:-1], out_features)


def _can_use_scaled_mm(x: torch.Tensor) -> bool:
    if os.environ.get("NANOVLLM_FP8_LINEAR_IMPL", "").strip().lower() == "dequant":
        return False
    return (
        x.is_cuda
        and hasattr(torch, "_scaled_mm")
        and x.dtype in {torch.float16, torch.bfloat16}
        and x.size(-1) % 16 == 0
    )


@triton.jit
def _fp8_row1_matvec_kernel(
    x_ptr,
    weight_fp8_t_ptr,
    weight_scale_inv_ptr,
    y_ptr,
    K: tl.constexpr,
    N: tl.constexpr,
    WEIGHT_STRIDE_K: tl.constexpr,
    WEIGHT_STRIDE_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QUANTIZE_X: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    if QUANTIZE_X:
        amax = tl.zeros((), tl.float32)
        for start in range(0, K, BLOCK_K):
            k = start + offs_k
            x = tl.load(x_ptr + k, mask=k < K, other=0.0).to(tl.float32)
            amax = tl.maximum(amax, tl.max(tl.abs(x), axis=0))
        x_scale = tl.where(amax > 0.0, FP8_MAX / amax, 1.0)
        x_scale_inv = 1.0 / x_scale
    else:
        x_scale = 1.0
        x_scale_inv = 1.0
    acc = tl.zeros((BLOCK_N,), tl.float32)
    for start in range(0, K, BLOCK_K):
        k = start + offs_k
        x = tl.load(x_ptr + k, mask=k < K, other=0.0).to(tl.float32)
        if QUANTIZE_X:
            x = (x * x_scale).to(tl.float8e4nv).to(tl.float32) * x_scale_inv
        w = tl.load(
            weight_fp8_t_ptr + k[:, None] * WEIGHT_STRIDE_K + n[None, :] * WEIGHT_STRIDE_N,
            mask=(k[:, None] < K) & (n[None, :] < N),
            other=0.0,
        )
        scale = tl.load(weight_scale_inv_ptr + n, mask=n < N, other=0.0).to(tl.float32)
        acc += tl.sum(x[:, None] * w.to(tl.float32) * scale[None, :], axis=0)
    tl.store(y_ptr + n, acc, mask=n < N)


@triton.jit
def _fp8_row1_matvec_dot_kernel(
    x_ptr,
    weight_fp8_t_ptr,
    weight_scale_inv_ptr,
    y_ptr,
    K: tl.constexpr,
    N: tl.constexpr,
    WEIGHT_STRIDE_K: tl.constexpr,
    WEIGHT_STRIDE_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    amax = tl.zeros((), tl.float32)
    for start in range(0, K, BLOCK_K):
        k = start + offs_k
        x = tl.load(x_ptr + k, mask=k < K, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(x), axis=0))
    x_scale = tl.where(amax > 0.0, FP8_MAX / amax, 1.0)
    x_scale_inv = 1.0 / x_scale

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for start in range(0, K, BLOCK_K):
        k = start + offs_k
        x = tl.load(x_ptr + k, mask=k < K, other=0.0).to(tl.float32)
        xq = (x * x_scale).to(tl.float8e4nv)
        x_tile = tl.where(offs_m[:, None] == 0, xq[None, :], 0.0).to(tl.float8e4nv)
        w = tl.load(
            weight_fp8_t_ptr + k[:, None] * WEIGHT_STRIDE_K + n[None, :] * WEIGHT_STRIDE_N,
            mask=(k[:, None] < K) & (n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(x_tile, w)

    scale = tl.load(weight_scale_inv_ptr + n, mask=n < N, other=0.0).to(tl.float32)
    row0 = tl.sum(tl.where(offs_m[:, None] == 0, acc, 0.0), axis=0) * x_scale_inv * scale
    tl.store(y_ptr + n, row0, mask=n < N)


def _fp8_row1_matvec(
    x: torch.Tensor,
    *,
    weight_fp8_t: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    bias: torch.Tensor | None = None,
    quantize_x: bool = False,
    impl: str | None = None,
) -> torch.Tensor:
    in_features = int(weight_fp8_t.size(0))
    out_features = int(weight_fp8_t.size(1))
    x_2d = x.reshape(1, in_features).contiguous()
    y_2d = torch.empty((1, out_features), device=x.device, dtype=x.dtype)
    if impl is None:
        impl = fp8_row1_matvec_impl()
    if quantize_x and impl == "dot":
        dot_block_k = fp8_row1_matvec_dot_block_k()
        dot_block_n = fp8_row1_matvec_dot_block_n()
        _fp8_row1_matvec_dot_kernel[(triton.cdiv(out_features, dot_block_n),)](
            x_2d,
            weight_fp8_t,
            weight_scale_inv,
            y_2d,
            in_features,
            out_features,
            weight_fp8_t.stride(0),
            weight_fp8_t.stride(1),
            BLOCK_M=16,
            BLOCK_K=dot_block_k,
            BLOCK_N=dot_block_n,
            FP8_MAX=_FP8_MAX,
            num_warps=4,
        )
    else:
        _fp8_row1_matvec_kernel[(triton.cdiv(out_features, 16),)](
            x_2d,
            weight_fp8_t,
            weight_scale_inv,
            y_2d,
            in_features,
            out_features,
            weight_fp8_t.stride(0),
            weight_fp8_t.stride(1),
            BLOCK_K=256,
            BLOCK_N=16,
            QUANTIZE_X=quantize_x,
            FP8_MAX=_FP8_MAX,
            num_warps=4,
        )
    if bias is not None:
        y_2d = y_2d + bias.to(dtype=y_2d.dtype)
    return y_2d.reshape(*x.shape[:-1], out_features)


def _fp8_linear(
    x: torch.Tensor,
    *,
    weight_fp8_t: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    in_features = int(weight_fp8_t.size(0))
    out_features = int(weight_fp8_t.size(1))
    x_2d = x.reshape(-1, in_features)
    if not _can_use_scaled_mm(x_2d):
        weight = (weight_fp8_t.to(dtype=x.dtype) * weight_scale_inv.to(dtype=x.dtype)).t().contiguous()
        return F.linear(x, weight, bias)

    x_2d = x_2d.contiguous()
    x_scale = _amax_to_scale(x_2d.abs().amax(dim=1).float())
    x_fp8 = (x_2d * x_scale.unsqueeze(1)).to(torch.float8_e4m3fn)
    y_2d = torch._scaled_mm(
        x_fp8,
        weight_fp8_t,
        scale_a=x_scale.reciprocal().reshape(-1, 1).contiguous(),
        scale_b=weight_scale_inv,
        out_dtype=x.dtype,
        use_fast_accum=False,
    )
    if bias is not None:
        y_2d = y_2d + bias.to(dtype=y_2d.dtype)
    return y_2d.reshape(*x.shape[:-1], out_features)


@triton.jit
def _fp8_small_m_linear_kernel(
    x_ptr,
    weight_fp8_t_ptr,
    weight_scale_inv_ptr,
    y_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    WEIGHT_STRIDE_K: tl.constexpr,
    WEIGHT_STRIDE_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    amax = tl.zeros((BLOCK_M,), tl.float32)
    for start in range(0, K, BLOCK_K):
        k = start + offs_k
        x = tl.load(
            x_ptr + m[:, None] * K + k[None, :],
            mask=(m[:, None] < M) & (k[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(x), axis=1))
    x_scale = tl.where(amax > 0.0, FP8_MAX / amax, 1.0)
    x_scale_inv = 1.0 / x_scale

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for start in range(0, K, BLOCK_K):
        k = start + offs_k
        x = tl.load(
            x_ptr + m[:, None] * K + k[None, :],
            mask=(m[:, None] < M) & (k[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        xq = (x * x_scale[:, None]).to(tl.float8e4nv)
        w = tl.load(
            weight_fp8_t_ptr + k[:, None] * WEIGHT_STRIDE_K + n[None, :] * WEIGHT_STRIDE_N,
            mask=(k[:, None] < K) & (n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(xq, w)

    scale = tl.load(weight_scale_inv_ptr + n, mask=n < N, other=0.0).to(tl.float32)
    y = acc * x_scale_inv[:, None] * scale[None, :]
    tl.store(
        y_ptr + m[:, None] * N + n[None, :],
        y,
        mask=(m[:, None] < M) & (n[None, :] < N),
    )


def _fp8_small_m_linear(
    x: torch.Tensor,
    *,
    weight_fp8_t: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    bias: torch.Tensor | None = None,
    max_rows: int | None = None,
) -> torch.Tensor:
    in_features = int(weight_fp8_t.size(0))
    out_features = int(weight_fp8_t.size(1))
    x_2d = x.reshape(-1, in_features).contiguous()
    rows = int(x_2d.size(0))
    if max_rows is None:
        max_rows = fp8_small_m_triton_max_rows()
    if rows <= 1 or rows > max_rows:
        return _fp8_linear(
            x,
            weight_fp8_t=weight_fp8_t,
            weight_scale_inv=weight_scale_inv,
            bias=bias,
        )
    y_2d = torch.empty((rows, out_features), device=x.device, dtype=x.dtype)
    block_n = fp8_small_m_triton_block_n()
    _fp8_small_m_linear_kernel[(triton.cdiv(out_features, block_n),)](
        x_2d,
        weight_fp8_t,
        weight_scale_inv,
        y_2d,
        rows,
        in_features,
        out_features,
        weight_fp8_t.stride(0),
        weight_fp8_t.stride(1),
        BLOCK_M=max_rows,
        BLOCK_K=256,
        BLOCK_N=block_n,
        FP8_MAX=_FP8_MAX,
        num_warps=4,
    )
    if bias is not None:
        y_2d = y_2d + bias.to(dtype=y_2d.dtype)
    return y_2d.reshape(*x.shape[:-1], out_features)


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


def _linear_profile_kind(module_name: str) -> str:
    if module_name.endswith(".qkv_proj"):
        return "qkv"
    if module_name.endswith(".o_proj"):
        return "attn_out"
    if module_name.endswith(".gate_up_proj"):
        return "gate_up"
    if module_name.endswith(".down_proj"):
        return "down"
    return "other"


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        self._fp8_weight_enabled = False
        self._fp8_min_rows = 1
        self._fp8_module_name = ""
        self._fp8_row1_only = False
        self._int4_row1_weight_enabled = False
        self._int4_row1_group_size = 0
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def quantize_weight_to_fp8(
        self,
        *,
        keep_bf16_weight: bool = False,
        module_name: str | None = None,
        row1_only: bool = False,
    ) -> None:
        if row1_only and not keep_bf16_weight:
            raise ValueError("row1-only FP8 shadow requires keeping the BF16 weight")
        weight = self.weight.detach().to(torch.float32)
        weight_scale = _amax_to_scale(weight.abs().amax(dim=1).float())
        weight_fp8_t = (weight.t() * weight_scale.unsqueeze(0)).to(torch.float8_e4m3fn)
        weight_fp8_t = _as_col_major_b(weight_fp8_t)
        weight_scale_inv = weight_scale.reciprocal().reshape(1, -1).contiguous()
        self._set_fp8_buffer("weight_fp8_t", weight_fp8_t)
        self._set_fp8_buffer("weight_scale_inv", weight_scale_inv)
        if int4_row1_weights_enabled() and int4_row1_module_in_scope(module_name):
            int4_weight = _quantize_weight_to_int4pack(weight)
            if int4_weight is not None:
                weight_int4pack, weight_int4_scale_zeros, q_group_size = int4_weight
                self._set_fp8_buffer("weight_int4pack", weight_int4pack)
                self._set_fp8_buffer("weight_int4_scale_zeros", weight_int4_scale_zeros)
                self._int4_row1_group_size = int(q_group_size)
                self._int4_row1_weight_enabled = True
        self._fp8_min_rows = fp8_module_min_rows(module_name)
        self._fp8_module_name = module_name or ""
        if not keep_bf16_weight:
            del self._parameters["weight"]
            self.register_buffer(
                "weight",
                torch.empty(0, device=weight_fp8_t.device, dtype=weight_fp8_t.dtype),
                persistent=False,
            )
        self._fp8_weight_enabled = True
        self._fp8_row1_only = bool(row1_only)

    def _set_fp8_buffer(self, name: str, value: torch.Tensor) -> None:
        if name in self._buffers:
            self._buffers[name] = value
        else:
            self.register_buffer(name, value, persistent=False)

    def _linear(self, x: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        prof = get_profiler()
        kind = _linear_profile_kind(self._fp8_module_name)
        if self._fp8_weight_enabled:
            rows = x.numel() // int(self.weight_fp8_t.size(0))
            row1_mode = "off"
            if self._fp8_module_name.endswith(".down_proj") and self.weight.numel() == 0:
                row1_mode = fp8_down_row1_triton_mode()
            elif self._fp8_module_name.endswith(".gate_up_proj"):
                row1_mode = fp8_gate_up_row1_triton_mode()
            elif ".self_attn." in self._fp8_module_name:
                row1_mode = fp8_attention_row1_triton_mode()
            elif fp8_row1_triton_enabled():
                row1_mode = fp8_row1_triton_mode()
            if (
                self._int4_row1_weight_enabled
                and rows >= 1
                and rows <= int4_row1_max_rows()
                and row1_mode != "off"
                and x.is_cuda
                and x.dtype is torch.bfloat16
            ):
                with prof.cuda_phase(f"linear_int4_{kind}"):
                    return _int4_row1_matvec(
                        x,
                        weight_int4pack=self.weight_int4pack,
                        weight_int4_scale_zeros=self.weight_int4_scale_zeros,
                        q_group_size=self._int4_row1_group_size,
                        bias=bias,
                    )
            if (
                rows == 1
                and row1_mode != "off"
                and x.is_cuda
                and x.dtype in {torch.float16, torch.bfloat16}
            ):
                with prof.cuda_phase(f"linear_fp8_row1_{kind}"):
                    return _fp8_row1_matvec(
                        x,
                        weight_fp8_t=self.weight_fp8_t,
                        weight_scale_inv=self.weight_scale_inv,
                        bias=bias,
                        quantize_x=row1_mode == "quantized",
                    )
            if self._fp8_row1_only:
                with prof.cuda_phase(f"linear_bf16_{kind}"):
                    return F.linear(x, self.weight, bias)
            if self.weight.numel() == 0 or rows >= self._fp8_min_rows:
                if (
                    fp8_small_m_triton_enabled()
                    and rows > 1
                    and rows <= fp8_small_m_triton_max_rows()
                    and x.is_cuda
                    and x.dtype in {torch.float16, torch.bfloat16}
                    and int(self.weight_fp8_t.size(0)) % 16 == 0
                ):
                    with prof.cuda_phase(f"linear_fp8_smallm_{kind}"):
                        return _fp8_small_m_linear(
                            x,
                            weight_fp8_t=self.weight_fp8_t,
                            weight_scale_inv=self.weight_scale_inv,
                            bias=bias,
                        )
                with prof.cuda_phase(f"linear_fp8_{kind}"):
                    return _fp8_linear(
                        x,
                        weight_fp8_t=self.weight_fp8_t,
                        weight_scale_inv=self.weight_scale_inv,
                        bias=bias,
                    )
        with prof.cuda_phase(f"linear_bf16_{kind}"):
            return F.linear(x, self.weight, bias)


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._linear(x, self.bias)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._linear(x, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        if param_data.ndim == 1:
            param_data.copy_(loaded_weight)
            return
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._linear(x, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y


def _fp8_module_in_scope(name: str, scope: str) -> bool:
    if scope in {"all", "transformer"}:
        return True
    if scope in {"lm_head", "head"}:
        return False
    if scope in {"mlp_lm_head", "mlp+lm_head", "mlp_head", "mlp+head"}:
        return ".mlp." in name
    if scope in {"mlp", "ffn"}:
        return ".mlp." in name
    if scope in {"attention", "attn"}:
        return ".self_attn." in name
    if scope == "qkv":
        return name.endswith(".qkv_proj")
    if scope in {"attn_out", "o_proj"}:
        return name.endswith(".o_proj")
    if scope in {"mlp_gate_up", "gate_up"}:
        return name.endswith(".gate_up_proj")
    if scope in {"mlp_down", "down"}:
        return name.endswith(".down_proj")
    raise ValueError(
        "NANOVLLM_FP8_WEIGHT_SCOPE must be one of all, mlp, attention, "
        f"qkv, attn_out, mlp_gate_up, mlp_down, lm_head, mlp_lm_head; got {scope!r}"
    )


def convert_linear_modules_to_fp8(module: nn.Module) -> int:
    scope = fp8_weight_scope()
    attention_shadow = fp8_attention_row1_shadow_enabled()
    attention_full = fp8_attention_full_enabled()
    converted = 0
    for name, child in module.named_modules():
        if isinstance(child, LinearBase):
            is_attention = ".self_attn." in name
            use_attention_shadow = attention_shadow and is_attention and not attention_full
            force_attention_full = attention_full and is_attention
            if not use_attention_shadow and not force_attention_full and not _fp8_module_in_scope(name, scope):
                continue
            keep_bf16_weight = True if use_attention_shadow else fp8_module_keep_bf16_weight(name)
            child.quantize_weight_to_fp8(
                keep_bf16_weight=keep_bf16_weight,
                module_name=name,
                row1_only=use_attention_shadow,
            )
            converted += 1
    return converted
