from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from importlib import metadata
from pathlib import Path

import torch

from fastdms import LLM, SamplingParams
from fastdms.engine.compact_kv import streaming_pack_cache_live_enabled
from fastdms.engine.compact_kv import streaming_pack_fused_rank_triton_enabled
from fastdms.engine.compact_kv import streaming_pack_triton_enabled
from fastdms.engine.model_runner import compact_greedy_fast_loop_enabled
from fastdms.engine.sequence import SequenceStatus
from fastdms.layers.compact_attention import compact_attention_inline_q_rope_enabled
from fastdms.layers.compact_attention import compact_attention_splitk_block_n
from fastdms.layers.compact_attention import compact_attention_splitk_enabled
from fastdms.layers.compact_attention import dms_decode_store_transient_k_enabled
from fastdms.layers.compact_attention import dms_fused_decode_preprocess_enabled
from fastdms.layers.embed_head import (
    fp8_embedding_enabled,
    fp8_embedding_share_lm_head_weight,
    fp8_keep_bf16_embedding_weight,
    fp8_keep_bf16_lm_head_weight,
    fp8_lm_head_argmax_block_n,
    fp8_lm_head_min_rows,
    fp8_lm_head_row1_triton_enabled,
    greedy_fused_argmax_enabled,
    int4_lm_head_rerank_topk,
)
from fastdms.layers.linear import (
    fp8_attention_full_enabled,
    fp8_attention_row1_shadow_enabled,
    fp8_attention_row1_triton_enabled,
    fp8_attention_row1_triton_mode,
    fp8_down_min_rows,
    fp8_down_row1_triton_enabled,
    fp8_down_row1_triton_mode,
    fp8_gate_up_row1_triton_enabled,
    fp8_gate_up_row1_triton_mode,
    fp8_gate_up_min_rows,
    fp8_keep_bf16_down_weights,
    fp8_keep_bf16_attention_weights,
    fp8_keep_bf16_gate_up_weights,
    fp8_keep_bf16_weights,
    fp8_min_rows,
    fp8_row1_matvec_dot_block_k,
    fp8_row1_matvec_dot_block_n,
    fp8_row1_matvec_impl,
    fp8_small_m_triton_block_n,
    fp8_small_m_triton_enabled,
    fp8_small_m_triton_max_rows,
    fp8_weight_scope,
    fp8_weights_enabled,
    int4_row1_group_size,
    int4_row1_inner_k_tiles,
    int4_row1_lm_head_enabled,
    int4_row1_max_rows,
    int4_row1_quant_mode,
    int4_row1_scope,
    int4_row1_weights_enabled,
)
from fastdms.utils.profiler import get_profiler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.bench_vllm_turboquant import _make_prompts


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def repo_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return proc.stdout.strip()


def repo_dirty() -> bool | None:
    try:
        unstaged = subprocess.run(["git", "diff", "--quiet"], cwd=REPO_ROOT)
        staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO_ROOT)
    except Exception:
        return None
    return unstaged.returncode != 0 or staged.returncode != 0


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def resolve_torch_dtype(value) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        dtype = getattr(torch, value.removeprefix("torch."), None)
        if isinstance(dtype, torch.dtype):
            return dtype
    raise ValueError(f"cannot resolve torch dtype from {value!r}")


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


def compact_prefill_summary(stats: list[dict]) -> dict:
    if not stats:
        return {
            "live_tokens_min": None,
            "live_tokens_mean": None,
            "live_tokens_max": None,
            "live_tokens_total": None,
            "live_tokens_total_peak": None,
            "eviction_decisions_true": None,
            "eviction_decisions_total": None,
        }
    return {
        "live_tokens_min": min(item["live_tokens_min"] for item in stats),
        "live_tokens_mean": sum(item["live_tokens_mean"] for item in stats) / len(stats),
        "live_tokens_max": max(item["live_tokens_max"] for item in stats),
        "live_tokens_total": sum(item["live_tokens_total"] for item in stats),
        "live_tokens_total_peak": max(item["live_tokens_total"] for item in stats),
        "eviction_decisions_true": sum(
            item.get("eviction_decisions_true", 0) for item in stats
            if item.get("eviction_decisions_true") is not None
        ),
        "eviction_decisions_total": sum(
            item.get("eviction_decisions_total", 0) for item in stats
            if item.get("eviction_decisions_total") is not None
        ),
    }


def dms_decode_summary(stats: list[dict]) -> dict:
    expire_rows = [item for item in stats if item.get("phase") == "expire"]
    record_rows = [item for item in stats if item.get("phase") == "record"]

    def sum_values(rows: list[dict], key: str) -> int:
        total = 0
        for item in rows:
            total = total + item.get(key, 0)
        return int(total.item()) if torch.is_tensor(total) else int(total)

    return {
        "expired_evictions_total": sum_values(expire_rows, "evicted_tokens"),
        "recorded_decisions_true": sum_values(record_rows, "eviction_decisions_true"),
        "recorded_decisions_total": sum_values(record_rows, "eviction_decisions_total"),
        "steps": len(record_rows),
    }


def _config_int(config, name: str, default: int | None = None) -> int:
    value = getattr(config, name, default)
    if value is None:
        raise ValueError(f"model config is missing {name}")
    return int(value)


def roofline_summary(
    *,
    hf_config,
    row: dict,
    kv_element_size: int,
    model_dtype_size: int,
    concurrency: int,
    ctx_len: int,
    phase_summary: dict | None,
) -> dict:
    hidden_size = _config_int(hf_config, "hidden_size")
    intermediate_size = _config_int(hf_config, "intermediate_size")
    num_layers = _config_int(hf_config, "num_hidden_layers")
    num_attention_heads = _config_int(hf_config, "num_attention_heads")
    num_kv_heads = _config_int(hf_config, "num_key_value_heads", num_attention_heads)
    head_dim = _config_int(
        hf_config,
        "head_dim",
        hidden_size // max(1, num_attention_heads),
    )
    vocab_size = _config_int(hf_config, "vocab_size")

    q_dim = num_attention_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    qkv_params = hidden_size * (q_dim + 2 * kv_dim)
    out_params = q_dim * hidden_size
    gate_up_params = 2 * hidden_size * intermediate_size
    down_params = intermediate_size * hidden_size
    layer_params = qkv_params + out_params + gate_up_params + down_params
    transformer_params = layer_params * num_layers
    lm_head_params = hidden_size * vocab_size

    transformer_flops = 2 * transformer_params
    lm_head_flops = 2 * lm_head_params
    decode_flops = transformer_flops + lm_head_flops
    transformer_weight_bytes = transformer_params * model_dtype_size
    decode_weight_bytes = (transformer_params + lm_head_params) * model_dtype_size

    bf16_tflops = 400.0
    fp8_tflops = 800.0
    hbm_tb_s = 1.7
    bf16_flops_s = bf16_tflops * 1e12
    fp8_flops_s = fp8_tflops * 1e12
    hbm_bytes_s = hbm_tb_s * 1e12

    prefill_tok_s = row.get("prefill_tok_s")
    decode_tok_s = row.get("decode_tok_s")
    decode_weight_bytes_per_token = decode_weight_bytes / max(1, concurrency)
    compact_scan_bytes_prefill = None
    compact_scan_bytes_peak = None
    if row.get("compact_live_tokens_prefill_total") is not None:
        compact_scan_bytes_prefill = (
            int(row["compact_live_tokens_prefill_total"]) * 2 * head_dim * kv_element_size
        ) / max(1, concurrency)
    if row.get("compact_live_tokens_peak") is not None:
        compact_scan_bytes_peak = (
            int(row["compact_live_tokens_peak"]) * 2 * head_dim * kv_element_size
        ) / max(1, concurrency)
    compact_scan_bytes_full_ctx = (
        ctx_len * num_layers * num_kv_heads * 2 * head_dim * kv_element_size
    )

    compact_attn_phase_tok_s = None
    compact_attn_hbm_utilization = None
    if phase_summary:
        compact_attn_ms = phase_summary.get("totals_ms", {}).get("compact_decode_attn")
        decode_tokens = row.get("decode_tokens")
        if compact_attn_ms and decode_tokens:
            compact_attn_phase_tok_s = float(decode_tokens) / (float(compact_attn_ms) / 1000.0)
            scan_bytes = compact_scan_bytes_peak or compact_scan_bytes_prefill
            if scan_bytes:
                compact_attn_hbm_utilization = compact_attn_phase_tok_s * scan_bytes / hbm_bytes_s

    return {
        "assumptions": {
            "bf16_fp16_tensor_tflops": bf16_tflops,
            "fp8_tensor_tflops": fp8_tflops,
            "hbm_tb_s": hbm_tb_s,
        },
        "model": {
            "transformer_params": transformer_params,
            "lm_head_params": lm_head_params,
            "transformer_flops_per_token": transformer_flops,
            "lm_head_flops_per_token": lm_head_flops,
            "decode_flops_per_token_with_lm_head": decode_flops,
            "transformer_weight_bytes": transformer_weight_bytes,
            "decode_weight_bytes_with_lm_head": decode_weight_bytes,
        },
        "achieved": {
            "prefill_tflops_s": (
                prefill_tok_s * transformer_flops / 1e12 if prefill_tok_s else None
            ),
            "decode_tflops_s": (
                decode_tok_s * decode_flops / 1e12 if decode_tok_s else None
            ),
            "decode_weight_gb_s": (
                decode_tok_s * decode_weight_bytes_per_token / 1e9 if decode_tok_s else None
            ),
            "compact_attention_phase_tok_s": compact_attn_phase_tok_s,
            "compact_attention_hbm_utilization": compact_attn_hbm_utilization,
        },
        "ceilings": {
            "bf16_prefill_compute_tok_s": bf16_flops_s / transformer_flops,
            "fp8_prefill_compute_tok_s": fp8_flops_s / transformer_flops,
            "bf16_decode_compute_tok_s": bf16_flops_s / decode_flops,
            "fp8_decode_compute_tok_s": fp8_flops_s / decode_flops,
            "decode_weight_bandwidth_tok_s": hbm_bytes_s / decode_weight_bytes_per_token,
            "compact_attention_prefill_live_hbm_tok_s": (
                hbm_bytes_s / compact_scan_bytes_prefill if compact_scan_bytes_prefill else None
            ),
            "compact_attention_peak_live_hbm_tok_s": (
                hbm_bytes_s / compact_scan_bytes_peak if compact_scan_bytes_peak else None
            ),
            "compact_attention_full_ctx_hbm_tok_s": (
                hbm_bytes_s / compact_scan_bytes_full_ctx
            ),
        },
        "utilization": {
            "prefill_bf16_compute": (
                (prefill_tok_s * transformer_flops / bf16_flops_s) if prefill_tok_s else None
            ),
            "decode_bf16_compute": (
                (decode_tok_s * decode_flops / bf16_flops_s) if decode_tok_s else None
            ),
            "decode_weight_bandwidth": (
                (decode_tok_s * decode_weight_bytes_per_token / hbm_bytes_s)
                if decode_tok_s else None
            ),
        },
        "compact_attention_scan_bytes_per_generated_token": {
            "prefill_live": compact_scan_bytes_prefill,
            "peak_live": compact_scan_bytes_peak,
            "full_ctx": compact_scan_bytes_full_ctx,
        },
    }


def _gib(num_bytes: int | float | None) -> float | None:
    return None if num_bytes is None else float(num_bytes) / 2**30


def _tensor_storage_key(tensor: torch.Tensor) -> tuple | None:
    if not torch.is_tensor(tensor) or tensor.numel() == 0:
        return None
    try:
        storage = tensor.untyped_storage()
    except RuntimeError:
        return None
    return (
        str(tensor.device),
        int(storage.data_ptr()),
        int(storage.nbytes()),
    )


def _module_memory_kind(module_name: str) -> str:
    if module_name.endswith(".gate_up_proj"):
        return "mlp_gate_up"
    if module_name.endswith(".down_proj"):
        return "mlp_down"
    if module_name.endswith(".qkv_proj"):
        return "attention_qkv"
    if module_name.endswith(".o_proj"):
        return "attention_o_proj"
    if module_name.endswith(".embed_tokens"):
        return "embedding"
    if module_name.endswith(".lm_head") or module_name == "lm_head":
        return "lm_head"
    return "other"


def memory_breakdown_summary(
    model: torch.nn.Module,
    *,
    kv_dense_bytes: int,
    compact_bytes: int,
    layer_major_metadata_bytes: int,
    torch_allocated_bytes: int,
    torch_reserved_bytes: int,
    cuda_peak_bytes: int,
) -> dict:
    param_roles_by_key: dict[tuple, set[str]] = defaultdict(set)
    param_bytes_by_key: dict[tuple, int] = {}
    fp8_buffer_seen: set[tuple] = set()
    int4_buffer_seen: set[tuple] = set()
    other_buffer_seen: set[tuple] = set()
    fp8_weight_bytes_by_kind: dict[str, int] = defaultdict(int)
    fp8_scale_bytes_by_kind: dict[str, int] = defaultdict(int)
    int4_weight_bytes_by_kind: dict[str, int] = defaultdict(int)
    int4_scale_bytes_by_kind: dict[str, int] = defaultdict(int)
    other_buffer_bytes = 0

    for module_name, module in model.named_modules():
        kind = _module_memory_kind(module_name)
        fp8_enabled = bool(getattr(module, "_fp8_weight_enabled", False))
        for param_name, param in module.named_parameters(recurse=False):
            key = _tensor_storage_key(param)
            if key is None:
                continue
            param_bytes_by_key.setdefault(key, key[2])
            if module_name.endswith(".embed_tokens") and param_name == "weight":
                param_roles_by_key[key].add("embedding")
            elif fp8_enabled and param_name == "weight":
                param_roles_by_key[key].add(f"fp8_bf16_backing:{kind}")
            else:
                param_roles_by_key[key].add(f"other_param:{kind}")

        for buffer_name, buffer in module.named_buffers(recurse=False):
            key = _tensor_storage_key(buffer)
            if key is None:
                continue
            if fp8_enabled and buffer_name in {"weight_fp8_t", "weight_scale_inv"}:
                if key in fp8_buffer_seen:
                    continue
                fp8_buffer_seen.add(key)
                if buffer_name == "weight_fp8_t":
                    fp8_weight_bytes_by_kind[kind] += key[2]
                else:
                    fp8_scale_bytes_by_kind[kind] += key[2]
                continue
            if fp8_enabled and buffer_name in {"weight_int4pack", "weight_int4_scale_zeros"}:
                if key in int4_buffer_seen:
                    continue
                int4_buffer_seen.add(key)
                if buffer_name == "weight_int4pack":
                    int4_weight_bytes_by_kind[kind] += key[2]
                else:
                    int4_scale_bytes_by_kind[kind] += key[2]
                continue
            if key not in other_buffer_seen:
                other_buffer_seen.add(key)
                other_buffer_bytes += key[2]

    for module_name, module in model.named_modules():
        if not bool(getattr(module, "_fp8_embedding_enabled", False)):
            continue
        kind = _module_memory_kind(module_name)
        for attr_name, target in (
            ("_fp8_embedding_weight", fp8_weight_bytes_by_kind),
            ("_fp8_embedding_scale_inv", fp8_scale_bytes_by_kind),
        ):
            tensor = getattr(module, attr_name, None)
            key = _tensor_storage_key(tensor) if torch.is_tensor(tensor) else None
            if key is None or key in fp8_buffer_seen:
                continue
            fp8_buffer_seen.add(key)
            target[kind] += key[2]

    embedding_or_shared_bytes = 0
    retained_bf16_backing_bytes = 0
    retained_bf16_backing_shared_with_embedding_bytes = 0
    retained_bf16_backing_by_kind: dict[str, int] = defaultdict(int)
    other_param_bytes = 0
    for key, roles in param_roles_by_key.items():
        storage_bytes = param_bytes_by_key[key]
        backing_roles = sorted(role for role in roles if role.startswith("fp8_bf16_backing:"))
        if "embedding" in roles:
            embedding_or_shared_bytes += storage_bytes
            if backing_roles:
                retained_bf16_backing_shared_with_embedding_bytes += storage_bytes
            continue
        if backing_roles:
            retained_bf16_backing_bytes += storage_bytes
            kind = backing_roles[0].split(":", 1)[1]
            retained_bf16_backing_by_kind[kind] += storage_bytes
            continue
        other_param_bytes += storage_bytes

    fp8_weight_bytes = sum(fp8_weight_bytes_by_kind.values())
    fp8_scale_bytes = sum(fp8_scale_bytes_by_kind.values())
    fp8_serving_bytes = fp8_weight_bytes + fp8_scale_bytes
    int4_weight_bytes = sum(int4_weight_bytes_by_kind.values())
    int4_scale_bytes = sum(int4_scale_bytes_by_kind.values())
    int4_serving_bytes = int4_weight_bytes + int4_scale_bytes
    model_parameter_bytes = (
        embedding_or_shared_bytes
        + retained_bf16_backing_bytes
        + other_param_bytes
    )
    known_current_bytes = (
        model_parameter_bytes
        + fp8_serving_bytes
        + int4_serving_bytes
        + other_buffer_bytes
        + kv_dense_bytes
        + compact_bytes
        + layer_major_metadata_bytes
    )

    def by_kind_gib(values: dict[str, int]) -> dict[str, float]:
        return {key: _gib(value) for key, value in sorted(values.items())}

    return {
        "model_unique_parameter_gib": _gib(model_parameter_bytes),
        "embedding_or_shared_weight_gib": _gib(embedding_or_shared_bytes),
        "retained_bf16_backing_gib": _gib(retained_bf16_backing_bytes),
        "retained_bf16_backing_by_kind_gib": by_kind_gib(retained_bf16_backing_by_kind),
        "retained_bf16_backing_shared_with_embedding_gib": _gib(
            retained_bf16_backing_shared_with_embedding_bytes
        ),
        "other_model_parameter_gib": _gib(other_param_bytes),
        "fp8_serving_gib": _gib(fp8_serving_bytes),
        "fp8_serving_weight_gib": _gib(fp8_weight_bytes),
        "fp8_serving_scale_gib": _gib(fp8_scale_bytes),
        "fp8_serving_weight_by_kind_gib": by_kind_gib(fp8_weight_bytes_by_kind),
        "fp8_serving_scale_by_kind_gib": by_kind_gib(fp8_scale_bytes_by_kind),
        "int4_row1_serving_gib": _gib(int4_serving_bytes),
        "int4_row1_serving_weight_gib": _gib(int4_weight_bytes),
        "int4_row1_serving_scale_gib": _gib(int4_scale_bytes),
        "int4_row1_serving_weight_by_kind_gib": by_kind_gib(int4_weight_bytes_by_kind),
        "int4_row1_serving_scale_by_kind_gib": by_kind_gib(int4_scale_bytes_by_kind),
        "other_model_buffer_gib": _gib(other_buffer_bytes),
        "kv_dense_gib": _gib(kv_dense_bytes),
        "kv_compact_allocated_gib": _gib(compact_bytes),
        "layer_major_metadata_gib": _gib(layer_major_metadata_bytes),
        "known_current_allocation_gib": _gib(known_current_bytes),
        "torch_allocated_gib": _gib(torch_allocated_bytes),
        "torch_allocated_unattributed_gib": _gib(
            max(0, torch_allocated_bytes - known_current_bytes)
        ),
        "cuda_peak_gib": _gib(cuda_peak_bytes),
        "cuda_peak_minus_current_gib": _gib(max(0, cuda_peak_bytes - torch_allocated_bytes)),
        "torch_reserved_gib": _gib(torch_reserved_bytes),
        "torch_reserved_minus_allocated_gib": _gib(
            max(0, torch_reserved_bytes - torch_allocated_bytes)
        ),
        "notes": [
            "retained_bf16_backing_gib excludes tied embedding storage; tied LM-head backing is reported as shared_with_embedding unless an FP8 embedding strategy removes it.",
            "torch_allocated_unattributed_gib includes CUDA graph pools, live runtime tensors, allocator bookkeeping, and any buffers not classified above.",
        ],
    }


def json_safe_stats(stats: list[dict]) -> list[dict]:
    result = []
    for item in stats:
        converted = {}
        for key, value in item.items():
            converted[key] = int(value.item()) if torch.is_tensor(value) else value
        result.append(converted)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark nano-vLLM Llama on WikiText-2.")
    parser.add_argument(
        "--model",
        default="results/dms/llama32-1b-cr8-v5-correctmask/final",
        help="Model/checkpoint path.",
    )
    parser.add_argument("--ctx-len", type=int, default=8192)
    parser.add_argument("--gen-len", type=int, default=128)
    parser.add_argument("--concurrency", type=int, choices=(1, 8), default=1)
    parser.add_argument("--prompt-source", choices=("wikitext2", "wikitext2_wrap", "synthetic"), default="wikitext2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0, help="Physical GPU id for artifact metadata.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.16)
    parser.add_argument(
        "--kv-cache-dtype",
        choices=("auto", "float16", "bfloat16", "float32", "float8_e4m3fn", "float8_e5m2"),
        default="auto",
    )
    parser.add_argument("--compact-kv", action="store_true")
    parser.add_argument("--compact-kv-capacity-tokens", type=int, default=0)
    parser.add_argument("--compact-retention-mode", choices=("all", "dms_stride", "dms"), default="all")
    parser.add_argument("--compact-retention-stride", type=int, default=8)
    parser.add_argument("--compact-retention-recent-tokens", type=int, default=256)
    parser.add_argument("--dms-metadata-path")
    parser.add_argument("--dms-window-size", type=int)
    parser.add_argument("--dms-alpha-scale", type=float)
    parser.add_argument("--dms-alpha-offset", type=float)
    parser.add_argument("--num-page-override", type=int, default=None)
    parser.add_argument("--token-pool-tokens", type=int, default=None)
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help=(
            "Override the benchmark's default ctx_len * concurrency prefill batch. "
            "Useful for max-context rows where the scheduler should chunk prefill."
        ),
    )
    parser.add_argument("--compact-kv-budget-bytes", type=int, default=None)
    parser.add_argument("--compact-kv-target-live-tokens-per-seq", type=int, default=None)
    parser.add_argument("--compact-kv-layer-major-metadata", action="store_true")
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help=(
            "Run this many full request batches before measuring prefill/decode. "
            "Startup and warmup time are reported separately."
        ),
    )
    parser.add_argument("--phase-profile", action="store_true")
    parser.add_argument("--phase-profile-output", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_request_batch(llm: LLM, prompts: list[dict], sampling_params: SamplingParams) -> dict:
    profiler = get_profiler()
    for prompt in prompts:
        llm.add_request(prompt["prompt_token_ids"], sampling_params)

    outputs: dict[int, list[int]] = {}
    prefill_elapsed_s = 0.0
    prefill_input_tokens = 0
    decode_elapsed_s = 0.0
    decode_tokens = 0
    ttft_ms: dict[int, float] = {}
    itl_ms: list[float] = []

    started = time.perf_counter()
    while not llm.is_finished():
        with profiler.wall_phase("scheduler"):
            seqs, is_prefill = llm.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else len(seqs)
        step_started = time.perf_counter()
        token_ids = llm.model_runner.call("run", seqs, is_prefill)
        step_elapsed_s = time.perf_counter() - step_started
        with profiler.wall_phase("postprocess"):
            llm.scheduler.postprocess(seqs, token_ids, is_prefill)
        if is_prefill:
            prefill_elapsed_s += step_elapsed_s
            prefill_input_tokens += num_tokens
            for seq in seqs:
                ttft_ms.setdefault(seq.seq_id, (time.perf_counter() - started) * 1000.0)
        else:
            decode_elapsed_s += step_elapsed_s
            decode_tokens += num_tokens
            itl_ms.extend([step_elapsed_s * 1000.0] * len(seqs))
        if (
            is_prefill
            and compact_greedy_fast_loop_enabled()
            and len(seqs) == 1
            and sampling_params.temperature <= 1e-10
            and sampling_params.ignore_eos
            and llm.model_runner.config.tensor_parallel_size == 1
            and llm.model_runner.config.compact_kv_enabled
            and llm.model_runner.config.compact_kv_retention_mode == "dms"
            and llm.model_runner.compact_decode_cudagraph
            and greedy_fused_argmax_enabled()
        ):
            seq = seqs[0]
            remaining_decode_tokens = seq.max_tokens - seq.num_completion_tokens
            if remaining_decode_tokens > 0 and seq.status == SequenceStatus.RUNNING:
                fast_started = time.perf_counter()
                fast_tokens = llm.model_runner.call(
                    "run_compact_greedy_decode_loop",
                    [seq],
                    remaining_decode_tokens,
                )
                fast_elapsed_s = time.perf_counter() - fast_started
                decode_elapsed_s += fast_elapsed_s
                decode_tokens += len(fast_tokens)
                if fast_tokens:
                    itl_ms.extend([fast_elapsed_s * 1000.0 / len(fast_tokens)] * len(fast_tokens))
                seq.num_scheduled_tokens = 0
                if seq.is_finished:
                    llm.scheduler.freed_seq_ids.append(seq.seq_id)
                    try:
                        llm.scheduler.running.remove(seq)
                    except ValueError:
                        pass
        for seq_id in llm.scheduler.consume_freed_seq_ids():
            if llm.model_runner.config.compact_kv_enabled:
                llm.model_runner.call("free_compact", [seq_id])
        for seq in seqs:
            if seq.is_finished:
                outputs[seq.seq_id] = seq.completion_token_ids

    elapsed_s = time.perf_counter() - started
    return {
        "outputs": outputs,
        "prefill_elapsed_s": prefill_elapsed_s,
        "prefill_input_tokens": prefill_input_tokens,
        "decode_elapsed_s": decode_elapsed_s,
        "decode_tokens": decode_tokens,
        "ttft_ms": ttft_ms,
        "itl_ms": itl_ms,
        "elapsed_s": elapsed_s,
    }


def summarize_request_batch(result: dict) -> dict:
    return {
        "prefill_elapsed_s": result["prefill_elapsed_s"],
        "prefill_input_tokens": result["prefill_input_tokens"],
        "prefill_tok_s": (
            result["prefill_input_tokens"] / result["prefill_elapsed_s"]
            if result["prefill_elapsed_s"] else None
        ),
        "decode_elapsed_s": result["decode_elapsed_s"],
        "decode_tokens": result["decode_tokens"],
        "decode_tok_s": (
            result["decode_tokens"] / result["decode_elapsed_s"] if result["decode_elapsed_s"] else None
        ),
        "elapsed_s": result["elapsed_s"],
    }


def run_benchmark(args: argparse.Namespace) -> dict:
    if args.compact_retention_mode != "all" and not args.compact_kv:
        raise ValueError("--compact-retention-mode requires --compact-kv")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be non-negative")
    prompt_started = time.perf_counter()
    prompts = _make_prompts(args.model, args.concurrency, args.ctx_len, args.seed, args.prompt_source)
    prompt_prep_elapsed_s = time.perf_counter() - prompt_started
    max_num_batched_tokens = args.max_num_batched_tokens or (args.ctx_len * args.concurrency)
    if max_num_batched_tokens <= 0:
        raise ValueError("--max-num-batched-tokens must be positive")
    startup_started = time.perf_counter()
    llm = LLM(
        args.model,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=args.ctx_len + args.gen_len,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=args.concurrency,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_dtype=args.kv_cache_dtype,
        compact_kv_enabled=args.compact_kv,
        compact_kv_capacity_tokens=args.compact_kv_capacity_tokens,
        compact_kv_retention_mode=args.compact_retention_mode,
        compact_kv_retention_stride=args.compact_retention_stride,
        compact_kv_retention_recent_tokens=args.compact_retention_recent_tokens,
        dms_metadata_path=args.dms_metadata_path,
        dms_window_size=args.dms_window_size,
        dms_alpha_scale=args.dms_alpha_scale,
        dms_alpha_offset=args.dms_alpha_offset,
        num_page_override=args.num_page_override,
        token_pool_tokens=args.token_pool_tokens,
        compact_kv_budget_bytes=args.compact_kv_budget_bytes,
        compact_kv_target_live_tokens_per_seq=args.compact_kv_target_live_tokens_per_seq,
        compact_kv_layer_major_metadata=args.compact_kv_layer_major_metadata,
    )
    cuda_sync()
    startup_elapsed_s = time.perf_counter() - startup_started
    profiler = get_profiler()
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.gen_len, ignore_eos=True)

    warmup_summaries = []
    warmup_elapsed_s = 0.0
    profiler.disable()
    for warmup_idx in range(args.warmup_runs):
        warmup_started = time.perf_counter()
        warmup_result = run_request_batch(llm, prompts, sampling_params)
        cuda_sync()
        warmup_wall_s = time.perf_counter() - warmup_started
        warmup_summary = summarize_request_batch(warmup_result)
        warmup_summary.update({
            "index": warmup_idx,
            "wall_elapsed_s": warmup_wall_s,
        })
        warmup_summaries.append(warmup_summary)
        warmup_elapsed_s += warmup_wall_s
        # This benchmark reports prefix_sharing=False; warmup is for kernels,
        # graphs, and allocator state, not for reusing prompt blocks during the
        # measured pass.
        llm.scheduler.block_manager.clear_prefix_cache()

    if llm.model_runner.config.compact_kv_enabled:
        llm.model_runner.last_compact_prefill_stats = []
        llm.model_runner.last_dms_decode_stats = []
        llm.model_runner.reset_compact_live_tokens_peak()
    torch.cuda.reset_peak_memory_stats()
    if args.phase_profile:
        profiler.reset()
        profiler.enable()

    measured_result = run_request_batch(llm, prompts, sampling_params)
    cuda_sync()
    phase_summary = profiler.summarize() if args.phase_profile else None
    if args.phase_profile:
        profiler.disable()

    outputs = measured_result["outputs"]
    prefill_elapsed_s = measured_result["prefill_elapsed_s"]
    prefill_input_tokens = measured_result["prefill_input_tokens"]
    decode_elapsed_s = measured_result["decode_elapsed_s"]
    decode_tokens = measured_result["decode_tokens"]
    ttft_ms = measured_result["ttft_ms"]
    itl_ms = measured_result["itl_ms"]
    elapsed_s = measured_result["elapsed_s"]
    generated_tokens = sum(len(tokens) for tokens in outputs.values())
    kv_total_bytes = llm.model_runner.kv_cache.numel() * llm.model_runner.kv_cache.element_size()
    allocated_tokens = llm.model_runner.config.num_kvcache_blocks * llm.model_runner.config.kvcache_block_size
    kv_total_gib = kv_total_bytes / 2**30
    allocated_bytes_per_token = kv_total_bytes / allocated_tokens if allocated_tokens else None
    hf_config = llm.model_runner.config.hf_config
    actual_kv_cache_dtype = dtype_name(llm.model_runner.kv_cache.dtype)
    compact_enabled = llm.model_runner.config.compact_kv_enabled
    compact_mode = llm.model_runner.config.compact_kv_retention_mode if compact_enabled else None
    compact_stats = llm.model_runner.last_compact_prefill_stats if compact_enabled else []
    compact_summary = compact_prefill_summary(compact_stats)
    dms_decode_stats = llm.model_runner.last_dms_decode_stats if compact_mode == "dms" else []
    decode_summary = dms_decode_summary(dms_decode_stats)
    dms_decode_stats_json = json_safe_stats(dms_decode_stats)
    dms_metadata = llm.model_runner.dms_metadata
    compact_bytes = 0
    if compact_enabled and llm.model_runner.compact_kv_cache is not None:
        compact_bytes = llm.model_runner.compact_kv_cache.numel() * llm.model_runner.compact_kv_cache.element_size()
    # J5: per-layer compact storage (when streaming pack mode is active) is
    # held in compact_kv_manager.compact_k_per_layer / compact_v_per_layer
    # rather than the legacy compact_kv_cache tensor. Sum those if present.
    if compact_enabled and llm.model_runner.compact_kv_manager is not None:
        for tens in (llm.model_runner.compact_kv_manager.compact_k_per_layer or []):
            if tens is not None:
                compact_bytes += tens.numel() * tens.element_size()
        for tens in (llm.model_runner.compact_kv_manager.compact_v_per_layer or []):
            if tens is not None:
                compact_bytes += tens.numel() * tens.element_size()
    staging_bytes = 0
    model_dtype = resolve_torch_dtype(hf_config.dtype)
    if llm.model_runner.kv_cache.numel() and llm.model_runner.kv_cache.dtype != model_dtype:
        staging_bytes = llm.model_runner.kv_cache.numel() * model_dtype.itemsize
    kv_staging_gib = staging_bytes / 2**30 if staging_bytes else None
    compact_gib = compact_bytes / 2**30 if compact_bytes else None
    layer_major_metadata_bytes = 0
    if compact_enabled and llm.model_runner.compact_kv_manager is not None:
        manager = llm.model_runner.compact_kv_manager
        for tens in (
            manager.layer_major_base_offsets,
            manager.layer_major_range_capacity,
            manager.layer_major_live_counts,
            manager.layer_major_token_positions,
            manager.layer_major_evict_mask,
        ):
            if tens is not None:
                layer_major_metadata_bytes += tens.numel() * tens.element_size()
    layer_major_metadata_gib = (
        layer_major_metadata_bytes / 2**30 if layer_major_metadata_bytes else None
    )
    effective_total_gib = kv_total_gib + (compact_gib or 0.0) + (kv_staging_gib or 0.0)
    compact_live_tokens_peak = (
        llm.model_runner.finalize_compact_live_tokens_peak() if compact_enabled else None
    )
    compact_live_bytes_peak = (
        compact_live_tokens_peak * 2 * getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        * llm.model_runner.kv_cache.element_size()
        if compact_live_tokens_peak is not None else None
    )
    compact_prefill_live_bytes = (
        compact_summary["live_tokens_total"] * 2
        * getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        * llm.model_runner.kv_cache.element_size()
        if compact_summary["live_tokens_total"] is not None else None
    )
    torch_allocated_bytes = torch.cuda.memory_allocated()
    torch_reserved_bytes = torch.cuda.memory_reserved()
    cuda_peak_bytes = torch.cuda.max_memory_allocated()
    memory_breakdown = memory_breakdown_summary(
        llm.model_runner.model,
        kv_dense_bytes=kv_total_bytes,
        compact_bytes=compact_bytes,
        layer_major_metadata_bytes=layer_major_metadata_bytes,
        torch_allocated_bytes=torch_allocated_bytes,
        torch_reserved_bytes=torch_reserved_bytes,
        cuda_peak_bytes=cuda_peak_bytes,
    )
    row = {
        "name": (
            f"fastdms compact {compact_mode} {actual_kv_cache_dtype}"
            if compact_enabled else f"fastdms dense {actual_kv_cache_dtype}"
        ),
        "status": "diagnostic",
        "mode": f"compact_{compact_mode}_{actual_kv_cache_dtype}" if compact_enabled else f"dense_{actual_kv_cache_dtype}",
        "model_type": hf_config.model_type,
        "torch_dtype": dtype_name(hf_config.dtype),
        "kv_cache_dtype": actual_kv_cache_dtype,
        "compact_kv_enabled": compact_enabled,
        "compact_retention_mode": compact_mode,
        "dms_source": dms_metadata.source_kind if dms_metadata is not None else None,
        "dms_checkpoint_metadata_found": (
            dms_metadata.packaged_metadata_found
            if dms_metadata is not None else (Path(args.model) / "dms_metadata.json").exists()
        ),
        "submitted_requests": args.concurrency,
        "completed_requests": len(outputs),
        "generated_tokens": generated_tokens,
        "prefill_input_tokens": prefill_input_tokens,
        "prefill_elapsed_s": prefill_elapsed_s,
        "prefill_tok_s": prefill_input_tokens / prefill_elapsed_s if prefill_elapsed_s else None,
        "decode_elapsed_s": decode_elapsed_s,
        "decode_tokens": decode_tokens,
        "decode_tok_s": decode_tokens / decode_elapsed_s if decode_elapsed_s else None,
        "aggregate_tok_s": generated_tokens / elapsed_s if elapsed_s else None,
        "elapsed_s": elapsed_s,
        "measurement_elapsed_s": elapsed_s,
        "prompt_prep_elapsed_s": prompt_prep_elapsed_s,
        "startup_elapsed_s": startup_elapsed_s,
        "warmup_runs": args.warmup_runs,
        "warmup_elapsed_s": warmup_elapsed_s,
        "startup_plus_warmup_elapsed_s": startup_elapsed_s + warmup_elapsed_s,
        "total_wall_elapsed_s": prompt_prep_elapsed_s + startup_elapsed_s + warmup_elapsed_s + elapsed_s,
        "ttft_ms_p50": percentile(list(ttft_ms.values()), 50),
        "ttft_ms_p90": percentile(list(ttft_ms.values()), 90),
        "ttft_ms_p99": percentile(list(ttft_ms.values()), 99),
        "ttft_ms_max": max(ttft_ms.values()) if ttft_ms else None,
        "itl_ms_p50": percentile(itl_ms, 50),
        "itl_ms_p90": percentile(itl_ms, 90),
        "itl_ms_p99": percentile(itl_ms, 99),
        "itl_ms_max": max(itl_ms) if itl_ms else None,
        "kv_total_gib": kv_total_gib,
        "kv_dense_gib": kv_total_gib,
        "dense_kv_released": False,
        "dense_scratch_resident": compact_enabled,
        "kv_dense_scratch_gib": kv_total_gib if compact_enabled else None,
        "kv_compact_gib": compact_gib,
        "kv_compact_allocated_gib": compact_gib,
        "kv_compact_live_gib_peak": compact_live_bytes_peak / 2**30 if compact_live_bytes_peak is not None else None,
        "kv_compact_live_gib_prefill": (
            compact_prefill_live_bytes / 2**30 if compact_prefill_live_bytes is not None else None
        ),
        "kv_staging_gib": kv_staging_gib,
        "allocated_bytes_per_token": allocated_bytes_per_token,
        "effective_total_gib": effective_total_gib,
        "allocator_visible_total_gib": effective_total_gib,
        "layer_major_metadata_gib": layer_major_metadata_gib,
        "allocator_visible_total_with_metadata_gib": (
            effective_total_gib + (layer_major_metadata_gib or 0.0)
        ),
        "effective_bytes_per_live_token": (
            None if compact_enabled else (kv_total_bytes + staging_bytes) / allocated_tokens
        ) if allocated_tokens else None,
        "torch_allocated_gib": torch_allocated_bytes / 2**30,
        "torch_reserved_gib": torch_reserved_bytes / 2**30,
        "cuda_peak_gib": cuda_peak_bytes / 2**30,
        "memory_breakdown": memory_breakdown,
        "num_dense_kv_blocks": llm.model_runner.config.num_kvcache_blocks,
        "compact_capacity_tokens": (
            llm.model_runner.compact_kv_manager.capacity_tokens
            if compact_enabled and llm.model_runner.compact_kv_manager is not None else None
        ),
        "compact_live_tokens_prefill_total": compact_summary["live_tokens_total"],
        "compact_live_tokens_peak": compact_live_tokens_peak,
        "dms_layer_head_live_tokens_min": compact_summary["live_tokens_min"] if compact_mode == "dms" else None,
        "dms_layer_head_live_tokens_mean": compact_summary["live_tokens_mean"] if compact_mode == "dms" else None,
        "dms_layer_head_live_tokens_max": compact_summary["live_tokens_max"] if compact_mode == "dms" else None,
        "dms_prefill_eviction_decisions_true": compact_summary["eviction_decisions_true"] if compact_mode == "dms" else None,
        "dms_prefill_eviction_decisions_total": compact_summary["eviction_decisions_total"] if compact_mode == "dms" else None,
        "dms_decode_expired_evictions_total": decode_summary["expired_evictions_total"] if compact_mode == "dms" else None,
        "dms_decode_eviction_decisions_true": decode_summary["recorded_decisions_true"] if compact_mode == "dms" else None,
        "dms_decode_eviction_decisions_total": decode_summary["recorded_decisions_total"] if compact_mode == "dms" else None,
        "dms_decode_decision_steps": decode_summary["steps"] if compact_mode == "dms" else None,
        "dms_physical_admission_peak_reserved_tokens": None,
        "dms_physical_admission_capacity_tokens": None,
        "dms_physical_admission_denied_headroom": None,
        "dms_memory_claim_status": (
            "compact_dms_prefill_decode_apply_dense_scratch_retained"
            if compact_mode == "dms" else (
                f"compact_{compact_mode}_no_dms_dense_scratch_retained"
                if compact_enabled else f"dense_{actual_kv_cache_dtype}_no_dms"
            )
        ),
        "quality_guard": None,
        "correctness_guard": None,
        "sample": {
            "first_completion_token_ids": next((outputs[key] for key in sorted(outputs)), [])[:16],
            "first_completion_token_ids_full": next((outputs[key] for key in sorted(outputs)), []),
        },
        "compact_prefill_stats": compact_stats,
        "dms_decode_stats": dms_decode_stats_json,
        "phase_profile": phase_summary,
        "warmup": warmup_summaries,
        "notes": [
            "Row is diagnostic until the matching correctness/quality guard passes.",
            "Compact DMS reports allocated dense scratch, allocated compact arena, and logical live compact bytes separately; promotion gates are separate PLAN14 items.",
        ],
    }
    row["roofline"] = roofline_summary(
        hf_config=hf_config,
        row=row,
        kv_element_size=llm.model_runner.kv_cache.element_size(),
        model_dtype_size=model_dtype.itemsize,
        concurrency=args.concurrency,
        ctx_len=args.ctx_len,
        phase_summary=phase_summary,
    )
    return {
        "schema_version": 2,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "engine": "fastdms",
        "repo_commit": repo_commit(),
        "repo_dirty": repo_dirty(),
        "gpu": args.gpu,
        "runtime": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": package_version("transformers"),
            "flash_attn": package_version("flash-attn"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "env": {
            "tensor_parallel_size": llm.model_runner.config.tensor_parallel_size,
            "enforce_eager": llm.model_runner.config.enforce_eager,
            "max_num_batched_tokens": llm.model_runner.config.max_num_batched_tokens,
            "max_num_seqs": llm.model_runner.config.max_num_seqs,
            "gpu_memory_utilization": llm.model_runner.config.gpu_memory_utilization,
            "kvcache_block_size": llm.model_runner.config.kvcache_block_size,
            "kv_cache_dtype_config": llm.model_runner.config.kv_cache_dtype,
            "compact_kv_enabled": llm.model_runner.config.compact_kv_enabled,
            "compact_retention_mode": compact_mode,
            "compact_kv_capacity_tokens": llm.model_runner.config.compact_kv_capacity_tokens,
            "compact_kv_layer_major_metadata": llm.model_runner.config.compact_kv_layer_major_metadata,
            "streaming_pack_cache_live": streaming_pack_cache_live_enabled(),
            "streaming_pack_fused_rank_triton": streaming_pack_fused_rank_triton_enabled(),
            "streaming_pack_triton": streaming_pack_triton_enabled(),
            "compact_decode_cudagraph": llm.model_runner.compact_decode_cudagraph,
            "compact_greedy_fast_loop": compact_greedy_fast_loop_enabled(),
            "compact_attention_splitk": compact_attention_splitk_enabled(),
            "compact_attention_splitk_block_n": compact_attention_splitk_block_n(),
            "compact_attention_inline_q_rope": compact_attention_inline_q_rope_enabled(),
            "compact_attention_inline_q_rope_batch1_only": True,
            "dms_fused_decode_preprocess": dms_fused_decode_preprocess_enabled(),
            "dms_decode_store_transient_k": dms_decode_store_transient_k_enabled(),
            "fp8_weights": fp8_weights_enabled(),
            "fp8_weight_scope": fp8_weight_scope(),
            "fp8_keep_bf16_weights": fp8_keep_bf16_weights(),
            "fp8_keep_bf16_gate_up": fp8_keep_bf16_gate_up_weights(),
            "fp8_keep_bf16_down": fp8_keep_bf16_down_weights(),
            "fp8_keep_bf16_attention": fp8_keep_bf16_attention_weights(),
            "fp8_min_rows": fp8_min_rows(),
            "fp8_gate_up_min_rows": fp8_gate_up_min_rows(),
            "fp8_down_min_rows": fp8_down_min_rows(),
            "fp8_down_row1_triton": fp8_down_row1_triton_enabled(),
            "fp8_down_row1_triton_mode": fp8_down_row1_triton_mode(),
            "fp8_gate_up_row1_triton": fp8_gate_up_row1_triton_enabled(),
            "fp8_gate_up_row1_triton_mode": fp8_gate_up_row1_triton_mode(),
            "fp8_attention_full": fp8_attention_full_enabled(),
            "fp8_attention_row1_shadow": fp8_attention_row1_shadow_enabled(),
            "fp8_attention_row1_triton": fp8_attention_row1_triton_enabled(),
            "fp8_attention_row1_triton_mode": fp8_attention_row1_triton_mode(),
            "fp8_row1_matvec_impl": fp8_row1_matvec_impl(),
            "fp8_row1_matvec_dot_block_k": fp8_row1_matvec_dot_block_k(),
            "fp8_row1_matvec_dot_block_n": fp8_row1_matvec_dot_block_n(),
            "fp8_small_m_triton": fp8_small_m_triton_enabled(),
            "fp8_small_m_triton_max_rows": fp8_small_m_triton_max_rows(),
            "fp8_small_m_triton_block_n": fp8_small_m_triton_block_n(),
            "int4_row1_weights": int4_row1_weights_enabled(),
            "int4_row1_scope": int4_row1_scope(),
            "int4_row1_lm_head": int4_row1_lm_head_enabled(),
            "int4_row1_group_size": int4_row1_group_size(),
            "int4_row1_inner_k_tiles": int4_row1_inner_k_tiles(),
            "int4_row1_max_rows": int4_row1_max_rows(),
            "int4_row1_quant_mode": int4_row1_quant_mode(),
            "int4_row1_lm_head_rerank_topk": int4_lm_head_rerank_topk(),
            "fp8_weight_modules": getattr(llm.model_runner, "fp8_weight_modules", 0),
            "fp8_embedding": getattr(llm.model_runner, "fp8_embedding", False),
            "fp8_embedding_env": fp8_embedding_enabled(),
            "fp8_embedding_share_lm_head": fp8_embedding_share_lm_head_weight(),
            "fp8_keep_bf16_embedding": fp8_keep_bf16_embedding_weight(),
            "fp8_lm_head": getattr(llm.model_runner, "fp8_lm_head", False),
            "fp8_keep_bf16_lm_head": fp8_keep_bf16_lm_head_weight(),
            "fp8_lm_head_min_rows": fp8_lm_head_min_rows(),
            "fp8_lm_head_row1_triton": fp8_lm_head_row1_triton_enabled(),
            "fp8_lm_head_argmax_block_n": fp8_lm_head_argmax_block_n(),
            "greedy_fused_argmax": greedy_fused_argmax_enabled(),
        },
        "model": {
            "path": args.model,
            "class": type(llm.model_runner.model).__name__,
            "model_type": hf_config.model_type,
            "num_hidden_layers": hf_config.num_hidden_layers,
            "num_attention_heads": hf_config.num_attention_heads,
            "num_key_value_heads": hf_config.num_key_value_heads,
            "hidden_size": hf_config.hidden_size,
            "head_dim": getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads),
            "max_position_embeddings": hf_config.max_position_embeddings,
        },
        "checkpoint_semantics": {
            "label": "corrected-mask-v5" if "cr8-v5-correctmask" in args.model else "unknown",
            "dms_metadata_path": dms_metadata.source_path if dms_metadata is not None else str(Path(args.model) / "dms_metadata.json"),
            "dms_metadata_source": dms_metadata.source_kind if dms_metadata is not None else None,
            "dms_checkpoint_metadata_found": (
                dms_metadata.packaged_metadata_found
                if dms_metadata is not None else (Path(args.model) / "dms_metadata.json").exists()
            ),
            "dms_window_size": dms_metadata.window_size if dms_metadata is not None else None,
            "dms_alpha_scale": dms_metadata.alpha_scale if dms_metadata is not None else None,
            "dms_alpha_offset": dms_metadata.alpha_offset if dms_metadata is not None else None,
        },
        "ctx_len": args.ctx_len,
        "gen_len": args.gen_len,
        "warmup_runs": args.warmup_runs,
        "prompt_prep_elapsed_s": prompt_prep_elapsed_s,
        "startup_elapsed_s": startup_elapsed_s,
        "warmup_elapsed_s": warmup_elapsed_s,
        "measurement_elapsed_s": elapsed_s,
        "total_wall_elapsed_s": prompt_prep_elapsed_s + startup_elapsed_s + warmup_elapsed_s + elapsed_s,
        "prompt_source": args.prompt_source,
        "concurrency": args.concurrency,
        "token_pool": (
            llm.model_runner.config.num_kvcache_blocks * llm.model_runner.config.kvcache_block_size
        ),
        "num_page_override": args.num_page_override,
        "token_pool_tokens": args.token_pool_tokens,
        "compact_kv_budget_bytes": args.compact_kv_budget_bytes,
        "compact_kv_layer_major_metadata": llm.model_runner.config.compact_kv_layer_major_metadata,
        "prefix_sharing": False,
        "rows": [row],
    }


def main() -> None:
    args = parse_args()
    result = run_benchmark(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.phase_profile_output:
        args.phase_profile_output.parent.mkdir(parents=True, exist_ok=True)
        args.phase_profile_output.write_text(
            json.dumps(result["rows"][0]["phase_profile"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if result["rows"][0]["completed_requests"] != args.concurrency:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
