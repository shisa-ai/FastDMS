import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from fastdms.layers.attention_backends import get_backend_name
from fastdms.layers.compact_attention import compact_decode_attention, store_compact_kvcache
from fastdms.utils.context import get_context
from fastdms.utils.profiler import get_profiler

# Resolved at import time; raises if FASTDMS_ATTENTION_BACKEND is set to an
# unimplemented value.
_BACKEND = get_backend_name()


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.compact_k_cache = self.compact_v_cache = torch.tensor([])
        self.layer_id = -1

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        compact_append_stored: bool = False,
        compact_q_rope: bool = False,
        positions: torch.Tensor | None = None,
        cos_sin_cache: torch.Tensor | None = None,
    ):
        context = get_context()
        prof = get_profiler()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel() and context.slot_mapping is not None:
            with prof.cuda_phase("dense_kvcache_store"):
                store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            with prof.cuda_phase("dense_prefill_attn"):
                o = flash_attn_varlen_func(q, k, v,
                                           max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                           max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                           softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        elif context.compact_kv_enabled:
            if context.compact_base_offsets is None or context.compact_live_counts is None:
                raise RuntimeError("compact KV decode requires compact_base_offsets and compact_live_counts")
            compact_base_offsets = context.compact_base_offsets
            compact_live_counts = context.compact_live_counts
            compact_slot_mapping = context.compact_slot_mapping
            if compact_base_offsets.ndim == 3:
                compact_base_offsets = compact_base_offsets[self.layer_id]
                compact_live_counts = compact_live_counts[self.layer_id]
                compact_slot_mapping = None if compact_slot_mapping is None else compact_slot_mapping[self.layer_id]
            if compact_slot_mapping is not None and not compact_append_stored:
                with prof.cuda_phase("compact_append_store"):
                    store_compact_kvcache(
                        k,
                        v,
                        self.compact_k_cache,
                        self.compact_v_cache,
                        compact_slot_mapping,
                    )
            with prof.cuda_phase("compact_decode_attn"):
                o = compact_decode_attention(
                    q,
                    self.compact_k_cache,
                    self.compact_v_cache,
                    compact_base_offsets,
                    compact_live_counts,
                    scale=self.scale,
                    max_live_count=context.compact_max_live_count,
                    q_rope_positions=positions if compact_q_rope else None,
                    q_rope_cos_sin_cache=cos_sin_cache if compact_q_rope else None,
                )
        else:    # decode
            if k_cache.dtype != q.dtype:
                # C1.1: flash_attn 2.x cannot consume FP8 KV directly, so we
                # still BF16-stage. Mitigation: only cast the blocks the
                # current decode batch references (deduplicated via
                # block_tables). Falls back to a full-arena cast when no
                # block_tables (single-seq cases without paging).
                with prof.cuda_phase("dense_decode_fp8_cast_stage"):
                    if context.block_tables is not None and context.block_tables.numel() > 0:
                        bt = context.block_tables
                        unique_blocks, inverse = torch.unique(bt.flatten(), return_inverse=True)
                        # Gather only the active blocks, cast, and remap
                        # block_tables to point into the reduced cache.
                        k_active = k_cache[unique_blocks].to(q.dtype)
                        v_active = v_cache[unique_blocks].to(q.dtype)
                        remapped_block_table = inverse.view(bt.shape).to(torch.int32)
                        k_cache, v_cache = k_active, v_active
                    else:
                        k_cache = k_cache.to(q.dtype)
                        v_cache = v_cache.to(q.dtype)
                        remapped_block_table = context.block_tables
            else:
                remapped_block_table = context.block_tables
            with prof.cuda_phase("dense_decode_attn"):
                o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                            cache_seqlens=context.context_lens, block_table=remapped_block_table,
                                            softmax_scale=self.scale, causal=True)
        return o
