import os
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from fastdms.engine.sequence import Sequence


def streaming_pack_cache_live_enabled() -> bool:
    return os.environ.get("NANOVLLM_STREAMING_PACK_CACHE_LIVE", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def streaming_pack_triton_enabled() -> bool:
    return os.environ.get("NANOVLLM_STREAMING_PACK_TRITON", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def streaming_pack_fused_rank_triton_enabled() -> bool:
    return os.environ.get("NANOVLLM_STREAMING_PACK_FUSED_RANK_TRITON", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@triton.jit
def _streaming_pack_scatter_kernel(
    k_ptr,
    v_ptr,
    keep_ptr,
    rank_ptr,
    seq_evict_ptr,
    compact_k_ptr,
    compact_v_ptr,
    token_positions_ptr,
    evict_mask_ptr,
    base_offsets_ptr,
    start_token: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    k_stride_t: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_d: tl.constexpr,
    v_stride_t: tl.constexpr,
    v_stride_h: tl.constexpr,
    v_stride_d: tl.constexpr,
    keep_stride_t: tl.constexpr,
    keep_stride_h: tl.constexpr,
    rank_stride_t: tl.constexpr,
    rank_stride_h: tl.constexpr,
    evict_stride_t: tl.constexpr,
    evict_stride_h: tl.constexpr,
    compact_stride_s: tl.constexpr,
    compact_stride_d: tl.constexpr,
    token_pos_stride_h: tl.constexpr,
    token_pos_stride_s: tl.constexpr,
    evict_mask_stride_h: tl.constexpr,
    evict_mask_stride_s: tl.constexpr,
    base_stride_h: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    chunk = tl.program_id(0)
    kv_head = tl.program_id(1)
    offsets_t = chunk * BLOCK_T + tl.arange(0, BLOCK_T)
    offsets_d = tl.arange(0, BLOCK_D)
    t_mask = offsets_t < seq_len
    d_mask = offsets_d < head_dim

    keep = tl.load(
        keep_ptr + offsets_t * keep_stride_t + kv_head * keep_stride_h,
        mask=t_mask,
        other=0,
    ).to(tl.int1)
    rank = tl.load(
        rank_ptr + offsets_t * rank_stride_t + kv_head * rank_stride_h,
        mask=t_mask,
        other=0,
    ).to(tl.int64)
    base = tl.load(base_offsets_ptr + kv_head * base_stride_h).to(tl.int64)
    dst_slot = base + rank
    src_token = start_token + offsets_t

    k_vals = tl.load(
        k_ptr
        + src_token[:, None] * k_stride_t
        + kv_head * k_stride_h
        + offsets_d[None, :] * k_stride_d,
        mask=keep[:, None] & d_mask[None, :],
        other=0.0,
    )
    v_vals = tl.load(
        v_ptr
        + src_token[:, None] * v_stride_t
        + kv_head * v_stride_h
        + offsets_d[None, :] * v_stride_d,
        mask=keep[:, None] & d_mask[None, :],
        other=0.0,
    )
    tl.store(
        compact_k_ptr + dst_slot[:, None] * compact_stride_s + offsets_d[None, :] * compact_stride_d,
        k_vals,
        mask=keep[:, None] & d_mask[None, :],
    )
    tl.store(
        compact_v_ptr + dst_slot[:, None] * compact_stride_s + offsets_d[None, :] * compact_stride_d,
        v_vals,
        mask=keep[:, None] & d_mask[None, :],
    )

    evict = tl.load(
        seq_evict_ptr + offsets_t * evict_stride_t + kv_head * evict_stride_h,
        mask=t_mask,
        other=0,
    )
    tl.store(
        token_positions_ptr + kv_head * token_pos_stride_h + rank * token_pos_stride_s,
        offsets_t.to(tl.int32),
        mask=keep,
    )
    tl.store(
        evict_mask_ptr + kv_head * evict_mask_stride_h + rank * evict_mask_stride_s,
        evict,
        mask=keep,
    )


@triton.jit
def _streaming_pack_count_kernel(
    seq_evict_ptr,
    block_counts_ptr,
    seq_len: tl.constexpr,
    window_size: tl.constexpr,
    evict_stride_t: tl.constexpr,
    evict_stride_h: tl.constexpr,
    counts_stride_h: tl.constexpr,
    counts_stride_c: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    chunk = tl.program_id(0)
    kv_head = tl.program_id(1)
    offsets_t = chunk * BLOCK_T + tl.arange(0, BLOCK_T)
    valid = offsets_t < seq_len
    recent_threshold = tl.maximum(0, seq_len - 1 - window_size)
    in_window = offsets_t >= recent_threshold
    evict = tl.load(
        seq_evict_ptr + offsets_t * evict_stride_t + kv_head * evict_stride_h,
        mask=valid,
        other=0,
    ).to(tl.int1)
    keep = valid & ((~evict) | in_window)
    count = tl.sum(keep.to(tl.int32), axis=0)
    tl.store(block_counts_ptr + kv_head * counts_stride_h + chunk * counts_stride_c, count)


@triton.jit
def _streaming_pack_rank_scatter_kernel(
    k_ptr,
    v_ptr,
    seq_evict_ptr,
    block_offsets_ptr,
    compact_k_ptr,
    compact_v_ptr,
    token_positions_ptr,
    evict_mask_ptr,
    base_offsets_ptr,
    start_token: tl.constexpr,
    seq_len: tl.constexpr,
    window_size: tl.constexpr,
    head_dim: tl.constexpr,
    k_stride_t: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_d: tl.constexpr,
    v_stride_t: tl.constexpr,
    v_stride_h: tl.constexpr,
    v_stride_d: tl.constexpr,
    evict_stride_t: tl.constexpr,
    evict_stride_h: tl.constexpr,
    offset_stride_h: tl.constexpr,
    offset_stride_c: tl.constexpr,
    compact_stride_s: tl.constexpr,
    compact_stride_d: tl.constexpr,
    token_pos_stride_h: tl.constexpr,
    token_pos_stride_s: tl.constexpr,
    evict_mask_stride_h: tl.constexpr,
    evict_mask_stride_s: tl.constexpr,
    base_stride_h: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    chunk = tl.program_id(0)
    kv_head = tl.program_id(1)
    offsets_t = chunk * BLOCK_T + tl.arange(0, BLOCK_T)
    offsets_d = tl.arange(0, BLOCK_D)
    t_mask = offsets_t < seq_len
    d_mask = offsets_d < head_dim
    recent_threshold = tl.maximum(0, seq_len - 1 - window_size)
    in_window = offsets_t >= recent_threshold

    evict = tl.load(
        seq_evict_ptr + offsets_t * evict_stride_t + kv_head * evict_stride_h,
        mask=t_mask,
        other=0,
    ).to(tl.int1)
    keep = t_mask & ((~evict) | in_window)
    local_rank = tl.cumsum(keep.to(tl.int32), 0) - 1
    block_offset = tl.load(
        block_offsets_ptr + kv_head * offset_stride_h + chunk * offset_stride_c,
    ).to(tl.int32)
    rank = block_offset + local_rank
    base = tl.load(base_offsets_ptr + kv_head * base_stride_h).to(tl.int64)
    dst_slot = base + rank.to(tl.int64)
    src_token = start_token + offsets_t

    k_vals = tl.load(
        k_ptr
        + src_token[:, None] * k_stride_t
        + kv_head * k_stride_h
        + offsets_d[None, :] * k_stride_d,
        mask=keep[:, None] & d_mask[None, :],
        other=0.0,
    )
    v_vals = tl.load(
        v_ptr
        + src_token[:, None] * v_stride_t
        + kv_head * v_stride_h
        + offsets_d[None, :] * v_stride_d,
        mask=keep[:, None] & d_mask[None, :],
        other=0.0,
    )
    tl.store(
        compact_k_ptr + dst_slot[:, None] * compact_stride_s + offsets_d[None, :] * compact_stride_d,
        k_vals,
        mask=keep[:, None] & d_mask[None, :],
    )
    tl.store(
        compact_v_ptr + dst_slot[:, None] * compact_stride_s + offsets_d[None, :] * compact_stride_d,
        v_vals,
        mask=keep[:, None] & d_mask[None, :],
    )
    tl.store(
        token_positions_ptr + kv_head * token_pos_stride_h + rank.to(tl.int64) * token_pos_stride_s,
        offsets_t.to(tl.int32),
        mask=keep,
    )
    tl.store(
        evict_mask_ptr + kv_head * evict_mask_stride_h + rank.to(tl.int64) * evict_mask_stride_s,
        evict,
        mask=keep,
    )


def _streaming_pack_fused_rank_prepare(
    *,
    seq_evict: torch.Tensor,
    seq_len: int,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if seq_evict.ndim != 2:
        raise ValueError(f"expected seq_evict [seq_len, kv_heads], got {tuple(seq_evict.shape)}")
    num_kv_heads = seq_evict.shape[1]
    block_t = 256
    num_chunks = triton.cdiv(seq_len, block_t)
    block_counts = torch.empty(
        (num_kv_heads, num_chunks),
        device=seq_evict.device,
        dtype=torch.int32,
    )
    _streaming_pack_count_kernel[(num_chunks, num_kv_heads)](
        seq_evict,
        block_counts,
        int(seq_len),
        int(window_size),
        seq_evict.stride(0),
        seq_evict.stride(1),
        block_counts.stride(0),
        block_counts.stride(1),
        block_t,
    )
    live_count_per_head = block_counts.sum(dim=1, dtype=torch.int32)
    block_offsets = block_counts.cumsum(dim=1) - block_counts
    return live_count_per_head, block_offsets.contiguous()


def _streaming_pack_rank_scatter_triton(
    *,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_evict: torch.Tensor,
    block_offsets: torch.Tensor,
    compact_k: torch.Tensor,
    compact_v: torch.Tensor,
    token_positions: torch.Tensor,
    evict_mask: torch.Tensor,
    base_offsets: torch.Tensor,
    start: int,
    seq_len: int,
    window_size: int,
) -> None:
    if seq_len <= 0:
        return
    num_kv_heads = k.shape[1]
    head_dim = k.shape[2]
    block_t = 256
    block_d = 1 << (head_dim - 1).bit_length()
    if block_d > 128:
        raise RuntimeError(f"streaming pack fused-rank Triton head_dim={head_dim} is unsupported")
    expected_chunks = triton.cdiv(seq_len, block_t)
    if block_offsets.shape != (num_kv_heads, expected_chunks):
        raise ValueError(
            f"expected block_offsets {(num_kv_heads, expected_chunks)}, got {tuple(block_offsets.shape)}"
        )
    grid = (expected_chunks, num_kv_heads)
    _streaming_pack_rank_scatter_kernel[grid](
        k,
        v,
        seq_evict,
        block_offsets,
        compact_k,
        compact_v,
        token_positions,
        evict_mask,
        base_offsets,
        int(start),
        int(seq_len),
        int(window_size),
        int(head_dim),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        seq_evict.stride(0),
        seq_evict.stride(1),
        block_offsets.stride(0),
        block_offsets.stride(1),
        compact_k.stride(0),
        compact_k.stride(1),
        token_positions.stride(0),
        token_positions.stride(1),
        evict_mask.stride(0),
        evict_mask.stride(1),
        base_offsets.stride(0),
        block_t,
        block_d,
    )


def _streaming_pack_scatter_triton(
    *,
    k: torch.Tensor,
    v: torch.Tensor,
    keep: torch.Tensor,
    rank: torch.Tensor,
    seq_evict: torch.Tensor,
    compact_k: torch.Tensor,
    compact_v: torch.Tensor,
    token_positions: torch.Tensor,
    evict_mask: torch.Tensor,
    base_offsets: torch.Tensor,
    start: int,
    seq_len: int,
) -> None:
    if seq_len <= 0:
        return
    num_kv_heads = k.shape[1]
    head_dim = k.shape[2]
    block_t = 256
    block_d = 1 << (head_dim - 1).bit_length()
    if block_d > 128:
        raise RuntimeError(f"streaming pack Triton head_dim={head_dim} is unsupported")
    grid = (triton.cdiv(seq_len, block_t), num_kv_heads)
    _streaming_pack_scatter_kernel[grid](
        k,
        v,
        keep,
        rank,
        seq_evict,
        compact_k,
        compact_v,
        token_positions,
        evict_mask,
        base_offsets,
        int(start),
        int(seq_len),
        int(head_dim),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        keep.stride(0),
        keep.stride(1),
        rank.stride(0),
        rank.stride(1),
        seq_evict.stride(0),
        seq_evict.stride(1),
        compact_k.stride(0),
        compact_k.stride(1),
        token_positions.stride(0),
        token_positions.stride(1),
        evict_mask.stride(0),
        evict_mask.stride(1),
        base_offsets.stride(0),
        block_t,
        block_d,
    )


@triton.jit
def _record_dms_decisions_layer_major_kernel(
    token_positions_ptr,
    evict_mask_ptr,
    live_counts_ptr,
    evict_all_ptr,
    positions_ptr,
    active_slot_start: tl.constexpr,
    total: tl.constexpr,
    batch: tl.constexpr,
    num_kv_heads: tl.constexpr,
    token_pos_stride_l: tl.constexpr,
    token_pos_stride_s: tl.constexpr,
    token_pos_stride_h: tl.constexpr,
    evict_mask_stride_l: tl.constexpr,
    evict_mask_stride_s: tl.constexpr,
    evict_mask_stride_h: tl.constexpr,
    live_stride_l: tl.constexpr,
    live_stride_s: tl.constexpr,
    live_stride_h: tl.constexpr,
    evict_all_stride_l: tl.constexpr,
    evict_all_stride_b: tl.constexpr,
    evict_all_stride_h: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    layer = offsets // (batch * num_kv_heads)
    rem = offsets - layer * batch * num_kv_heads
    batch_idx = rem // num_kv_heads
    head = rem - batch_idx * num_kv_heads
    slot = active_slot_start + batch_idx

    live_offset = layer * live_stride_l + slot * live_stride_s + head * live_stride_h
    append_idx = tl.load(live_counts_ptr + live_offset, mask=mask, other=1).to(tl.int64) - 1
    positions = tl.load(positions_ptr + batch_idx, mask=mask, other=0).to(tl.int32)
    evict = tl.load(
        evict_all_ptr
        + layer * evict_all_stride_l
        + batch_idx * evict_all_stride_b
        + head * evict_all_stride_h,
        mask=mask,
        other=0,
    )

    token_pos_offset = (
        layer * token_pos_stride_l
        + slot * token_pos_stride_s
        + head * token_pos_stride_h
        + append_idx
    )
    evict_mask_offset = (
        layer * evict_mask_stride_l
        + slot * evict_mask_stride_s
        + head * evict_mask_stride_h
        + append_idx
    )
    tl.store(token_positions_ptr + token_pos_offset, positions, mask=mask)
    tl.store(evict_mask_ptr + evict_mask_offset, evict, mask=mask)


@dataclass(slots=True)
class CompactKVMetadata:
    base_offsets: torch.Tensor
    live_counts: torch.Tensor
    token_positions: torch.Tensor | None = None

    @property
    def max_live_count(self) -> int:
        if self.live_counts.numel() == 0:
            return 0
        return int(self.live_counts.max().item())


@dataclass(slots=True)
class CompactKVSequenceState:
    seq_id: int
    # base_offsets: [num_layers, num_kv_heads] int32 — per-(layer, head) start
    # offset within that layer's pool. -1 if not yet allocated for this layer.
    base_offsets: torch.Tensor
    # range_capacity: [num_layers, num_kv_heads] int32 — per-(layer, head)
    # reservation size in slots. 0 if not yet allocated.
    range_capacity: torch.Tensor
    live_counts: torch.Tensor
    token_positions: torch.Tensor
    evict_mask: torch.Tensor
    # max_capacity_per_lh: storage tensors' third-dim size (worst-case
    # across all (layer, head) — uniform for tensor-shape compatibility).
    max_capacity_per_lh: int
    # Decode headroom available per (layer, head) — i.e., gen_len at allocate
    # time. Used by per-layer alloc to determine reservation size.
    decode_headroom: int
    active_slot: int | None = None
    dms_initial_expiry_done: bool = False
    last_dms_expiry_position: int = -1


def build_emulated_dms_live_mask(
    *,
    num_layers: int,
    num_kv_heads: int,
    num_tokens: int,
    mode: str,
    stride: int,
    recent_tokens: int,
    device: torch.device | str,
) -> torch.Tensor:
    if mode == "all":
        return torch.ones((num_layers, num_kv_heads, num_tokens), device=device, dtype=torch.bool)
    if mode != "dms_stride":
        raise ValueError(f"unknown compact KV retention mode: {mode}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if recent_tokens < 0:
        raise ValueError(f"recent_tokens must be non-negative, got {recent_tokens}")

    positions = torch.arange(num_tokens, device=device)
    mask = torch.zeros((num_layers, num_kv_heads, num_tokens), device=device, dtype=torch.bool)
    if recent_tokens:
        mask[..., max(0, num_tokens - recent_tokens):] = True
    if num_tokens:
        for layer_id in range(num_layers):
            for kv_head in range(num_kv_heads):
                mask[layer_id, kv_head] |= (positions + layer_id + kv_head) % stride == 0
                mask[layer_id, kv_head, -1] = True
    return mask


class CompactKVManager:
    """Owns flat compact KV spans for decode-time attention.

    The prototype reserves one contiguous span per KV head and reuses that same
    base-offset layout for every layer. Per-layer live counts are kept separate
    so later DMS masks can diverge by layer/head without changing the ABI.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_kv_heads: int,
        capacity_tokens: int,
        max_model_len: int,
        device: torch.device | str,
        target_live_tokens_per_seq: int | None = None,
        head_dim: int | None = None,
        compact_dtype: torch.dtype | None = None,
        per_layer_storage: bool = False,
        layer_major_metadata: bool = False,
        max_active_seqs: int | None = None,
    ):
        if capacity_tokens <= 0:
            raise ValueError(f"capacity_tokens must be positive, got {capacity_tokens}")
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.capacity_tokens = capacity_tokens
        self.max_model_len = max_model_len
        self.device = torch.device(device)
        # J5 per-layer separate storage: each layer gets its own (compact_k,
        # compact_v) tensor sized for its needs. capacity_tokens becomes a
        # per-layer growth-cap, not a uniform tensor dim.
        self.per_layer_storage = per_layer_storage
        self.head_dim = head_dim
        self.compact_dtype = compact_dtype
        self.layer_major_metadata = layer_major_metadata
        self.max_active_seqs = int(max_active_seqs or 0)
        self.compact_k_per_layer: list[torch.Tensor | None] = [None] * num_layers
        self.compact_v_per_layer: list[torch.Tensor | None] = [None] * num_layers
        self.layer_major_base_offsets: torch.Tensor | None = None
        self.layer_major_range_capacity: torch.Tensor | None = None
        self.layer_major_live_counts: torch.Tensor | None = None
        self.layer_major_token_positions: torch.Tensor | None = None
        self.layer_major_evict_mask: torch.Tensor | None = None
        self.active_slot_by_seq: dict[int, int] = {}
        self.free_active_slots: list[int] = []
        if self.layer_major_metadata:
            if self.max_active_seqs <= 0:
                raise ValueError("layer-major metadata requires max_active_seqs > 0")
            metadata_capacity = int(max_model_len)
            self.layer_major_base_offsets = torch.full(
                (num_layers, self.max_active_seqs, num_kv_heads),
                -1,
                device=self.device,
                dtype=torch.int32,
            )
            self.layer_major_range_capacity = torch.zeros(
                (num_layers, self.max_active_seqs, num_kv_heads),
                device=self.device,
                dtype=torch.int32,
            )
            self.layer_major_live_counts = torch.zeros_like(self.layer_major_range_capacity)
            self.layer_major_token_positions = torch.full(
                (num_layers, self.max_active_seqs, num_kv_heads, metadata_capacity),
                -1,
                device=self.device,
                dtype=torch.int32,
            )
            self.layer_major_evict_mask = torch.zeros(
                (num_layers, self.max_active_seqs, num_kv_heads, metadata_capacity),
                device=self.device,
                dtype=torch.bool,
            )
            self.free_active_slots = list(range(self.max_active_seqs))
        # Modules that need their compact_k_cache pointers updated after a
        # per-layer storage tensor is allocated.
        self._attn_modules: list = []
        # B1.4: explicit target live-token budget per seq. When set, allocate()
        # reserves only this many tokens per kv_head instead of worst-case
        # max_model_len. Admission denial routes back to the scheduler.
        self.target_live_tokens_per_seq = target_live_tokens_per_seq
        # J5: per-(seq, head, layer) allocation. Each layer has its own free
        # pool of slots so per-(seq, head) ranges in different layers can be
        # different sizes. self.free_ranges_per_layer[L] = list[(start, length)].
        self.free_ranges_per_layer: list[list[tuple[int, int]]] = [
            [(0, capacity_tokens)] for _ in range(num_layers)
        ]
        self.seq_states: dict[int, CompactKVSequenceState] = {}
        self.admission_denied_count = 0
        self.debug_checks = os.environ.get("NANOVLLM_COMPACT_DEBUG_CHECKS") == "1"
        # J1 streaming pack: manager owns compact storage tensors directly so
        # the streaming pack and apply_dms_evictions don't need them passed.
        self.compact_k: torch.Tensor | None = None
        self.compact_v: torch.Tensor | None = None

    def attach_compact_storage(self, compact_kv_cache: torch.Tensor) -> None:
        """Wire the compact arena tensor (shape [2, num_layers, capacity, head_dim])."""
        self.compact_k = compact_kv_cache[0]
        self.compact_v = compact_kv_cache[1]

    def attach_attention_modules(self, attn_modules: list) -> None:
        """J5 register attention modules so manager can update their
        compact_k_cache / compact_v_cache pointers when per-layer storage is
        lazily allocated. attn_modules must be ordered by layer_id."""
        self._attn_modules = attn_modules

    def _ensure_layer_storage(self, layer_id: int, min_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
        """J5 lazy-allocate per-layer compact storage. If existing storage is
        too small for the requested min_tokens, reallocate larger and migrate.
        Returns (compact_k[layer], compact_v[layer]) for this layer.

        Allocations exit inference_mode so the resulting tensors are normal
        tensors mutable across both inference and non-inference call sites.
        """
        existing_k = self.compact_k_per_layer[layer_id]
        if existing_k is not None and existing_k.shape[0] >= min_tokens:
            return existing_k, self.compact_v_per_layer[layer_id]
        if self.head_dim is None or self.compact_dtype is None:
            raise RuntimeError(
                "per-layer storage requires head_dim and compact_dtype set on manager"
            )
        # Grow strategy: round up to a multiple of 256 (cache-line / kernel
        # alignment) rather than next power of 2 - pow2 wasted ~30% at our
        # workload sizes. Keep growth exact so small per-layer live-count
        # variation does not double compact storage after warmup.
        new_size = max(256, ((min_tokens + 255) // 256) * 256)
        with torch.inference_mode(False):
            new_k = torch.empty(new_size, self.head_dim, device=self.device, dtype=self.compact_dtype)
            new_v = torch.empty(new_size, self.head_dim, device=self.device, dtype=self.compact_dtype)
            if existing_k is not None:
                n = existing_k.shape[0]
                new_k[:n].copy_(existing_k)
                new_v[:n].copy_(self.compact_v_per_layer[layer_id])
        self.compact_k_per_layer[layer_id] = new_k
        self.compact_v_per_layer[layer_id] = new_v
        # Update attention module bindings if registered
        if self._attn_modules and layer_id < len(self._attn_modules):
            module = self._attn_modules[layer_id]
            module.compact_k_cache = new_k
            module.compact_v_cache = new_v
        return new_k, new_v

    def free_tokens_in_layer(self, layer_id: int) -> int:
        return sum(length for _, length in self.free_ranges_per_layer[layer_id])

    @property
    def free_tokens(self) -> int:
        # Total free across all layers (rough indicator).
        return sum(self.free_tokens_in_layer(L) for L in range(self.num_layers))

    def _alloc_range(self, layer_id: int, length: int) -> int:
        free = self.free_ranges_per_layer[layer_id]
        for idx, (start, available) in enumerate(free):
            if available < length:
                continue
            if available == length:
                free.pop(idx)
            else:
                free[idx] = (start + length, available - length)
            return start
        raise RuntimeError(
            f"compact KV capacity exhausted in layer {layer_id}: requested {length}, "
            f"free {self.free_tokens_in_layer(layer_id)}"
        )

    def _free_range(self, layer_id: int, start: int, length: int) -> None:
        if length <= 0:
            return
        free = self.free_ranges_per_layer[layer_id]
        free.append((start, length))
        free.sort()
        merged: list[tuple[int, int]] = []
        for range_start, range_len in free:
            if not merged:
                merged.append((range_start, range_len))
                continue
            prev_start, prev_len = merged[-1]
            prev_end = prev_start + prev_len
            if range_start <= prev_end:
                merged[-1] = (prev_start, max(prev_end, range_start + range_len) - prev_start)
            else:
                merged.append((range_start, range_len))
        self.free_ranges_per_layer[layer_id] = merged

    def can_allocate(self, seq: Sequence) -> bool:
        # Conservative admission: need reservation_tokens * num_kv_heads slots
        # available in EVERY layer.
        needed = self._reservation_tokens(seq) * self.num_kv_heads
        return all(
            self.free_tokens_in_layer(L) >= needed for L in range(self.num_layers)
        )

    def _reservation_tokens(self, seq: Sequence) -> int:
        worst_case = min(self.max_model_len, seq.num_prompt_tokens + seq.max_tokens)
        if self.target_live_tokens_per_seq is not None:
            return min(worst_case, max(1, int(self.target_live_tokens_per_seq)))
        return worst_case

    def allocate(self, seq: Sequence) -> CompactKVSequenceState:
        # J5 lightweight registration: actual per-(layer, head) range
        # allocation happens lazily in streaming_pack_layer based on the
        # observed live counts. Storage tensors are sized to worst-case
        # reservation for shape uniformity (a few MB of metadata, not GB).
        existing = self.seq_states.get(seq.seq_id)
        if existing is not None:
            return existing

        max_capacity_per_lh = self._reservation_tokens(seq)
        if max_capacity_per_lh <= 0:
            raise ValueError(f"invalid reservation {max_capacity_per_lh} for seq {seq.seq_id}")

        active_slot = None
        if self.layer_major_metadata:
            if not self.free_active_slots:
                raise RuntimeError(
                    f"compact KV active-slot capacity exhausted: max_active_seqs={self.max_active_seqs}"
                )
            active_slot = self.free_active_slots.pop(0)
            self.active_slot_by_seq[seq.seq_id] = active_slot
            base_offsets = self.layer_major_base_offsets[:, active_slot, :]
            range_capacity = self.layer_major_range_capacity[:, active_slot, :]
            live_counts = self.layer_major_live_counts[:, active_slot, :]
            token_positions = self.layer_major_token_positions[
                :, active_slot, :, :max_capacity_per_lh
            ]
            evict_mask = self.layer_major_evict_mask[
                :, active_slot, :, :max_capacity_per_lh
            ]
            base_offsets.fill_(-1)
            range_capacity.zero_()
            live_counts.zero_()
            token_positions.fill_(-1)
            evict_mask.zero_()
        else:
            base_offsets = torch.full(
                (self.num_layers, self.num_kv_heads),
                -1,
                device=self.device,
                dtype=torch.int32,
            )
            range_capacity = torch.zeros(
                (self.num_layers, self.num_kv_heads),
                device=self.device,
                dtype=torch.int32,
            )
            live_counts = torch.zeros(
                (self.num_layers, self.num_kv_heads),
                device=self.device,
                dtype=torch.int32,
            )
            token_positions = torch.full(
                (self.num_layers, self.num_kv_heads, max_capacity_per_lh),
                -1,
                device=self.device,
                dtype=torch.int32,
            )
            evict_mask = torch.zeros(
                (self.num_layers, self.num_kv_heads, max_capacity_per_lh),
                device=self.device,
                dtype=torch.bool,
            )
        state = CompactKVSequenceState(
            seq_id=seq.seq_id,
            base_offsets=base_offsets,
            range_capacity=range_capacity,
            live_counts=live_counts,
            token_positions=token_positions,
            evict_mask=evict_mask,
            active_slot=active_slot,
            max_capacity_per_lh=max_capacity_per_lh,
            decode_headroom=int(seq.max_tokens),
        )
        self.seq_states[seq.seq_id] = state
        return state

    def _allocate_layer_ranges_for_seq(
        self, state: CompactKVSequenceState, layer_id: int, live_count_per_head: torch.Tensor
    ) -> torch.Tensor:
        """Lazy per-(seq, head, layer) allocation. Sizes each (head)'s range to
        (live_count + decode_headroom, capped at max_capacity_per_lh). Returns
        the allocated bases as an int32 tensor [num_kv_heads]."""
        live_cpu = live_count_per_head.tolist()
        bases = []
        sizes = []
        for h in range(self.num_kv_heads):
            size = int(live_cpu[h]) + int(state.decode_headroom)
            size = min(size, state.max_capacity_per_lh)
            size = max(size, 1)
            sizes.append(size)
        # First pass: try to allocate. If a size fails, deallocate prior heads
        # and re-raise.
        try:
            for h, size in enumerate(sizes):
                bases.append(self._alloc_range(layer_id, size))
        except Exception:
            for h2, base2 in enumerate(bases):
                self._free_range(layer_id, base2, sizes[h2])
            raise
        bases_t = torch.tensor(bases, device=self.device, dtype=torch.int32)
        sizes_t = torch.tensor(sizes, device=self.device, dtype=torch.int32)
        state.base_offsets[layer_id].copy_(bases_t)
        state.range_capacity[layer_id].copy_(sizes_t)
        return bases_t

    def mark_all_heads_live(self, seq_id: int, live_count: int) -> None:
        live_counts = torch.full(
            (self.num_layers, self.num_kv_heads),
            live_count,
            device=self.device,
            dtype=torch.int32,
        )
        self.mark_live_counts(seq_id, live_counts)

    def mark_live_counts(self, seq_id: int, live_counts: torch.Tensor) -> None:
        state = self.seq_states[seq_id]
        if live_counts.shape != (self.num_layers, self.num_kv_heads):
            raise ValueError(
                f"expected live_counts {(self.num_layers, self.num_kv_heads)}, "
                f"got {tuple(live_counts.shape)}"
            )
        live_counts = live_counts.to(device=self.device, dtype=torch.int32)
        # J5: per-(layer, head) capacity check against range_capacity.
        if bool(torch.any(live_counts > state.range_capacity).item()):
            max_live = int(live_counts.max().item())
            raise RuntimeError(
                f"compact KV live count {max_live} exceeds per-(layer,head) range capacity"
            )
        state.live_counts.copy_(live_counts)

    def mark_live_metadata(
        self,
        seq_id: int,
        layer_id: int,
        kv_head: int,
        positions: torch.Tensor,
        evict_mask: torch.Tensor | None = None,
    ) -> None:
        state = self.seq_states[seq_id]
        live = int(positions.numel())
        if live > state.max_capacity_per_lh:
            raise RuntimeError(
                f"compact KV metadata live count {live} exceeds metadata capacity "
                f"{state.max_capacity_per_lh}"
            )
        if live:
            state.token_positions[layer_id, kv_head, :live] = positions.to(
                device=self.device,
                dtype=torch.int32,
            )
            if evict_mask is None:
                state.evict_mask[layer_id, kv_head, :live] = False
            else:
                state.evict_mask[layer_id, kv_head, :live] = evict_mask.to(
                    device=self.device,
                    dtype=torch.bool,
                )
        if live < state.max_capacity_per_lh:
            state.token_positions[layer_id, kv_head, live:] = -1
            state.evict_mask[layer_id, kv_head, live:] = False

    def streaming_pack_layer(
        self,
        *,
        layer_id: int,
        k: torch.Tensor,
        v: torch.Tensor,
        evict_mask: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        seq_ids: list[int],
        window_size: int,
    ) -> None:
        """J1 streaming compact pack inside attention forward.

        For each seq in the prefill batch, compute the live mask from this
        layer's evict_mask + DMS window, cumsum-rank-scatter k/v into the
        compact arena at base_offsets[h] + rank, and update state metadata.
        Eliminates the need for a persistent dense KV scratch.

        Args:
          layer_id: which layer is being packed.
          k, v: BF16 [total_prefill_tokens, num_kv_heads, head_dim] post-RoPE.
          evict_mask: bool [total_prefill_tokens, num_kv_heads] DMS decisions.
          cu_seqlens_q: int32 [num_seqs+1] prefill boundaries.
          seq_ids: list of seq_ids in batch order.
          window_size: DMS retention window.
        """
        triton_pack = streaming_pack_triton_enabled()
        fused_rank_pack = (
            triton_pack
            and streaming_pack_fused_rank_triton_enabled()
            and k.is_cuda
            and evict_mask.is_cuda
        )
        if self.per_layer_storage:
            cache_live = streaming_pack_cache_live_enabled()
            live_cache: dict[int, tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]] = {}
            # Lazy allocate per-layer storage to the high-water allocated range
            # for this layer. Free-range bases may sit after prior active seqs,
            # so sizing only to the current batch's live-token sum is unsafe.
            layer_high_water = 0
            for batch_idx, seq_id in enumerate(seq_ids):
                state = self.seq_states.get(seq_id)
                if state is None:
                    continue
                start = int(cu_seqlens_q[batch_idx].item())
                end = int(cu_seqlens_q[batch_idx + 1].item())
                seq_len = end - start
                seq_evict = evict_mask[start:end]
                if fused_rank_pack:
                    keep = None
                    live_per_head, block_offsets = _streaming_pack_fused_rank_prepare(
                        seq_evict=seq_evict,
                        seq_len=seq_len,
                        window_size=int(window_size),
                    )
                else:
                    positions = torch.arange(seq_len, device=k.device, dtype=torch.int32)
                    recent_threshold = max(0, seq_len - 1 - int(window_size))
                    in_window = positions >= recent_threshold
                    keep = (~seq_evict) | in_window.unsqueeze(1)
                    live_per_head = keep.sum(dim=0, dtype=torch.int32)
                    block_offsets = None
                if cache_live:
                    live_cache[seq_id] = (seq_evict, keep, live_per_head, block_offsets)
                self._allocate_layer_ranges_for_seq(state, layer_id, live_per_head)
                range_end = (
                    state.base_offsets[layer_id].to(torch.int64)
                    + state.range_capacity[layer_id].to(torch.int64)
                ).max()
                layer_high_water = max(layer_high_water, int(range_end.item()))
            self._ensure_layer_storage(layer_id, layer_high_water)
            compact_k = self.compact_k_per_layer[layer_id]
            compact_v = self.compact_v_per_layer[layer_id]
        else:
            if self.compact_k is None or self.compact_v is None:
                raise RuntimeError("CompactKVManager.attach_compact_storage was not called")
            compact_k = self.compact_k[layer_id]
            compact_v = self.compact_v[layer_id]
        device = k.device
        compact_dtype = compact_k.dtype
        num_kv_heads = k.shape[1]
        head_dim = k.shape[2]
        for batch_idx, seq_id in enumerate(seq_ids):
            state = self.seq_states.get(seq_id)
            if state is None:
                raise RuntimeError(f"sequence {seq_id} has no compact KV allocation")
            start = int(cu_seqlens_q[batch_idx].item())
            end = int(cu_seqlens_q[batch_idx + 1].item())
            seq_len = end - start
            if seq_len <= 0:
                continue
            cached_live = live_cache.get(seq_id) if self.per_layer_storage else None
            if cached_live is not None:
                seq_evict, keep, live_count_per_head, block_offsets = cached_live
            else:
                seq_evict = evict_mask[start:end]  # [seq_len, num_kv_heads]
                positions = torch.arange(seq_len, device=device, dtype=torch.int32)
                recent_threshold = max(0, seq_len - 1 - int(window_size))
                in_window = positions >= recent_threshold  # [seq_len]
                keep = (~seq_evict) | in_window.unsqueeze(1)  # [seq_len, num_kv_heads]
                live_count_per_head = keep.sum(dim=0, dtype=torch.int32)  # [num_kv_heads]
                block_offsets = None

            max_live = int(live_count_per_head.max().item())
            if max_live > state.max_capacity_per_lh:
                raise RuntimeError(
                    f"streaming pack live count {max_live} exceeds metadata capacity "
                    f"{state.max_capacity_per_lh} (layer {layer_id}, seq {seq_id})"
                )

            # J5: per-layer storage already allocated ranges in the pre-scan so
            # storage can grow to the true high-water offset before scattering.
            if not self.per_layer_storage:
                self._allocate_layer_ranges_for_seq(state, layer_id, live_count_per_head)

            # Reset this layer's metadata before scattering.
            state.token_positions[layer_id].fill_(-1)
            state.evict_mask[layer_id].fill_(False)

            if max_live == 0:
                state.live_counts[layer_id].zero_()
                continue

            if fused_rank_pack and block_offsets is not None:
                _streaming_pack_rank_scatter_triton(
                    k=k,
                    v=v,
                    seq_evict=seq_evict,
                    block_offsets=block_offsets,
                    compact_k=compact_k,
                    compact_v=compact_v,
                    token_positions=state.token_positions[layer_id],
                    evict_mask=state.evict_mask[layer_id],
                    base_offsets=state.base_offsets[layer_id],
                    start=start,
                    seq_len=seq_len,
                    window_size=int(window_size),
                )
                state.live_counts[layer_id].copy_(live_count_per_head)
                continue

            # Per-(token, head) rank via cumsum down the token axis.
            rank = keep.to(torch.int32).cumsum(dim=0) - 1  # [seq_len, num_kv_heads]
            if triton_pack and k.is_cuda and compact_k.is_cuda:
                _streaming_pack_scatter_triton(
                    k=k,
                    v=v,
                    keep=keep,
                    rank=rank,
                    seq_evict=seq_evict,
                    compact_k=compact_k,
                    compact_v=compact_v,
                    token_positions=state.token_positions[layer_id],
                    evict_mask=state.evict_mask[layer_id],
                    base_offsets=state.base_offsets[layer_id],
                    start=start,
                    seq_len=seq_len,
                )
                state.live_counts[layer_id].copy_(live_count_per_head)
                continue

            t_idx, h_idx = torch.nonzero(keep, as_tuple=True)
            t_long = t_idx.to(torch.long)
            h_long = h_idx.to(torch.long)
            dst_rank = rank[t_long, h_long].to(torch.long)
            base_per_lh = state.base_offsets[layer_id].to(torch.long)
            dst_slot = base_per_lh[h_long] + dst_rank

            src_k = k[start + t_long, h_long, :]
            src_v = v[start + t_long, h_long, :]
            if src_k.dtype != compact_dtype:
                src_k = src_k.to(compact_dtype)
                src_v = src_v.to(compact_dtype)
            # When per_layer_storage, compact_k/compact_v are 2D for the layer
            # (slot, head_dim). Otherwise they are 3D [layer, slot, head_dim].
            if self.per_layer_storage:
                compact_k[dst_slot] = src_k
                compact_v[dst_slot] = src_v
            else:
                compact_k[dst_slot] = src_k
                compact_v[dst_slot] = src_v

            state.token_positions[layer_id, h_long, dst_rank] = t_idx.to(torch.int32)
            state.evict_mask[layer_id, h_long, dst_rank] = seq_evict[t_long, h_long]
            state.live_counts[layer_id].copy_(live_count_per_head)

    def mark_live_metadata_all(
        self,
        seq_id: int,
        live_mask: torch.Tensor,
        dense_token_positions: torch.Tensor,
        evict_mask_full: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """D1.4 batched (layer, kv_head) metadata writer.

        Inputs:
          live_mask: [num_layers, num_kv_heads, packed_tokens] bool, the result
            of DMS / stride retention. True means token is kept.
          dense_token_positions: [packed_tokens] int32, the original positions.
          evict_mask_full (optional): [num_layers, num_kv_heads, packed_tokens]
            bool, per-token DMS decision aligned with live_mask.

        Returns the live_counts tensor [num_layers, num_kv_heads] int32.
        """
        state = self.seq_states[seq_id]
        num_layers, num_kv_heads, packed_tokens = live_mask.shape
        device = self.device
        # Per (layer, head) live counts.
        live_counts = live_mask.sum(dim=-1, dtype=torch.int32)
        max_live = int(live_counts.max().item())
        if max_live > state.max_capacity_per_lh:
            raise RuntimeError(
                f"compact KV metadata live count {max_live} exceeds metadata capacity "
                f"{state.max_capacity_per_lh}"
            )

        # Reset destinations: token_positions=-1, evict_mask=False
        state.token_positions.fill_(-1)
        state.evict_mask.fill_(False)

        if max_live == 0:
            return live_counts

        # Build dense kept-position tensor per (layer, head). The trick:
        # rank[i] = cumulative count of trues up to i in each (layer, head)
        # row. That gives the destination slot for each token.
        cum = live_mask.to(torch.int32).cumsum(dim=-1)
        # rank is 1-indexed where mask is True; convert to 0-indexed slot.
        rank = cum - 1
        # Build (layer, head, packed_tokens) meshgrids.
        layer_idx = torch.arange(num_layers, device=device).view(-1, 1, 1).expand(num_layers, num_kv_heads, packed_tokens)
        head_idx = torch.arange(num_kv_heads, device=device).view(1, -1, 1).expand(num_layers, num_kv_heads, packed_tokens)
        # Where live_mask is True, write dense_token_positions[token_idx] into
        # state.token_positions[layer_idx, head_idx, rank].
        if dense_token_positions.dtype != torch.int32:
            dense_token_positions = dense_token_positions.to(torch.int32)
        # Broadcast positions across [num_layers, num_kv_heads, packed_tokens]
        broadcast_positions = dense_token_positions.view(1, 1, packed_tokens).expand(num_layers, num_kv_heads, packed_tokens)
        l_flat = layer_idx[live_mask]
        h_flat = head_idx[live_mask]
        r_flat = rank[live_mask].to(torch.long)
        p_flat = broadcast_positions[live_mask]
        state.token_positions[l_flat, h_flat, r_flat] = p_flat
        if evict_mask_full is not None:
            state.evict_mask[l_flat, h_flat, r_flat] = evict_mask_full[live_mask]
        return live_counts

    def _contiguous_active_slot_start(self, states: list[CompactKVSequenceState]) -> int | None:
        if not self.layer_major_metadata or not states:
            return None
        slots = [state.active_slot for state in states]
        if any(slot is None for slot in slots):
            return None
        start = int(slots[0])
        expected = list(range(start, start + len(slots)))
        if [int(slot) for slot in slots] != expected:
            return None
        return start

    def apply_dms_evictions(
        self,
        seqs: list[Sequence],
        current_positions: list[int],
        *,
        window_size: int,
        compact_k: torch.Tensor | None = None,
        compact_v: torch.Tensor | None = None,
        sync_stats: bool = True,
    ) -> dict:
        # J2 Triton expiry path for per-layer storage. The default auto mode
        # uses it for c=1 and batched decode, where window-cadence expiry
        # removes substantially more work than the launch overhead adds; set
        # NANOVLLM_DMS_EXPIRY_TRITON=1 to force it or 0 to disable it.
        import os
        triton_mode = os.environ.get("NANOVLLM_DMS_EXPIRY_TRITON", "auto").strip().lower()
        use_triton = (
            self.per_layer_storage
            and len(seqs) > 0
            and (
                triton_mode in {"1", "true", "yes", "on"}
                or (triton_mode in {"", "auto"} and (len(seqs) == 1 or len(seqs) >= 4))
            )
        )
        if use_triton:
            states = [self.seq_states.get(seq.seq_id) for seq in seqs]
            if any(state is None for state in states):
                missing = [seq.seq_id for seq, state in zip(seqs, states) if state is None]
                raise RuntimeError(f"sequence(s) {missing} have no compact KV allocation")
            sequential = all(
                state.dms_initial_expiry_done
                and int(current_position) == state.last_dms_expiry_position + 1
                for state, current_position in zip(states, current_positions)
            )
            if sequential:
                # Batched auto decode is launch-bound here, and short expiry
                # intervals repeatedly compact tokens that attention could
                # safely ignore until the DMS retention window is crossed. Use
                # one full expiry per DMS window by default; keep forced Triton
                # exact unless the interval env override is explicitly set.
                interval_env = os.environ.get("NANOVLLM_DMS_EXPIRY_INTERVAL")
                auto_windowed = triton_mode in {"", "auto"} and (len(seqs) == 1 or len(seqs) >= 4)
                expiry_interval = int(interval_env) if interval_env is not None else (
                    max(1, int(window_size)) if auto_windowed else 1
                )
                if expiry_interval > 1:
                    should_expire = any(
                        int(current_position) % expiry_interval == 0
                        for current_position in current_positions
                    )
                    if not should_expire:
                        return self._skip_dms_evictions_triton(
                            states,
                            current_positions,
                            sync_stats=sync_stats,
                        )
                    return self._apply_dms_evictions_triton(
                        seqs,
                        current_positions,
                        window_size=window_size,
                        sync_stats=sync_stats,
                    )
                return self._apply_dms_evictions_triton_incremental(
                    seqs,
                    current_positions,
                    window_size=window_size,
                    sync_stats=sync_stats,
                )
        # Legacy single-tensor path:
        if not self.per_layer_storage:
            if compact_k is None:
                compact_k = self.compact_k
            if compact_v is None:
                compact_v = self.compact_v
            if compact_k is None or compact_v is None:
                raise RuntimeError("apply_dms_evictions needs compact storage attached or passed in")
        # E1.2: vectorized eviction. For each seq, compute keep mask across all
        # (layer, kv_head, slot), use cumsum-based rank to materialize a
        # compacted layout, then gather/scatter compact_k/v + state metadata
        # in a small fixed number of GPU ops.
        if len(seqs) != len(current_positions):
            raise ValueError("seq/current_positions length mismatch")
        device = self.device
        total_evicted_tensor = torch.zeros((), device=device, dtype=torch.int64)
        live_before_min_t = torch.full((), 1 << 30, device=device, dtype=torch.int64)
        live_before_max_t = torch.zeros((), device=device, dtype=torch.int64)
        live_after_min_t = torch.full((), 1 << 30, device=device, dtype=torch.int64)
        live_after_max_t = torch.zeros((), device=device, dtype=torch.int64)
        any_seq = False
        for seq, current_position in zip(seqs, current_positions):
            state = self.seq_states.get(seq.seq_id)
            if state is None:
                raise RuntimeError(f"sequence {seq.seq_id} has no compact KV allocation")
            any_seq = True
            capacity = state.max_capacity_per_lh
            # slot_idx broadcast for live mask
            slot_idx = torch.arange(capacity, device=device, dtype=torch.int32).view(1, 1, -1)
            live_mask = slot_idx < state.live_counts.unsqueeze(-1)  # [L, H, C]
            positions = state.token_positions
            evict = state.evict_mask
            # Eligibility: not yet evicted, or still inside the DMS window.
            within_window = (
                int(current_position) - positions.to(torch.long)
            ) <= int(window_size)
            # Use long cmp; positions may be -1 for empty slots — those get
            # masked off by live_mask anyway.
            new_keep = live_mask & ((~evict) | within_window)
            old_live = state.live_counts.to(torch.int64)
            new_live = new_keep.sum(dim=-1, dtype=torch.int32)  # [L, H]
            evicted_per_lh = old_live - new_live.to(torch.int64)  # [L, H]
            seq_evicted = evicted_per_lh.sum()
            total_evicted_tensor += seq_evicted

            live_before_min_t = torch.minimum(live_before_min_t, old_live.min())
            live_before_max_t = torch.maximum(live_before_max_t, old_live.max())
            live_after_min_t = torch.minimum(live_after_min_t, new_live.to(torch.int64).min())
            live_after_max_t = torch.maximum(live_after_max_t, new_live.to(torch.int64).max())

            # Rank assignment: where new_keep is True, dst_slot = cumsum-1
            rank = (new_keep.to(torch.int32).cumsum(dim=-1) - 1).clamp(min=0)
            # Build new layouts using gather/scatter via boolean mask
            l_flat, h_flat, src_flat = torch.nonzero(new_keep, as_tuple=True)
            l_long = l_flat.to(torch.long)
            h_long = h_flat.to(torch.long)
            src_long = src_flat.to(torch.long)
            dst_long = rank[l_long, h_long, src_long].to(torch.long)

            # K/V gather-then-scatter.
            # Source slot index = base[layer, h] + src_slot;
            # dst = base[layer, h] + dst_slot.
            base_per_lh = state.base_offsets.to(torch.long)
            seq_base_for_indices = base_per_lh[l_long, h_long]
            src_compact_idx = seq_base_for_indices + src_long
            dst_compact_idx = seq_base_for_indices + dst_long
            if self.per_layer_storage:
                # Per-layer separate tensors. Iterate per layer, gather/scatter
                # within that layer's tensor. Skip the .any().item() sync;
                # empty masks produce 0-length scatters which are cheap.
                for layer_id in range(self.num_layers):
                    ck_l = self.compact_k_per_layer[layer_id]
                    cv_l = self.compact_v_per_layer[layer_id]
                    if ck_l is None:
                        continue
                    layer_mask = (l_long == layer_id)
                    src_idx_l = src_compact_idx[layer_mask]
                    dst_idx_l = dst_compact_idx[layer_mask]
                    src_k_l = ck_l[src_idx_l].clone()
                    src_v_l = cv_l[src_idx_l].clone()
                    ck_l[dst_idx_l] = src_k_l
                    cv_l[dst_idx_l] = src_v_l
            else:
                src_k = compact_k[l_long, src_compact_idx].clone()
                src_v = compact_v[l_long, src_compact_idx].clone()
                compact_k[l_long, dst_compact_idx] = src_k
                compact_v[l_long, dst_compact_idx] = src_v

            new_token_positions = torch.full_like(positions, -1)
            new_evict_mask = torch.zeros_like(evict)
            new_token_positions[l_long, h_long, dst_long] = positions[l_long, h_long, src_long]
            new_evict_mask[l_long, h_long, dst_long] = evict[l_long, h_long, src_long]
            del dst_long
            state.token_positions.copy_(new_token_positions)
            state.evict_mask.copy_(new_evict_mask)
            state.live_counts.copy_(new_live)

        if not any_seq:
            return {
                "evicted_tokens": 0,
                "live_tokens_before_min": 0,
                "live_tokens_before_max": 0,
                "live_tokens_after_min": 0,
                "live_tokens_after_max": 0,
            }
        for seq, current_position in zip(seqs, current_positions):
            state = self.seq_states[seq.seq_id]
            state.dms_initial_expiry_done = True
            state.last_dms_expiry_position = int(current_position)

        stats = {
            "evicted_tokens": total_evicted_tensor,
            "live_tokens_before_min": live_before_min_t,
            "live_tokens_before_max": live_before_max_t,
            "live_tokens_after_min": live_after_min_t,
            "live_tokens_after_max": live_after_max_t,
        }
        if sync_stats:
            return {key: int(value.item()) for key, value in stats.items()}
        return stats

    def _skip_dms_evictions_triton(
        self,
        states: list[CompactKVSequenceState],
        current_positions: list[int],
        *,
        sync_stats: bool = True,
    ) -> dict:
        for state, current_position in zip(states, current_positions):
            state.last_dms_expiry_position = int(current_position)
        if not sync_stats:
            return {
                "evicted_tokens": 0,
                "live_tokens_before_min": 0,
                "live_tokens_before_max": 0,
                "live_tokens_after_min": 0,
                "live_tokens_after_max": 0,
            }
        device = self.device
        live_counts = torch.stack([state.live_counts for state in states], dim=0)
        live_min = live_counts.to(torch.int64).min()
        live_max = live_counts.to(torch.int64).max()
        zero = torch.zeros((), device=device, dtype=torch.int64)
        stats = {
            "evicted_tokens": zero,
            "live_tokens_before_min": live_min,
            "live_tokens_before_max": live_max,
            "live_tokens_after_min": live_min,
            "live_tokens_after_max": live_max,
        }
        if sync_stats:
            return {key: int(value.item()) for key, value in stats.items()}
        return stats

    def _apply_dms_evictions_triton(
        self,
        seqs: list[Sequence],
        current_positions: list[int],
        *,
        window_size: int,
        sync_stats: bool = True,
    ) -> dict:
        """J2 diagnostic full-expiry Triton kernel, one launch per layer.

        The correctness bug is fixed, but the full pass is slower than the
        Python vectorized path on the post-J5 per-layer layout. The production
        Triton path uses this only as retained diagnostic/profiling code and
        relies on the incremental kernel for steady-state batched decode.
        """
        from fastdms.layers.compact_attention import dms_expiry_one_layer
        from fastdms.utils.profiler import get_profiler

        device = self.device
        prof = get_profiler()
        states = []
        for seq in seqs:
            state = self.seq_states.get(seq.seq_id)
            if state is None:
                raise RuntimeError(f"sequence {seq.seq_id} has no compact KV allocation")
            states.append(state)
        live_before = torch.stack([s.live_counts for s in states], dim=0)
        live_before_min = live_before.to(torch.int64).min()
        live_before_max = live_before.to(torch.int64).max()
        cur_pos_t = torch.tensor(current_positions, device=device, dtype=torch.int64)
        active_slot_start = self._contiguous_active_slot_start(states)
        use_layer_major_views = active_slot_start is not None
        max_capacity = max(state.max_capacity_per_lh for state in states)
        for layer_id in range(self.num_layers):
            ck_l = self.compact_k_per_layer[layer_id]
            cv_l = self.compact_v_per_layer[layer_id]
            if ck_l is None:
                continue
            if use_layer_major_views:
                slot_slice = slice(active_slot_start, active_slot_start + len(states))
                positions_per_lh = self.layer_major_token_positions[
                    layer_id, slot_slice, :, :max_capacity
                ]
                evict_per_lh = self.layer_major_evict_mask[
                    layer_id, slot_slice, :, :max_capacity
                ]
                live_counts_layer = self.layer_major_live_counts[layer_id, slot_slice, :]
                base_offsets_layer = self.layer_major_base_offsets[layer_id, slot_slice, :]
            else:
                positions_per_lh = torch.stack([s.token_positions[layer_id] for s in states], dim=0).contiguous()
                evict_per_lh = torch.stack([s.evict_mask[layer_id] for s in states], dim=0).contiguous()
                live_counts_layer = torch.stack([s.live_counts[layer_id] for s in states], dim=0).contiguous()
                base_offsets_layer = torch.stack([s.base_offsets[layer_id] for s in states], dim=0).contiguous()
            with prof.cuda_phase("dms_decode_expire_full_layer"):
                dms_expiry_one_layer(
                    positions_per_lh=positions_per_lh,
                    evict_per_lh=evict_per_lh,
                    live_counts=live_counts_layer,
                    base_offsets=base_offsets_layer,
                    compact_k=ck_l,
                    compact_v=cv_l,
                    current_positions=cur_pos_t,
                    window_size=int(window_size),
                )
            if not use_layer_major_views:
                for batch_idx, state in enumerate(states):
                    state.token_positions[layer_id].copy_(positions_per_lh[batch_idx])
                    state.evict_mask[layer_id].copy_(evict_per_lh[batch_idx])
                    state.live_counts[layer_id].copy_(live_counts_layer[batch_idx])
        live_after = torch.stack([s.live_counts for s in states], dim=0)
        live_after_min = live_after.to(torch.int64).min()
        live_after_max = live_after.to(torch.int64).max()
        evicted = (live_before.to(torch.int64) - live_after.to(torch.int64)).sum()
        for state, current_position in zip(states, current_positions):
            state.dms_initial_expiry_done = True
            state.last_dms_expiry_position = int(current_position)
        stats = {
            "evicted_tokens": evicted,
            "live_tokens_before_min": live_before_min,
            "live_tokens_before_max": live_before_max,
            "live_tokens_after_min": live_after_min,
            "live_tokens_after_max": live_after_max,
        }
        if sync_stats:
            return {key: int(value.item()) for key, value in stats.items()}
        return stats

    def _apply_dms_evictions_triton_incremental(
        self,
        seqs: list[Sequence],
        current_positions: list[int],
        *,
        window_size: int,
        sync_stats: bool = True,
    ) -> dict:
        from fastdms.layers.compact_attention import dms_expiry_incremental_one_layer
        from fastdms.utils.profiler import get_profiler

        device = self.device
        prof = get_profiler()
        states = []
        for seq in seqs:
            state = self.seq_states.get(seq.seq_id)
            if state is None:
                raise RuntimeError(f"sequence {seq.seq_id} has no compact KV allocation")
            states.append(state)
        live_before = torch.stack([s.live_counts for s in states], dim=0)
        live_before_min = live_before.to(torch.int64).min()
        live_before_max = live_before.to(torch.int64).max()
        cur_pos_t = torch.tensor(current_positions, device=device, dtype=torch.int64)
        active_slot_start = self._contiguous_active_slot_start(states)
        use_layer_major_views = active_slot_start is not None
        max_capacity = max(state.max_capacity_per_lh for state in states)
        for layer_id in range(self.num_layers):
            ck_l = self.compact_k_per_layer[layer_id]
            cv_l = self.compact_v_per_layer[layer_id]
            if ck_l is None:
                continue
            if use_layer_major_views:
                slot_slice = slice(active_slot_start, active_slot_start + len(states))
                positions_per_lh = self.layer_major_token_positions[
                    layer_id, slot_slice, :, :max_capacity
                ]
                evict_per_lh = self.layer_major_evict_mask[
                    layer_id, slot_slice, :, :max_capacity
                ]
                live_counts_layer = self.layer_major_live_counts[layer_id, slot_slice, :]
                base_offsets_layer = self.layer_major_base_offsets[layer_id, slot_slice, :]
            else:
                positions_per_lh = torch.stack([s.token_positions[layer_id] for s in states], dim=0).contiguous()
                evict_per_lh = torch.stack([s.evict_mask[layer_id] for s in states], dim=0).contiguous()
                live_counts_layer = torch.stack([s.live_counts[layer_id] for s in states], dim=0).contiguous()
                base_offsets_layer = torch.stack([s.base_offsets[layer_id] for s in states], dim=0).contiguous()
            with prof.cuda_phase("dms_decode_expire_incremental_layer"):
                dms_expiry_incremental_one_layer(
                    positions_per_lh=positions_per_lh,
                    evict_per_lh=evict_per_lh,
                    live_counts=live_counts_layer,
                    base_offsets=base_offsets_layer,
                    compact_k=ck_l,
                    compact_v=cv_l,
                    current_positions=cur_pos_t,
                    window_size=int(window_size),
                )
            if not use_layer_major_views:
                for batch_idx, state in enumerate(states):
                    state.token_positions[layer_id].copy_(positions_per_lh[batch_idx])
                    state.evict_mask[layer_id].copy_(evict_per_lh[batch_idx])
                    state.live_counts[layer_id].copy_(live_counts_layer[batch_idx])
        live_after = torch.stack([s.live_counts for s in states], dim=0)
        live_after_min = live_after.to(torch.int64).min()
        live_after_max = live_after.to(torch.int64).max()
        evicted = (live_before.to(torch.int64) - live_after.to(torch.int64)).sum()
        for state, current_position in zip(states, current_positions):
            state.last_dms_expiry_position = int(current_position)
        stats = {
            "evicted_tokens": evicted,
            "live_tokens_before_min": live_before_min,
            "live_tokens_before_max": live_before_max,
            "live_tokens_after_min": live_after_min,
            "live_tokens_after_max": live_after_max,
        }
        if sync_stats:
            return {key: int(value.item()) for key, value in stats.items()}
        return stats

    def record_appended_dms_decisions(
        self,
        seqs: list[Sequence],
        current_positions: list[int] | torch.Tensor,
        layer_evictions: dict[int, torch.Tensor] | torch.Tensor,
        *,
        sync_stats: bool = True,
    ) -> dict:
        # D1.5: vectorized decode decision recording. Single GPU-resident
        # scatter writes (layer, batch, kv_head, position) triples in one go,
        # eliminating the per-(layer, seq, head) .item() syncs that dominated
        # decode wallclock.
        if len(seqs) != len(current_positions):
            raise ValueError("seq/current_positions length mismatch")
        batch = len(seqs)
        if batch == 0:
            return {"eviction_decisions_true": 0, "eviction_decisions_total": 0}

        device = self.device
        expected_all_shape = (self.num_layers, batch, self.num_kv_heads)
        if isinstance(layer_evictions, torch.Tensor):
            if tuple(layer_evictions.shape) != expected_all_shape:
                raise RuntimeError(
                    f"expected DMS decode decisions {expected_all_shape}, got {tuple(layer_evictions.shape)}"
                )
            evict_all = layer_evictions
            if evict_all.device != device or evict_all.dtype != torch.bool:
                evict_all = evict_all.to(device=device, dtype=torch.bool)
        else:
            # Stack layer_evictions to [num_layers, batch, num_kv_heads]
            evict_layers = []
            for layer_id in range(self.num_layers):
                evict = layer_evictions.get(layer_id)
                if evict is None:
                    raise RuntimeError(f"missing DMS decode eviction decisions for layer {layer_id}")
                expected_shape = (batch, self.num_kv_heads)
                if tuple(evict.shape) != expected_shape:
                    raise RuntimeError(
                        f"expected DMS decode decisions {expected_shape}, got {tuple(evict.shape)}"
                    )
                evict_layers.append(evict)
            evict_all = torch.stack(evict_layers, dim=0).to(device=device, dtype=torch.bool)

        if isinstance(current_positions, torch.Tensor):
            if current_positions.numel() != batch:
                raise ValueError("current_positions tensor length mismatch")
            positions_t = current_positions
            if positions_t.device != device:
                positions_t = positions_t.to(device=device, non_blocking=True)
            if positions_t.dtype not in (torch.int32, torch.int64):
                positions_t = positions_t.to(dtype=torch.int32)
            if positions_t.stride(0) != 1:
                positions_t = positions_t.contiguous()
        else:
            positions_t = torch.tensor(current_positions, device=device, dtype=torch.int32)

        states = []
        for batch_idx, seq in enumerate(seqs):
            state = self.seq_states.get(seq.seq_id)
            if state is None:
                raise RuntimeError(f"sequence {seq.seq_id} has no compact KV allocation")
            states.append(state)
        active_slot_start = self._contiguous_active_slot_start(states)
        if active_slot_start is not None:
            slot_slice = slice(active_slot_start, active_slot_start + batch)
            live = self.layer_major_live_counts[:, slot_slice, :]
            if self.debug_checks and bool(torch.any(live <= 0).item()):
                raise RuntimeError("at least one active (layer, slot, head) has no append slot")
            if evict_all.is_cuda and self.layer_major_token_positions.is_cuda:
                total = self.num_layers * batch * self.num_kv_heads
                block = 256
                _record_dms_decisions_layer_major_kernel[(triton.cdiv(total, block),)](
                    self.layer_major_token_positions,
                    self.layer_major_evict_mask,
                    self.layer_major_live_counts,
                    evict_all,
                    positions_t,
                    int(active_slot_start),
                    total,
                    batch,
                    self.num_kv_heads,
                    self.layer_major_token_positions.stride(0),
                    self.layer_major_token_positions.stride(1),
                    self.layer_major_token_positions.stride(2),
                    self.layer_major_evict_mask.stride(0),
                    self.layer_major_evict_mask.stride(1),
                    self.layer_major_evict_mask.stride(2),
                    self.layer_major_live_counts.stride(0),
                    self.layer_major_live_counts.stride(1),
                    self.layer_major_live_counts.stride(2),
                    evict_all.stride(0),
                    evict_all.stride(1),
                    evict_all.stride(2),
                    block,
                )
            else:
                layer_grid, head_grid = torch.meshgrid(
                    torch.arange(self.num_layers, device=device),
                    torch.arange(self.num_kv_heads, device=device),
                    indexing="ij",
                )
                idx = (live - 1).to(torch.long)
                batch_grid = torch.arange(batch, device=device).view(1, -1, 1).expand(
                    self.num_layers,
                    batch,
                    self.num_kv_heads,
                )
                layer_grid_3d = layer_grid.view(self.num_layers, 1, self.num_kv_heads).expand(
                    self.num_layers,
                    batch,
                    self.num_kv_heads,
                )
                head_grid_3d = head_grid.view(self.num_layers, 1, self.num_kv_heads).expand(
                    self.num_layers,
                    batch,
                    self.num_kv_heads,
                )
                self.layer_major_token_positions[
                    layer_grid_3d,
                    batch_grid + active_slot_start,
                    head_grid_3d,
                    idx,
                ] = positions_t.to(dtype=torch.int32).view(1, batch, 1)
                self.layer_major_evict_mask[
                    layer_grid_3d,
                    batch_grid + active_slot_start,
                    head_grid_3d,
                    idx,
                ] = evict_all
            total = self.num_layers * batch * self.num_kv_heads
            total_true = evict_all.sum()
            if sync_stats:
                total_true = int(total_true.item())
            return {
                "eviction_decisions_true": total_true,
                "eviction_decisions_total": total,
            }

        layer_grid, head_grid = torch.meshgrid(
            torch.arange(self.num_layers, device=device),
            torch.arange(self.num_kv_heads, device=device),
            indexing="ij",
        )
        for batch_idx, seq in enumerate(seqs):
            state = states[batch_idx]
            # live_counts shape [num_layers, num_kv_heads]; idx = live - 1.
            live = state.live_counts
            if self.debug_checks and bool(torch.any(live <= 0).item()):
                raise RuntimeError(
                    f"sequence {seq.seq_id} has at least one (layer, head) without an append slot"
                )
            idx = (live - 1).to(torch.long)
            # Scatter token_positions[layer, head, idx[layer, head]] = current_position
            state.token_positions[layer_grid, head_grid, idx] = positions_t[batch_idx]
            state.evict_mask[layer_grid, head_grid, idx] = evict_all[:, batch_idx, :]

        total = self.num_layers * batch * self.num_kv_heads
        total_true = evict_all.sum()
        if sync_stats:
            total_true = int(total_true.item())
        return {
            "eviction_decisions_true": total_true,
            "eviction_decisions_total": total,
        }

    def prepare_decode_metadata(
        self,
        seqs: list[Sequence],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        # J3+J5: pure on-device metadata prep with per-(seq, head, layer)
        # base offsets. max_live_count is a fixed constexpr equal to per-seq
        # max_capacity_per_lh; the kernel masks invalid slots so padding is
        # safe. No .item() syncs.
        batch = len(seqs)
        device = self.device
        base_offsets = torch.empty(
            (self.num_layers, batch, self.num_kv_heads),
            device=device,
            dtype=torch.int32,
        )
        live_counts = torch.empty_like(base_offsets)
        slot_mapping = torch.empty_like(base_offsets)

        max_capacity = 0
        states = []
        for seq in seqs:
            state = self.seq_states.get(seq.seq_id)
            if state is None:
                raise RuntimeError(f"sequence {seq.seq_id} has no compact KV allocation")
            states.append(state)
        active_slot_start = self._contiguous_active_slot_start(states)
        if active_slot_start is not None:
            slot_slice = slice(active_slot_start, active_slot_start + batch)
            base_offsets = self.layer_major_base_offsets[:, slot_slice, :]
            live_counts = self.layer_major_live_counts[:, slot_slice, :]
            slot_mapping = base_offsets + live_counts
            live_counts.add_(1)
            max_capacity = max(state.max_capacity_per_lh for state in states)
            return base_offsets, live_counts, slot_mapping, max_capacity

        for batch_idx, state in enumerate(states):
            old_live_counts = state.live_counts
            # base_offsets is now per-(layer, head) — copy directly.
            base_offsets[:, batch_idx, :] = state.base_offsets
            slot_mapping[:, batch_idx, :] = state.base_offsets + old_live_counts
            state.live_counts += 1
            live_counts[:, batch_idx, :] = state.live_counts
            max_capacity = max(max_capacity, state.max_capacity_per_lh)
        return base_offsets, live_counts, slot_mapping, max_capacity

    def free(self, seq_id: int) -> None:
        state = self.seq_states.pop(seq_id, None)
        if state is None:
            return
        # J5: per-(layer, head) ranges. Free each one in its layer's pool.
        bases = state.base_offsets.cpu().tolist()
        sizes = state.range_capacity.cpu().tolist()
        for layer_id in range(self.num_layers):
            for h in range(self.num_kv_heads):
                base = int(bases[layer_id][h])
                size = int(sizes[layer_id][h])
                if base >= 0 and size > 0:
                    self._free_range(layer_id, base, size)
        if state.active_slot is not None:
            slot = int(state.active_slot)
            self.active_slot_by_seq.pop(seq_id, None)
            if self.layer_major_base_offsets is not None:
                self.layer_major_base_offsets[:, slot, :].fill_(-1)
                self.layer_major_range_capacity[:, slot, :].zero_()
                self.layer_major_live_counts[:, slot, :].zero_()
                self.layer_major_token_positions[:, slot, :, :].fill_(-1)
                self.layer_major_evict_mask[:, slot, :, :].zero_()
            self.free_active_slots.append(slot)
            self.free_active_slots.sort()


def pack_compact_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    live_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, CompactKVMetadata]:
    """Pack dense per-request K/V into flat compact per-(request, kv-head) spans.

    Args:
        k: Dense keys shaped [batch, tokens, kv_heads, head_dim].
        v: Dense values shaped [batch, tokens, kv_heads, head_dim].
        live_mask: Boolean retention mask shaped [batch, kv_heads, tokens].

    Returns:
        Flat compact K/V shaped [total_live, head_dim] plus base/live metadata.
    """
    if k.shape != v.shape:
        raise ValueError(f"k/v shape mismatch: {k.shape} != {v.shape}")
    if k.ndim != 4:
        raise ValueError(f"expected k/v as [batch, tokens, kv_heads, head_dim], got {k.shape}")
    batch, tokens, kv_heads, head_dim = k.shape
    if live_mask.shape != (batch, kv_heads, tokens):
        raise ValueError(
            f"expected live_mask {(batch, kv_heads, tokens)}, got {tuple(live_mask.shape)}"
        )
    if live_mask.device != k.device:
        live_mask = live_mask.to(device=k.device)

    live_counts = live_mask.sum(dim=-1, dtype=torch.int32)
    total_live = int(live_counts.sum().item())
    compact_k = torch.empty((total_live, head_dim), device=k.device, dtype=k.dtype)
    compact_v = torch.empty((total_live, head_dim), device=v.device, dtype=v.dtype)
    base_offsets = torch.empty((batch, kv_heads), device=k.device, dtype=torch.int32)
    token_positions = torch.empty((total_live,), device=k.device, dtype=torch.int32)

    cursor = 0
    for batch_idx in range(batch):
        for kv_head in range(kv_heads):
            keep = torch.nonzero(live_mask[batch_idx, kv_head], as_tuple=False).flatten()
            live = int(keep.numel())
            base_offsets[batch_idx, kv_head] = cursor
            if live:
                end = cursor + live
                compact_k[cursor:end] = k[batch_idx, keep, kv_head]
                compact_v[cursor:end] = v[batch_idx, keep, kv_head]
                token_positions[cursor:end] = keep.to(dtype=torch.int32)
                cursor = end

    metadata = CompactKVMetadata(
        base_offsets=base_offsets,
        live_counts=live_counts.contiguous(),
        token_positions=token_positions,
    )
    return compact_k, compact_v, metadata
