import os

import torch
import triton
import triton.language as tl

_COMPACT_DEBUG_CHECKS = os.environ.get("NANOVLLM_COMPACT_DEBUG_CHECKS") == "1"
_COMPACT_ATTN_SPLITK = os.environ.get("NANOVLLM_COMPACT_ATTN_SPLITK", "auto").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
_COMPACT_ATTN_SPLITK_DEFAULT_BLOCK_N = 256
_COMPACT_ATTN_SPLITK_BLOCK_N_ENV = os.environ.get("NANOVLLM_COMPACT_ATTN_SPLITK_BLOCK_N")


def _parse_block_n_override(raw: str | None) -> int | None:
    if raw is None or raw.strip() == "":
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"NANOVLLM_COMPACT_ATTN_SPLITK_BLOCK_N must be an integer, got {raw!r}") from exc
    if value not in {16, 32, 64, 128, 256, 512}:
        raise ValueError(
            "NANOVLLM_COMPACT_ATTN_SPLITK_BLOCK_N must be one of "
            f"16, 32, 64, 128, 256, 512; got {value}"
        )
    return value


_COMPACT_ATTN_SPLITK_BLOCK_N = _parse_block_n_override(_COMPACT_ATTN_SPLITK_BLOCK_N_ENV)
_DMS_FUSED_DECODE_PREPROCESS = os.environ.get(
    "NANOVLLM_DMS_FUSED_DECODE_PREPROCESS", "1"
).strip().lower() not in {"0", "false", "no", "off"}
_COMPACT_ATTN_INLINE_Q_ROPE = os.environ.get(
    "NANOVLLM_COMPACT_ATTN_INLINE_Q_ROPE", "1"
).strip().lower() in {"1", "true", "yes", "on"}


def compact_attention_splitk_enabled() -> bool:
    return _COMPACT_ATTN_SPLITK


def compact_attention_splitk_block_n() -> int:
    return _COMPACT_ATTN_SPLITK_BLOCK_N or _COMPACT_ATTN_SPLITK_DEFAULT_BLOCK_N


def dms_fused_decode_preprocess_enabled() -> bool:
    return _DMS_FUSED_DECODE_PREPROCESS


def dms_decode_store_transient_k_enabled() -> bool:
    return os.environ.get("NANOVLLM_DMS_DECODE_STORE_TRANSIENT_K", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def compact_attention_inline_q_rope_enabled() -> bool:
    return _COMPACT_ATTN_INLINE_Q_ROPE


@triton.jit
def _dms_rope_store_compact_decode_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    cos_sin_ptr,
    compact_k_ptr,
    compact_v_ptr,
    slot_mapping_ptr,
    evict_out_ptr,
    positions_ptr,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    k_stride_b: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_d: tl.constexpr,
    v_stride_b: tl.constexpr,
    v_stride_h: tl.constexpr,
    v_stride_d: tl.constexpr,
    cos_sin_stride_p: tl.constexpr,
    cos_sin_stride_d: tl.constexpr,
    compact_stride_s: tl.constexpr,
    compact_stride_d: tl.constexpr,
    slot_stride_b: tl.constexpr,
    slot_stride_h: tl.constexpr,
    evict_stride_b: tl.constexpr,
    evict_stride_h: tl.constexpr,
    num_kv_heads: tl.constexpr,
    group_size: tl.constexpr,
    head_dim: tl.constexpr,
    alpha_scale: tl.constexpr,
    alpha_offset: tl.constexpr,
    positive_scale: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
    STORE_Q: tl.constexpr,
    STORE_TRANSIENT_K: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // num_kv_heads
    kv_head = pid - batch_idx * num_kv_heads
    q_head_base = kv_head * group_size

    half_dim = head_dim // 2
    offsets_g = tl.arange(0, BLOCK_G)
    offsets_d = tl.arange(0, BLOCK_HALF)
    dim_mask = offsets_d < half_dim
    group_mask = offsets_g < group_size
    position = tl.load(positions_ptr + batch_idx).to(tl.int64)
    cos = tl.load(
        cos_sin_ptr + position * cos_sin_stride_p + offsets_d * cos_sin_stride_d,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)
    sin = tl.load(
        cos_sin_ptr + position * cos_sin_stride_p + (offsets_d + half_dim) * cos_sin_stride_d,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)

    decision_value = tl.load(
        q_ptr
        + batch_idx * q_stride_b
        + q_head_base * q_stride_h
        + (head_dim - 1) * q_stride_d,
    ).to(tl.float32)
    if STORE_Q:
        q_heads = q_head_base + offsets_g
        q_base = q_ptr + batch_idx * q_stride_b + q_heads[:, None] * q_stride_h
        q1 = tl.load(
            q_base + offsets_d[None, :] * q_stride_d,
            mask=group_mask[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        q2 = tl.load(
            q_base + (offsets_d[None, :] + half_dim) * q_stride_d,
            mask=group_mask[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        zero_decision_lane = (offsets_g[:, None] == 0) & (offsets_d[None, :] == (half_dim - 1))
        q2_clean = tl.where(zero_decision_lane, 0.0, q2)
        q_rot1 = q1 * cos[None, :] - q2_clean * sin[None, :]
        q_rot2 = q2_clean * cos[None, :] + q1 * sin[None, :]
        tl.store(
            q_base + offsets_d[None, :] * q_stride_d,
            q_rot1,
            mask=group_mask[:, None] & dim_mask[None, :],
        )
        tl.store(
            q_base + (offsets_d[None, :] + half_dim) * q_stride_d,
            q_rot2,
            mask=group_mask[:, None] & dim_mask[None, :],
        )

    if positive_scale:
        evict = decision_value > (alpha_offset / alpha_scale)
    else:
        evict = (decision_value * alpha_scale - alpha_offset) > 0.0
    tl.store(evict_out_ptr + batch_idx * evict_stride_b + kv_head * evict_stride_h, evict)

    k_base = k_ptr + batch_idx * k_stride_b + kv_head * k_stride_h
    k1 = tl.load(
        k_base + offsets_d * k_stride_d,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)
    k2 = tl.load(
        k_base + (offsets_d + half_dim) * k_stride_d,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)
    k_rot1 = k1 * cos - k2 * sin
    k_rot2 = k2 * cos + k1 * sin
    if STORE_TRANSIENT_K:
        tl.store(k_base + offsets_d * k_stride_d, k_rot1, mask=dim_mask)
        tl.store(k_base + (offsets_d + half_dim) * k_stride_d, k_rot2, mask=dim_mask)

    slot = tl.load(slot_mapping_ptr + batch_idx * slot_stride_b + kv_head * slot_stride_h)
    compact_base = slot.to(tl.int64) * compact_stride_s
    tl.store(
        compact_k_ptr + compact_base + offsets_d * compact_stride_d,
        k_rot1,
        mask=(slot >= 0) & dim_mask,
    )
    tl.store(
        compact_k_ptr + compact_base + (offsets_d + half_dim) * compact_stride_d,
        k_rot2,
        mask=(slot >= 0) & dim_mask,
    )
    v_base = v_ptr + batch_idx * v_stride_b + kv_head * v_stride_h
    v1 = tl.load(v_base + offsets_d * v_stride_d, mask=dim_mask, other=0.0)
    v2 = tl.load(v_base + (offsets_d + half_dim) * v_stride_d, mask=dim_mask, other=0.0)
    tl.store(
        compact_v_ptr + compact_base + offsets_d * compact_stride_d,
        v1,
        mask=(slot >= 0) & dim_mask,
    )
    tl.store(
        compact_v_ptr + compact_base + (offsets_d + half_dim) * compact_stride_d,
        v2,
        mask=(slot >= 0) & dim_mask,
    )


@triton.jit
def _store_compact_kvcache_kernel(
    key_ptr,
    key_stride_b: tl.constexpr,
    key_stride_h: tl.constexpr,
    key_stride_d: tl.constexpr,
    value_ptr,
    value_stride_b: tl.constexpr,
    value_stride_h: tl.constexpr,
    value_stride_d: tl.constexpr,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // num_kv_heads
    kv_head = pid - batch_idx * num_kv_heads
    slot = tl.load(slot_mapping_ptr + batch_idx * num_kv_heads + kv_head)
    offsets_d = tl.arange(0, BLOCK_D)

    key = tl.load(
        key_ptr + batch_idx * key_stride_b + kv_head * key_stride_h + offsets_d * key_stride_d,
        mask=(slot >= 0) & (offsets_d < head_dim),
        other=0.0,
    )
    value = tl.load(
        value_ptr + batch_idx * value_stride_b + kv_head * value_stride_h + offsets_d * value_stride_d,
        mask=(slot >= 0) & (offsets_d < head_dim),
        other=0.0,
    )
    tl.store(k_cache_ptr + slot * head_dim + offsets_d, key, mask=(slot >= 0) & (offsets_d < head_dim))
    tl.store(v_cache_ptr + slot * head_dim + offsets_d, value, mask=(slot >= 0) & (offsets_d < head_dim))


@triton.jit
def _compact_decode_grouped_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    base_offsets_ptr,
    live_counts_ptr,
    scale: tl.constexpr,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    kv_stride_s: tl.constexpr,
    kv_stride_d: tl.constexpr,
    out_stride_b: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_d: tl.constexpr,
    num_kv_heads: tl.constexpr,
    group_size: tl.constexpr,
    head_dim: tl.constexpr,
    max_live_count: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // num_kv_heads
    kv_head = pid - batch_idx * num_kv_heads
    q_head_base = kv_head * group_size

    offsets_g = tl.arange(0, BLOCK_G)
    offsets_d = tl.arange(0, BLOCK_D)
    q = tl.load(
        q_ptr
        + batch_idx * q_stride_b
        + (q_head_base + offsets_g[:, None]) * q_stride_h
        + offsets_d[None, :] * q_stride_d,
        mask=(offsets_g[:, None] < group_size) & (offsets_d[None, :] < head_dim),
        other=0.0,
    ).to(tl.float32)

    meta_offset = batch_idx * num_kv_heads + kv_head
    base = tl.load(base_offsets_ptr + meta_offset)
    live_count = tl.load(live_counts_ptr + meta_offset)

    m_i = tl.full((BLOCK_G,), -float("inf"), tl.float32)
    l_i = tl.full((BLOCK_G,), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_G, BLOCK_D), tl.float32)

    for start in range(0, max_live_count, BLOCK_N):
        offsets_n = start + tl.arange(0, BLOCK_N)
        live_mask = offsets_n < live_count
        k = tl.load(
            k_ptr + (base + offsets_n[:, None]) * kv_stride_s + offsets_d[None, :] * kv_stride_d,
            mask=live_mask[:, None] & (offsets_d[None, :] < head_dim),
            other=0.0,
        ).to(tl.float32)
        scores = tl.dot(k, tl.trans(q)) * scale
        scores = tl.where(live_mask[:, None], scores, -float("inf"))

        m_next = tl.maximum(m_i, tl.max(scores, axis=0))
        p = tl.exp(scores - m_next[None, :])
        alpha = tl.exp(m_i - m_next)
        l_next = l_i * alpha + tl.sum(p, axis=0)

        values = tl.load(
            v_ptr + (base + offsets_n[:, None]) * kv_stride_s + offsets_d[None, :] * kv_stride_d,
            mask=live_mask[:, None] & (offsets_d[None, :] < head_dim),
            other=0.0,
        ).to(tl.float32)
        acc = acc * alpha[:, None] + tl.dot(tl.trans(p), values)
        m_i = m_next
        l_i = l_next

    acc = acc / l_i[:, None]
    tl.store(
        out_ptr
        + batch_idx * out_stride_b
        + (q_head_base + offsets_g[:, None]) * out_stride_h
        + offsets_d[None, :] * out_stride_d,
        acc,
        mask=(offsets_g[:, None] < group_size) & (offsets_d[None, :] < head_dim),
    )


@triton.jit
def _compact_decode_grouped_splitk_partial_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    partial_m_ptr,
    partial_l_ptr,
    partial_acc_ptr,
    base_offsets_ptr,
    live_counts_ptr,
    scale: tl.constexpr,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    kv_stride_s: tl.constexpr,
    kv_stride_d: tl.constexpr,
    num_kv_heads: tl.constexpr,
    group_size: tl.constexpr,
    head_dim: tl.constexpr,
    num_splits: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    split_id = pid % num_splits
    kv_program = pid // num_splits
    batch_idx = kv_program // num_kv_heads
    kv_head = kv_program - batch_idx * num_kv_heads
    q_head_base = kv_head * group_size

    offsets_g = tl.arange(0, BLOCK_G)
    offsets_d = tl.arange(0, BLOCK_D)
    q = tl.load(
        q_ptr
        + batch_idx * q_stride_b
        + (q_head_base + offsets_g[:, None]) * q_stride_h
        + offsets_d[None, :] * q_stride_d,
        mask=(offsets_g[:, None] < group_size) & (offsets_d[None, :] < head_dim),
        other=0.0,
    ).to(tl.float32)

    meta_offset = batch_idx * num_kv_heads + kv_head
    base = tl.load(base_offsets_ptr + meta_offset)
    live_count = tl.load(live_counts_ptr + meta_offset)

    offsets_n = split_id * BLOCK_N + tl.arange(0, BLOCK_N)
    live_mask = offsets_n < live_count
    has_live = split_id * BLOCK_N < live_count
    k = tl.load(
        k_ptr + (base + offsets_n[:, None]) * kv_stride_s + offsets_d[None, :] * kv_stride_d,
        mask=live_mask[:, None] & (offsets_d[None, :] < head_dim),
        other=0.0,
    ).to(tl.float32)
    scores = tl.dot(k, tl.trans(q)) * scale
    scores = tl.where(live_mask[:, None], scores, -float("inf"))
    m = tl.max(scores, axis=0)
    m_safe = tl.where(has_live, m, 0.0)
    p = tl.exp(scores - m_safe[None, :])
    l = tl.sum(p, axis=0)

    values = tl.load(
        v_ptr + (base + offsets_n[:, None]) * kv_stride_s + offsets_d[None, :] * kv_stride_d,
        mask=live_mask[:, None] & (offsets_d[None, :] < head_dim),
        other=0.0,
    ).to(tl.float32)
    acc = tl.dot(tl.trans(p), values)

    base_ml = ((batch_idx * num_kv_heads + kv_head) * num_splits + split_id) * BLOCK_G
    tl.store(
        partial_m_ptr + base_ml + offsets_g,
        tl.where(has_live, m, -float("inf")),
        mask=offsets_g < group_size,
    )
    tl.store(partial_l_ptr + base_ml + offsets_g, l, mask=offsets_g < group_size)

    acc_base = (((batch_idx * num_kv_heads + kv_head) * num_splits + split_id) * BLOCK_G) * BLOCK_D
    tl.store(
        partial_acc_ptr + acc_base + offsets_g[:, None] * BLOCK_D + offsets_d[None, :],
        acc,
        mask=(offsets_g[:, None] < group_size) & (offsets_d[None, :] < head_dim),
    )


@triton.jit
def _compact_decode_grouped_splitk_partial_qrope_kernel(
    q_ptr,
    cos_sin_ptr,
    positions_ptr,
    k_ptr,
    v_ptr,
    partial_m_ptr,
    partial_l_ptr,
    partial_acc_ptr,
    base_offsets_ptr,
    live_counts_ptr,
    scale: tl.constexpr,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    cos_sin_stride_p: tl.constexpr,
    cos_sin_stride_d: tl.constexpr,
    kv_stride_s: tl.constexpr,
    kv_stride_d: tl.constexpr,
    num_kv_heads: tl.constexpr,
    group_size: tl.constexpr,
    head_dim: tl.constexpr,
    num_splits: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
):
    pid = tl.program_id(0)
    split_id = pid % num_splits
    kv_program = pid // num_splits
    batch_idx = kv_program // num_kv_heads
    kv_head = kv_program - batch_idx * num_kv_heads
    q_head_base = kv_head * group_size

    half_dim = head_dim // 2
    offsets_g = tl.arange(0, BLOCK_G)
    offsets_h = tl.arange(0, BLOCK_HALF)
    offsets_d = tl.arange(0, BLOCK_D)
    group_mask = offsets_g < group_size
    half_mask = offsets_h < half_dim

    position = tl.load(positions_ptr + batch_idx).to(tl.int64)
    cos = tl.load(
        cos_sin_ptr + position * cos_sin_stride_p + offsets_h * cos_sin_stride_d,
        mask=half_mask,
        other=0.0,
    ).to(tl.float32)
    sin = tl.load(
        cos_sin_ptr + position * cos_sin_stride_p + (offsets_h + half_dim) * cos_sin_stride_d,
        mask=half_mask,
        other=0.0,
    ).to(tl.float32)

    q_heads = q_head_base + offsets_g
    q_base = q_ptr + batch_idx * q_stride_b + q_heads[:, None] * q_stride_h
    q1 = tl.load(
        q_base + offsets_h[None, :] * q_stride_d,
        mask=group_mask[:, None] & half_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    q2 = tl.load(
        q_base + (offsets_h[None, :] + half_dim) * q_stride_d,
        mask=group_mask[:, None] & half_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    zero_decision_lane = (offsets_g[:, None] == 0) & (offsets_h[None, :] == (half_dim - 1))
    q2_clean = tl.where(zero_decision_lane, 0.0, q2)
    q_rot1 = q1 * cos[None, :] - q2_clean * sin[None, :]
    q_rot2 = q2_clean * cos[None, :] + q1 * sin[None, :]

    meta_offset = batch_idx * num_kv_heads + kv_head
    base = tl.load(base_offsets_ptr + meta_offset)
    live_count = tl.load(live_counts_ptr + meta_offset)

    offsets_n = split_id * BLOCK_N + tl.arange(0, BLOCK_N)
    live_mask = offsets_n < live_count
    has_live = split_id * BLOCK_N < live_count
    k1 = tl.load(
        k_ptr + (base + offsets_n[:, None]) * kv_stride_s + offsets_h[None, :] * kv_stride_d,
        mask=live_mask[:, None] & half_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    k2 = tl.load(
        k_ptr + (base + offsets_n[:, None]) * kv_stride_s + (offsets_h[None, :] + half_dim) * kv_stride_d,
        mask=live_mask[:, None] & half_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    scores = (tl.dot(k1, tl.trans(q_rot1)) + tl.dot(k2, tl.trans(q_rot2))) * scale
    scores = tl.where(live_mask[:, None], scores, -float("inf"))
    m = tl.max(scores, axis=0)
    m_safe = tl.where(has_live, m, 0.0)
    p = tl.exp(scores - m_safe[None, :])
    l = tl.sum(p, axis=0)

    values = tl.load(
        v_ptr + (base + offsets_n[:, None]) * kv_stride_s + offsets_d[None, :] * kv_stride_d,
        mask=live_mask[:, None] & (offsets_d[None, :] < head_dim),
        other=0.0,
    ).to(tl.float32)
    acc = tl.dot(tl.trans(p), values)

    base_ml = ((batch_idx * num_kv_heads + kv_head) * num_splits + split_id) * BLOCK_G
    tl.store(
        partial_m_ptr + base_ml + offsets_g,
        tl.where(has_live, m, -float("inf")),
        mask=offsets_g < group_size,
    )
    tl.store(partial_l_ptr + base_ml + offsets_g, l, mask=offsets_g < group_size)

    acc_base = (((batch_idx * num_kv_heads + kv_head) * num_splits + split_id) * BLOCK_G) * BLOCK_D
    tl.store(
        partial_acc_ptr + acc_base + offsets_g[:, None] * BLOCK_D + offsets_d[None, :],
        acc,
        mask=(offsets_g[:, None] < group_size) & (offsets_d[None, :] < head_dim),
    )


@triton.jit
def _compact_decode_grouped_splitk_reduce_kernel(
    partial_m_ptr,
    partial_l_ptr,
    partial_acc_ptr,
    out_ptr,
    out_stride_b: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_d: tl.constexpr,
    num_kv_heads: tl.constexpr,
    group_size: tl.constexpr,
    head_dim: tl.constexpr,
    num_splits: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // num_kv_heads
    kv_head = pid - batch_idx * num_kv_heads
    q_head_base = kv_head * group_size

    offsets_g = tl.arange(0, BLOCK_G)
    offsets_d = tl.arange(0, BLOCK_D)

    m_i = tl.full((BLOCK_G,), -float("inf"), tl.float32)
    l_i = tl.full((BLOCK_G,), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_G, BLOCK_D), tl.float32)

    for split_id in range(0, num_splits):
        base_ml = ((batch_idx * num_kv_heads + kv_head) * num_splits + split_id) * BLOCK_G
        m_tile = tl.load(
            partial_m_ptr + base_ml + offsets_g,
            mask=offsets_g < group_size,
            other=-float("inf"),
        ).to(tl.float32)
        l_tile = tl.load(
            partial_l_ptr + base_ml + offsets_g,
            mask=offsets_g < group_size,
            other=0.0,
        ).to(tl.float32)
        acc_base = (((batch_idx * num_kv_heads + kv_head) * num_splits + split_id) * BLOCK_G) * BLOCK_D
        acc_tile = tl.load(
            partial_acc_ptr + acc_base + offsets_g[:, None] * BLOCK_D + offsets_d[None, :],
            mask=(offsets_g[:, None] < group_size) & (offsets_d[None, :] < head_dim),
            other=0.0,
        ).to(tl.float32)
        if split_id == 0:
            m_i = m_tile
            l_i = l_tile
            acc = acc_tile
        else:
            m_next = tl.maximum(m_i, m_tile)
            alpha_old = tl.exp(m_i - m_next)
            alpha_tile = tl.exp(m_tile - m_next)
            l_i = l_i * alpha_old + l_tile * alpha_tile
            acc = acc * alpha_old[:, None] + acc_tile * alpha_tile[:, None]
            m_i = m_next

    acc = acc / l_i[:, None]
    tl.store(
        out_ptr
        + batch_idx * out_stride_b
        + (q_head_base + offsets_g[:, None]) * out_stride_h
        + offsets_d[None, :] * out_stride_d,
        acc,
        mask=(offsets_g[:, None] < group_size) & (offsets_d[None, :] < head_dim),
    )


@triton.jit
def _compact_decode_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    base_offsets_ptr,
    live_counts_ptr,
    scale: tl.constexpr,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    kv_stride_s: tl.constexpr,
    kv_stride_d: tl.constexpr,
    out_stride_b: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_d: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    group_size: tl.constexpr,
    head_dim: tl.constexpr,
    max_live_count: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // num_q_heads
    q_head = pid - batch_idx * num_q_heads
    kv_head = q_head // group_size

    offsets_d = tl.arange(0, BLOCK_D)
    q = tl.load(
        q_ptr + batch_idx * q_stride_b + q_head * q_stride_h + offsets_d * q_stride_d,
        mask=offsets_d < head_dim,
        other=0.0,
    ).to(tl.float32)

    meta_offset = batch_idx * num_kv_heads + kv_head
    base = tl.load(base_offsets_ptr + meta_offset)
    live_count = tl.load(live_counts_ptr + meta_offset)

    m_i = tl.full((), -float("inf"), tl.float32)
    l_i = tl.full((), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_D,), tl.float32)

    for start in range(0, max_live_count, BLOCK_N):
        offsets_n = start + tl.arange(0, BLOCK_N)
        live_mask = offsets_n < live_count
        k = tl.load(
            k_ptr + (base + offsets_n[:, None]) * kv_stride_s + offsets_d[None, :] * kv_stride_d,
            mask=live_mask[:, None] & (offsets_d[None, :] < head_dim),
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(k * q[None, :], axis=1) * scale
        scores = tl.where(live_mask, scores, -float("inf"))

        m_next = tl.maximum(m_i, tl.max(scores, axis=0))
        p = tl.exp(scores - m_next)
        alpha = tl.exp(m_i - m_next)
        l_next = l_i * alpha + tl.sum(p, axis=0)

        values = tl.load(
            v_ptr + (base + offsets_n[:, None]) * kv_stride_s + offsets_d[None, :] * kv_stride_d,
            mask=live_mask[:, None] & (offsets_d[None, :] < head_dim),
            other=0.0,
        ).to(tl.float32)
        acc = acc * alpha + tl.sum(values * p[:, None], axis=0)
        m_i = m_next
        l_i = l_next

    acc = acc / l_i
    tl.store(
        out_ptr + batch_idx * out_stride_b + q_head * out_stride_h + offsets_d * out_stride_d,
        acc,
        mask=offsets_d < head_dim,
    )


def _next_power_of_2(value: int) -> int:
    if value < 1:
        raise ValueError(f"value must be positive, got {value}")
    return 1 << (value - 1).bit_length()


@triton.jit
def _dms_expiry_kernel(
    positions_ptr,          # [num_seqs, num_kv_heads, max_capacity] int32
    pos_stride_s,
    pos_stride_h,
    pos_stride_slot,
    evict_ptr,              # [num_seqs, num_kv_heads, max_capacity] bool
    ev_stride_s,
    ev_stride_h,
    ev_stride_slot,
    live_counts_ptr,        # [num_seqs, num_kv_heads] int32
    lc_stride_s,
    lc_stride_h,
    base_offsets_ptr,       # [num_seqs, num_kv_heads] int32
    bo_stride_s,
    bo_stride_h,
    compact_k_ptr,          # [layer_capacity_tokens, head_dim]
    compact_v_ptr,
    ck_stride_slot,
    ck_stride_d,
    current_positions_ptr,  # [num_seqs] int64
    window_size: tl.constexpr,
    head_dim: tl.constexpr,
    num_kv_heads: tl.constexpr,
    max_capacity: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    seq_idx = pid // num_kv_heads
    kv_head = pid - seq_idx * num_kv_heads

    cur = tl.load(current_positions_ptr + seq_idx)
    base = tl.load(base_offsets_ptr + seq_idx * bo_stride_s + kv_head * bo_stride_h)
    live = tl.load(live_counts_ptr + seq_idx * lc_stride_s + kv_head * lc_stride_h)

    s = tl.arange(0, BLOCK_S)
    valid = s < live
    pos_off = seq_idx * pos_stride_s + kv_head * pos_stride_h + s * pos_stride_slot
    ev_off = seq_idx * ev_stride_s + kv_head * ev_stride_h + s * ev_stride_slot
    pos = tl.load(positions_ptr + pos_off, mask=valid, other=-1).to(tl.int64)
    evict = tl.load(evict_ptr + ev_off, mask=valid, other=0).to(tl.int1)

    in_window = (cur - pos) <= window_size
    keep = valid & ((~evict) | in_window)
    rank = tl.cumsum(keep.to(tl.int32)) - 1
    new_live = tl.sum(keep.to(tl.int32))

    offsets_d = tl.arange(0, BLOCK_D)
    d_valid = offsets_d < head_dim
    src_compact = (base + s).to(tl.int64)
    dst_compact = (base + rank).to(tl.int64)

    src_k = tl.load(
        compact_k_ptr + src_compact[:, None] * ck_stride_slot + offsets_d[None, :] * ck_stride_d,
        mask=keep[:, None] & d_valid[None, :],
        other=0.0,
    )
    src_v = tl.load(
        compact_v_ptr + src_compact[:, None] * ck_stride_slot + offsets_d[None, :] * ck_stride_d,
        mask=keep[:, None] & d_valid[None, :],
        other=0.0,
    )

    # Reset trailing metadata slots. BLOCK_S is rounded up to a power of two,
    # so padding lanes must not store past max_capacity into the next
    # head/sequence row.
    trailing = (s >= new_live) & (s < max_capacity)
    tl.store(positions_ptr + pos_off, tl.full((BLOCK_S,), -1, tl.int32), mask=trailing)
    tl.store(evict_ptr + ev_off, tl.zeros((BLOCK_S,), tl.int1), mask=trailing)

    # Write compacted metadata at rank slots.
    dst_pos_off = seq_idx * pos_stride_s + kv_head * pos_stride_h + rank.to(tl.int64) * pos_stride_slot
    dst_ev_off = seq_idx * ev_stride_s + kv_head * ev_stride_h + rank.to(tl.int64) * ev_stride_slot
    tl.store(positions_ptr + dst_pos_off, pos.to(tl.int32), mask=keep)
    tl.store(evict_ptr + dst_ev_off, evict.to(tl.int1), mask=keep)

    # Write compacted K/V at dst slots.
    dst_kv_off = dst_compact[:, None] * ck_stride_slot + offsets_d[None, :] * ck_stride_d
    tl.store(compact_k_ptr + dst_kv_off, src_k, mask=keep[:, None] & d_valid[None, :])
    tl.store(compact_v_ptr + dst_kv_off, src_v, mask=keep[:, None] & d_valid[None, :])

    # Per-(seq, head) new live count: written once per program by thread 0.
    tl.store(live_counts_ptr + seq_idx * lc_stride_s + kv_head * lc_stride_h, new_live)


@triton.jit
def _dms_expiry_incremental_kernel(
    positions_ptr,          # [num_seqs, num_kv_heads, max_capacity] int32
    pos_stride_s,
    pos_stride_h,
    pos_stride_slot,
    evict_ptr,              # [num_seqs, num_kv_heads, max_capacity] bool
    ev_stride_s,
    ev_stride_h,
    ev_stride_slot,
    live_counts_ptr,        # [num_seqs, num_kv_heads] int32
    lc_stride_s,
    lc_stride_h,
    base_offsets_ptr,       # [num_seqs, num_kv_heads] int32
    bo_stride_s,
    bo_stride_h,
    compact_k_ptr,          # [layer_capacity_tokens, head_dim]
    compact_v_ptr,
    ck_stride_slot,
    ck_stride_d,
    current_positions_ptr,  # [num_seqs] int64
    window_size: tl.constexpr,
    head_dim: tl.constexpr,
    num_kv_heads: tl.constexpr,
    max_capacity: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    seq_idx = pid // num_kv_heads
    kv_head = pid - seq_idx * num_kv_heads

    cur = tl.load(current_positions_ptr + seq_idx)
    cutoff = cur - window_size
    base = tl.load(base_offsets_ptr + seq_idx * bo_stride_s + kv_head * bo_stride_h)
    live = tl.load(live_counts_ptr + seq_idx * lc_stride_s + kv_head * lc_stride_h)

    s = tl.arange(0, BLOCK_S)
    in_bounds = s < max_capacity
    valid = s < live
    pos_off = seq_idx * pos_stride_s + kv_head * pos_stride_h + s * pos_stride_slot
    ev_off = seq_idx * ev_stride_s + kv_head * ev_stride_h + s * ev_stride_slot
    pos = tl.load(positions_ptr + pos_off, mask=valid & in_bounds, other=-1).to(tl.int64)
    evict = tl.load(evict_ptr + ev_off, mask=valid & in_bounds, other=0).to(tl.int1)

    expired = valid & evict & (pos < cutoff)
    expire_slot = tl.min(tl.where(expired, s, max_capacity))
    has_expire = expire_slot < live

    move = has_expire & (s >= expire_slot) & (s < (live - 1)) & in_bounds
    next_s = s + 1
    next_pos_off = seq_idx * pos_stride_s + kv_head * pos_stride_h + next_s * pos_stride_slot
    next_ev_off = seq_idx * ev_stride_s + kv_head * ev_stride_h + next_s * ev_stride_slot
    next_pos = tl.load(positions_ptr + next_pos_off, mask=move, other=-1)
    next_evict = tl.load(evict_ptr + next_ev_off, mask=move, other=0).to(tl.int1)
    tl.store(positions_ptr + pos_off, next_pos, mask=move)
    tl.store(evict_ptr + ev_off, next_evict, mask=move)

    offsets_d = tl.arange(0, BLOCK_D)
    d_valid = offsets_d < head_dim
    src_compact = (base + next_s).to(tl.int64)
    dst_compact = (base + s).to(tl.int64)
    src_k = tl.load(
        compact_k_ptr + src_compact[:, None] * ck_stride_slot + offsets_d[None, :] * ck_stride_d,
        mask=move[:, None] & d_valid[None, :],
        other=0.0,
    )
    src_v = tl.load(
        compact_v_ptr + src_compact[:, None] * ck_stride_slot + offsets_d[None, :] * ck_stride_d,
        mask=move[:, None] & d_valid[None, :],
        other=0.0,
    )
    dst_kv_off = dst_compact[:, None] * ck_stride_slot + offsets_d[None, :] * ck_stride_d
    tl.store(compact_k_ptr + dst_kv_off, src_k, mask=move[:, None] & d_valid[None, :])
    tl.store(compact_v_ptr + dst_kv_off, src_v, mask=move[:, None] & d_valid[None, :])

    reset = has_expire & (s == (live - 1)) & in_bounds
    tl.store(positions_ptr + pos_off, tl.full((BLOCK_S,), -1, tl.int32), mask=reset)
    tl.store(evict_ptr + ev_off, tl.zeros((BLOCK_S,), tl.int1), mask=reset)
    tl.store(
        live_counts_ptr + seq_idx * lc_stride_s + kv_head * lc_stride_h,
        live - has_expire.to(tl.int32),
    )


def dms_expiry_one_layer(
    *,
    positions_per_lh: torch.Tensor,    # [num_seqs, num_kv_heads, max_capacity] int32
    evict_per_lh: torch.Tensor,        # [num_seqs, num_kv_heads, max_capacity] bool
    live_counts: torch.Tensor,         # [num_seqs, num_kv_heads] int32
    base_offsets: torch.Tensor,        # [num_seqs, num_kv_heads] int32
    compact_k: torch.Tensor,           # [layer_capacity, head_dim]
    compact_v: torch.Tensor,
    current_positions: torch.Tensor,   # [num_seqs] int64
    window_size: int,
) -> None:
    """J2 Triton fused DMS expiry for one layer.

    Compacts each (seq, kv_head)'s slots in-place via single-tile cumsum-rank
    scatter. Writes back compacted (positions, evict, live_counts, compact_k,
    compact_v) for this layer.
    """
    num_seqs, num_kv_heads, max_capacity = positions_per_lh.shape
    head_dim = compact_k.shape[-1]
    block_s = _next_power_of_2(max_capacity)
    block_d = _next_power_of_2(head_dim)
    if block_s > 32768:
        raise RuntimeError(f"max_capacity={max_capacity} too large for single-tile expiry kernel")
    grid = (num_seqs * num_kv_heads,)
    _dms_expiry_kernel[grid](
        positions_per_lh,
        positions_per_lh.stride(0),
        positions_per_lh.stride(1),
        positions_per_lh.stride(2),
        evict_per_lh,
        evict_per_lh.stride(0),
        evict_per_lh.stride(1),
        evict_per_lh.stride(2),
        live_counts,
        live_counts.stride(0),
        live_counts.stride(1),
        base_offsets,
        base_offsets.stride(0),
        base_offsets.stride(1),
        compact_k,
        compact_v,
        compact_k.stride(0),
        compact_k.stride(1),
        current_positions,
        int(window_size),
        head_dim,
        num_kv_heads,
        max_capacity,
        block_s,
        block_d,
    )


def dms_expiry_incremental_one_layer(
    *,
    positions_per_lh: torch.Tensor,
    evict_per_lh: torch.Tensor,
    live_counts: torch.Tensor,
    base_offsets: torch.Tensor,
    compact_k: torch.Tensor,
    compact_v: torch.Tensor,
    current_positions: torch.Tensor,
    window_size: int,
) -> None:
    """Steady-state DMS expiry for one layer.

    After the first full post-prefill compaction, decode advances one position
    at a time. At most one slot per (seq, kv_head) newly ages out of the DMS
    window, so this shifts only the tail after that slot instead of rebuilding
    the whole compact span.
    """
    num_seqs, num_kv_heads, max_capacity = positions_per_lh.shape
    head_dim = compact_k.shape[-1]
    block_s = _next_power_of_2(max_capacity)
    block_d = _next_power_of_2(head_dim)
    if block_s > 32768:
        raise RuntimeError(f"max_capacity={max_capacity} too large for incremental expiry kernel")
    grid = (num_seqs * num_kv_heads,)
    _dms_expiry_incremental_kernel[grid](
        positions_per_lh,
        positions_per_lh.stride(0),
        positions_per_lh.stride(1),
        positions_per_lh.stride(2),
        evict_per_lh,
        evict_per_lh.stride(0),
        evict_per_lh.stride(1),
        evict_per_lh.stride(2),
        live_counts,
        live_counts.stride(0),
        live_counts.stride(1),
        base_offsets,
        base_offsets.stride(0),
        base_offsets.stride(1),
        compact_k,
        compact_v,
        compact_k.stride(0),
        compact_k.stride(1),
        current_positions,
        int(window_size),
        head_dim,
        num_kv_heads,
        max_capacity,
        block_s,
        block_d,
    )


def store_compact_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    compact_k: torch.Tensor,
    compact_v: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    if key.shape != value.shape:
        raise ValueError(f"key/value shape mismatch: {key.shape} != {value.shape}")
    if key.ndim != 3:
        raise ValueError(f"expected key/value as [batch, kv_heads, head_dim], got {key.shape}")
    batch, num_kv_heads, head_dim = key.shape
    if compact_k.shape != compact_v.shape:
        raise ValueError(f"compact cache shape mismatch: {compact_k.shape} != {compact_v.shape}")
    if compact_k.ndim != 2 or compact_k.shape[1] != head_dim:
        raise ValueError(f"expected compact cache [slots, {head_dim}], got {compact_k.shape}")
    if slot_mapping.shape != (batch, num_kv_heads):
        raise ValueError(f"expected slot_mapping {(batch, num_kv_heads)}, got {tuple(slot_mapping.shape)}")

    block_d = _next_power_of_2(head_dim)
    if block_d > 128:
        raise ValueError(f"head_dim={head_dim} is above the prototype kernel limit")
    slot_mapping = slot_mapping.to(device=key.device, dtype=torch.int32).contiguous()

    _store_compact_kvcache_kernel[(batch * num_kv_heads,)](
        key,
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value,
        value.stride(0),
        value.stride(1),
        value.stride(2),
        compact_k,
        compact_v,
        slot_mapping,
        num_kv_heads,
        head_dim,
        block_d,
    )


def fused_dms_rope_store_compact_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    compact_k: torch.Tensor,
    compact_v: torch.Tensor,
    slot_mapping: torch.Tensor,
    evict_out: torch.Tensor,
    positions: torch.Tensor,
    *,
    num_kv_heads: int,
    alpha_scale: float,
    alpha_offset: float,
    store_q: bool = True,
    store_transient_k: bool = True,
) -> None:
    """Decode-only DMS extraction + RoPE + compact append-store.

    k is updated in-place by default to match RotaryEmbedding.forward, and the
    post-RoPE k plus raw v are appended into compact_k/compact_v at
    slot_mapping. Compact-DMS decode callers can set store_transient_k=False
    when no downstream path reads the transient k tensor. q is updated in-place
    by default; compact attention can opt to rotate q in the attention kernel
    instead.
    """
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("expected q/k/v as [batch, heads, head_dim]")
    if k.shape != v.shape:
        raise ValueError(f"k/value shape mismatch: {k.shape} != {v.shape}")
    batch, num_q_heads, head_dim = q.shape
    if k.shape[0] != batch or k.shape[1] != num_kv_heads or k.shape[2] != head_dim:
        raise ValueError(f"q/k shape mismatch: q={q.shape}, k={k.shape}, num_kv_heads={num_kv_heads}")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(f"q_heads={num_q_heads} must be divisible by kv_heads={num_kv_heads}")
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE head_dim must be even, got {head_dim}")
    if compact_k.shape != compact_v.shape or compact_k.ndim != 2 or compact_k.shape[1] != head_dim:
        raise ValueError(f"expected compact cache [slots, {head_dim}], got {compact_k.shape}/{compact_v.shape}")
    if slot_mapping.shape != (batch, num_kv_heads):
        raise ValueError(f"expected slot_mapping {(batch, num_kv_heads)}, got {tuple(slot_mapping.shape)}")
    if evict_out.shape != (batch, num_kv_heads) or evict_out.dtype != torch.bool:
        raise ValueError(f"expected evict_out bool {(batch, num_kv_heads)}, got {tuple(evict_out.shape)} {evict_out.dtype}")
    if positions.shape != (batch,):
        raise ValueError(f"expected positions {(batch,)}, got {tuple(positions.shape)}")
    if head_dim > 256:
        raise ValueError(f"head_dim={head_dim} is above the fused decode preprocess kernel limit")
    if not (q.is_cuda and k.is_cuda and v.is_cuda and compact_k.is_cuda and positions.is_cuda):
        raise RuntimeError("fused decode preprocess requires CUDA tensors")

    slot_mapping = slot_mapping.to(device=q.device, dtype=torch.int32).contiguous()
    if evict_out.device != q.device:
        raise ValueError(f"expected evict_out on {q.device}, got {evict_out.device}")
    if positions.device != q.device:
        positions = positions.to(device=q.device, non_blocking=True)
    if positions.dtype not in (torch.int32, torch.int64):
        positions = positions.to(dtype=torch.int64)
    if cos_sin_cache.device != q.device:
        cos_sin_cache = cos_sin_cache.to(device=q.device)
    if cos_sin_cache.ndim != 3 or cos_sin_cache.shape[2] != head_dim:
        raise ValueError(f"expected cos_sin_cache [positions, 1, {head_dim}], got {tuple(cos_sin_cache.shape)}")

    block_half = _next_power_of_2(head_dim // 2)
    block_g = _next_power_of_2(num_q_heads // num_kv_heads)
    if block_g > 8:
        raise ValueError(f"group_size={num_q_heads // num_kv_heads} is above the fused prototype limit")
    _dms_rope_store_compact_decode_kernel[(batch * num_kv_heads,)](
        q,
        k,
        v,
        cos_sin_cache,
        compact_k,
        compact_v,
        slot_mapping,
        evict_out,
        positions,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        cos_sin_cache.stride(0),
        cos_sin_cache.stride(2),
        compact_k.stride(0),
        compact_k.stride(1),
        slot_mapping.stride(0),
        slot_mapping.stride(1),
        evict_out.stride(0),
        evict_out.stride(1),
        num_kv_heads,
        num_q_heads // num_kv_heads,
        head_dim,
        float(alpha_scale),
        float(alpha_offset),
        bool(alpha_scale > 0),
        block_g,
        block_half,
        bool(store_q),
        bool(store_transient_k),
    )


def compact_decode_attention(
    q: torch.Tensor,
    compact_k: torch.Tensor,
    compact_v: torch.Tensor,
    base_offsets: torch.Tensor,
    live_counts: torch.Tensor,
    scale: float | None = None,
    max_live_count: int | None = None,
    block_n: int = 128,
    grouped: bool = True,
    q_rope_positions: torch.Tensor | None = None,
    q_rope_cos_sin_cache: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode attention over flat compact K/V spans.

    q is shaped [batch, q_heads, head_dim]. compact_k and compact_v are flat
    [total_live, head_dim] tensors. base_offsets/live_counts are shaped
    [batch, kv_heads], with grouped-query mapping q_head // group_size.
    """
    if q.ndim != 3:
        raise ValueError(f"expected q as [batch, q_heads, head_dim], got {q.shape}")
    if compact_k.shape != compact_v.shape:
        raise ValueError(f"compact k/v shape mismatch: {compact_k.shape} != {compact_v.shape}")
    if compact_k.ndim != 2:
        raise ValueError(f"expected compact K/V as [total_live, head_dim], got {compact_k.shape}")
    batch, num_q_heads, head_dim = q.shape
    if compact_k.shape[1] != head_dim:
        raise ValueError(f"K/V head_dim {compact_k.shape[1]} does not match q head_dim {head_dim}")
    if base_offsets.shape != live_counts.shape or base_offsets.ndim != 2:
        raise ValueError("base_offsets and live_counts must both be [batch, kv_heads]")
    if base_offsets.shape[0] != batch:
        raise ValueError(f"metadata batch {base_offsets.shape[0]} does not match q batch {batch}")

    num_kv_heads = base_offsets.shape[1]
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(f"q_heads={num_q_heads} must be divisible by kv_heads={num_kv_heads}")
    if max_live_count is None:
        max_live_count = int(live_counts.max().item())
    if max_live_count <= 0:
        raise ValueError("compact decode requires at least one live token per active KV head")
    if _COMPACT_DEBUG_CHECKS and bool(torch.any(live_counts <= 0).item()):
        raise ValueError("compact decode prototype expects every active KV head to have live_count > 0")
    block_d = _next_power_of_2(head_dim)
    if block_d > 128:
        raise ValueError(f"head_dim={head_dim} is above the prototype kernel limit")

    compact_k = compact_k.contiguous()
    compact_v = compact_v.contiguous()
    base_offsets = base_offsets.to(device=q.device, dtype=torch.int32).contiguous()
    live_counts = live_counts.to(device=q.device, dtype=torch.int32).contiguous()
    out = torch.empty_like(q)
    group_size = num_q_heads // num_kv_heads
    if scale is None:
        scale = head_dim ** -0.5
    inline_q_rope = q_rope_positions is not None or q_rope_cos_sin_cache is not None
    if inline_q_rope:
        if q_rope_positions is None or q_rope_cos_sin_cache is None:
            raise ValueError("inline q RoPE requires both positions and cos_sin_cache")
        if not (grouped and _COMPACT_ATTN_SPLITK):
            raise ValueError("inline q RoPE is implemented only for grouped split-K compact attention")
        if head_dim % 2 != 0:
            raise ValueError(f"inline q RoPE requires even head_dim, got {head_dim}")
        if head_dim < 32:
            raise ValueError(f"inline q RoPE requires head_dim >= 32 for Triton dot, got {head_dim}")
        if q_rope_positions.shape != (batch,):
            raise ValueError(f"expected q_rope_positions {(batch,)}, got {tuple(q_rope_positions.shape)}")
        if q_rope_positions.device != q.device:
            q_rope_positions = q_rope_positions.to(device=q.device, non_blocking=True)
        if q_rope_positions.dtype not in (torch.int32, torch.int64):
            q_rope_positions = q_rope_positions.to(dtype=torch.int64)
        if q_rope_cos_sin_cache.device != q.device:
            q_rope_cos_sin_cache = q_rope_cos_sin_cache.to(device=q.device)
        if q_rope_cos_sin_cache.ndim != 3 or q_rope_cos_sin_cache.shape[2] != head_dim:
            raise ValueError(
                f"expected q_rope_cos_sin_cache [positions, 1, {head_dim}], "
                f"got {tuple(q_rope_cos_sin_cache.shape)}"
            )

    if grouped and _COMPACT_ATTN_SPLITK:
        if _COMPACT_ATTN_SPLITK_BLOCK_N is not None:
            block_n = _COMPACT_ATTN_SPLITK_BLOCK_N
        elif block_n == 128:
            block_n = _COMPACT_ATTN_SPLITK_DEFAULT_BLOCK_N
        block_g = _next_power_of_2(group_size)
        if block_g > 8:
            raise ValueError(f"group_size={group_size} is above the grouped prototype limit")
        num_splits = triton.cdiv(max_live_count, block_n)
        partial_m = torch.empty(
            (batch, num_kv_heads, num_splits, block_g),
            device=q.device,
            dtype=torch.float32,
        )
        partial_l = torch.empty_like(partial_m)
        partial_acc = torch.empty(
            (batch, num_kv_heads, num_splits, block_g, block_d),
            device=q.device,
            dtype=torch.float32,
        )
        if inline_q_rope:
            block_half = _next_power_of_2(head_dim // 2)
            _compact_decode_grouped_splitk_partial_qrope_kernel[(batch * num_kv_heads * num_splits,)](
                q,
                q_rope_cos_sin_cache,
                q_rope_positions,
                compact_k,
                compact_v,
                partial_m,
                partial_l,
                partial_acc,
                base_offsets,
                live_counts,
                float(scale),
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q_rope_cos_sin_cache.stride(0),
                q_rope_cos_sin_cache.stride(2),
                compact_k.stride(0),
                compact_k.stride(1),
                num_kv_heads,
                group_size,
                head_dim,
                num_splits,
                block_n,
                block_g,
                block_d,
                block_half,
            )
        else:
            _compact_decode_grouped_splitk_partial_kernel[(batch * num_kv_heads * num_splits,)](
                q,
                compact_k,
                compact_v,
                partial_m,
                partial_l,
                partial_acc,
                base_offsets,
                live_counts,
                float(scale),
                q.stride(0),
                q.stride(1),
                q.stride(2),
                compact_k.stride(0),
                compact_k.stride(1),
                num_kv_heads,
                group_size,
                head_dim,
                num_splits,
                block_n,
                block_g,
                block_d,
            )
        _compact_decode_grouped_splitk_reduce_kernel[(batch * num_kv_heads,)](
            partial_m,
            partial_l,
            partial_acc,
            out,
            out.stride(0),
            out.stride(1),
            out.stride(2),
            num_kv_heads,
            group_size,
            head_dim,
            num_splits,
            block_g,
            block_d,
        )
    elif grouped:
        block_g = _next_power_of_2(group_size)
        if block_g > 8:
            raise ValueError(f"group_size={group_size} is above the grouped prototype limit")
        _compact_decode_grouped_kernel[(batch * num_kv_heads,)](
            q,
            compact_k,
            compact_v,
            out,
            base_offsets,
            live_counts,
            float(scale),
            q.stride(0),
            q.stride(1),
            q.stride(2),
            compact_k.stride(0),
            compact_k.stride(1),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            num_kv_heads,
            group_size,
            head_dim,
            triton.cdiv(max_live_count, block_n) * block_n,
            block_n,
            block_g,
            block_d,
        )
    else:
        _compact_decode_kernel[(batch * num_q_heads,)](
            q,
            compact_k,
            compact_v,
            out,
            base_offsets,
            live_counts,
            float(scale),
            q.stride(0),
            q.stride(1),
            q.stride(2),
            compact_k.stride(0),
            compact_k.stride(1),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            num_q_heads,
            num_kv_heads,
            group_size,
            head_dim,
            triton.cdiv(max_live_count, block_n) * block_n,
            block_n,
            block_d,
        )
    return out


def compact_decode_attention_ref(
    q: torch.Tensor,
    compact_k: torch.Tensor,
    compact_v: torch.Tensor,
    base_offsets: torch.Tensor,
    live_counts: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    batch, num_q_heads, head_dim = q.shape
    num_kv_heads = base_offsets.shape[1]
    group_size = num_q_heads // num_kv_heads
    if scale is None:
        scale = head_dim ** -0.5

    out = torch.empty_like(q)
    for batch_idx in range(batch):
        for q_head in range(num_q_heads):
            kv_head = q_head // group_size
            base = int(base_offsets[batch_idx, kv_head].item())
            live = int(live_counts[batch_idx, kv_head].item())
            keys = compact_k[base:base + live].float()
            values = compact_v[base:base + live].float()
            scores = torch.matmul(keys, q[batch_idx, q_head].float()) * scale
            probs = torch.softmax(scores, dim=-1)
            out[batch_idx, q_head] = torch.matmul(probs, values).to(dtype=q.dtype)
    return out
