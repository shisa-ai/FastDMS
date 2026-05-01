import os

import torch

from fastdms.engine.compact_kv import (
    CompactKVManager,
    _streaming_pack_fused_rank_prepare,
    _streaming_pack_rank_scatter_triton,
    _streaming_pack_scatter_triton,
    build_emulated_dms_live_mask,
    pack_compact_kv,
    streaming_pack_cache_live_enabled,
    streaming_pack_fused_rank_triton_enabled,
)
from fastdms.engine.dms import extract_dms_eviction_decisions
from fastdms.engine.dms import load_dms_metadata
from fastdms.engine.sequence import Sequence
from fastdms.sampling_params import SamplingParams
from fastdms.layers import compact_attention
from fastdms.layers.compact_attention import compact_decode_attention
from fastdms.layers.compact_attention import compact_decode_attention_ref
from fastdms.layers.compact_attention import dms_decode_store_transient_k_enabled
from fastdms.layers.compact_attention import fused_dms_rope_store_compact_decode
from fastdms.layers.embed_head import ParallelLMHead
from fastdms.layers.embed_head import VocabParallelEmbedding
from fastdms.layers.embed_head import _fp8_lm_head_row1_triton_allowed
from fastdms.layers.embed_head import _fp8_row1_argmax
from fastdms.layers.embed_head import _fp8_row1_rerank_topk
from fastdms.layers.embed_head import fp8_embedding_enabled
from fastdms.layers.embed_head import fp8_embedding_share_lm_head_weight
from fastdms.layers.embed_head import fp8_keep_bf16_embedding_weight
from fastdms.layers.embed_head import fp8_lm_head_row1_triton_enabled
from fastdms.layers.embed_head import int4_lm_head_rerank_topk
from fastdms.layers.linear import _amax_to_scale
from fastdms.layers.linear import _as_col_major_b
from fastdms.layers.linear import _fp8_row1_matvec
from fastdms.layers.linear import _int4_row1_matvec
from fastdms.layers.linear import _quantize_weight_to_int4pack
from fastdms.layers.linear import fp8_attention_full_enabled
from fastdms.layers.linear import fp8_attention_row1_shadow_enabled
from fastdms.layers.linear import fp8_attention_row1_triton_enabled
from fastdms.layers.linear import fp8_attention_row1_triton_mode
from fastdms.layers.linear import fp8_down_row1_triton_mode
from fastdms.layers.linear import fp8_gate_up_row1_triton_enabled
from fastdms.layers.linear import fp8_gate_up_row1_triton_mode
from fastdms.layers.linear import fp8_keep_bf16_down_weights
from fastdms.layers.linear import fp8_keep_bf16_gate_up_weights
from fastdms.layers.linear import fp8_row1_matvec_impl
from fastdms.layers.linear import fp8_row1_triton_enabled
from fastdms.layers.linear import fp8_row1_triton_mode
from fastdms.layers.linear import int4_row1_lm_head_enabled
from fastdms.layers.linear import int4_row1_max_rows
from fastdms.layers.linear import int4_row1_module_in_scope
from fastdms.layers.linear import int4_row1_quant_mode
from fastdms.layers.rotary_embedding import apply_rotary_emb
from fastdms.layers.rotary_embedding import get_rope


def test_pack_compact_kv_reference_decode():
    k = torch.arange(1 * 5 * 2 * 4, dtype=torch.float32).reshape(1, 5, 2, 4)
    v = k + 100
    live_mask = torch.tensor([[[True, False, True, False, True], [False, True, True, False, False]]])
    compact_k, compact_v, metadata = pack_compact_kv(k, v, live_mask)

    assert compact_k.shape == (5, 4)
    assert compact_v.shape == (5, 4)
    assert metadata.base_offsets.tolist() == [[0, 3]]
    assert metadata.live_counts.tolist() == [[3, 2]]
    assert metadata.token_positions.tolist() == [0, 2, 4, 1, 2]

    q = torch.ones((1, 4, 4), dtype=torch.float32)
    out = compact_decode_attention_ref(
        q,
        compact_k,
        compact_v,
        metadata.base_offsets,
        metadata.live_counts,
        scale=1.0,
    )
    assert out.shape == q.shape


def test_streaming_pack_cache_live_env():
    old_env = os.environ.get("FASTDMS_STREAMING_PACK_CACHE_LIVE")
    try:
        os.environ.pop("FASTDMS_STREAMING_PACK_CACHE_LIVE", None)
        assert streaming_pack_cache_live_enabled()
        os.environ["FASTDMS_STREAMING_PACK_CACHE_LIVE"] = "0"
        assert not streaming_pack_cache_live_enabled()
        os.environ["FASTDMS_STREAMING_PACK_CACHE_LIVE"] = "1"
        assert streaming_pack_cache_live_enabled()
    finally:
        if old_env is None:
            os.environ.pop("FASTDMS_STREAMING_PACK_CACHE_LIVE", None)
        else:
            os.environ["FASTDMS_STREAMING_PACK_CACHE_LIVE"] = old_env


def test_streaming_pack_fused_rank_env():
    old_env = os.environ.get("FASTDMS_STREAMING_PACK_FUSED_RANK_TRITON")
    try:
        os.environ.pop("FASTDMS_STREAMING_PACK_FUSED_RANK_TRITON", None)
        assert streaming_pack_fused_rank_triton_enabled()
        os.environ["FASTDMS_STREAMING_PACK_FUSED_RANK_TRITON"] = "0"
        assert not streaming_pack_fused_rank_triton_enabled()
        os.environ["FASTDMS_STREAMING_PACK_FUSED_RANK_TRITON"] = "1"
        assert streaming_pack_fused_rank_triton_enabled()
    finally:
        if old_env is None:
            os.environ.pop("FASTDMS_STREAMING_PACK_FUSED_RANK_TRITON", None)
        else:
            os.environ["FASTDMS_STREAMING_PACK_FUSED_RANK_TRITON"] = old_env


def test_load_dms_metadata_reads_hf_config_dms_aliases(tmp_path):
    model_dir = tmp_path / "qwen3-dms"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        """
{
  "model_type": "qwen3",
  "dms_window_size": 512,
  "dms_alpha_scale": 100.0,
  "dms_initial_alpha_offset": 5.0,
  "dms_cr": 8,
  "max_position_embeddings": 40960
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    metadata = load_dms_metadata(model_dir)

    assert metadata.source_kind == "hf_config_dms"
    assert metadata.source_path.endswith("config.json")
    assert metadata.packaged_metadata_found is False
    assert metadata.window_size == 512
    assert metadata.alpha_scale == 100.0
    assert metadata.alpha_offset == 5.0
    assert metadata.target_cr == 8
    assert metadata.context_len == 40960


def test_dms_decode_store_transient_k_env():
    old_env = os.environ.get("FASTDMS_DMS_DECODE_STORE_TRANSIENT_K")
    try:
        os.environ.pop("FASTDMS_DMS_DECODE_STORE_TRANSIENT_K", None)
        assert not dms_decode_store_transient_k_enabled()
        os.environ["FASTDMS_DMS_DECODE_STORE_TRANSIENT_K"] = "1"
        assert dms_decode_store_transient_k_enabled()
        os.environ["FASTDMS_DMS_DECODE_STORE_TRANSIENT_K"] = "0"
        assert not dms_decode_store_transient_k_enabled()
    finally:
        if old_env is None:
            os.environ.pop("FASTDMS_DMS_DECODE_STORE_TRANSIENT_K", None)
        else:
            os.environ["FASTDMS_DMS_DECODE_STORE_TRANSIENT_K"] = old_env


def test_compact_kv_manager_allocates_and_releases_sequence_spans():
    manager = CompactKVManager(
        num_layers=3,
        num_kv_heads=2,
        capacity_tokens=64,
        max_model_len=32,
        device="cpu",
    )
    seq = Sequence([1, 2, 3], SamplingParams(max_tokens=5))

    state = manager.allocate(seq)

    assert state.seq_id == seq.seq_id
    assert state.max_capacity_per_lh == 8
    assert state.base_offsets.tolist() == [[-1, -1], [-1, -1], [-1, -1]]
    assert state.range_capacity.tolist() == [[0, 0], [0, 0], [0, 0]]
    assert state.live_counts.shape == (3, 2)
    assert state.live_counts.sum().item() == 0
    assert manager.free_tokens == 3 * 64
    assert manager.allocate(seq) is state

    manager.free(seq.seq_id)

    assert manager.free_tokens == 3 * 64
    assert manager.free_ranges_per_layer == [[(0, 64)], [(0, 64)], [(0, 64)]]


def test_layer_major_metadata_state_views_reuse_active_slots():
    manager = CompactKVManager(
        num_layers=2,
        num_kv_heads=2,
        capacity_tokens=32,
        max_model_len=16,
        device="cpu",
        layer_major_metadata=True,
        max_active_seqs=2,
    )
    seq0 = Sequence([1, 2, 3], SamplingParams(max_tokens=4))
    seq1 = Sequence([4, 5], SamplingParams(max_tokens=4))

    state0 = manager.allocate(seq0)
    state1 = manager.allocate(seq1)

    assert state0.active_slot == 0
    assert state1.active_slot == 1
    state0.live_counts[0, 1] = 7
    state0.token_positions[0, 1, 0] = 11
    state0.evict_mask[0, 1, 0] = True
    assert manager.layer_major_live_counts[0, 0, 1].item() == 7
    assert manager.layer_major_token_positions[0, 0, 1, 0].item() == 11
    assert manager.layer_major_evict_mask[0, 0, 1, 0].item() is True
    assert manager._contiguous_active_slot_start([state0, state1]) == 0
    assert manager._contiguous_active_slot_start([state1, state0]) is None

    manager.free(seq0.seq_id)
    seq2 = Sequence([6], SamplingParams(max_tokens=4))
    state2 = manager.allocate(seq2)

    assert state2.active_slot == 0
    assert state2.live_counts.sum().item() == 0
    assert state2.token_positions.max().item() == -1


def test_compact_kv_manager_prepares_decode_metadata_and_append_slots():
    manager = CompactKVManager(
        num_layers=2,
        num_kv_heads=2,
        capacity_tokens=32,
        max_model_len=16,
        device="cpu",
    )
    seq = Sequence([10, 11], SamplingParams(max_tokens=4))
    state = manager.allocate(seq)
    for layer_id in range(manager.num_layers):
        manager._allocate_layer_ranges_for_seq(
            state,
            layer_id,
            torch.full((manager.num_kv_heads,), 2, dtype=torch.int32),
        )
    manager.mark_all_heads_live(seq.seq_id, 2)

    base_offsets, live_counts, slot_mapping, max_live_count = manager.prepare_decode_metadata([seq])

    assert base_offsets.tolist() == [[[0, 6]], [[0, 6]]]
    assert slot_mapping.tolist() == [[[2, 8]], [[2, 8]]]
    assert live_counts.tolist() == [[[3, 3]], [[3, 3]]]
    assert state.live_counts.tolist() == [[3, 3], [3, 3]]
    assert max_live_count == state.max_capacity_per_lh


def test_record_dms_decode_decisions_accepts_layer_major_tensor():
    manager = CompactKVManager(
        num_layers=2,
        num_kv_heads=2,
        capacity_tokens=32,
        max_model_len=16,
        device="cpu",
        layer_major_metadata=True,
        max_active_seqs=2,
    )
    seq0 = Sequence([10, 11], SamplingParams(max_tokens=4))
    seq1 = Sequence([20, 21], SamplingParams(max_tokens=4))
    state0 = manager.allocate(seq0)
    state1 = manager.allocate(seq1)
    state0.live_counts.fill_(1)
    state1.live_counts.fill_(1)

    evictions = torch.tensor(
        [
            [[True, False], [False, True]],
            [[False, False], [True, True]],
        ],
        dtype=torch.bool,
    )

    stats = manager.record_appended_dms_decisions([seq0, seq1], [10, 20], evictions)

    assert stats == {"eviction_decisions_true": 4, "eviction_decisions_total": 8}
    assert manager.layer_major_token_positions[:, 0, :, 0].tolist() == [[10, 10], [10, 10]]
    assert manager.layer_major_token_positions[:, 1, :, 0].tolist() == [[20, 20], [20, 20]]
    assert torch.equal(manager.layer_major_evict_mask[:, :2, :, 0], evictions)


def test_record_dms_decode_decisions_layer_major_cuda_uses_position_tensor():
    if not torch.cuda.is_available():
        return
    manager = CompactKVManager(
        num_layers=2,
        num_kv_heads=2,
        capacity_tokens=32,
        max_model_len=16,
        device="cuda",
        layer_major_metadata=True,
        max_active_seqs=2,
    )
    seq0 = Sequence([10, 11], SamplingParams(max_tokens=4))
    seq1 = Sequence([20, 21], SamplingParams(max_tokens=4))
    state0 = manager.allocate(seq0)
    state1 = manager.allocate(seq1)
    state0.live_counts.fill_(1)
    state1.live_counts.fill_(1)

    evictions = torch.tensor(
        [
            [[True, False], [False, True]],
            [[False, False], [True, True]],
        ],
        device="cuda",
        dtype=torch.bool,
    )
    positions = torch.tensor([10, 20], device="cuda", dtype=torch.int64)

    stats = manager.record_appended_dms_decisions([seq0, seq1], positions, evictions)
    torch.cuda.synchronize()

    assert stats == {"eviction_decisions_true": 4, "eviction_decisions_total": 8}
    assert manager.layer_major_token_positions[:, 0, :, 0].cpu().tolist() == [[10, 10], [10, 10]]
    assert manager.layer_major_token_positions[:, 1, :, 0].cpu().tolist() == [[20, 20], [20, 20]]
    assert torch.equal(manager.layer_major_evict_mask[:, :2, :, 0].cpu(), evictions.cpu())


def test_compact_kv_manager_accepts_layer_head_live_counts():
    manager = CompactKVManager(
        num_layers=2,
        num_kv_heads=2,
        capacity_tokens=32,
        max_model_len=16,
        device="cpu",
    )
    seq = Sequence([10, 11, 12], SamplingParams(max_tokens=5))
    state = manager.allocate(seq)
    live_counts = torch.tensor([[2, 3], [4, 5]], dtype=torch.int32)
    for layer_id in range(manager.num_layers):
        manager._allocate_layer_ranges_for_seq(state, layer_id, live_counts[layer_id])

    manager.mark_live_counts(seq.seq_id, live_counts)

    assert state.live_counts.tolist() == [[2, 3], [4, 5]]


def test_emulated_dms_live_mask_keeps_shifted_stride_and_recent_tokens():
    mask = build_emulated_dms_live_mask(
        num_layers=2,
        num_kv_heads=2,
        num_tokens=8,
        mode="dms_stride",
        stride=3,
        recent_tokens=2,
        device="cpu",
    )

    expected = []
    for layer_id in range(2):
        layer = []
        for kv_head in range(2):
            layer.append([
                pos >= 6 or (pos + layer_id + kv_head) % 3 == 0 or pos == 7
                for pos in range(8)
            ])
        expected.append(layer)
    assert mask.tolist() == expected


def _build_cuda_expiry_case(dtype=torch.float32, layer_major_metadata: bool = False):
    device = "cuda"
    manager = CompactKVManager(
        num_layers=2,
        num_kv_heads=2,
        capacity_tokens=128,
        max_model_len=9,
        device=device,
        head_dim=4,
        compact_dtype=dtype,
        per_layer_storage=True,
        layer_major_metadata=layer_major_metadata,
        max_active_seqs=2,
    )
    seq = Sequence([1, 2, 3, 4], SamplingParams(max_tokens=5))
    state = manager.allocate(seq)
    for layer_id in range(manager.num_layers):
        manager._ensure_layer_storage(layer_id, 64)
        manager.compact_k_per_layer[layer_id].zero_()
        manager.compact_v_per_layer[layer_id].zero_()
        live_counts = torch.tensor([6, 7], device=device, dtype=torch.int32)
        manager._allocate_layer_ranges_for_seq(state, layer_id, live_counts)
        state.live_counts[layer_id].copy_(live_counts)
        for head in range(manager.num_kv_heads):
            live = int(live_counts[head].item())
            positions = torch.arange(live, device=device, dtype=torch.int32)
            evict = positions % 2 == 0
            state.token_positions[layer_id, head, :live] = positions
            state.evict_mask[layer_id, head, :live] = evict
            base = int(state.base_offsets[layer_id, head].item())
            values = (
                layer_id * 10.0
                + head
                + torch.arange(live, device=device, dtype=torch.float32) * 0.01
            ).view(live, 1).expand(live, manager.head_dim)
            manager.compact_k_per_layer[layer_id][base:base + live] = values.to(dtype)
            manager.compact_v_per_layer[layer_id][base:base + live] = (values + 0.5).to(dtype)
    return manager, [seq]


def _append_cuda_expiry_token(manager: CompactKVManager, position: int, dtype: torch.dtype) -> None:
    for state in manager.seq_states.values():
        for layer_id in range(manager.num_layers):
            for head in range(manager.num_kv_heads):
                idx = int(state.live_counts[layer_id, head].item())
                base = int(state.base_offsets[layer_id, head].item())
                state.token_positions[layer_id, head, idx] = position
                state.evict_mask[layer_id, head, idx] = (position % 2) == 0
                value = torch.full(
                    (manager.head_dim,),
                    layer_id * 10.0 + head + position * 0.01,
                    device=manager.device,
                    dtype=torch.float32,
                ).to(dtype)
                manager.compact_k_per_layer[layer_id][base + idx] = value
                manager.compact_v_per_layer[layer_id][base + idx] = (value.float() + 0.5).to(dtype)
                state.live_counts[layer_id, head] += 1


def _assert_cuda_expiry_managers_equal(py_manager: CompactKVManager, triton_manager: CompactKVManager) -> None:
    py_state = next(iter(py_manager.seq_states.values()))
    triton_state = next(iter(triton_manager.seq_states.values()))
    assert torch.equal(py_state.live_counts.cpu(), triton_state.live_counts.cpu())
    assert torch.equal(py_state.token_positions.cpu(), triton_state.token_positions.cpu())
    assert torch.equal(py_state.evict_mask.cpu(), triton_state.evict_mask.cpu())
    for layer_id in range(py_manager.num_layers):
        assert torch.equal(
            py_manager.compact_k_per_layer[layer_id].cpu(),
            triton_manager.compact_k_per_layer[layer_id].cpu(),
        )
        assert torch.equal(
            py_manager.compact_v_per_layer[layer_id].cpu(),
            triton_manager.compact_v_per_layer[layer_id].cpu(),
        )


def test_dms_extract_triton_matches_torch_and_zeroes_borrowed_channel():
    if not torch.cuda.is_available():
        return
    q = torch.randn(11, 32, 64, dtype=torch.float32)
    q[:, 0::4, -1] = torch.linspace(-0.1, 0.2, q.shape[0]).view(-1, 1)
    q[0, 0::4, -1] = 0.05
    q[1, 0::4, -1] = torch.nextafter(torch.tensor(0.05), torch.tensor(1.0))
    q[2, 0::4, -1] = torch.nextafter(torch.tensor(0.05), torch.tensor(0.0))
    q_cpu, evict_cpu = extract_dms_eviction_decisions(
        q.clone(),
        num_kv_heads=8,
        num_qo_heads=32,
        head_dim=64,
        alpha_scale=100.0,
        alpha_offset=5.0,
        inplace=True,
    )
    q_gpu, evict_gpu = extract_dms_eviction_decisions(
        q.cuda(),
        num_kv_heads=8,
        num_qo_heads=32,
        head_dim=64,
        alpha_scale=100.0,
        alpha_offset=5.0,
        inplace=True,
    )
    out_gpu = torch.empty((q.shape[0], 8), device="cuda", dtype=torch.bool)
    q_gpu_out, evict_gpu_out = extract_dms_eviction_decisions(
        q.cuda(),
        num_kv_heads=8,
        num_qo_heads=32,
        head_dim=64,
        alpha_scale=100.0,
        alpha_offset=5.0,
        inplace=True,
        out=out_gpu,
    )
    torch.cuda.synchronize()
    assert torch.equal(evict_gpu.cpu(), evict_cpu)
    assert torch.equal(q_gpu.cpu(), q_cpu)
    assert evict_gpu_out.data_ptr() == out_gpu.data_ptr()
    assert torch.equal(evict_gpu_out.cpu(), evict_cpu)
    assert torch.equal(q_gpu_out.cpu(), q_cpu)


def test_fp8_row1_matvec_matches_dequantized_linear():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(0)
    in_features = 256
    out_features = 64
    x = torch.randn((1, in_features), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((out_features, in_features), device="cuda", dtype=torch.bfloat16)
    weight_scale = _amax_to_scale(weight.float().abs().amax(dim=1))
    weight_fp8_t = _as_col_major_b((weight.float().t() * weight_scale.unsqueeze(0)).to(torch.float8_e4m3fn))
    weight_scale_inv = weight_scale.reciprocal().reshape(1, -1).contiguous()

    actual = _fp8_row1_matvec(
        x,
        weight_fp8_t=weight_fp8_t,
        weight_scale_inv=weight_scale_inv,
    )
    expected = (x.float() @ (weight_fp8_t.float() * weight_scale_inv.float()).float()).to(dtype=x.dtype)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual.float(), expected.float(), atol=0.25, rtol=0.02)

    x_scale = _amax_to_scale(x.float().abs().amax(dim=1))
    x_fp8 = (x * x_scale.unsqueeze(1)).to(torch.float8_e4m3fn)
    quantized_actual = _fp8_row1_matvec(
        x,
        weight_fp8_t=weight_fp8_t,
        weight_scale_inv=weight_scale_inv,
        quantize_x=True,
    )
    quantized_expected = torch._scaled_mm(
        x_fp8,
        weight_fp8_t,
        scale_a=x_scale.reciprocal().reshape(-1, 1).contiguous(),
        scale_b=weight_scale_inv,
        out_dtype=x.dtype,
        use_fast_accum=False,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(quantized_actual.float(), quantized_expected.float(), atol=0.25, rtol=0.02)

    quantized_dot_actual = _fp8_row1_matvec(
        x,
        weight_fp8_t=weight_fp8_t,
        weight_scale_inv=weight_scale_inv,
        quantize_x=True,
        impl="dot",
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(quantized_dot_actual.float(), quantized_expected.float(), atol=0.25, rtol=0.02)


def test_fp8_row1_matvec_impl_env():
    old = os.environ.get("FASTDMS_FP8_ROW1_MATVEC_IMPL")
    try:
        os.environ["FASTDMS_FP8_ROW1_MATVEC_IMPL"] = "dot"
        assert fp8_row1_matvec_impl() == "dot"
        os.environ["FASTDMS_FP8_ROW1_MATVEC_IMPL"] = "scalar"
        assert fp8_row1_matvec_impl() == "scalar"
    finally:
        if old is None:
            os.environ.pop("FASTDMS_FP8_ROW1_MATVEC_IMPL", None)
        else:
            os.environ["FASTDMS_FP8_ROW1_MATVEC_IMPL"] = old


def test_int4_row1_matvec_matches_manual_quantized_linear():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(0)
    in_features = 64
    out_features = 16
    group_size = 32
    x = torch.randn((4, in_features), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((out_features, in_features), device="cuda", dtype=torch.float32)

    int4_weight = _quantize_weight_to_int4pack(weight, group_size=group_size, inner_k_tiles=2)
    assert int4_weight is not None
    weight_int4pack, weight_int4_scale_zeros, q_group_size = int4_weight
    actual = _int4_row1_matvec(
        x,
        weight_int4pack=weight_int4pack,
        weight_int4_scale_zeros=weight_int4_scale_zeros,
        q_group_size=q_group_size,
    )

    grouped = weight.reshape(out_features, in_features // group_size, group_size)
    amax = grouped.abs().amax(dim=2)
    scale = torch.where(amax > 0, amax / 7.0, torch.ones((), device=weight.device))
    quantized = torch.round(grouped / scale.unsqueeze(2)).clamp(-8, 7) * scale.unsqueeze(2)
    expected = x.float() @ quantized.reshape(out_features, in_features).t()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual.float(), expected.float(), atol=0.05, rtol=0.02)


def test_int4_row1_asymmetric_matvec_matches_manual_quantized_linear():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(0)
    in_features = 64
    out_features = 16
    group_size = 32
    x = torch.randn((4, in_features), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((out_features, in_features), device="cuda", dtype=torch.float32) + 0.35

    int4_weight = _quantize_weight_to_int4pack(
        weight,
        group_size=group_size,
        inner_k_tiles=2,
        quant_mode="asymmetric",
    )
    assert int4_weight is not None
    weight_int4pack, weight_int4_scale_zeros, q_group_size = int4_weight
    actual = _int4_row1_matvec(
        x,
        weight_int4pack=weight_int4pack,
        weight_int4_scale_zeros=weight_int4_scale_zeros,
        q_group_size=q_group_size,
    )

    grouped = weight.reshape(out_features, in_features // group_size, group_size)
    w_min = grouped.amin(dim=2)
    w_max = grouped.amax(dim=2)
    span = w_max - w_min
    scale = torch.where(span > 0, span / 15.0, torch.ones((), device=weight.device))
    zero = torch.where(span > 0, w_min + 8.0 * scale, w_min)
    q = torch.round((grouped - zero.unsqueeze(2)) / scale.unsqueeze(2) + 8.0).clamp(0, 15)
    quantized = (q - 8.0) * scale.unsqueeze(2) + zero.unsqueeze(2)
    expected = x.float() @ quantized.reshape(out_features, in_features).t()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual.float(), expected.float(), atol=0.08, rtol=0.03)


def test_int4_row1_quant_mode_env():
    old = os.environ.get("FASTDMS_INT4_ROW1_QUANT_MODE")
    try:
        os.environ.pop("FASTDMS_INT4_ROW1_QUANT_MODE", None)
        assert int4_row1_quant_mode() == "symmetric"
        os.environ["FASTDMS_INT4_ROW1_QUANT_MODE"] = "asymmetric"
        assert int4_row1_quant_mode() == "asymmetric"
        os.environ["FASTDMS_INT4_ROW1_QUANT_MODE"] = "affine"
        assert int4_row1_quant_mode() == "asymmetric"
    finally:
        if old is None:
            os.environ.pop("FASTDMS_INT4_ROW1_QUANT_MODE", None)
        else:
            os.environ["FASTDMS_INT4_ROW1_QUANT_MODE"] = old


def test_int4_row1_max_rows_env():
    old = os.environ.get("FASTDMS_INT4_ROW1_MAX_ROWS")
    try:
        os.environ.pop("FASTDMS_INT4_ROW1_MAX_ROWS", None)
        assert int4_row1_max_rows() == 1
        os.environ["FASTDMS_INT4_ROW1_MAX_ROWS"] = "8"
        assert int4_row1_max_rows() == 8
    finally:
        if old is None:
            os.environ.pop("FASTDMS_INT4_ROW1_MAX_ROWS", None)
        else:
            os.environ["FASTDMS_INT4_ROW1_MAX_ROWS"] = old


def test_int4_row1_scope_env():
    old_enabled = os.environ.get("FASTDMS_INT4_ROW1_WEIGHTS")
    old_scope = os.environ.get("FASTDMS_INT4_ROW1_SCOPE")
    try:
        os.environ.pop("FASTDMS_INT4_ROW1_WEIGHTS", None)
        os.environ.pop("FASTDMS_INT4_ROW1_SCOPE", None)
        assert not int4_row1_lm_head_enabled()
        assert int4_row1_module_in_scope("model.layers.0.mlp.gate_up_proj")

        os.environ["FASTDMS_INT4_ROW1_WEIGHTS"] = "1"
        os.environ["FASTDMS_INT4_ROW1_SCOPE"] = "mlp"
        assert int4_row1_module_in_scope("model.layers.0.mlp.gate_up_proj")
        assert int4_row1_module_in_scope("model.layers.0.mlp.down_proj")
        assert not int4_row1_module_in_scope("model.layers.0.self_attn.qkv_proj")
        assert not int4_row1_lm_head_enabled()

        os.environ["FASTDMS_INT4_ROW1_SCOPE"] = "mlp_lm_head"
        assert int4_row1_module_in_scope("model.layers.0.mlp.gate_up_proj")
        assert int4_row1_lm_head_enabled()

        os.environ["FASTDMS_INT4_ROW1_SCOPE"] = "mlp,qkv"
        assert int4_row1_module_in_scope("model.layers.0.mlp.gate_up_proj")
        assert int4_row1_module_in_scope("model.layers.0.self_attn.qkv_proj")
        assert not int4_row1_module_in_scope("model.layers.0.self_attn.o_proj")
        assert not int4_row1_lm_head_enabled()

        os.environ["FASTDMS_INT4_ROW1_SCOPE"] = "mlp+attn_out+lm_head"
        assert int4_row1_module_in_scope("model.layers.0.mlp.down_proj")
        assert not int4_row1_module_in_scope("model.layers.0.self_attn.qkv_proj")
        assert int4_row1_module_in_scope("model.layers.0.self_attn.o_proj")
        assert int4_row1_lm_head_enabled()

        os.environ["FASTDMS_INT4_ROW1_SCOPE"] = "all"
        assert int4_row1_module_in_scope("model.layers.0.self_attn.qkv_proj")
        assert int4_row1_lm_head_enabled()

        os.environ["FASTDMS_INT4_ROW1_SCOPE"] = "mlp+qkv+attn_out@0-7"
        assert int4_row1_module_in_scope("model.layers.3.mlp.gate_up_proj")
        assert int4_row1_module_in_scope("model.layers.12.mlp.down_proj")
        assert int4_row1_module_in_scope("model.layers.4.self_attn.qkv_proj")
        assert int4_row1_module_in_scope("model.layers.7.self_attn.o_proj")
        assert not int4_row1_module_in_scope("model.layers.8.self_attn.o_proj")
        assert not int4_row1_lm_head_enabled()

        os.environ["FASTDMS_INT4_ROW1_SCOPE"] = "attn_out@even"
        assert int4_row1_module_in_scope("model.layers.2.self_attn.o_proj")
        assert not int4_row1_module_in_scope("model.layers.3.self_attn.o_proj")
        assert not int4_row1_module_in_scope("lm_head")
    finally:
        for name, value in (
            ("FASTDMS_INT4_ROW1_WEIGHTS", old_enabled),
            ("FASTDMS_INT4_ROW1_SCOPE", old_scope),
        ):
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_fp8_down_row1_triton_truthy_env_uses_quantized():
    old = os.environ.get("FASTDMS_FP8_DOWN_ROW1_TRITON")
    try:
        os.environ.pop("FASTDMS_FP8_DOWN_ROW1_TRITON", None)
        assert fp8_down_row1_triton_mode() == "quantized"
        os.environ["FASTDMS_FP8_DOWN_ROW1_TRITON"] = "1"
        assert fp8_down_row1_triton_mode() == "quantized"
        os.environ["FASTDMS_FP8_DOWN_ROW1_TRITON"] = "raw_unsafe"
        assert fp8_down_row1_triton_mode() == "raw"
    finally:
        if old is None:
            os.environ.pop("FASTDMS_FP8_DOWN_ROW1_TRITON", None)
        else:
            os.environ["FASTDMS_FP8_DOWN_ROW1_TRITON"] = old


def test_fp8_row1_triton_env():
    old = os.environ.get("FASTDMS_FP8_ROW1_TRITON")
    try:
        os.environ.pop("FASTDMS_FP8_ROW1_TRITON", None)
        assert fp8_row1_triton_mode() == "off"
        assert not fp8_row1_triton_enabled()
        os.environ["FASTDMS_FP8_ROW1_TRITON"] = "1"
        assert fp8_row1_triton_mode() == "quantized"
        assert fp8_row1_triton_enabled()
        os.environ["FASTDMS_FP8_ROW1_TRITON"] = "raw_unsafe"
        assert fp8_row1_triton_mode() == "raw"
    finally:
        if old is None:
            os.environ.pop("FASTDMS_FP8_ROW1_TRITON", None)
        else:
            os.environ["FASTDMS_FP8_ROW1_TRITON"] = old


def test_fp8_gate_up_row1_triton_env():
    old = os.environ.get("FASTDMS_FP8_GATE_UP_ROW1_TRITON")
    try:
        os.environ.pop("FASTDMS_FP8_GATE_UP_ROW1_TRITON", None)
        assert fp8_gate_up_row1_triton_mode() == "quantized"
        assert fp8_gate_up_row1_triton_enabled()
        os.environ["FASTDMS_FP8_GATE_UP_ROW1_TRITON"] = "0"
        assert fp8_gate_up_row1_triton_mode() == "off"
        assert not fp8_gate_up_row1_triton_enabled()
        os.environ["FASTDMS_FP8_GATE_UP_ROW1_TRITON"] = "raw_unsafe"
        assert fp8_gate_up_row1_triton_mode() == "raw"
    finally:
        if old is None:
            os.environ.pop("FASTDMS_FP8_GATE_UP_ROW1_TRITON", None)
        else:
            os.environ["FASTDMS_FP8_GATE_UP_ROW1_TRITON"] = old


def test_fp8_attention_row1_shadow_env():
    old_full = os.environ.get("FASTDMS_FP8_ATTENTION_FULL")
    old_shadow = os.environ.get("FASTDMS_FP8_ATTENTION_ROW1_SHADOW")
    old_triton = os.environ.get("FASTDMS_FP8_ATTENTION_ROW1_TRITON")
    try:
        os.environ.pop("FASTDMS_FP8_ATTENTION_FULL", None)
        os.environ.pop("FASTDMS_FP8_ATTENTION_ROW1_SHADOW", None)
        os.environ.pop("FASTDMS_FP8_ATTENTION_ROW1_TRITON", None)
        assert fp8_attention_full_enabled()
        assert fp8_attention_row1_shadow_enabled()
        assert fp8_attention_row1_triton_mode() == "quantized"
        assert fp8_attention_row1_triton_enabled()

        os.environ["FASTDMS_FP8_ATTENTION_ROW1_SHADOW"] = "0"
        assert not fp8_attention_row1_shadow_enabled()
        assert fp8_attention_row1_triton_mode() == "quantized"
        assert fp8_attention_row1_triton_enabled()

        os.environ["FASTDMS_FP8_ATTENTION_FULL"] = "0"
        assert fp8_attention_row1_triton_mode() == "off"
        assert not fp8_attention_row1_triton_enabled()

        os.environ["FASTDMS_FP8_ATTENTION_ROW1_SHADOW"] = "1"
        os.environ.pop("FASTDMS_FP8_ATTENTION_ROW1_TRITON", None)

        os.environ["FASTDMS_FP8_ATTENTION_ROW1_TRITON"] = "0"
        assert fp8_attention_row1_triton_mode() == "off"
        assert not fp8_attention_row1_triton_enabled()

        os.environ["FASTDMS_FP8_ATTENTION_ROW1_TRITON"] = "raw_unsafe"
        assert fp8_attention_row1_triton_mode() == "raw"
        assert fp8_attention_row1_triton_enabled()
    finally:
        for name, value in (
            ("FASTDMS_FP8_ATTENTION_FULL", old_full),
            ("FASTDMS_FP8_ATTENTION_ROW1_SHADOW", old_shadow),
            ("FASTDMS_FP8_ATTENTION_ROW1_TRITON", old_triton),
        ):
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_fp8_mlp_bf16_backing_defaults_memory_pure():
    old_global = os.environ.get("FASTDMS_FP8_KEEP_BF16_WEIGHTS")
    old_gate = os.environ.get("FASTDMS_FP8_KEEP_BF16_GATE_UP")
    old_down = os.environ.get("FASTDMS_FP8_KEEP_BF16_DOWN")
    try:
        os.environ.pop("FASTDMS_FP8_KEEP_BF16_WEIGHTS", None)
        os.environ.pop("FASTDMS_FP8_KEEP_BF16_GATE_UP", None)
        os.environ.pop("FASTDMS_FP8_KEEP_BF16_DOWN", None)
        assert not fp8_keep_bf16_gate_up_weights()
        assert not fp8_keep_bf16_down_weights()

        os.environ["FASTDMS_FP8_KEEP_BF16_WEIGHTS"] = "1"
        assert fp8_keep_bf16_gate_up_weights()
        assert fp8_keep_bf16_down_weights()

        os.environ["FASTDMS_FP8_KEEP_BF16_GATE_UP"] = "0"
        os.environ["FASTDMS_FP8_KEEP_BF16_DOWN"] = "0"
        assert not fp8_keep_bf16_gate_up_weights()
        assert not fp8_keep_bf16_down_weights()
    finally:
        for name, value in (
            ("FASTDMS_FP8_KEEP_BF16_WEIGHTS", old_global),
            ("FASTDMS_FP8_KEEP_BF16_GATE_UP", old_gate),
            ("FASTDMS_FP8_KEEP_BF16_DOWN", old_down),
        ):
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_fp8_lm_head_row1_triton_env():
    old = os.environ.get("FASTDMS_FP8_LM_HEAD_ROW1_TRITON")
    try:
        os.environ["FASTDMS_FP8_LM_HEAD_ROW1_TRITON"] = "1"
        assert fp8_lm_head_row1_triton_enabled()
        assert _fp8_lm_head_row1_triton_allowed(rows=1, min_rows=1, has_bf16_weight=True)
        assert not _fp8_lm_head_row1_triton_allowed(rows=1, min_rows=2, has_bf16_weight=True)
        assert _fp8_lm_head_row1_triton_allowed(rows=1, min_rows=2, has_bf16_weight=False)
        os.environ["FASTDMS_FP8_LM_HEAD_ROW1_TRITON"] = "0"
        assert not fp8_lm_head_row1_triton_enabled()
        assert not _fp8_lm_head_row1_triton_allowed(rows=1, min_rows=1, has_bf16_weight=False)
    finally:
        if old is None:
            os.environ.pop("FASTDMS_FP8_LM_HEAD_ROW1_TRITON", None)
        else:
            os.environ["FASTDMS_FP8_LM_HEAD_ROW1_TRITON"] = old


def test_int4_lm_head_rerank_topk_env():
    old = os.environ.get("FASTDMS_INT4_ROW1_LM_HEAD_RERANK_TOPK")
    try:
        os.environ.pop("FASTDMS_INT4_ROW1_LM_HEAD_RERANK_TOPK", None)
        assert int4_lm_head_rerank_topk() == 0
        os.environ["FASTDMS_INT4_ROW1_LM_HEAD_RERANK_TOPK"] = "off"
        assert int4_lm_head_rerank_topk() == 0
        os.environ["FASTDMS_INT4_ROW1_LM_HEAD_RERANK_TOPK"] = "2"
        assert int4_lm_head_rerank_topk() == 2
        os.environ["FASTDMS_INT4_ROW1_LM_HEAD_RERANK_TOPK"] = "16"
        assert int4_lm_head_rerank_topk() == 16
    finally:
        if old is None:
            os.environ.pop("FASTDMS_INT4_ROW1_LM_HEAD_RERANK_TOPK", None)
        else:
            os.environ["FASTDMS_INT4_ROW1_LM_HEAD_RERANK_TOPK"] = old


def test_fp8_lm_head_rerank_topk_matches_full_argmax():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(0)
    in_features = 64
    out_features = 64
    x = torch.randn((1, in_features), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((out_features, in_features), device="cuda", dtype=torch.float32)
    weight_scale = _amax_to_scale(weight.abs().amax(dim=1).float())
    weight_fp8_t = _as_col_major_b((weight.t() * weight_scale.unsqueeze(0)).to(torch.float8_e4m3fn))
    weight_scale_inv = weight_scale.reciprocal().reshape(1, -1).contiguous()

    expected = _fp8_row1_argmax(
        x,
        weight_fp8_t=weight_fp8_t,
        weight_scale_inv=weight_scale_inv,
    )
    other = (int(expected.item()) + 1) % out_features
    candidates = torch.tensor([[other, int(expected.item())]], device=x.device, dtype=torch.int64)
    actual = _fp8_row1_rerank_topk(
        x,
        candidates,
        weight_fp8_t=weight_fp8_t,
        weight_scale_inv=weight_scale_inv,
    )
    torch.cuda.synchronize()
    assert actual.item() == expected.item()


def test_fp8_embedding_env_defaults_memory_pure():
    old_enabled = os.environ.get("FASTDMS_FP8_EMBEDDING")
    old_keep = os.environ.get("FASTDMS_FP8_KEEP_BF16_EMBEDDING")
    old_share = os.environ.get("FASTDMS_FP8_EMBEDDING_SHARE_LM_HEAD")
    try:
        os.environ.pop("FASTDMS_FP8_EMBEDDING", None)
        os.environ.pop("FASTDMS_FP8_KEEP_BF16_EMBEDDING", None)
        os.environ.pop("FASTDMS_FP8_EMBEDDING_SHARE_LM_HEAD", None)
        assert fp8_embedding_enabled()
        assert not fp8_keep_bf16_embedding_weight()
        assert not fp8_embedding_share_lm_head_weight()
        os.environ["FASTDMS_FP8_EMBEDDING"] = "0"
        assert not fp8_embedding_enabled()
        os.environ["FASTDMS_FP8_EMBEDDING"] = "1"
        os.environ["FASTDMS_FP8_KEEP_BF16_EMBEDDING"] = "1"
        assert fp8_keep_bf16_embedding_weight()
        os.environ["FASTDMS_FP8_EMBEDDING_SHARE_LM_HEAD"] = "1"
        assert fp8_embedding_share_lm_head_weight()
    finally:
        for name, value in (
            ("FASTDMS_FP8_EMBEDDING", old_enabled),
            ("FASTDMS_FP8_KEEP_BF16_EMBEDDING", old_keep),
            ("FASTDMS_FP8_EMBEDDING_SHARE_LM_HEAD", old_share),
        ):
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _manual_embedding(weight: torch.Tensor) -> VocabParallelEmbedding:
    module = VocabParallelEmbedding.__new__(VocabParallelEmbedding)
    torch.nn.Module.__init__(module)
    module.tp_size = 1
    module.weight = torch.nn.Parameter(weight)
    module._fp8_embedding_enabled = False
    module._fp8_embedding_weight = None
    module._fp8_embedding_scale_inv = None
    module._fp8_embedding_output_dtype = None
    return module


def _manual_lm_head(weight: torch.Tensor) -> ParallelLMHead:
    module = ParallelLMHead.__new__(ParallelLMHead)
    torch.nn.Module.__init__(module)
    module.tp_size = 1
    module.tp_rank = 0
    module.weight = torch.nn.Parameter(weight)
    module._fp8_weight_enabled = False
    module._fp8_min_rows = 1
    return module


def test_fp8_embedding_can_drop_bf16_and_share_lm_head_storage():
    weight = torch.randn((16, 8), dtype=torch.bfloat16)
    lm_head = _manual_lm_head(weight.clone())
    lm_head.quantize_weight_to_fp8(keep_bf16_weight=False)
    embedding = _manual_embedding(weight.clone())
    embedding.enable_fp8_embedding_from_lm_head(lm_head, keep_bf16_weight=False)

    assert embedding.weight.numel() == 0
    assert embedding._fp8_embedding_weight.untyped_storage().data_ptr() == lm_head.weight_fp8_t.untyped_storage().data_ptr()

    ids = torch.tensor([0, 3, 7, 3], dtype=torch.long)
    out = embedding(ids)
    assert out.shape == (4, 8)
    assert out.dtype == torch.bfloat16


def test_fused_dms_rope_store_compact_decode_matches_reference():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(19)
    batch = 3
    num_kv_heads = 2
    group_size = 2
    num_q_heads = num_kv_heads * group_size
    head_dim = 32
    q = torch.randn((batch, num_q_heads, head_dim), device="cuda", dtype=torch.float32)
    k = torch.randn((batch, num_kv_heads, head_dim), device="cuda", dtype=torch.float32)
    v = torch.randn_like(k)
    q[:, 0::group_size, -1] = torch.tensor(
        [[0.01, 0.06], [0.05, 0.08], [0.10, -0.01]],
        device="cuda",
        dtype=torch.float32,
    )
    positions = torch.tensor([0, 3, 7], device="cuda", dtype=torch.int64)
    slot_mapping = torch.tensor(
        [[0, 3], [6, 9], [12, 15]],
        device="cuda",
        dtype=torch.int32,
    )
    rope = get_rope(head_dim, rotary_dim=head_dim, max_position=16, base=10000).to("cuda")

    q_ref, evict_ref = extract_dms_eviction_decisions(
        q.clone(),
        num_kv_heads=num_kv_heads,
        num_qo_heads=num_q_heads,
        head_dim=head_dim,
        alpha_scale=100.0,
        alpha_offset=5.0,
        inplace=True,
    )
    cos_sin = rope.cos_sin_cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    q_ref = apply_rotary_emb(q_ref, cos, sin)
    k_ref = apply_rotary_emb(k.clone(), cos, sin)
    compact_k_ref = torch.zeros((18, head_dim), device="cuda", dtype=torch.float32)
    compact_v_ref = torch.zeros_like(compact_k_ref)
    compact_attention.store_compact_kvcache(k_ref, v, compact_k_ref, compact_v_ref, slot_mapping)

    q_fused = q.clone()
    k_fused = k.clone()
    evict_fused = torch.empty((batch, num_kv_heads), device="cuda", dtype=torch.bool)
    compact_k_fused = torch.zeros_like(compact_k_ref)
    compact_v_fused = torch.zeros_like(compact_v_ref)
    fused_dms_rope_store_compact_decode(
        q_fused,
        k_fused,
        v,
        rope.cos_sin_cache,
        compact_k_fused,
        compact_v_fused,
        slot_mapping,
        evict_fused,
        positions,
        num_kv_heads=num_kv_heads,
        alpha_scale=100.0,
        alpha_offset=5.0,
    )
    torch.cuda.synchronize()

    assert torch.equal(evict_fused.cpu(), evict_ref.cpu())
    assert torch.allclose(q_fused.cpu(), q_ref.cpu(), atol=1e-5, rtol=1e-5)
    assert torch.allclose(k_fused.cpu(), k_ref.cpu(), atol=1e-5, rtol=1e-5)
    assert torch.allclose(compact_k_fused.cpu(), compact_k_ref.cpu(), atol=1e-5, rtol=1e-5)
    assert torch.allclose(compact_v_fused.cpu(), compact_v_ref.cpu(), atol=1e-5, rtol=1e-5)

    q_skip_k = q.clone()
    k_skip_k = k.clone()
    evict_skip_k = torch.empty_like(evict_fused)
    compact_k_skip_k = torch.zeros_like(compact_k_ref)
    compact_v_skip_k = torch.zeros_like(compact_v_ref)
    fused_dms_rope_store_compact_decode(
        q_skip_k,
        k_skip_k,
        v,
        rope.cos_sin_cache,
        compact_k_skip_k,
        compact_v_skip_k,
        slot_mapping,
        evict_skip_k,
        positions,
        num_kv_heads=num_kv_heads,
        alpha_scale=100.0,
        alpha_offset=5.0,
        store_transient_k=False,
    )
    torch.cuda.synchronize()

    assert torch.equal(evict_skip_k.cpu(), evict_ref.cpu())
    assert torch.allclose(q_skip_k.cpu(), q_ref.cpu(), atol=1e-5, rtol=1e-5)
    assert torch.equal(k_skip_k.cpu(), k.cpu())
    assert torch.allclose(compact_k_skip_k.cpu(), compact_k_ref.cpu(), atol=1e-5, rtol=1e-5)
    assert torch.allclose(compact_v_skip_k.cpu(), compact_v_ref.cpu(), atol=1e-5, rtol=1e-5)

    q_inline = q.clone()
    k_inline = k.clone()
    evict_inline = torch.empty_like(evict_fused)
    compact_k_inline = torch.zeros_like(compact_k_ref)
    compact_v_inline = torch.zeros_like(compact_v_ref)
    fused_dms_rope_store_compact_decode(
        q_inline,
        k_inline,
        v,
        rope.cos_sin_cache,
        compact_k_inline,
        compact_v_inline,
        slot_mapping,
        evict_inline,
        positions,
        num_kv_heads=num_kv_heads,
        alpha_scale=100.0,
        alpha_offset=5.0,
        store_q=False,
    )
    torch.cuda.synchronize()

    live_counts = torch.ones_like(slot_mapping)
    out_ref = compact_decode_attention(
        q_ref,
        compact_k_ref,
        compact_v_ref,
        slot_mapping,
        live_counts,
        max_live_count=1,
        block_n=16,
    )
    out_inline = compact_decode_attention(
        q_inline,
        compact_k_inline,
        compact_v_inline,
        slot_mapping,
        live_counts,
        max_live_count=1,
        block_n=16,
        q_rope_positions=positions,
        q_rope_cos_sin_cache=rope.cos_sin_cache,
    )
    torch.cuda.synchronize()

    assert torch.equal(evict_inline.cpu(), evict_ref.cpu())
    assert torch.allclose(q_inline.cpu(), q.cpu(), atol=0, rtol=0)
    assert torch.allclose(k_inline.cpu(), k_ref.cpu(), atol=1e-5, rtol=1e-5)
    assert torch.allclose(compact_k_inline.cpu(), compact_k_ref.cpu(), atol=1e-5, rtol=1e-5)
    assert torch.allclose(compact_v_inline.cpu(), compact_v_ref.cpu(), atol=1e-5, rtol=1e-5)
    assert torch.allclose(out_inline.cpu(), out_ref.cpu(), atol=1e-4, rtol=1e-4)


def test_streaming_pack_scatter_triton_matches_indexed_reference():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(23)
    for dtype in (torch.float32, torch.float8_e4m3fn):
        seq_len = 37
        num_kv_heads = 3
        head_dim = 16
        k = torch.randn((seq_len, num_kv_heads, head_dim), device="cuda", dtype=torch.float32).to(dtype)
        v = torch.randn((seq_len, num_kv_heads, head_dim), device="cuda", dtype=torch.float32).to(dtype)
        keep = torch.rand((seq_len, num_kv_heads), device="cuda") > 0.35
        keep[-1] = True
        rank = keep.to(torch.int32).cumsum(dim=0) - 1
        live = keep.sum(dim=0, dtype=torch.int32)
        base_offsets = torch.tensor([0, 48, 96], device="cuda", dtype=torch.int32)
        seq_evict = torch.rand((seq_len, num_kv_heads), device="cuda") > 0.5

        compact_k_ref = torch.zeros((144, head_dim), device="cuda", dtype=dtype)
        compact_v_ref = torch.zeros_like(compact_k_ref)
        token_pos_ref = torch.full((num_kv_heads, 64), -1, device="cuda", dtype=torch.int32)
        evict_ref = torch.zeros((num_kv_heads, 64), device="cuda", dtype=torch.bool)

        t_idx, h_idx = torch.nonzero(keep, as_tuple=True)
        dst_rank = rank[t_idx, h_idx].to(torch.long)
        dst_slot = base_offsets.to(torch.long)[h_idx] + dst_rank
        compact_k_ref[dst_slot] = k[t_idx, h_idx]
        compact_v_ref[dst_slot] = v[t_idx, h_idx]
        token_pos_ref[h_idx, dst_rank] = t_idx.to(torch.int32)
        evict_ref[h_idx, dst_rank] = seq_evict[t_idx, h_idx]

        compact_k_out = torch.zeros_like(compact_k_ref)
        compact_v_out = torch.zeros_like(compact_v_ref)
        token_pos_out = torch.full_like(token_pos_ref, -1)
        evict_out = torch.zeros_like(evict_ref)
        _streaming_pack_scatter_triton(
            k=k,
            v=v,
            keep=keep,
            rank=rank,
            seq_evict=seq_evict,
            compact_k=compact_k_out,
            compact_v=compact_v_out,
            token_positions=token_pos_out,
            evict_mask=evict_out,
            base_offsets=base_offsets,
            start=0,
            seq_len=seq_len,
        )
        torch.cuda.synchronize()

        assert live.tolist() == [int(keep[:, h].sum().item()) for h in range(num_kv_heads)]
        assert torch.allclose(compact_k_out.float().cpu(), compact_k_ref.float().cpu(), atol=0, rtol=0)
        assert torch.allclose(compact_v_out.float().cpu(), compact_v_ref.float().cpu(), atol=0, rtol=0)
        assert torch.equal(token_pos_out.cpu(), token_pos_ref.cpu())
        assert torch.equal(evict_out.cpu(), evict_ref.cpu())


def test_streaming_pack_fused_rank_triton_matches_indexed_reference():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(29)
    for dtype in (torch.float32, torch.float8_e4m3fn):
        seq_len = 541
        window_size = 31
        num_kv_heads = 3
        head_dim = 16
        k = torch.randn((seq_len, num_kv_heads, head_dim), device="cuda", dtype=torch.float32).to(dtype)
        v = torch.randn((seq_len, num_kv_heads, head_dim), device="cuda", dtype=torch.float32).to(dtype)
        seq_evict = torch.rand((seq_len, num_kv_heads), device="cuda") > 0.5
        positions = torch.arange(seq_len, device="cuda", dtype=torch.int32)
        in_window = positions >= max(0, seq_len - 1 - window_size)
        keep = (~seq_evict) | in_window.unsqueeze(1)
        rank = keep.to(torch.int32).cumsum(dim=0) - 1
        live = keep.sum(dim=0, dtype=torch.int32)
        base_offsets = torch.tensor([0, 640, 1280], device="cuda", dtype=torch.int32)

        compact_k_ref = torch.zeros((1920, head_dim), device="cuda", dtype=dtype)
        compact_v_ref = torch.zeros_like(compact_k_ref)
        token_pos_ref = torch.full((num_kv_heads, 700), -1, device="cuda", dtype=torch.int32)
        evict_ref = torch.zeros((num_kv_heads, 700), device="cuda", dtype=torch.bool)

        t_idx, h_idx = torch.nonzero(keep, as_tuple=True)
        dst_rank = rank[t_idx, h_idx].to(torch.long)
        dst_slot = base_offsets.to(torch.long)[h_idx] + dst_rank
        compact_k_ref[dst_slot] = k[t_idx, h_idx]
        compact_v_ref[dst_slot] = v[t_idx, h_idx]
        token_pos_ref[h_idx, dst_rank] = t_idx.to(torch.int32)
        evict_ref[h_idx, dst_rank] = seq_evict[t_idx, h_idx]

        live_out, block_offsets = _streaming_pack_fused_rank_prepare(
            seq_evict=seq_evict,
            seq_len=seq_len,
            window_size=window_size,
        )
        compact_k_out = torch.zeros_like(compact_k_ref)
        compact_v_out = torch.zeros_like(compact_v_ref)
        token_pos_out = torch.full_like(token_pos_ref, -1)
        evict_out = torch.zeros_like(evict_ref)
        _streaming_pack_rank_scatter_triton(
            k=k,
            v=v,
            seq_evict=seq_evict,
            block_offsets=block_offsets,
            compact_k=compact_k_out,
            compact_v=compact_v_out,
            token_positions=token_pos_out,
            evict_mask=evict_out,
            base_offsets=base_offsets,
            start=0,
            seq_len=seq_len,
            window_size=window_size,
        )
        torch.cuda.synchronize()

        assert torch.equal(live_out.cpu(), live.cpu())
        assert torch.allclose(compact_k_out.float().cpu(), compact_k_ref.float().cpu(), atol=0, rtol=0)
        assert torch.allclose(compact_v_out.float().cpu(), compact_v_ref.float().cpu(), atol=0, rtol=0)
        assert torch.equal(token_pos_out.cpu(), token_pos_ref.cpu())
        assert torch.equal(evict_out.cpu(), evict_ref.cpu())


def test_compact_decode_attention_splitk_matches_default():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(17)
    q = torch.randn((2, 8, 16), device="cuda", dtype=torch.float32)
    compact_k = torch.randn((96, 16), device="cuda", dtype=torch.float32)
    compact_v = torch.randn((96, 16), device="cuda", dtype=torch.float32)
    base_offsets = torch.tensor([[0, 24], [48, 72]], device="cuda", dtype=torch.int32)
    live_counts = torch.tensor([[17, 13], [19, 11]], device="cuda", dtype=torch.int32)

    old_splitk = compact_attention._COMPACT_ATTN_SPLITK
    try:
        compact_attention._COMPACT_ATTN_SPLITK = False
        default_out = compact_decode_attention(
            q,
            compact_k,
            compact_v,
            base_offsets,
            live_counts,
            max_live_count=20,
            block_n=16,
        )
        compact_attention._COMPACT_ATTN_SPLITK = True
        splitk_out = compact_decode_attention(
            q,
            compact_k,
            compact_v,
            base_offsets,
            live_counts,
            max_live_count=20,
            block_n=16,
        )
    finally:
        compact_attention._COMPACT_ATTN_SPLITK = old_splitk
    torch.cuda.synchronize()

    assert torch.allclose(splitk_out.cpu(), default_out.cpu(), atol=1e-4, rtol=1e-4)


def test_dms_expiry_triton_matches_python_for_non_power_of_two_capacity():
    if not torch.cuda.is_available():
        return
    dtypes = [torch.float32]
    if hasattr(torch, "float8_e4m3fn"):
        dtypes.append(torch.float8_e4m3fn)
    for dtype in dtypes:
        for layer_major_metadata in (False, True):
            py_manager, py_seqs = _build_cuda_expiry_case(dtype, layer_major_metadata)
            triton_manager, triton_seqs = _build_cuda_expiry_case(dtype, layer_major_metadata)

            old_env = os.environ.get("FASTDMS_DMS_EXPIRY_TRITON")
            old_interval = os.environ.get("FASTDMS_DMS_EXPIRY_INTERVAL")
            try:
                os.environ["FASTDMS_DMS_EXPIRY_TRITON"] = "0"
                py_stats = py_manager.apply_dms_evictions(py_seqs, [8], window_size=3)
                triton_stats = triton_manager._apply_dms_evictions_triton(triton_seqs, [8], window_size=3)
                torch.cuda.synchronize()
                assert py_stats == triton_stats
                _assert_cuda_expiry_managers_equal(py_manager, triton_manager)

                _append_cuda_expiry_token(py_manager, 8, dtype)
                _append_cuda_expiry_token(triton_manager, 8, dtype)
                os.environ["FASTDMS_DMS_EXPIRY_TRITON"] = "0"
                py_manager.apply_dms_evictions(py_seqs, [9], window_size=3)
                os.environ["FASTDMS_DMS_EXPIRY_TRITON"] = "1"
                os.environ["FASTDMS_DMS_EXPIRY_INTERVAL"] = "1"
                triton_manager.apply_dms_evictions(triton_seqs, [9], window_size=3)
                _append_cuda_expiry_token(py_manager, 9, dtype)
                _append_cuda_expiry_token(triton_manager, 9, dtype)
                os.environ["FASTDMS_DMS_EXPIRY_TRITON"] = "0"
                py_manager.apply_dms_evictions(py_seqs, [10], window_size=3)
                os.environ["FASTDMS_DMS_EXPIRY_TRITON"] = "1"
                os.environ["FASTDMS_DMS_EXPIRY_INTERVAL"] = "1"
                triton_manager.apply_dms_evictions(triton_seqs, [10], window_size=3)
            finally:
                if old_env is None:
                    os.environ.pop("FASTDMS_DMS_EXPIRY_TRITON", None)
                else:
                    os.environ["FASTDMS_DMS_EXPIRY_TRITON"] = old_env
                if old_interval is None:
                    os.environ.pop("FASTDMS_DMS_EXPIRY_INTERVAL", None)
                else:
                    os.environ["FASTDMS_DMS_EXPIRY_INTERVAL"] = old_interval

            torch.cuda.synchronize()
            _assert_cuda_expiry_managers_equal(py_manager, triton_manager)


if __name__ == "__main__":
    test_pack_compact_kv_reference_decode()
    test_streaming_pack_cache_live_env()
    test_streaming_pack_fused_rank_env()
    test_dms_decode_store_transient_k_env()
    test_compact_kv_manager_allocates_and_releases_sequence_spans()
    test_layer_major_metadata_state_views_reuse_active_slots()
    test_compact_kv_manager_prepares_decode_metadata_and_append_slots()
    test_record_dms_decode_decisions_accepts_layer_major_tensor()
    test_compact_kv_manager_accepts_layer_head_live_counts()
    test_emulated_dms_live_mask_keeps_shifted_stride_and_recent_tokens()
    test_dms_extract_triton_matches_torch_and_zeroes_borrowed_channel()
    test_fp8_row1_matvec_matches_dequantized_linear()
    test_fp8_row1_matvec_impl_env()
    test_int4_row1_matvec_matches_manual_quantized_linear()
    test_int4_row1_asymmetric_matvec_matches_manual_quantized_linear()
    test_int4_row1_quant_mode_env()
    test_int4_row1_max_rows_env()
    test_int4_row1_scope_env()
    test_fp8_down_row1_triton_truthy_env_uses_quantized()
    test_fp8_row1_triton_env()
    test_fp8_gate_up_row1_triton_env()
    test_fp8_attention_row1_shadow_env()
    test_fp8_mlp_bf16_backing_defaults_memory_pure()
    test_fp8_lm_head_row1_triton_env()
    test_int4_lm_head_rerank_topk_env()
    test_fp8_lm_head_rerank_topk_matches_full_argmax()
    test_fp8_embedding_env_defaults_memory_pure()
    test_fp8_embedding_can_drop_bf16_and_share_lm_head_storage()
    test_fused_dms_rope_store_compact_decode_matches_reference()
    test_streaming_pack_scatter_triton_matches_indexed_reference()
    test_streaming_pack_fused_rank_triton_matches_indexed_reference()
    test_compact_decode_attention_splitk_matches_default()
    test_dms_expiry_triton_matches_python_for_non_power_of_two_capacity()
