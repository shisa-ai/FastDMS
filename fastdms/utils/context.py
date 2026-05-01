from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    compact_kv_enabled: bool = False
    compact_base_offsets: torch.Tensor | None = None
    compact_live_counts: torch.Tensor | None = None
    compact_slot_mapping: torch.Tensor | None = None
    compact_max_live_count: int = 0
    dms_capture_enabled: bool = False
    dms_alpha_scale: float = 100.0
    dms_alpha_offset: float = 5.0
    dms_layer_evictions: dict[int, torch.Tensor] | None = None
    dms_layer_evictions_buffer: torch.Tensor | None = None
    # J1 streaming pack hook
    streaming_pack_manager: object | None = None
    streaming_pack_seq_ids: list[int] | None = None
    streaming_pack_window_size: int = 0

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(
    is_prefill,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    context_lens=None,
    block_tables=None,
    compact_kv_enabled=False,
    compact_base_offsets=None,
    compact_live_counts=None,
    compact_slot_mapping=None,
    compact_max_live_count=0,
    dms_capture_enabled=False,
    dms_alpha_scale=100.0,
    dms_alpha_offset=5.0,
    dms_layer_evictions=None,
    dms_layer_evictions_buffer=None,
    streaming_pack_manager=None,
    streaming_pack_seq_ids=None,
    streaming_pack_window_size=0,
):
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        slot_mapping,
        context_lens,
        block_tables,
        compact_kv_enabled,
        compact_base_offsets,
        compact_live_counts,
        compact_slot_mapping,
        compact_max_live_count,
        dms_capture_enabled,
        dms_alpha_scale,
        dms_alpha_offset,
        dms_layer_evictions,
        dms_layer_evictions_buffer,
        streaming_pack_manager,
        streaming_pack_seq_ids,
        streaming_pack_window_size,
    )

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
