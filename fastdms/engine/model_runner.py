import os
import pickle
from dataclasses import replace

import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from fastdms.config import Config
from fastdms.engine.sequence import Sequence, SequenceStatus
from fastdms.engine.compact_kv import CompactKVManager, build_emulated_dms_live_mask
from fastdms.engine.dms import DMSMetadata, build_dms_live_mask, load_dms_metadata
from fastdms.models.registry import build_model
from fastdms.layers.embed_head import (
    fp8_embedding_enabled,
    fp8_embedding_share_lm_head_weight,
    fp8_keep_bf16_embedding_weight,
    fp8_keep_bf16_lm_head_weight,
    fp8_lm_head_enabled,
    greedy_fused_argmax_enabled,
)
from fastdms.layers.linear import convert_linear_modules_to_fp8, fp8_weights_enabled
from fastdms.layers.sampler import Sampler
from fastdms.utils.context import set_context, get_context, reset_context
from fastdms.utils.loader import load_model
from fastdms.utils.profiler import get_profiler


def compact_greedy_fast_loop_enabled() -> bool:
    value = os.environ.get("FASTDMS_COMPACT_GREEDY_FAST_LOOP", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.compact_kv_manager = None
        self.compact_kv_cache = None
        self.dms_metadata: DMSMetadata | None = self.load_dms_metadata()
        self.last_compact_prefill_stats: list[dict] = []
        self.last_dms_decode_stats: list[dict] = []
        self.compact_live_tokens_peak = 0
        self.compact_live_tokens_peak_device: torch.Tensor | None = None
        graph_mode = os.environ.get("FASTDMS_COMPACT_DECODE_CUDAGRAPH", "auto").strip().lower()
        graph_default = config.compact_kv_enabled and config.compact_kv_retention_mode == "dms"
        self.compact_decode_cudagraph = graph_mode in {"1", "true", "yes", "on"} or (
            graph_mode in {"", "auto"} and graph_default
        )
        self.compact_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.compact_graph_vars: dict[int, dict] = {}
        self.compact_graph_storage_signature: dict[int, tuple] = {}
        self.compact_greedy_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.compact_greedy_graph_vars: dict[int, dict] = {}
        self.compact_greedy_graph_storage_signature: dict[int, tuple] = {}
        self.compact_graph_pool = None

        dist_port = int(os.environ.get("FASTDMS_DIST_PORT", "2333"))
        dist.init_process_group("nccl", f"tcp://localhost:{dist_port}", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = build_model(hf_config)
        load_model(self.model, config.model)
        self.fp8_weight_modules = convert_linear_modules_to_fp8(self.model) if fp8_weights_enabled() else 0
        self.fp8_lm_head = False
        self.fp8_embedding = False
        fp8_embedding = fp8_weights_enabled() and fp8_embedding_enabled()
        if fp8_weights_enabled() and fp8_lm_head_enabled():
            lm_head = getattr(self.model, "lm_head", None)
            if lm_head is None or not hasattr(lm_head, "quantize_weight_to_fp8"):
                raise RuntimeError("FASTDMS_FP8_LM_HEAD requires a quantizable lm_head")
            embed_tokens = getattr(getattr(self.model, "model", None), "embed_tokens", None)
            tied_embedding = (
                fp8_embedding
                and embed_tokens is not None
                and hasattr(embed_tokens, "weight")
                and hasattr(lm_head, "weight")
                and embed_tokens.weight.untyped_storage().data_ptr() == lm_head.weight.untyped_storage().data_ptr()
            )
            keep_lm_head_weight = fp8_keep_bf16_lm_head_weight()
            if tied_embedding and not fp8_keep_bf16_embedding_weight():
                keep_lm_head_weight = False
            lm_head.quantize_weight_to_fp8(keep_bf16_weight=keep_lm_head_weight)
            self.fp8_lm_head = True
        if fp8_embedding:
            embed_tokens = getattr(getattr(self.model, "model", None), "embed_tokens", None)
            if embed_tokens is None or not hasattr(embed_tokens, "quantize_embedding_to_fp8"):
                raise RuntimeError("FASTDMS_FP8_EMBEDDING requires quantizable embed_tokens")
            lm_head = getattr(self.model, "lm_head", None)
            use_lm_head_storage = bool(
                fp8_embedding_share_lm_head_weight()
                and self.fp8_lm_head
                and lm_head is not None
                and hasattr(lm_head, "weight_fp8_t")
                and hasattr(lm_head, "weight_scale_inv")
            )
            if use_lm_head_storage:
                embed_tokens.enable_fp8_embedding_from_lm_head(
                    lm_head,
                    keep_bf16_weight=fp8_keep_bf16_embedding_weight(),
                )
            else:
                embed_tokens.quantize_embedding_to_fp8(
                    keep_bf16_weight=fp8_keep_bf16_embedding_weight(),
                )
            self.fp8_embedding = True
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="fastdms", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="fastdms")
                self.loop()

    def load_dms_metadata(self) -> DMSMetadata | None:
        if self.config.compact_kv_retention_mode != "dms":
            return None
        metadata = load_dms_metadata(self.config.model, self.config.dms_metadata_path)
        overrides = {}
        if self.config.dms_window_size is not None:
            overrides["window_size"] = self.config.dms_window_size
        if self.config.dms_alpha_scale is not None:
            overrides["alpha_scale"] = self.config.dms_alpha_scale
        if self.config.dms_alpha_offset is not None:
            overrides["alpha_offset"] = self.config.dms_alpha_offset
        if overrides:
            metadata = replace(metadata, **overrides)
        return metadata

    def reset_compact_live_tokens_peak(self) -> None:
        self.compact_live_tokens_peak = 0
        if self.compact_live_tokens_peak_device is not None:
            self.compact_live_tokens_peak_device.zero_()

    def update_compact_live_tokens_peak(self, live_counts: torch.Tensor) -> None:
        if self.compact_live_tokens_peak_device is None:
            self.compact_live_tokens_peak_device = torch.zeros((), dtype=torch.int64, device=live_counts.device)
        current_live_tokens = torch.sum(live_counts, dtype=torch.int64)
        torch.maximum(self.compact_live_tokens_peak_device, current_live_tokens, out=self.compact_live_tokens_peak_device)

    def finalize_compact_live_tokens_peak(self) -> int:
        if self.compact_live_tokens_peak_device is not None:
            self.compact_live_tokens_peak = max(
                self.compact_live_tokens_peak,
                int(self.compact_live_tokens_peak_device.item()),
            )
        return self.compact_live_tokens_peak

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        if self.compact_graphs:
            del self.compact_graphs, self.compact_graph_vars
        if self.compact_greedy_graphs:
            del self.compact_greedy_graphs, self.compact_greedy_graph_vars
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        kv_cache_dtype = config.resolve_kv_cache_dtype()
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * kv_cache_dtype.itemsize
        available_bytes = int(total * config.gpu_memory_utilization - used - peak + current)
        # P0.3: explicit token-pool override forces a fixed dense KV pool independent
        # of gpu_memory_utilization. Useful for byte-identical comparisons against
        # mini-sglang anchors. token_pool_tokens is converted to a block count.
        if config.num_page_override is not None:
            forced_blocks = int(config.num_page_override)
        elif config.token_pool_tokens is not None:
            forced_blocks = int(config.token_pool_tokens) // self.block_size
        else:
            forced_blocks = None
        # J1 streaming compact pack: when DMS mode is active, dense KV is not
        # used at all (per-layer pack writes straight to compact arena from
        # in-flight post-RoPE k/v). Use a 1-block placeholder so module
        # k_cache numel()==block_size>0... actually we want numel()==0 so the
        # store_kvcache gate skips. Use 0 blocks and special-case kv_cache.
        # B1.1 (legacy): when compact_kv_enabled but NOT streaming pack mode,
        # dense KV is *prefill scratch only*. Size it to one prefill batch.
        streaming_pack_mode = config.compact_kv_enabled and config.compact_kv_retention_mode == "dms"
        compact_dense_scratch = config.compact_kv_enabled and config.compact_kv_retention_mode == "dms"
        if compact_dense_scratch:
            scratch_blocks = (config.max_num_batched_tokens + self.block_size - 1) // self.block_size
            scratch_blocks = max(scratch_blocks, 1)
        if streaming_pack_mode:
            # Override: streaming pack does not need any dense scratch; use 0
            # blocks so the kv_cache tensor is shape-zero on dim 2 (numel == 0).
            scratch_blocks = 0
        compact_capacity_tokens = 0
        if config.compact_kv_enabled:
            # B1.3: explicit compact arena budget. If unset, fall back to legacy
            # half-memory split and arena = num_blocks * block_size * num_kv_heads.
            if config.compact_kv_budget_bytes is not None:
                compact_bytes = int(config.compact_kv_budget_bytes)
                compact_capacity_tokens = compact_bytes // (
                    2 * hf_config.num_hidden_layers * head_dim * kv_cache_dtype.itemsize
                )
                assert compact_capacity_tokens > 0, "compact_kv_budget_bytes too small"
                available_bytes -= compact_bytes
                if compact_dense_scratch:
                    config.num_kvcache_blocks = scratch_blocks
                elif forced_blocks is not None:
                    config.num_kvcache_blocks = forced_blocks
                else:
                    config.num_kvcache_blocks = available_bytes // block_bytes
            elif config.compact_kv_capacity_tokens > 0:
                compact_capacity_tokens = config.compact_kv_capacity_tokens
                compact_bytes = 2 * hf_config.num_hidden_layers * compact_capacity_tokens * head_dim * kv_cache_dtype.itemsize
                available_bytes -= compact_bytes
                if compact_dense_scratch:
                    config.num_kvcache_blocks = scratch_blocks
                elif forced_blocks is not None:
                    config.num_kvcache_blocks = forced_blocks
                else:
                    config.num_kvcache_blocks = available_bytes // block_bytes
            else:
                if compact_dense_scratch:
                    config.num_kvcache_blocks = scratch_blocks
                    # Reserve the rest of the budget (minus scratch) for the
                    # compact arena.
                    scratch_bytes = scratch_blocks * block_bytes
                    available_for_compact = max(0, available_bytes - scratch_bytes)
                    compact_capacity_tokens = available_for_compact // (
                        2 * hf_config.num_hidden_layers * head_dim * kv_cache_dtype.itemsize
                    )
                elif forced_blocks is not None:
                    config.num_kvcache_blocks = forced_blocks
                    compact_capacity_tokens = config.num_kvcache_blocks * self.block_size * num_kv_heads
                else:
                    config.num_kvcache_blocks = available_bytes // (2 * block_bytes)
                    compact_capacity_tokens = config.num_kvcache_blocks * self.block_size * num_kv_heads
        else:
            if forced_blocks is not None:
                config.num_kvcache_blocks = forced_blocks
            else:
                config.num_kvcache_blocks = available_bytes // block_bytes
        if streaming_pack_mode:
            # Override post-block-sizing: zero dense scratch.
            config.num_kvcache_blocks = 0
        else:
            assert config.num_kvcache_blocks > 0
        # When streaming_pack_mode is active, num_kvcache_blocks==0 => the
        # tensor has numel()==0, attention.py's store_kvcache gate is a no-op,
        # and per-module k_cache slices are zero-element views.
        self.kv_cache = torch.empty(
            2,
            hf_config.num_hidden_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
            dtype=kv_cache_dtype,
        )
        if config.compact_kv_enabled:
            assert compact_capacity_tokens > 0
            # J5: per-layer separate compact storage when DMS streaming pack
            # mode. Each layer's compact tensor is allocated lazily to fit
            # only that layer's actual (seq, head) live counts. capacity_tokens
            # acts as a per-layer growth cap rather than a uniform allocation.
            use_per_layer = streaming_pack_mode
            if use_per_layer:
                self.compact_kv_cache = torch.empty(0, dtype=kv_cache_dtype, device="cuda")
            else:
                self.compact_kv_cache = torch.empty(
                    2,
                    hf_config.num_hidden_layers,
                    compact_capacity_tokens,
                    head_dim,
                    dtype=kv_cache_dtype,
                )
            use_layer_major_metadata = (
                use_per_layer
                and (
                    config.compact_kv_layer_major_metadata
                    or os.environ.get("FASTDMS_COMPACT_LAYER_MAJOR_METADATA") == "1"
                )
            )
            config.compact_kv_layer_major_metadata = use_layer_major_metadata
            self.compact_kv_manager = CompactKVManager(
                num_layers=hf_config.num_hidden_layers,
                num_kv_heads=num_kv_heads,
                capacity_tokens=compact_capacity_tokens,
                max_model_len=config.max_model_len,
                device="cuda",
                target_live_tokens_per_seq=config.compact_kv_target_live_tokens_per_seq,
                head_dim=head_dim,
                compact_dtype=kv_cache_dtype,
                per_layer_storage=use_per_layer,
                layer_major_metadata=use_layer_major_metadata,
                max_active_seqs=config.max_num_seqs,
            )
            if not use_per_layer:
                self.compact_kv_manager.attach_compact_storage(self.compact_kv_cache)
        layer_id = 0
        attn_modules = []
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.layer_id = layer_id
                module.k_cache = self.kv_cache[0, layer_id] if self.kv_cache.numel() else self.kv_cache
                module.v_cache = self.kv_cache[1, layer_id] if self.kv_cache.numel() else self.kv_cache
                if config.compact_kv_enabled:
                    if streaming_pack_mode:
                        # J5: per-layer storage. compact_k_cache stays as the
                        # default empty tensor; manager will replace it on
                        # first lazy allocation.
                        attn_modules.append(module)
                    else:
                        module.compact_k_cache = self.compact_kv_cache[0, layer_id]
                        module.compact_v_cache = self.compact_kv_cache[1, layer_id]
                layer_id += 1
        if config.compact_kv_enabled and streaming_pack_mode:
            self.compact_kv_manager.attach_attention_modules(attn_modules)

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def seq_dense_slots(self, seq: Sequence, num_tokens: int) -> torch.Tensor:
        slots = []
        for block_idx, block_id in enumerate(seq.block_table):
            block_start = block_idx * self.block_size
            if block_start >= num_tokens:
                break
            block_end = min(block_start + self.block_size, num_tokens)
            slots.extend(range(block_id * self.block_size, block_id * self.block_size + block_end - block_start))
        if len(slots) != num_tokens:
            raise RuntimeError(
                f"sequence {seq.seq_id} has {len(slots)} dense slots for {num_tokens} compact tokens"
            )
        return torch.tensor(slots, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)

    def build_dms_prefill_live_mask(
        self,
        seq: Sequence,
        seq_idx: int,
        packed_tokens: int,
        total_prefill_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        metadata = self.dms_metadata
        context = get_context()
        if metadata is None:
            raise RuntimeError("compact_kv_retention_mode='dms' requires DMS metadata")
        if context.dms_layer_evictions is None or context.cu_seqlens_q is None:
            raise RuntimeError("DMS compact prefill requires captured layer eviction decisions")
        if seq.num_cached_tokens != 0 or packed_tokens != seq.num_scheduled_tokens:
            raise RuntimeError(
                "DMS compact prefill currently requires an unchunked full-prompt prefill"
            )

        start = int(context.cu_seqlens_q[seq_idx].item())
        end = int(context.cu_seqlens_q[seq_idx + 1].item())
        if end - start != packed_tokens:
            raise RuntimeError(
                f"DMS prefill slice length {end - start} does not match packed_tokens={packed_tokens}"
            )

        # D1.3: stack all layer evict tensors once and slice + permute in a
        # single op so we avoid per-layer .T.contiguous() copies.
        num_layers = self.kv_cache.size(1)
        num_kv_heads = self.kv_cache.size(4)
        layer_tensors = []
        for layer_id in range(num_layers):
            evict = context.dms_layer_evictions.get(layer_id)
            if evict is None:
                raise RuntimeError(f"missing DMS eviction decisions for layer {layer_id}")
            expected_shape = (total_prefill_tokens, num_kv_heads)
            if tuple(evict.shape) != expected_shape:
                raise RuntimeError(
                    f"expected layer {layer_id} DMS evict mask {expected_shape}, got {tuple(evict.shape)}"
                )
            layer_tensors.append(evict)
        # [num_layers, total_prefill_tokens, num_kv_heads]
        all_evict = torch.stack(layer_tensors, dim=0)
        # Slice [num_layers, packed_tokens, num_kv_heads] then permute to
        # [num_layers, num_kv_heads, packed_tokens] in one fused op.
        evict_mask = all_evict[:, start:end, :].permute(0, 2, 1).contiguous()
        live_mask = build_dms_live_mask(
            evict_mask,
            current_position=packed_tokens - 1,
            window_size=metadata.window_size,
        )
        return live_mask, evict_mask

    def pack_compact_prefill(self, seqs: list[Sequence]):
        if self.compact_kv_manager is None:
            return
        # J1: under DMS retention, the per-layer streaming pack inside
        # LlamaAttention.forward already populated compact storage, so this
        # post-forward pack is a no-op. We still capture per-batch stats from
        # what the streaming pack wrote.
        if self.config.compact_kv_retention_mode == "dms":
            self._record_streaming_pack_stats(seqs)
            return
        prof = get_profiler()
        with prof.cuda_phase("compact_pack_prefill"):
            return self._pack_compact_prefill_impl(seqs)

    def _record_streaming_pack_stats(self, seqs: list[Sequence]) -> None:
        self.last_compact_prefill_stats = []
        self.last_dms_decode_stats = []
        for seq in seqs:
            packed_tokens = min(seq.num_cached_tokens + seq.num_scheduled_tokens, seq.num_tokens)
            if packed_tokens < seq.num_tokens:
                continue
            state = self.compact_kv_manager.seq_states.get(seq.seq_id)
            if state is None:
                continue
            # Final live counts after streaming pack has run all layers.
            live_counts = state.live_counts
            stats = {
                "seq_id": seq.seq_id,
                "mode": "dms",
                "packed_tokens": packed_tokens,
                "live_tokens_min": int(live_counts.min().item()),
                "live_tokens_mean": float(live_counts.float().mean().item()),
                "live_tokens_max": int(live_counts.max().item()),
                "live_tokens_total": int(live_counts.sum().item()),
            }
            self.last_compact_prefill_stats.append(stats)
        if self.last_compact_prefill_stats:
            batch_peak = sum(item["live_tokens_total"] for item in self.last_compact_prefill_stats)
            self.compact_live_tokens_peak = max(self.compact_live_tokens_peak, batch_peak)

    def _pack_compact_prefill_impl(self, seqs: list[Sequence]):
        # A1.6: do not reset compact_live_tokens_peak here; under continuous
        # batching a new prefill batch must not erase the running peak from
        # previously running sequences. Per-batch stats are still cleared.
        context = get_context()
        total_prefill_tokens = int(context.cu_seqlens_q[-1].item()) if context.cu_seqlens_q is not None else 0
        self.last_compact_prefill_stats = []
        self.last_dms_decode_stats = []
        dense_k = self.kv_cache[0].reshape(
            self.kv_cache.size(1),
            self.kv_cache.size(2) * self.kv_cache.size(3),
            self.kv_cache.size(4),
            self.kv_cache.size(5),
        )
        dense_v = self.kv_cache[1].reshape_as(dense_k)
        compact_k = self.compact_kv_cache[0]
        compact_v = self.compact_kv_cache[1]

        for seq_idx, seq in enumerate(seqs):
            packed_tokens = min(seq.num_cached_tokens + seq.num_scheduled_tokens, seq.num_tokens)
            if packed_tokens < seq.num_tokens:
                continue
            state = self.compact_kv_manager.allocate(seq)
            dense_slots = self.seq_dense_slots(seq, packed_tokens)
            evict_mask = None
            if self.config.compact_kv_retention_mode == "dms":
                live_mask, evict_mask = self.build_dms_prefill_live_mask(
                    seq,
                    seq_idx,
                    packed_tokens,
                    total_prefill_tokens,
                )
            else:
                live_mask = build_emulated_dms_live_mask(
                    num_layers=self.kv_cache.size(1),
                    num_kv_heads=self.kv_cache.size(4),
                    num_tokens=packed_tokens,
                    mode=self.config.compact_kv_retention_mode,
                    stride=self.config.compact_kv_retention_stride,
                    recent_tokens=self.config.compact_kv_retention_recent_tokens,
                    device=dense_slots.device,
                )
            # D1.4: write all per-(layer, kv_head) metadata in one batched call.
            dense_positions = torch.arange(packed_tokens, device=dense_slots.device, dtype=torch.int32)
            live_counts = self.compact_kv_manager.mark_live_metadata_all(
                seq.seq_id,
                live_mask,
                dense_positions,
                evict_mask_full=evict_mask,
            )
            # F1.1: vectorize the per-(layer, kv_head) K/V copy. Use the same
            # cumsum-based rank trick to compute destination compact slots,
            # then do a single fancy-index scatter for K and one for V.
            num_layers_local = self.kv_cache.size(1)
            num_kv_heads_local = self.kv_cache.size(4)
            cum = live_mask.to(torch.int32).cumsum(dim=-1)
            rank_full = (cum - 1).clamp(min=0)
            # nonzero gives (l, h, token_idx) triples where live_mask is True.
            l_flat, h_flat, t_flat = torch.nonzero(live_mask, as_tuple=True)
            l_long = l_flat.to(torch.long)
            h_long = h_flat.to(torch.long)
            t_long = t_flat.to(torch.long)
            base_per_h = state.base_offsets.to(torch.long)
            dst_slot = base_per_h[h_long] + rank_full[l_long, h_long, t_long].to(torch.long)
            dense_token_slot = dense_slots[t_long]
            compact_k[l_long, dst_slot] = dense_k[l_long, dense_token_slot, h_long]
            compact_v[l_long, dst_slot] = dense_v[l_long, dense_token_slot, h_long]
            self.compact_kv_manager.mark_live_counts(seq.seq_id, live_counts)
            stats = {
                "seq_id": seq.seq_id,
                "mode": self.config.compact_kv_retention_mode,
                "packed_tokens": packed_tokens,
                "live_tokens_min": int(live_counts.min().item()),
                "live_tokens_mean": float(live_counts.float().mean().item()),
                "live_tokens_max": int(live_counts.max().item()),
                "live_tokens_total": int(live_counts.sum().item()),
            }
            if evict_mask is not None:
                stats.update({
                    "dms_source": self.dms_metadata.source_kind if self.dms_metadata is not None else None,
                    "dms_window_size": self.dms_metadata.window_size if self.dms_metadata is not None else None,
                    "eviction_decisions_true": int(evict_mask.sum().item()),
                    "eviction_decisions_total": int(evict_mask.numel()),
                })
            self.last_compact_prefill_stats.append(stats)
        if self.last_compact_prefill_stats:
            # Sum across this batch's prefills and merge against any cross-batch
            # decode peak so continuous-batching peaks are preserved.
            batch_peak = sum(item["live_tokens_total"] for item in self.last_compact_prefill_stats)
            self.compact_live_tokens_peak = max(self.compact_live_tokens_peak, batch_peak)
        # B1.2: mark dense block_tables for release. Actual block_manager
        # deallocation happens post-step in LLMEngine.step / the bench harness
        # via Scheduler.release_dense_blocks_after_compact_pack.
        # The runner cannot reach the scheduler's BlockManager without coupling,
        # so we just leave a flag and let the engine release.

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        streaming_pack_mode = (
            self.config.compact_kv_enabled and self.config.compact_kv_retention_mode == "dms"
        )
        for seq in seqs:
            seqlen = len(seq)
            start = min(seq.num_cached_tokens, seqlen - 1)
            seqlen_q = seq.num_scheduled_tokens
            seqlen_k = seqlen
            end = start + seqlen_q
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if streaming_pack_mode:
                # No dense scratch; slot_mapping placeholders so downstream
                # tensor builds still get a fixed-size array. attention.py's
                # store_kvcache gate is off because k_cache.numel()==0.
                slot_mapping.extend([-1] * seqlen_q)
                continue
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        dms_capture_enabled = (
            self.compact_kv_manager is not None
            and self.config.compact_kv_retention_mode == "dms"
        )
        # J1: when DMS is active, allocate compact ranges for the prefilling
        # seqs and plumb the streaming pack hook so the per-layer pack runs
        # inline inside attention.forward.
        streaming_pack_seq_ids: list[int] | None = None
        if dms_capture_enabled and self.compact_kv_manager is not None:
            streaming_pack_seq_ids = []
            for seq in seqs:
                packed_tokens = min(seq.num_cached_tokens + seq.num_scheduled_tokens, seq.num_tokens)
                if packed_tokens < seq.num_tokens:
                    continue
                self.compact_kv_manager.allocate(seq)
                streaming_pack_seq_ids.append(seq.seq_id)
        set_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            None,
            block_tables,
            dms_capture_enabled=dms_capture_enabled,
            dms_alpha_scale=self.dms_metadata.alpha_scale if dms_capture_enabled and self.dms_metadata is not None else 100.0,
            dms_alpha_offset=self.dms_metadata.alpha_offset if dms_capture_enabled and self.dms_metadata is not None else 5.0,
            dms_layer_evictions={} if dms_capture_enabled else None,
            streaming_pack_manager=self.compact_kv_manager if dms_capture_enabled else None,
            streaming_pack_seq_ids=streaming_pack_seq_ids,
            streaming_pack_window_size=(
                self.dms_metadata.window_size if dms_capture_enabled and self.dms_metadata is not None else 0
            ),
        )
        return input_ids, positions

    def record_dms_decode_decisions(
        self,
        seqs: list[Sequence],
        positions: torch.Tensor | None = None,
    ) -> None:
        if self.compact_kv_manager is None or self.config.compact_kv_retention_mode != "dms":
            return
        context = get_context()
        layer_evictions = (
            context.dms_layer_evictions_buffer
            if context.dms_layer_evictions_buffer is not None
            else context.dms_layer_evictions
        )
        if layer_evictions is None:
            raise RuntimeError("DMS decode requires captured layer eviction decisions")
        prof = get_profiler()
        current_positions = [len(seq) - 1 for seq in seqs]
        position_source = positions if positions is not None else current_positions
        with prof.cuda_phase("record_dms_decode_decisions"):
            stats = self.compact_kv_manager.record_appended_dms_decisions(
                seqs,
                position_source,
                layer_evictions,
                sync_stats=False,
            )
        stats.update({
            "phase": "record",
            "positions_min": min(current_positions) if current_positions else None,
            "positions_max": max(current_positions) if current_positions else None,
        })
        self.last_dms_decode_stats.append(stats)

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        compact_active = self.config.compact_kv_enabled and self.compact_kv_manager is not None
        context_lens = None if compact_active else []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            if not compact_active:
                assert context_lens is not None
                context_lens.append(len(seq))
                slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
        decode_positions = positions.copy()
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        if self.config.compact_kv_enabled:
            assert self.compact_kv_manager is not None
            prof = get_profiler()
            dms_decode_enabled = self.config.compact_kv_retention_mode == "dms"
            if dms_decode_enabled:
                if self.dms_metadata is None:
                    raise RuntimeError("compact DMS decode requires DMS metadata")
                with prof.cuda_phase("dms_decode_expire"):
                    expire_stats = self.compact_kv_manager.apply_dms_evictions(
                        seqs,
                        decode_positions,
                        window_size=self.dms_metadata.window_size,
                        sync_stats=False,
                    )
                expire_stats.update({
                    "phase": "expire",
                    "positions_min": min(decode_positions) if decode_positions else None,
                    "positions_max": max(decode_positions) if decode_positions else None,
                })
                self.last_dms_decode_stats.append(expire_stats)
            with prof.cuda_phase("prepare_decode_metadata"):
                compact_base_offsets, compact_live_counts, compact_slot_mapping, compact_max_live_count = (
                    self.compact_kv_manager.prepare_decode_metadata(seqs)
                )
            self.update_compact_live_tokens_peak(compact_live_counts)
            set_context(
                False,
                compact_kv_enabled=True,
                compact_base_offsets=compact_base_offsets,
                compact_live_counts=compact_live_counts,
                compact_slot_mapping=compact_slot_mapping,
                compact_max_live_count=compact_max_live_count,
                dms_capture_enabled=dms_decode_enabled,
                dms_alpha_scale=self.dms_metadata.alpha_scale if dms_decode_enabled and self.dms_metadata is not None else 100.0,
                dms_alpha_offset=self.dms_metadata.alpha_offset if dms_decode_enabled and self.dms_metadata is not None else 5.0,
                dms_layer_evictions={} if dms_decode_enabled else None,
            )
        else:
            assert context_lens is not None
            context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            block_tables = self.prepare_block_tables(seqs)
            set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        if all(temperature <= 1e-10 for temperature in temperatures):
            return None
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    def _compact_storage_signature_current(self) -> tuple | None:
        manager = self.compact_kv_manager
        if manager is None or not manager.per_layer_storage:
            return None
        signature = []
        for layer_id in range(manager.num_layers):
            compact_k = manager.compact_k_per_layer[layer_id]
            compact_v = manager.compact_v_per_layer[layer_id]
            if compact_k is None or compact_v is None:
                return None
            signature.append((
                compact_k.data_ptr(),
                compact_v.data_ptr(),
                tuple(compact_k.shape),
                tuple(compact_v.shape),
            ))
        return tuple(signature)

    def _set_compact_graph_context(self, graph_vars: dict) -> None:
        set_context(
            False,
            compact_kv_enabled=True,
            compact_base_offsets=graph_vars["compact_base_offsets"],
            compact_live_counts=graph_vars["compact_live_counts"],
            compact_slot_mapping=graph_vars["compact_slot_mapping"],
            compact_max_live_count=graph_vars["compact_max_live_count"],
            dms_capture_enabled=graph_vars["dms_capture_enabled"],
            dms_alpha_scale=graph_vars["dms_alpha_scale"],
            dms_alpha_offset=graph_vars["dms_alpha_offset"],
            dms_layer_evictions=None,
            dms_layer_evictions_buffer=graph_vars["dms_layer_evictions_buffer"],
        )

    def _copy_compact_graph_inputs(
        self,
        graph_vars: dict,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        context = get_context()
        if (
            context.compact_base_offsets is None
            or context.compact_live_counts is None
            or context.compact_slot_mapping is None
        ):
            raise RuntimeError("compact decode graph requires complete compact decode context")
        graph_vars["input_ids"].copy_(input_ids)
        graph_vars["positions"].copy_(positions)
        graph_vars["compact_base_offsets"].copy_(context.compact_base_offsets)
        graph_vars["compact_live_counts"].copy_(context.compact_live_counts)
        graph_vars["compact_slot_mapping"].copy_(context.compact_slot_mapping)

    def _capture_compact_cudagraph(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        storage_signature: tuple,
    ) -> None:
        context = get_context()
        batch = input_ids.size(0)
        if context.compact_base_offsets is None or context.compact_live_counts is None:
            raise RuntimeError("compact decode graph capture requires compact metadata")
        if context.compact_slot_mapping is None:
            raise RuntimeError("compact decode graph capture requires compact slot mapping")
        model_dtype = next(self.model.parameters()).dtype
        graph_vars = {
            "input_ids": torch.empty_like(input_ids),
            "positions": torch.empty_like(positions),
            "compact_base_offsets": torch.empty_like(context.compact_base_offsets),
            "compact_live_counts": torch.empty_like(context.compact_live_counts),
            "compact_slot_mapping": torch.empty_like(context.compact_slot_mapping),
            "compact_max_live_count": context.compact_max_live_count,
            "dms_capture_enabled": context.dms_capture_enabled,
            "dms_alpha_scale": context.dms_alpha_scale,
            "dms_alpha_offset": context.dms_alpha_offset,
            "dms_layer_evictions_buffer": (
                torch.empty(
                    (self.compact_kv_manager.num_layers, batch, self.compact_kv_manager.num_kv_heads),
                    device=input_ids.device,
                    dtype=torch.bool,
                )
                if context.dms_capture_enabled
                else None
            ),
            "logits": torch.empty(
                batch,
                self.config.hf_config.vocab_size,
                device=input_ids.device,
                dtype=model_dtype,
            ),
        }
        self._copy_compact_graph_inputs(graph_vars, input_ids, positions)
        self._set_compact_graph_context(graph_vars)
        outputs = self.model(graph_vars["input_ids"], graph_vars["positions"])
        graph_vars["logits"][:] = self.model.compute_logits(outputs)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, self.compact_graph_pool):
            outputs = self.model(graph_vars["input_ids"], graph_vars["positions"])
            graph_vars["logits"][:] = self.model.compute_logits(outputs)
        if self.compact_graph_pool is None:
            self.compact_graph_pool = graph.pool()
        self.compact_graphs[batch] = graph
        self.compact_graph_vars[batch] = graph_vars
        self.compact_graph_storage_signature[batch] = storage_signature
        torch.cuda.synchronize()

    def run_compact_cudagraph(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        storage_signature = self._compact_storage_signature_current()
        if storage_signature is None:
            raise RuntimeError("compact decode graph requires allocated per-layer compact storage")
        batch = input_ids.size(0)
        if (
            batch not in self.compact_graphs
            or self.compact_graph_storage_signature.get(batch) != storage_signature
        ):
            self._capture_compact_cudagraph(input_ids, positions, storage_signature)
        graph_vars = self.compact_graph_vars[batch]
        self._copy_compact_graph_inputs(graph_vars, input_ids, positions)
        self._set_compact_graph_context(graph_vars)
        self.compact_graphs[batch].replay()
        return graph_vars["logits"]

    def _capture_compact_greedy_cudagraph(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        storage_signature: tuple,
    ) -> None:
        context = get_context()
        batch = input_ids.size(0)
        if context.compact_base_offsets is None or context.compact_live_counts is None:
            raise RuntimeError("compact decode graph capture requires compact metadata")
        if context.compact_slot_mapping is None:
            raise RuntimeError("compact decode graph capture requires compact slot mapping")
        graph_vars = {
            "input_ids": torch.empty_like(input_ids),
            "positions": torch.empty_like(positions),
            "compact_base_offsets": torch.empty_like(context.compact_base_offsets),
            "compact_live_counts": torch.empty_like(context.compact_live_counts),
            "compact_slot_mapping": torch.empty_like(context.compact_slot_mapping),
            "compact_max_live_count": context.compact_max_live_count,
            "dms_capture_enabled": context.dms_capture_enabled,
            "dms_alpha_scale": context.dms_alpha_scale,
            "dms_alpha_offset": context.dms_alpha_offset,
            "dms_layer_evictions_buffer": (
                torch.empty(
                    (self.compact_kv_manager.num_layers, batch, self.compact_kv_manager.num_kv_heads),
                    device=input_ids.device,
                    dtype=torch.bool,
                )
                if context.dms_capture_enabled
                else None
            ),
            "token_ids": torch.empty(batch, device=input_ids.device, dtype=torch.int64),
        }
        self._copy_compact_graph_inputs(graph_vars, input_ids, positions)
        self._set_compact_graph_context(graph_vars)
        outputs = self.model(graph_vars["input_ids"], graph_vars["positions"])
        graph_vars["token_ids"][:] = self.model.compute_greedy_token_ids(outputs)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, self.compact_graph_pool):
            outputs = self.model(graph_vars["input_ids"], graph_vars["positions"])
            graph_vars["token_ids"][:] = self.model.compute_greedy_token_ids(outputs)
        if self.compact_graph_pool is None:
            self.compact_graph_pool = graph.pool()
        self.compact_greedy_graphs[batch] = graph
        self.compact_greedy_graph_vars[batch] = graph_vars
        self.compact_greedy_graph_storage_signature[batch] = storage_signature
        torch.cuda.synchronize()

    def run_compact_greedy_cudagraph(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        storage_signature = self._compact_storage_signature_current()
        if storage_signature is None:
            raise RuntimeError("compact decode graph requires allocated per-layer compact storage")
        batch = input_ids.size(0)
        if (
            batch not in self.compact_greedy_graphs
            or self.compact_greedy_graph_storage_signature.get(batch) != storage_signature
        ):
            self._capture_compact_greedy_cudagraph(input_ids, positions, storage_signature)
        graph_vars = self.compact_greedy_graph_vars[batch]
        self._copy_compact_graph_inputs(graph_vars, input_ids, positions)
        self._set_compact_graph_context(graph_vars)
        self.compact_greedy_graphs[batch].replay()
        return graph_vars["token_ids"]

    def run_compact_greedy_decode_loop(self, seqs: list[Sequence], max_steps: int) -> list[int]:
        """Run a c=1 compact greedy decode tail without per-token CPU syncs.

        This is deliberately narrow and env-gated by callers: compact DMS,
        greedy sampling, ignore-eos, batch size 1, world size 1. The scheduler
        still owns prefill and final teardown; this method only keeps token
        feedback on GPU for the fixed-length decode tail.
        """
        if max_steps <= 0:
            return []
        if len(seqs) != 1:
            raise RuntimeError("compact greedy fast loop only supports batch size 1")
        if self.rank != 0 or self.world_size != 1:
            raise RuntimeError("compact greedy fast loop only supports rank0/world_size1")
        if not self.compact_decode_cudagraph:
            raise RuntimeError("compact greedy fast loop requires compact CUDA graph replay")
        if self.compact_kv_manager is None or self.config.compact_kv_retention_mode != "dms":
            raise RuntimeError("compact greedy fast loop requires compact DMS")
        if self.dms_metadata is None:
            raise RuntimeError("compact greedy fast loop requires DMS metadata")
        if not greedy_fused_argmax_enabled() or not hasattr(self.model, "compute_greedy_token_ids"):
            raise RuntimeError("compact greedy fast loop requires the fused greedy argmax path")

        seq = seqs[0]
        input_ids = torch.empty((1,), device="cuda", dtype=torch.int64)
        positions = torch.empty((1,), device="cuda", dtype=torch.int64)
        input_ids.fill_(int(seq.last_token))
        token_ids_out = torch.empty((max_steps,), device="cuda", dtype=torch.int64)
        prof = get_profiler()

        for step in range(max_steps):
            current_position = len(seq) - 1
            positions.fill_(int(current_position))
            decode_positions = [current_position]

            with prof.cuda_phase("dms_decode_expire"):
                expire_stats = self.compact_kv_manager.apply_dms_evictions(
                    seqs,
                    decode_positions,
                    window_size=self.dms_metadata.window_size,
                    sync_stats=False,
                )
            expire_stats.update({
                "phase": "expire",
                "positions_min": current_position,
                "positions_max": current_position,
            })
            self.last_dms_decode_stats.append(expire_stats)

            with prof.cuda_phase("prepare_decode_metadata"):
                compact_base_offsets, compact_live_counts, compact_slot_mapping, compact_max_live_count = (
                    self.compact_kv_manager.prepare_decode_metadata(seqs)
                )
            self.update_compact_live_tokens_peak(compact_live_counts)
            set_context(
                False,
                compact_kv_enabled=True,
                compact_base_offsets=compact_base_offsets,
                compact_live_counts=compact_live_counts,
                compact_slot_mapping=compact_slot_mapping,
                compact_max_live_count=compact_max_live_count,
                dms_capture_enabled=True,
                dms_alpha_scale=self.dms_metadata.alpha_scale,
                dms_alpha_offset=self.dms_metadata.alpha_offset,
                dms_layer_evictions={},
            )

            token_ids = self.run_model_greedy_token_ids(input_ids, positions, False)
            self.record_dms_decode_decisions(seqs, positions)
            reset_context()
            token_ids_out[step].copy_(token_ids[0])
            input_ids.copy_(token_ids)
            seq.num_tokens += 1
            seq.num_cached_tokens += 1
            seq.num_scheduled_tokens = 0

        tokens = token_ids_out.cpu().tolist()
        seq.token_ids.extend(tokens)
        if tokens:
            seq.last_token = int(tokens[-1])
        if seq.num_completion_tokens >= seq.max_tokens:
            seq.status = SequenceStatus.FINISHED
        return tokens

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        context = get_context()
        prof = get_profiler()
        phase_name = "model_forward_prefill" if is_prefill else "model_forward_decode"
        if (
            not is_prefill
            and context.compact_kv_enabled
            and self.compact_decode_cudagraph
            and input_ids.size(0) <= 512
        ):
            with prof.cuda_phase(phase_name):
                return self.run_compact_cudagraph(input_ids, positions)
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512 or context.compact_kv_enabled:
            with prof.cuda_phase(phase_name):
                return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            with prof.cuda_phase(phase_name):
                graph_vars["input_ids"][:bs] = input_ids
                graph_vars["positions"][:bs] = positions
                graph_vars["slot_mapping"].fill_(-1)
                graph_vars["slot_mapping"][:bs] = context.slot_mapping
                graph_vars["context_lens"].zero_()
                graph_vars["context_lens"][:bs] = context.context_lens
                graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
                graph.replay()
                return self.model.compute_logits(graph_vars["outputs"][:bs])

    @torch.inference_mode()
    def run_model_greedy_token_ids(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        context = get_context()
        prof = get_profiler()
        phase_name = "model_forward_prefill" if is_prefill else "model_forward_decode"
        if (
            not is_prefill
            and context.compact_kv_enabled
            and self.compact_decode_cudagraph
            and input_ids.size(0) <= 512
        ):
            with prof.cuda_phase(phase_name):
                return self.run_compact_greedy_cudagraph(input_ids, positions)
        if context.compact_kv_enabled or is_prefill or self.enforce_eager:
            with prof.cuda_phase(phase_name):
                return self.model.compute_greedy_token_ids(self.model(input_ids, positions))
        with prof.cuda_phase(phase_name):
            return self.run_model(input_ids, positions, is_prefill).argmax(dim=-1)

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        prof = get_profiler()
        with prof.wall_phase("prepare_prefill" if is_prefill else "prepare_decode"):
            input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        context = get_context()
        use_greedy_fused_argmax = (
            temperatures is None
            and self.rank == 0
            and self.world_size == 1
            and input_ids.size(0) == 1
            and context.compact_kv_enabled
            and greedy_fused_argmax_enabled()
            and hasattr(self.model, "compute_greedy_token_ids")
        )
        if use_greedy_fused_argmax:
            token_ids_tensor = self.run_model_greedy_token_ids(input_ids, positions, is_prefill)
        else:
            logits = self.run_model(input_ids, positions, is_prefill)
        if is_prefill and self.compact_kv_manager is not None:
            self.pack_compact_prefill(seqs)
        if not is_prefill:
            self.record_dms_decode_decisions(seqs, positions)
        with prof.cuda_phase("sampler"):
            if self.rank == 0:
                if use_greedy_fused_argmax:
                    token_ids = token_ids_tensor.tolist()
                elif temperatures is None:
                    token_ids = logits.argmax(dim=-1).tolist()
                else:
                    greedy_token_ids = logits.argmax(dim=-1)
                    sampled_token_ids = self.sampler(logits, temperatures.clamp_min(1e-10))
                    token_ids = torch.where(
                        temperatures <= 1e-10,
                        greedy_token_ids,
                        sampled_token_ids,
                    ).tolist()
            else:
                token_ids = None
        reset_context()
        return token_ids

    def free_compact(self, seq_ids: list[int]):
        if self.compact_kv_manager is None:
            return
        for seq_id in seq_ids:
            self.compact_kv_manager.free(seq_id)

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
