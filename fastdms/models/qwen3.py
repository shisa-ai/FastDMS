import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from fastdms.layers.activation import SiluAndMul
from fastdms.layers.attention import Attention
from fastdms.layers.layernorm import RMSNorm
from fastdms.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from fastdms.layers.rotary_embedding import get_rope
from fastdms.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from fastdms.engine.dms import extract_dms_eviction_decisions
from fastdms.layers.compact_attention import (
    compact_attention_inline_q_rope_enabled,
    compact_attention_splitk_enabled,
    dms_decode_store_transient_k_enabled,
    dms_fused_decode_preprocess_enabled,
    fused_dms_rope_store_compact_decode,
)
from fastdms.utils.context import get_context
from fastdms.utils.profiler import get_profiler


class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        context = get_context()
        fused_decode_preprocess = False
        inline_q_rope = False
        if context.dms_capture_enabled:
            can_fuse_decode_preprocess = (
                dms_fused_decode_preprocess_enabled()
                and not context.is_prefill
                and context.compact_kv_enabled
                and context.compact_slot_mapping is not None
                and self.attn.compact_k_cache.numel() > 0
                and self.attn.compact_v_cache.numel() > 0
            )
            if (
                context.dms_layer_evictions is None
                and context.dms_layer_evictions_buffer is None
                and not can_fuse_decode_preprocess
            ):
                raise RuntimeError("DMS capture requires a layer eviction recorder")
            layer_id = self.attn.layer_id
            if layer_id < 0:
                raise RuntimeError("DMS capture requires assigned attention layer ids")
            evict_out = None
            if context.dms_layer_evictions_buffer is not None:
                evict_out = context.dms_layer_evictions_buffer[layer_id]
            if can_fuse_decode_preprocess:
                inline_q_rope = (
                    compact_attention_inline_q_rope_enabled()
                    and compact_attention_splitk_enabled()
                    and q.shape[0] == 1
                    and self.head_dim % 2 == 0
                    and self.head_dim >= 32
                )
                compact_slot_mapping = context.compact_slot_mapping
                if compact_slot_mapping.ndim == 3:
                    compact_slot_mapping = compact_slot_mapping[layer_id]
                if evict_out is None:
                    evict_out = torch.empty(
                        (q.shape[0], self.num_kv_heads),
                        device=q.device,
                        dtype=torch.bool,
                    )
                with get_profiler().cuda_phase("dms_rope_store"):
                    fused_dms_rope_store_compact_decode(
                        q,
                        k,
                        v,
                        self.rotary_emb.cos_sin_cache,
                        self.attn.compact_k_cache,
                        self.attn.compact_v_cache,
                        compact_slot_mapping,
                        evict_out,
                        positions,
                        num_kv_heads=self.num_kv_heads,
                        alpha_scale=context.dms_alpha_scale,
                        alpha_offset=context.dms_alpha_offset,
                        store_q=not inline_q_rope,
                        store_transient_k=dms_decode_store_transient_k_enabled(),
                    )
                evict_mask = evict_out
                fused_decode_preprocess = True
            else:
                with get_profiler().cuda_phase("dms_capture"):
                    q, evict_mask = extract_dms_eviction_decisions(
                        q,
                        num_kv_heads=self.num_kv_heads,
                        num_qo_heads=self.num_heads,
                        head_dim=self.head_dim,
                        alpha_scale=context.dms_alpha_scale,
                        alpha_offset=context.dms_alpha_offset,
                        inplace=True,
                        out=evict_out,
                    )
            if context.dms_layer_evictions is not None:
                context.dms_layer_evictions[layer_id] = evict_mask
        if not fused_decode_preprocess:
            q, k = self.rotary_emb(positions, q, k)
        if (
            context.is_prefill
            and context.streaming_pack_manager is not None
            and context.streaming_pack_seq_ids
            and context.dms_capture_enabled
        ):
            with get_profiler().cuda_phase("streaming_pack"):
                context.streaming_pack_manager.streaming_pack_layer(
                    layer_id=self.attn.layer_id,
                    k=k,
                    v=v,
                    evict_mask=evict_mask,
                    cu_seqlens_q=context.cu_seqlens_q,
                    seq_ids=context.streaming_pack_seq_ids,
                    window_size=context.streaming_pack_window_size,
                )
        o = self.attn(
            q,
            k,
            v,
            compact_append_stored=fused_decode_preprocess,
            compact_q_rope=fused_decode_preprocess and inline_q_rope,
            positions=positions,
            cos_sin_cache=self.rotary_emb.cos_sin_cache,
        )
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def compute_greedy_token_ids(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head.greedy_token_ids(hidden_states)
