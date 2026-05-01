import os

import torch
import triton
import triton.language as tl
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from fastdms.layers.linear import (
    _FP8_MAX,
    _amax_to_scale,
    _as_col_major_b,
    _fp8_linear,
    _fp8_row1_matvec,
    _env_bool,
    _int4_row1_matvec,
    _quantize_weight_to_int4pack,
    fp8_keep_bf16_weights,
    fp8_min_rows,
    fp8_weight_scope,
    int4_row1_lm_head_enabled,
)
from fastdms.utils.context import get_context
from fastdms.utils.profiler import get_profiler


def fp8_lm_head_enabled() -> bool:
    explicit = os.environ.get("NANOVLLM_FP8_LM_HEAD")
    if explicit is not None:
        return explicit.strip().lower() in {"1", "true", "yes", "on"}
    return fp8_weight_scope() in {"lm_head", "head", "mlp_lm_head", "mlp+lm_head", "mlp_head", "mlp+head"}


def fp8_lm_head_min_rows() -> int:
    raw = os.environ.get("NANOVLLM_FP8_LM_HEAD_MIN_ROWS")
    if raw is None or raw.strip() == "":
        return fp8_min_rows()
    return max(1, int(raw))


def fp8_keep_bf16_lm_head_weight() -> bool:
    return _env_bool("NANOVLLM_FP8_KEEP_BF16_LM_HEAD", fp8_keep_bf16_weights())


def fp8_embedding_enabled() -> bool:
    return _env_bool("NANOVLLM_FP8_EMBEDDING", True)


def fp8_keep_bf16_embedding_weight() -> bool:
    return _env_bool("NANOVLLM_FP8_KEEP_BF16_EMBEDDING", False)


def fp8_embedding_share_lm_head_weight() -> bool:
    return _env_bool("NANOVLLM_FP8_EMBEDDING_SHARE_LM_HEAD", False)


def fp8_lm_head_row1_triton_enabled() -> bool:
    return _env_bool("NANOVLLM_FP8_LM_HEAD_ROW1_TRITON", True)


def greedy_fused_argmax_enabled() -> bool:
    return _env_bool("NANOVLLM_GREEDY_FUSED_ARGMAX", True)


def fp8_lm_head_argmax_block_n() -> int:
    raw = os.environ.get("NANOVLLM_FP8_LM_HEAD_ARGMAX_BLOCK_N")
    if raw is None or raw.strip() == "":
        return 128
    value = int(raw)
    if value not in {16, 32, 64, 128}:
        raise ValueError("NANOVLLM_FP8_LM_HEAD_ARGMAX_BLOCK_N must be one of 16, 32, 64, 128")
    return value


def int4_lm_head_rerank_topk() -> int:
    raw = os.environ.get("NANOVLLM_INT4_ROW1_LM_HEAD_RERANK_TOPK")
    if raw is None or raw.strip() == "":
        return 0
    if raw.strip().lower() in {"0", "false", "no", "off"}:
        return 0
    value = int(raw)
    if value not in {2, 4, 8, 16, 32}:
        raise ValueError("NANOVLLM_INT4_ROW1_LM_HEAD_RERANK_TOPK must be one of 0, 2, 4, 8, 16, 32")
    return value


def _fp8_lm_head_row1_triton_allowed(rows: int, min_rows: int, has_bf16_weight: bool) -> bool:
    return (
        rows == 1
        and (not has_bf16_weight or rows >= min_rows)
        and fp8_lm_head_row1_triton_enabled()
    )


@triton.jit
def _fp8_row1_argmax_stage_kernel(
    x_ptr,
    weight_fp8_t_ptr,
    weight_scale_inv_ptr,
    block_max_ptr,
    block_idx_ptr,
    K: tl.constexpr,
    N: tl.constexpr,
    WEIGHT_STRIDE_K: tl.constexpr,
    WEIGHT_STRIDE_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    block_id = tl.program_id(0)
    n = block_id * BLOCK_N + tl.arange(0, BLOCK_N)
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
    scores = tl.sum(tl.where(offs_m[:, None] == 0, acc, 0.0), axis=0) * x_scale_inv * scale
    scores = tl.where(n < N, scores, -float("inf"))
    block_max = tl.max(scores, axis=0)
    block_idx = tl.min(tl.where(scores == block_max, n, N), axis=0)
    tl.store(block_max_ptr + block_id, block_max)
    tl.store(block_idx_ptr + block_id, block_idx)


@triton.jit
def _fp8_row1_argmax_reduce_kernel(
    block_max_ptr,
    block_idx_ptr,
    out_ptr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_B)
    mask = offsets < NUM_BLOCKS
    vals = tl.load(block_max_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    idxs = tl.load(block_idx_ptr + offsets, mask=mask, other=2147483647).to(tl.int64)
    max_val = tl.max(vals, axis=0)
    token_id = tl.min(tl.where(vals == max_val, idxs, 2147483647), axis=0)
    tl.store(out_ptr, token_id)


def _next_power_of_2(value: int) -> int:
    if value < 1:
        raise ValueError(f"value must be positive, got {value}")
    return 1 << (value - 1).bit_length()


def _fp8_row1_argmax(
    x: torch.Tensor,
    *,
    weight_fp8_t: torch.Tensor,
    weight_scale_inv: torch.Tensor,
) -> torch.Tensor:
    in_features = int(weight_fp8_t.size(0))
    out_features = int(weight_fp8_t.size(1))
    x_2d = x.reshape(1, in_features).contiguous()
    block_n = fp8_lm_head_argmax_block_n()
    num_blocks = triton.cdiv(out_features, block_n)
    block_max = torch.empty((num_blocks,), device=x.device, dtype=torch.float32)
    block_idx = torch.empty((num_blocks,), device=x.device, dtype=torch.int64)
    out = torch.empty((1,), device=x.device, dtype=torch.int64)
    prof = get_profiler()
    with prof.cuda_phase("lm_head_fp8_argmax_stage"):
        _fp8_row1_argmax_stage_kernel[(num_blocks,)](
            x_2d,
            weight_fp8_t,
            weight_scale_inv,
            block_max,
            block_idx,
            in_features,
            out_features,
            weight_fp8_t.stride(0),
            weight_fp8_t.stride(1),
            BLOCK_M=16,
            BLOCK_K=256,
            BLOCK_N=block_n,
            FP8_MAX=_FP8_MAX,
            num_warps=4,
        )
    with prof.cuda_phase("lm_head_fp8_argmax_reduce"):
        _fp8_row1_argmax_reduce_kernel[(1,)](
            block_max,
            block_idx,
            out,
            NUM_BLOCKS=num_blocks,
            BLOCK_B=_next_power_of_2(num_blocks),
            num_warps=8,
        )
    return out


@triton.jit
def _fp8_row1_rerank_topk_kernel(
    x_ptr,
    weight_fp8_t_ptr,
    weight_scale_inv_ptr,
    candidate_idx_ptr,
    out_ptr,
    K: tl.constexpr,
    N: tl.constexpr,
    WEIGHT_STRIDE_K: tl.constexpr,
    WEIGHT_STRIDE_N: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    offs_t = tl.arange(0, BLOCK_TOPK)
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    candidate_ids = tl.load(candidate_idx_ptr + offs_t, mask=offs_t < TOPK, other=N).to(tl.int64)

    amax = tl.zeros((), tl.float32)
    for start in range(0, K, BLOCK_K):
        k = start + offs_k
        x = tl.load(x_ptr + k, mask=k < K, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(x), axis=0))
    x_scale = tl.where(amax > 0.0, FP8_MAX / amax, 1.0)
    x_scale_inv = 1.0 / x_scale

    acc = tl.zeros((BLOCK_M, BLOCK_TOPK), tl.float32)
    for start in range(0, K, BLOCK_K):
        k = start + offs_k
        x = tl.load(x_ptr + k, mask=k < K, other=0.0).to(tl.float32)
        xq = (x * x_scale).to(tl.float8e4nv)
        x_tile = tl.where(offs_m[:, None] == 0, xq[None, :], 0.0).to(tl.float8e4nv)
        w = tl.load(
            weight_fp8_t_ptr
            + k[:, None] * WEIGHT_STRIDE_K
            + candidate_ids[None, :] * WEIGHT_STRIDE_N,
            mask=(k[:, None] < K) & (offs_t[None, :] < TOPK) & (candidate_ids[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(x_tile, w)

    scale = tl.load(weight_scale_inv_ptr + candidate_ids, mask=(offs_t < TOPK) & (candidate_ids < N), other=0.0)
    scores = tl.sum(tl.where(offs_m[:, None] == 0, acc, 0.0), axis=0) * x_scale_inv * scale.to(tl.float32)
    scores = tl.where((offs_t < TOPK) & (candidate_ids < N), scores, -float("inf"))
    max_val = tl.max(scores, axis=0)
    token_id = tl.min(tl.where(scores == max_val, candidate_ids, N), axis=0)
    tl.store(out_ptr, token_id)


def _fp8_row1_rerank_topk(
    x: torch.Tensor,
    candidate_ids: torch.Tensor,
    *,
    weight_fp8_t: torch.Tensor,
    weight_scale_inv: torch.Tensor,
) -> torch.Tensor:
    in_features = int(weight_fp8_t.size(0))
    out_features = int(weight_fp8_t.size(1))
    x_2d = x.reshape(1, in_features).contiguous()
    candidates = candidate_ids.reshape(-1).to(dtype=torch.int64).contiguous()
    topk = int(candidates.numel())
    if topk not in {2, 4, 8, 16, 32}:
        raise ValueError(f"topk must be one of 2, 4, 8, 16, 32, got {topk}")
    out = torch.empty((1,), device=x.device, dtype=torch.int64)
    block_topk = max(16, _next_power_of_2(topk))
    with get_profiler().cuda_phase("lm_head_fp8_rerank"):
        _fp8_row1_rerank_topk_kernel[(1,)](
            x_2d,
            weight_fp8_t,
            weight_scale_inv,
            candidates,
            out,
            in_features,
            out_features,
            weight_fp8_t.stride(0),
            weight_fp8_t.stride(1),
            topk,
            BLOCK_M=16,
            BLOCK_K=256,
            BLOCK_TOPK=block_topk,
            FP8_MAX=_FP8_MAX,
            num_warps=4,
        )
    return out


def _int4_row1_argmax(
    x: torch.Tensor,
    *,
    weight_int4pack: torch.Tensor,
    weight_int4_scale_zeros: torch.Tensor,
    q_group_size: int,
) -> torch.Tensor:
    with get_profiler().cuda_phase("lm_head_int4_logits"):
        logits = _int4_row1_matvec(
            x,
            weight_int4pack=weight_int4pack,
            weight_int4_scale_zeros=weight_int4_scale_zeros,
            q_group_size=q_group_size,
        )
    return logits.argmax(dim=-1)


def _int4_row1_argmax_rerank_topk(
    x: torch.Tensor,
    *,
    weight_int4pack: torch.Tensor,
    weight_int4_scale_zeros: torch.Tensor,
    q_group_size: int,
    weight_fp8_t: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    rerank_topk: int,
) -> torch.Tensor:
    prof = get_profiler()
    with prof.cuda_phase("lm_head_int4_logits"):
        logits = _int4_row1_matvec(
            x,
            weight_int4pack=weight_int4pack,
            weight_int4_scale_zeros=weight_int4_scale_zeros,
            q_group_size=q_group_size,
        )
    with prof.cuda_phase("lm_head_topk"):
        candidates = logits.topk(rerank_topk, dim=-1, sorted=False).indices
    return _fp8_row1_rerank_topk(
        x,
        candidates,
        weight_fp8_t=weight_fp8_t,
        weight_scale_inv=weight_scale_inv,
    )


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader
        self._fp8_embedding_enabled = False
        self._fp8_embedding_weight: torch.Tensor | None = None
        self._fp8_embedding_scale_inv: torch.Tensor | None = None
        self._fp8_embedding_output_dtype: torch.dtype | None = None

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def quantize_embedding_to_fp8(self, *, keep_bf16_weight: bool = False) -> None:
        weight = self.weight.detach().to(torch.float32)
        weight_scale = _amax_to_scale(weight.abs().amax(dim=1).float())
        self._set_fp8_embedding_weight(
            (weight * weight_scale.unsqueeze(1)).to(torch.float8_e4m3fn).contiguous(),
            weight_scale.reciprocal().reshape(-1, 1).contiguous(),
            output_dtype=self.weight.dtype,
            keep_bf16_weight=keep_bf16_weight,
        )

    def enable_fp8_embedding_from_lm_head(
        self,
        lm_head: "ParallelLMHead",
        *,
        keep_bf16_weight: bool = False,
    ) -> None:
        if not getattr(lm_head, "_fp8_weight_enabled", False):
            raise RuntimeError("FP8 embedding sharing requires an FP8-quantized lm_head")
        self._set_fp8_embedding_weight(
            lm_head.weight_fp8_t.t(),
            lm_head.weight_scale_inv.t(),
            output_dtype=self.weight.dtype,
            keep_bf16_weight=keep_bf16_weight,
        )

    def _set_fp8_embedding_weight(
        self,
        weight_fp8: torch.Tensor,
        scale_inv: torch.Tensor,
        *,
        output_dtype: torch.dtype,
        keep_bf16_weight: bool,
    ) -> None:
        self._fp8_embedding_weight = weight_fp8
        self._fp8_embedding_scale_inv = scale_inv
        self._fp8_embedding_output_dtype = output_dtype
        if not keep_bf16_weight and "weight" in self._parameters:
            del self._parameters["weight"]
            self.register_buffer(
                "weight",
                torch.empty(0, device=weight_fp8.device, dtype=weight_fp8.dtype),
                persistent=False,
            )
        self._fp8_embedding_enabled = True

    def _fp8_embedding(self, x: torch.Tensor) -> torch.Tensor:
        weight_fp8 = self._fp8_embedding_weight
        scale_inv = self._fp8_embedding_scale_inv
        if weight_fp8 is None or scale_inv is None or self._fp8_embedding_output_dtype is None:
            raise RuntimeError("FP8 embedding is enabled without quantized embedding storage")
        flat = x.reshape(-1).to(torch.long)
        y_fp8 = weight_fp8.index_select(0, flat)
        y = y_fp8.to(self._fp8_embedding_output_dtype) * scale_inv.index_select(0, flat).to(
            self._fp8_embedding_output_dtype
        )
        return y.reshape(*x.shape, int(weight_fp8.size(1)))

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        if self._fp8_embedding_enabled:
            y = self._fp8_embedding(x)
        else:
            y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)
        self._fp8_weight_enabled = False
        self._fp8_min_rows = 1
        self._int4_row1_weight_enabled = False
        self._int4_row1_group_size = 0

    def quantize_weight_to_fp8(self, *, keep_bf16_weight: bool = True) -> None:
        weight = self.weight.detach().to(torch.float32)
        weight_scale = _amax_to_scale(weight.abs().amax(dim=1).float())
        weight_fp8_t = (weight.t() * weight_scale.unsqueeze(0)).to(torch.float8_e4m3fn)
        weight_fp8_t = _as_col_major_b(weight_fp8_t)
        weight_scale_inv = weight_scale.reciprocal().reshape(1, -1).contiguous()
        self.register_buffer("weight_fp8_t", weight_fp8_t, persistent=False)
        self.register_buffer("weight_scale_inv", weight_scale_inv, persistent=False)
        if int4_row1_lm_head_enabled():
            int4_weight = _quantize_weight_to_int4pack(weight)
            if int4_weight is not None:
                weight_int4pack, weight_int4_scale_zeros, q_group_size = int4_weight
                self.register_buffer("weight_int4pack", weight_int4pack, persistent=False)
                self.register_buffer("weight_int4_scale_zeros", weight_int4_scale_zeros, persistent=False)
                self._int4_row1_group_size = int(q_group_size)
                self._int4_row1_weight_enabled = True
        self._fp8_min_rows = fp8_lm_head_min_rows()
        if not keep_bf16_weight:
            del self._parameters["weight"]
            self.register_buffer(
                "weight",
                torch.empty(0, device=weight_fp8_t.device, dtype=weight_fp8_t.dtype),
                persistent=False,
            )
        self._fp8_weight_enabled = True

    def greedy_token_ids(self, x: torch.Tensor) -> torch.Tensor:
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        rerank_topk = int4_lm_head_rerank_topk()
        if (
            self.tp_size == 1
            and self._fp8_weight_enabled
            and x.numel() == int(self.weight_fp8_t.size(0))
            and self._int4_row1_weight_enabled
            and _fp8_lm_head_row1_triton_allowed(
                1,
                self._fp8_min_rows,
                self.weight.numel() > 0,
            )
            and x.is_cuda
            and x.dtype is torch.bfloat16
        ):
            if rerank_topk > 0:
                return _int4_row1_argmax_rerank_topk(
                    x,
                    weight_int4pack=self.weight_int4pack,
                    weight_int4_scale_zeros=self.weight_int4_scale_zeros,
                    q_group_size=self._int4_row1_group_size,
                    weight_fp8_t=self.weight_fp8_t,
                    weight_scale_inv=self.weight_scale_inv,
                    rerank_topk=rerank_topk,
                )
            return _int4_row1_argmax(
                x,
                weight_int4pack=self.weight_int4pack,
                weight_int4_scale_zeros=self.weight_int4_scale_zeros,
                q_group_size=self._int4_row1_group_size,
            )
        if (
            self.tp_size == 1
            and self._fp8_weight_enabled
            and x.numel() == int(self.weight_fp8_t.size(0))
            and _fp8_lm_head_row1_triton_allowed(
                1,
                self._fp8_min_rows,
                self.weight.numel() > 0,
            )
            and x.is_cuda
            and x.dtype in {torch.float16, torch.bfloat16}
        ):
            return _fp8_row1_argmax(
                x,
                weight_fp8_t=self.weight_fp8_t,
                weight_scale_inv=self.weight_scale_inv,
            )
        return self.forward(x).argmax(dim=-1)

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        if self._fp8_weight_enabled:
            rows = x.numel() // int(self.weight_fp8_t.size(0))
            if (
                _fp8_lm_head_row1_triton_allowed(
                    rows,
                    self._fp8_min_rows,
                    self.weight.numel() > 0,
                )
                and self._int4_row1_weight_enabled
                and int4_lm_head_rerank_topk() == 0
                and x.is_cuda
                and x.dtype is torch.bfloat16
            ):
                logits = _int4_row1_matvec(
                    x,
                    weight_int4pack=self.weight_int4pack,
                    weight_int4_scale_zeros=self.weight_int4_scale_zeros,
                    q_group_size=self._int4_row1_group_size,
                )
            elif (
                _fp8_lm_head_row1_triton_allowed(
                    rows,
                    self._fp8_min_rows,
                    self.weight.numel() > 0,
                )
                and x.is_cuda
                and x.dtype in {torch.float16, torch.bfloat16}
            ):
                logits = _fp8_row1_matvec(
                    x,
                    weight_fp8_t=self.weight_fp8_t,
                    weight_scale_inv=self.weight_scale_inv,
                    quantize_x=True,
                )
            elif self.weight.numel() == 0 or rows >= self._fp8_min_rows:
                logits = _fp8_linear(
                    x,
                    weight_fp8_t=self.weight_fp8_t,
                    weight_scale_inv=self.weight_scale_inv,
                )
            else:
                logits = F.linear(x, self.weight)
        else:
            logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
