from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def _extract_dms_eviction_decisions_kernel(
    q_ptr,
    evict_ptr,
    q_stride_t: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    evict_stride_t: tl.constexpr,
    evict_stride_h: tl.constexpr,
    num_kv_heads: tl.constexpr,
    group_size: tl.constexpr,
    head_dim: tl.constexpr,
    alpha_scale: tl.constexpr,
    alpha_offset: tl.constexpr,
    positive_scale: tl.constexpr,
    total: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    token_idx = offsets // num_kv_heads
    kv_head = offsets - token_idx * num_kv_heads
    q_head = kv_head * group_size
    q_offset = token_idx * q_stride_t + q_head * q_stride_h + (head_dim - 1) * q_stride_d
    decision_value = tl.load(q_ptr + q_offset, mask=mask, other=0.0).to(tl.float32)
    if positive_scale:
        evict = decision_value > (alpha_offset / alpha_scale)
    else:
        evict = (decision_value * alpha_scale - alpha_offset) > 0.0
    tl.store(evict_ptr + token_idx * evict_stride_t + kv_head * evict_stride_h, evict, mask=mask)
    tl.store(q_ptr + q_offset, 0.0, mask=mask)


@dataclass(frozen=True, slots=True)
class DMSMetadata:
    model_path: str
    source_path: str
    source_kind: str
    packaged_metadata_found: bool
    window_size: int
    alpha_scale: float
    alpha_offset: float
    target_cr: int | None = None
    context_len: int | None = None
    compression_mode: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _metadata_from_config(
    *,
    model_path: Path,
    source_path: Path,
    source_kind: str,
    packaged_metadata_found: bool,
    config: dict,
) -> DMSMetadata:
    window_size = config.get("window_size", config.get("dms_window_size", 256))
    alpha_scale = config.get("alpha_scale", config.get("dms_alpha_scale", 100.0))
    alpha_offset = config.get(
        "alpha_offset",
        config.get("dms_alpha_offset", config.get("dms_initial_alpha_offset", 5.0)),
    )
    target_cr = config.get("target_cr", config.get("dms_cr"))
    context_len = config.get("context_len", config.get("max_position_embeddings"))
    return DMSMetadata(
        model_path=str(model_path),
        source_path=str(source_path),
        source_kind=source_kind,
        packaged_metadata_found=packaged_metadata_found,
        window_size=int(window_size),
        alpha_scale=float(alpha_scale),
        alpha_offset=float(alpha_offset),
        target_cr=None if target_cr is None else int(target_cr),
        context_len=None if context_len is None else int(context_len),
        compression_mode=config.get("compression_mode"),
    )


def load_dms_metadata(model_path: str | Path, metadata_path: str | Path | None = None) -> DMSMetadata:
    """Load DMS runtime metadata for corrected-mask checkpoints.

    Preferred packaging is ``dms_metadata.json`` beside the model weights.  The
    retained corrected-mask v5 checkpoint lacks that file, so the fallback reads
    the parent ``training_log.json`` config and labels the source explicitly.
    """
    model_path = Path(model_path)
    if metadata_path is not None:
        path = Path(metadata_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        config = data.get("config", data)
        return _metadata_from_config(
            model_path=model_path,
            source_path=path,
            source_kind="explicit_metadata",
            packaged_metadata_found=path.name == "dms_metadata.json",
            config=config,
        )

    packaged = model_path / "dms_metadata.json"
    if packaged.exists():
        data = json.loads(packaged.read_text(encoding="utf-8"))
        config = data.get("config", data)
        return _metadata_from_config(
            model_path=model_path,
            source_path=packaged,
            source_kind="packaged_metadata",
            packaged_metadata_found=True,
            config=config,
        )

    training_log = model_path.parent / "training_log.json"
    if training_log.exists():
        data = json.loads(training_log.read_text(encoding="utf-8"))
        return _metadata_from_config(
            model_path=model_path,
            source_path=training_log,
            source_kind="parent_training_log",
            packaged_metadata_found=False,
            config=data.get("config", {}),
        )

    hf_config = model_path / "config.json"
    if hf_config.exists():
        data = json.loads(hf_config.read_text(encoding="utf-8"))
        if any(key in data for key in ("dms_window_size", "dms_alpha_scale", "dms_initial_alpha_offset", "dms_cr")):
            return _metadata_from_config(
                model_path=model_path,
                source_path=hf_config,
                source_kind="hf_config_dms",
                packaged_metadata_found=False,
                config=data,
            )

    raise FileNotFoundError(
        f"no DMS metadata found for {model_path}; expected {packaged} or {training_log}"
    )


def extract_dms_eviction_decisions(
    q: torch.Tensor,
    *,
    num_kv_heads: int,
    num_qo_heads: int,
    head_dim: int,
    alpha_scale: float,
    alpha_offset: float,
    inplace: bool = False,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract per-KV-head DMS eviction decisions and zero the borrowed neuron.

    The corrected-mask DMS convention stores one scalar decision in the last
    channel of the first query head for each GQA group.  Returned ``evict_mask``
    is shaped ``[tokens, kv_heads]`` with ``True`` meaning this token should be
    evicted for that KV head once it leaves the DMS retention window.
    """
    if num_qo_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_qo_heads={num_qo_heads} must be divisible by num_kv_heads={num_kv_heads}"
        )
    if q.ndim == 2:
        if q.shape[1] != num_qo_heads * head_dim:
            raise ValueError(f"expected flat q width {num_qo_heads * head_dim}, got {q.shape[1]}")
        q_clean = q if inplace else q.clone()
        q_view = q_clean.view(q.shape[0], num_qo_heads, head_dim)
    elif q.ndim == 3:
        if q.shape[1:] != (num_qo_heads, head_dim):
            raise ValueError(f"expected q shape [tokens, {num_qo_heads}, {head_dim}], got {tuple(q.shape)}")
        q_clean = q if inplace else q.clone()
        q_view = q_clean
    else:
        raise ValueError(f"expected q as [tokens, heads, dim] or [tokens, heads*dim], got {tuple(q.shape)}")

    if out is not None:
        expected_shape = (q.shape[0], num_kv_heads)
        if tuple(out.shape) != expected_shape:
            raise ValueError(f"expected DMS output shape {expected_shape}, got {tuple(out.shape)}")
        if out.dtype != torch.bool:
            raise ValueError(f"expected DMS output dtype torch.bool, got {out.dtype}")
        if out.device != q_clean.device:
            raise ValueError(f"expected DMS output on {q_clean.device}, got {out.device}")

    if q_clean.is_cuda and q_clean.numel() > 0:
        evict_mask = out
        if evict_mask is None:
            evict_mask = torch.empty((q.shape[0], num_kv_heads), device=q_clean.device, dtype=torch.bool)
        group_size = num_qo_heads // num_kv_heads
        total = q.shape[0] * num_kv_heads
        block = 256
        _extract_dms_eviction_decisions_kernel[(triton.cdiv(total, block),)](
            q_view,
            evict_mask,
            q_view.stride(0),
            q_view.stride(1),
            q_view.stride(2),
            evict_mask.stride(0),
            evict_mask.stride(1),
            num_kv_heads,
            group_size,
            head_dim,
            float(alpha_scale),
            float(alpha_offset),
            bool(alpha_scale > 0),
            total,
            block,
        )
        return q_clean, evict_mask

    grouped = q_view.view(q.shape[0], num_kv_heads, num_qo_heads // num_kv_heads, head_dim)
    decision_values = grouped[:, :, 0, -1].float()
    if alpha_scale > 0:
        evict_mask = decision_values > (alpha_offset / alpha_scale)
    else:
        evict_mask = (decision_values * alpha_scale - alpha_offset) > 0
    grouped[:, :, 0, -1] = 0
    if out is not None:
        out.copy_(evict_mask)
        evict_mask = out
    return q_clean, evict_mask


def build_dms_live_mask(
    evict_mask: torch.Tensor,
    *,
    current_position: int,
    window_size: int,
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build a compact-KV live mask from recorded DMS eviction decisions.

    ``evict_mask`` may be ``[kv_heads, tokens]`` or
    ``[layers, kv_heads, tokens]``. A token remains live if it has not been
    marked for eviction for that layer/head, or if it is still inside the
    sliding retention window relative to ``current_position``.
    """
    if evict_mask.ndim not in (2, 3):
        raise ValueError(f"expected evict_mask rank 2 or 3, got {evict_mask.ndim}")
    if window_size < 0:
        raise ValueError(f"window_size must be non-negative, got {window_size}")
    tokens = evict_mask.shape[-1]
    if positions is None:
        positions = torch.arange(tokens, device=evict_mask.device)
    else:
        positions = positions.to(device=evict_mask.device)
    if positions.shape != (tokens,):
        raise ValueError(f"expected positions shape {(tokens,)}, got {tuple(positions.shape)}")

    inside_window = (int(current_position) - positions.to(torch.long)) <= int(window_size)
    return (~evict_mask.bool()) | inside_window.view((1,) * (evict_mask.ndim - 1) + (tokens,))
