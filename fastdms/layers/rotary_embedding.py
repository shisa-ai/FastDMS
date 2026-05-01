import math
from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


def _apply_llama3_freq_scaling(
    inv_freq: torch.Tensor,
    factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    original_max_position: int,
) -> torch.Tensor:
    # Mirrors transformers _compute_llama3_parameters: per-frequency piecewise
    # scaling with smooth interpolation between low_freq_wavelen and
    # high_freq_wavelen so long-context positions degrade gracefully.
    low_freq_wavelen = original_max_position / low_freq_factor
    high_freq_wavelen = original_max_position / high_freq_factor
    wavelen = 2 * math.pi / inv_freq
    inv_freq_low = inv_freq / factor
    smooth = (original_max_position / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smooth_freq = (1.0 - smooth) * inv_freq_low + smooth * inv_freq
    out = torch.where(wavelen < high_freq_wavelen, inv_freq, smooth_freq)
    out = torch.where(wavelen > low_freq_wavelen, inv_freq_low, out)
    return out


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        if rope_scaling is not None:
            rope_type = rope_scaling.get("rope_type") or rope_scaling.get("type")
            if rope_type == "llama3":
                inv_freq = _apply_llama3_freq_scaling(
                    inv_freq,
                    factor=float(rope_scaling["factor"]),
                    low_freq_factor=float(rope_scaling.get("low_freq_factor", 1.0)),
                    high_freq_factor=float(rope_scaling.get("high_freq_factor", 4.0)),
                    original_max_position=int(rope_scaling["original_max_position_embeddings"]),
                )
            elif rope_type in (None, "default", "linear"):
                if rope_type == "linear":
                    inv_freq = inv_freq / float(rope_scaling.get("factor", 1.0))
            else:
                raise NotImplementedError(f"rope_type={rope_type!r} is not implemented")
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


_ROPE_CACHE: dict[tuple, RotaryEmbedding] = {}


def _rope_scaling_key(rope_scaling: dict | None) -> tuple | None:
    if rope_scaling is None:
        return None
    return tuple(sorted(
        (k, tuple(v) if isinstance(v, list) else v)
        for k, v in rope_scaling.items()
    ))


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    cache_key = (head_size, rotary_dim, max_position, base, _rope_scaling_key(rope_scaling))
    cached = _ROPE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    rope = RotaryEmbedding(head_size, rotary_dim, max_position, base, rope_scaling=rope_scaling)
    _ROPE_CACHE[cache_key] = rope
    return rope
