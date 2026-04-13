"""
HIGGS vector quantizer: Hadamard rotation + EDEN lattice nearest-neighbor lookup.

This is the core quantize/dequantize engine. For cache integration, see packed_cache.py.
"""
from collections import namedtuple
from functools import partial
from typing import TypeVar

import torch
from fast_hadamard_transform import hadamard_transform

from .edenn import get_grid, get_grid_norms_squared, pad_to_block

QuantizedTensor = namedtuple("QuantizedTensor", ["idx", "scales"])


class QuantizerBase:
    QuantizedState = TypeVar("QuantizedState")

    def quantize(self, x: torch.Tensor) -> QuantizedState:
        raise NotImplementedError

    def dequantize(self, quantized: QuantizedState) -> torch.Tensor:
        raise NotImplementedError

    def quantize_dequantize(self, x: torch.Tensor) -> torch.Tensor:
        return self.dequantize(self.quantize(x)).to(dtype=x.dtype, device=x.device)


class HiggsQuantizer(QuantizerBase):
    """
    HIGGS vector quantization (Ashkboos et al., arXiv:2411.17525).

    1. Pad input to a multiple of hadamard_groupsize.
    2. Apply Hadamard transform per group, normalize by L2 norm (stored as scale).
    3. Nearest-neighbor lookup against a precomputed EDEN lattice grid.
    4. Store uint8 grid indices + float32 scales.

    Dequantize reverses: grid lookup → rescale → inverse Hadamard → trim padding.

    :param hadamard_groupsize: group size for the Hadamard rotation
    :param edenn_d: EDEN grid dimension (always 2 for our grids)
    :param edenn_n: EDEN grid size: 16=2bit, 64=3bit, 256=4bit
    :param channel_size: actual channel dim before padding (e.g. head_dim * num_kv_heads)
    :param chunk_size: batch chunk size for quantization to limit peak memory
    """

    def __init__(
        self,
        hadamard_groupsize: int,
        edenn_d: int,
        edenn_n: int,
        channel_size: int = 1024,
        chunk_size: int = 64,
    ) -> None:
        super().__init__()
        self.hadamard_groupsize = hadamard_groupsize
        self.channel_size = channel_size
        self.edenn_d = edenn_d
        self.edenn_n = edenn_n
        self.chunk_size = chunk_size
        self.hadamard_scale = 1.0 / hadamard_groupsize
        self._grid = partial(get_grid, dim=edenn_d, size=edenn_n)
        self._grid_norm = partial(get_grid_norms_squared, dim=edenn_d, size=edenn_n)

    @torch.no_grad()
    def quantize(self, x: torch.Tensor) -> QuantizedTensor:
        """
        Quantize x of shape [B, C] → QuantizedTensor(idx=uint8[B, mult, groups], scales=float32[B, mult]).
        """
        batch_size, channel_size = x.shape
        assert channel_size == self.channel_size, (
            f"Expected channel_size={self.channel_size}, got {channel_size}"
        )
        device = x.device

        x = x.to(dtype=torch.float32)
        x = pad_to_block(x, [1], self.hadamard_groupsize)
        mult = x.shape[1] // self.hadamard_groupsize
        x = x.reshape(-1, mult, self.hadamard_groupsize)

        scales = torch.linalg.norm(x, axis=-1)  # [B, mult]
        x = hadamard_transform(x) / scales[:, :, None]

        x = pad_to_block(x, [2], self.edenn_d).reshape(batch_size, mult, -1, self.edenn_d)

        result_idx = torch.empty((batch_size, mult, x.shape[2]), dtype=torch.uint8)
        for i, chunk in enumerate(torch.split(x, self.chunk_size, dim=0)):
            chunk_idx = torch.argmax(
                2 * chunk @ self._grid(device=device).T - self._grid_norm(device=device),
                dim=-1,
            )
            result_idx[i * self.chunk_size : (i + 1) * self.chunk_size] = chunk_idx

        return QuantizedTensor(result_idx, scales)

    @torch.no_grad()
    def dequantize(self, quantized: QuantizedTensor) -> torch.Tensor:
        """
        Dequantize QuantizedTensor → float tensor of shape [B, channel_size].
        """
        idx, scales = quantized.idx, quantized.scales
        device = scales.device

        x = self._grid(device=device)[idx.int()].flatten(start_dim=2)  # [B, mult, groupsize]
        x = x[..., : self.hadamard_groupsize]  # trim EDEN padding
        x = x * scales.unsqueeze(dim=2)
        x = hadamard_transform(x, scale=self.hadamard_scale).flatten(start_dim=1)  # [B, C]
        return x[:, : self.channel_size]

    def packed_nbytes_per_row(self) -> int:
        """Estimate bytes per quantized row (for memory accounting)."""
        padded = ((self.channel_size - 1) // self.hadamard_groupsize + 1) * self.hadamard_groupsize
        mult = padded // self.hadamard_groupsize
        groups_per_mult = ((self.hadamard_groupsize - 1) // self.edenn_d + 1)
        idx_bytes = mult * groups_per_mult * 1  # uint8
        scale_bytes = mult * 4  # float32
        return idx_bytes + scale_bytes

    def bf16_nbytes_per_row(self) -> int:
        """Bytes per row in BF16 (for comparison)."""
        return self.channel_size * 2
