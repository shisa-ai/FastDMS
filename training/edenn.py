"""
EDEN lattice grid loading for HIGGS quantization.

Grids are precomputed MSE-optimal quantization lattices from the HIGGS paper
(Ashkboos et al., arXiv:2411.17525). Only EDEN2 grids are shipped:
  EDEN2-16   → 2-bit (16 codewords in 2D)
  EDEN2-64   → 3-bit (64 codewords in 2D)
  EDEN2-256  → 4-bit (256 codewords in 2D)

Based on https://gist.github.com/galqiwi/d8fdeb2c6603ad3e54d72a0801416ad3
"""
import functools
import pathlib
import warnings

import torch
import torch.nn.functional as F

_GRIDS_DIR = pathlib.Path(__file__).parent / "grids"

# Load grids at import time — only EDEN2 files are shipped.
GRIDS: dict[int, dict[int, torch.Tensor]] = {}

for _file in _GRIDS_DIR.iterdir():
    if _file.suffix != ".pt" or not _file.name.startswith("EDEN"):
        continue
    try:
        _dim, _size = map(int, _file.stem[4:].split("-"))
    except ValueError:
        warnings.warn(f"Failed to parse grid file: {_file}")
        continue
    GRIDS.setdefault(_dim, {})[_size] = torch.load(_file, map_location="cpu", weights_only=True)


@functools.lru_cache()
def get_grid(dim: int, size: int, device: torch.device) -> torch.Tensor:
    """Return the EDEN grid of shape [size, dim] on the requested device."""
    return GRIDS[dim][size].to(device)


@functools.lru_cache()
def get_grid_norms_squared(dim: int, size: int, device: torch.device) -> torch.Tensor:
    """Return squared L2 norms of each grid codeword, shape [size]."""
    return torch.linalg.norm(get_grid(dim, size, device), dim=1).square()


def pad_to_block(tensor: torch.Tensor, dims: list[int], block_size: int, value: float = 0) -> torch.Tensor:
    """Pad tensor so that each specified dim is a multiple of block_size."""
    pad_dims = [0] * (2 * tensor.ndim)
    for dim in dims:
        size = tensor.shape[dim]
        next_multiple = ((size - 1) // block_size + 1) * block_size
        pad_dims[-2 * dim - 1] = next_multiple - size
    return F.pad(tensor, pad_dims, "constant", value)
