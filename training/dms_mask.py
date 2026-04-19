"""Shared DMS attention-mask helpers."""

from __future__ import annotations

import torch


def dms_outside_window_mask(
    seq_len: int,
    window_size: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return [query, key] mask for past keys outside the DMS grace window."""
    positions = torch.arange(seq_len, device=device)
    positions_diff = positions.unsqueeze(1) - positions.unsqueeze(0)
    return positions_diff > window_size
