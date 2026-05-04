from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from fastdms.engine.dms import (
    build_dms_live_mask,
    extract_dms_eviction_decisions,
    load_dms_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate FastDMS DMS metadata and mask semantics.")
    parser.add_argument("--model", type=Path, default=Path("results/dms/llama32-1b-cr8-v5-correctmask/final"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = load_dms_metadata(args.model)

    tokens = 4
    num_kv_heads = 8
    q_per_kv = 4
    num_qo_heads = num_kv_heads * q_per_kv
    head_dim = 64
    q = torch.zeros(tokens, num_qo_heads, head_dim, dtype=torch.float32)
    grouped = q.view(tokens, num_kv_heads, q_per_kv, head_dim)
    threshold = metadata.alpha_offset / metadata.alpha_scale
    grouped[0, 0, 0, -1] = threshold + 0.01
    grouped[1, 0, 0, -1] = threshold - 0.01
    grouped[2, 3, 0, -1] = threshold + 0.01

    q_clean, evict = extract_dms_eviction_decisions(
        q,
        num_kv_heads=num_kv_heads,
        num_qo_heads=num_qo_heads,
        head_dim=head_dim,
        alpha_scale=metadata.alpha_scale,
        alpha_offset=metadata.alpha_offset,
    )
    live = build_dms_live_mask(
        evict.T.unsqueeze(0),
        current_position=tokens - 1,
        window_size=1,
    )

    borrowed_after_zero = q_clean.view(tokens, num_kv_heads, q_per_kv, head_dim)[:, :, 0, -1]
    checks = {
        "metadata_window_size": metadata.window_size == 256,
        "metadata_alpha_scale": metadata.alpha_scale == 100.0,
        "metadata_alpha_offset": metadata.alpha_offset == 5.0,
        "parent_training_log_fallback": metadata.source_kind == "parent_training_log",
        "evict_true_token0_head0": bool(evict[0, 0].item()),
        "evict_false_token1_head0": not bool(evict[1, 0].item()),
        "evict_true_token2_head3": bool(evict[2, 3].item()),
        "borrowed_neuron_zeroed": float(borrowed_after_zero.abs().max().item()) == 0.0,
        "old_evicted_token_not_live": not bool(live[0, 0, 0].item()),
        "old_non_evicted_token_live": bool(live[0, 0, 1].item()),
        "recent_evicted_token_live": bool(live[0, 3, 2].item()),
    }
    ok = all(checks.values())
    result = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "model": str(args.model),
        "metadata": metadata.to_dict(),
        "threshold": threshold,
        "evict_shape": list(evict.shape),
        "live_shape": list(live.shape),
        "checks": checks,
        "ok": ok,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
