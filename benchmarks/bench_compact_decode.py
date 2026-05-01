import argparse
import json
import math
import time

import torch

from fastdms.layers.compact_attention import (
    compact_attention_splitk_block_n,
    compact_decode_attention,
    compact_decode_attention_ref,
)


def parse_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float8_e4m3fn":
        return torch.float8_e4m3fn
    raise ValueError(f"unsupported dtype {name!r}")


def randn_dtype(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    data = torch.randn(shape, device="cuda", dtype=torch.float32)
    if dtype == torch.float8_e4m3fn:
        data = data.clamp(-448.0, 448.0)
    return data.to(dtype=dtype)


def build_inputs(
    batch: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    live: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
):
    q = randn_dtype((batch, q_heads, head_dim), q_dtype)
    total_spans = batch * kv_heads
    compact_k = randn_dtype((total_spans * live, head_dim), kv_dtype)
    compact_v = randn_dtype((total_spans * live, head_dim), kv_dtype)
    base_offsets = torch.arange(total_spans, device="cuda", dtype=torch.int32).reshape(batch, kv_heads) * live
    live_counts = torch.full((batch, kv_heads), live, device="cuda", dtype=torch.int32)
    return q, compact_k, compact_v, base_offsets, live_counts


def build_rope_inputs(batch: int, head_dim: int, max_position: int = 8192):
    positions = torch.full((batch,), max_position, device="cuda", dtype=torch.int64)
    half = head_dim // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, device="cuda", dtype=torch.float32) / half))
    pos = torch.arange(max_position + 1, device="cuda", dtype=torch.float32)
    freqs = pos[:, None] * inv_freq[None, :]
    cos_sin = torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=1).to(torch.float32).unsqueeze(1)
    return positions, cos_sin


def apply_inline_q_rope_ref(
    q: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    kv_heads: int,
) -> torch.Tensor:
    batch, q_heads, head_dim = q.shape
    group_size = q_heads // kv_heads
    half = head_dim // 2
    q_float = q.float().clone()
    cos = cos_sin_cache[positions, 0, :half].float()
    sin = cos_sin_cache[positions, 0, half:].float()
    q1 = q_float[:, :, :half]
    q2 = q_float[:, :, half:].clone()
    group_first_heads = torch.arange(0, q_heads, group_size, device=q.device)
    q2[:, group_first_heads, half - 1] = 0.0
    rot1 = q1 * cos[:, None, :] - q2 * sin[:, None, :]
    rot2 = q2 * cos[:, None, :] + q1 * sin[:, None, :]
    return torch.cat([rot1, rot2], dim=-1).to(dtype=q.dtype)


def time_cuda(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--live", type=int, nargs="+", default=[2048, 4096, 8192, 16384])
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="")
    parser.add_argument("--q-dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--kv-dtype", choices=["float16", "bfloat16", "float8_e4m3fn"], default="float8_e4m3fn")
    parser.add_argument("--inline-q-rope", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--verify-live", type=int, default=256)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--scalar", action="store_true")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    if args.dtype:
        args.q_dtype = args.dtype
        args.kv_dtype = args.dtype
    q_dtype = parse_dtype(args.q_dtype)
    kv_dtype = parse_dtype(args.kv_dtype)
    torch.manual_seed(0)
    torch.cuda.synchronize()
    rope_positions, rope_cache = (None, None)
    if args.inline_q_rope:
        if args.head_dim % 2 != 0:
            raise ValueError("--inline-q-rope requires an even head dimension")
        rope_positions, rope_cache = build_rope_inputs(args.batch, args.head_dim)

    q, compact_k, compact_v, base_offsets, live_counts = build_inputs(
        args.batch,
        args.q_heads,
        args.kv_heads,
        args.head_dim,
        args.verify_live,
        q_dtype,
        kv_dtype,
    )
    ref_q = (
        apply_inline_q_rope_ref(q, rope_positions, rope_cache, args.kv_heads)
        if args.inline_q_rope
        else q
    )
    ref = compact_decode_attention_ref(ref_q, compact_k, compact_v, base_offsets, live_counts)
    out = compact_decode_attention(
        q,
        compact_k,
        compact_v,
        base_offsets,
        live_counts,
        grouped=not args.scalar,
        q_rope_positions=rope_positions,
        q_rope_cos_sin_cache=rope_cache,
    )
    torch.cuda.synchronize()
    max_abs_err = float((out.float() - ref.float()).abs().max().item())
    mean_abs_err = float((out.float() - ref.float()).abs().mean().item())

    rows = []
    block_n = compact_attention_splitk_block_n()
    for live in args.live:
        q, compact_k, compact_v, base_offsets, live_counts = build_inputs(
            args.batch,
            args.q_heads,
            args.kv_heads,
            args.head_dim,
            live,
            q_dtype,
            kv_dtype,
        )
        if args.inline_q_rope:
            rope_positions, rope_cache = build_rope_inputs(args.batch, args.head_dim)
        latency_ms = time_cuda(
            lambda: compact_decode_attention(
                q,
                compact_k,
                compact_v,
                base_offsets,
                live_counts,
                grouped=not args.scalar,
                q_rope_positions=rope_positions,
                q_rope_cos_sin_cache=rope_cache,
            ),
            args.warmup,
            args.iters,
        )
        rows.append(
            {
                "batch": args.batch,
                "q_heads": args.q_heads,
                "kv_heads": args.kv_heads,
                "head_dim": args.head_dim,
                "live": live,
                "q_dtype": args.q_dtype,
                "kv_dtype": args.kv_dtype,
                "kernel": "scalar" if args.scalar else "grouped_splitk",
                "inline_q_rope": args.inline_q_rope,
                "block_n": block_n,
                "num_splits": int(math.ceil(live / block_n)),
                "latency_ms": latency_ms,
                "decode_tokens_per_s": args.batch / (latency_ms / 1000.0),
                "projected_attention_only_tokens_per_s": args.batch / ((latency_ms * args.layers) / 1000.0),
            }
        )

    result = {
        "device": torch.cuda.get_device_name(),
        "timestamp_unix": int(time.time()),
        "verify": {
            "live": args.verify_live,
            "max_abs_err": max_abs_err,
            "mean_abs_err": mean_abs_err,
        },
        "q_dtype": args.q_dtype,
        "kv_dtype": args.kv_dtype,
        "inline_q_rope": args.inline_q_rope,
        "warmup": args.warmup,
        "iters": args.iters,
        "layers_for_projection": args.layers,
        "rows": rows,
    }
    print(json.dumps(result, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
