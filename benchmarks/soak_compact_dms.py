"""G1.4 ≥60-min compact DMS soak.

Runs a continuous stream of unique-content multi-tenant batches at FP8 KV +
compact DMS retention. Captures per-batch:
  * torch.cuda.memory_allocated / reserved
  * compact_kv_manager free range count
  * decode tok/s
  * elapsed_s

Exits cleanly when soak duration reached. Bounded RSS drift = (final
torch_reserved_gib - initial torch_reserved_gib) / initial. Free-range count
return = final == initial (no leaked ranges).
"""
from __future__ import annotations

import argparse
import atexit
import json
import random
import time
from pathlib import Path

import torch

from fastdms import LLM, SamplingParams
from fastdms.engine.sequence import Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("results/dms/llama32-1b-cr8-v5-correctmask/final"))
    parser.add_argument("--soak-seconds", type=int, default=3600)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--gen-len", type=int, default=64)
    parser.add_argument("--ctx-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.30)
    parser.add_argument("--compact-kv-budget-bytes", type=int, default=2415919104)
    parser.add_argument("--compact-kv-target-live-tokens-per-seq", type=int, default=None)
    parser.add_argument("--compact-kv-layer-major-metadata", action="store_true")
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _make_random_prompts(num: int, length: int, seed: int) -> list[list[int]]:
    return [
        [random.Random(seed + idx * 1009 + 7).randint(64, 100000) for _ in range(length)]
        for idx in range(num)
    ]


def _free_range_count(manager) -> int:
    if hasattr(manager, "free_ranges"):
        return len(manager.free_ranges)
    if hasattr(manager, "free_ranges_per_layer"):
        return sum(len(ranges) for ranges in manager.free_ranges_per_layer)
    return 0


def _free_token_count(manager) -> int | None:
    value = getattr(manager, "free_tokens", None)
    if value is None:
        return None
    return int(value)


def _free_active_slot_count(manager) -> int | None:
    slots = getattr(manager, "free_active_slots", None)
    if slots is None:
        return None
    return len(slots)


def _run_batch(llm: LLM, args: argparse.Namespace, *, batch_idx: int, seed: int, started: float) -> dict:
    manager = llm.model_runner.compact_kv_manager
    prompts = _make_random_prompts(args.concurrency, args.ctx_len, seed)
    for prompt in prompts:
        llm.scheduler.add(
            Sequence(prompt, SamplingParams(temperature=1.0, max_tokens=args.gen_len, ignore_eos=True))
        )

    batch_started = time.perf_counter()
    prefill_tokens = 0
    prefill_elapsed = 0.0
    decode_tokens = 0
    decode_elapsed = 0.0
    while not llm.is_finished():
        seqs, is_prefill = llm.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else len(seqs)
        step_started = time.perf_counter()
        tokens = llm.model_runner.call("run", seqs, is_prefill)
        step_elapsed = time.perf_counter() - step_started
        llm.scheduler.postprocess(seqs, tokens, is_prefill)
        for sid in llm.scheduler.consume_freed_seq_ids():
            llm.model_runner.call("free_compact", [sid])
        if is_prefill:
            prefill_tokens += num_tokens
            prefill_elapsed += step_elapsed
        else:
            decode_tokens += num_tokens
            decode_elapsed += step_elapsed

    batch_elapsed = time.perf_counter() - batch_started
    return {
        "batch_idx": batch_idx,
        "elapsed_s": batch_elapsed,
        "prefill_tokens": prefill_tokens,
        "prefill_elapsed_s": prefill_elapsed,
        "prefill_tok_s": prefill_tokens / prefill_elapsed if prefill_elapsed > 0 else None,
        "decode_tokens": decode_tokens,
        "decode_elapsed_s": decode_elapsed,
        "decode_tok_s": decode_tokens / decode_elapsed if decode_elapsed > 0 else None,
        "torch_alloc_gib": torch.cuda.memory_allocated() / 2**30,
        "torch_reserved_gib": torch.cuda.memory_reserved() / 2**30,
        "free_ranges": _free_range_count(manager),
        "free_tokens": _free_token_count(manager),
        "free_active_slots": _free_active_slot_count(manager),
        "wall_time_s": time.perf_counter() - started,
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    llm = LLM(
        str(args.model),
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=args.ctx_len + args.gen_len,
        max_num_batched_tokens=args.ctx_len * args.concurrency,
        max_num_seqs=args.concurrency,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_dtype="float8_e4m3fn",
        compact_kv_enabled=True,
        compact_kv_retention_mode="dms",
        compact_kv_budget_bytes=args.compact_kv_budget_bytes,
        compact_kv_target_live_tokens_per_seq=args.compact_kv_target_live_tokens_per_seq,
        compact_kv_layer_major_metadata=args.compact_kv_layer_major_metadata,
    )
    startup_seconds = time.perf_counter() - started
    manager = llm.model_runner.compact_kv_manager
    seed_counter = args.seed * 100
    warmup_started = time.perf_counter()
    warmup_samples = []
    for warmup_idx in range(max(0, args.warmup_batches)):
        seed_counter += 1
        warmup_samples.append(
            _run_batch(llm, args, batch_idx=warmup_idx, seed=seed_counter * 7919, started=started)
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - warmup_started

    initial_alloc = torch.cuda.memory_allocated() / 2**30
    initial_reserved = torch.cuda.memory_reserved() / 2**30
    initial_free_ranges = _free_range_count(manager)
    initial_free_tokens = _free_token_count(manager)
    initial_free_active_slots = _free_active_slot_count(manager)
    samples: list[dict] = []
    batch_idx = 0
    measurement_started = time.perf_counter()
    deadline = measurement_started + args.soak_seconds
    try:
        while time.perf_counter() < deadline:
            seed_counter += 1
            samples.append(
                _run_batch(llm, args, batch_idx=batch_idx, seed=seed_counter * 7919, started=started)
            )
            batch_idx += 1

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        final_alloc = torch.cuda.memory_allocated() / 2**30
        final_reserved = torch.cuda.memory_reserved() / 2**30
        final_free_ranges = _free_range_count(manager)
        final_free_tokens = _free_token_count(manager)
        final_free_active_slots = _free_active_slot_count(manager)
        rss_drift_pct = (
            (final_reserved - initial_reserved) / initial_reserved * 100.0
            if initial_reserved > 0 else None
        )
        alloc_drift_pct = (
            (final_alloc - initial_alloc) / initial_alloc * 100.0
            if initial_alloc > 0 else None
        )
        result = {
            "schema_version": 1,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "soak_seconds_target": args.soak_seconds,
            "soak_seconds_actual": time.perf_counter() - measurement_started,
            "total_wall_seconds": time.perf_counter() - started,
            "startup_seconds": startup_seconds,
            "warmup_batches": args.warmup_batches,
            "warmup_seconds": warmup_seconds,
            "warmup_samples": warmup_samples,
            "batches": batch_idx,
            "concurrency": args.concurrency,
            "gen_len": args.gen_len,
            "ctx_len": args.ctx_len,
            "compact_kv_budget_bytes": args.compact_kv_budget_bytes,
            "compact_kv_target_live_tokens_per_seq": args.compact_kv_target_live_tokens_per_seq,
            "compact_kv_layer_major_metadata": args.compact_kv_layer_major_metadata,
            "initial_alloc_gib": initial_alloc,
            "initial_reserved_gib": initial_reserved,
            "initial_free_ranges": initial_free_ranges,
            "initial_free_tokens": initial_free_tokens,
            "initial_free_active_slots": initial_free_active_slots,
            "final_alloc_gib": final_alloc,
            "final_reserved_gib": final_reserved,
            "final_free_ranges": final_free_ranges,
            "final_free_tokens": final_free_tokens,
            "final_free_active_slots": final_free_active_slots,
            "rss_drift_pct": rss_drift_pct,
            "alloc_drift_pct": alloc_drift_pct,
            "free_ranges_returned": final_free_ranges == initial_free_ranges,
            "free_tokens_returned": (
                final_free_tokens == initial_free_tokens
                if initial_free_tokens is not None and final_free_tokens is not None
                else None
            ),
            "free_active_slots_returned": (
                final_free_active_slots == initial_free_active_slots
                if initial_free_active_slots is not None and final_free_active_slots is not None
                else None
            ),
            "alloc_drift_bounded_5pct": (alloc_drift_pct is not None and abs(alloc_drift_pct) <= 5.0),
            "rss_drift_bounded_5pct": (rss_drift_pct is not None and abs(rss_drift_pct) <= 5.0),
            "decode_tok_s_first": samples[0]["decode_tok_s"] if samples else None,
            "decode_tok_s_last": samples[-1]["decode_tok_s"] if samples else None,
            "decode_tok_s_mean": (
                sum(s["decode_tok_s"] for s in samples if s["decode_tok_s"]) / len(samples)
                if samples else None
            ),
            "prefill_tok_s_mean": (
                sum(s["prefill_tok_s"] for s in samples if s["prefill_tok_s"]) / len(samples)
                if samples else None
            ),
            "samples": samples[:5] + samples[-5:] if len(samples) > 10 else samples,
            "samples_total": len(samples),
        }
        g6_pass = bool(
            result["samples_total"] > 0
            and result["free_ranges_returned"]
            and result["free_tokens_returned"] is not False
            and result["free_active_slots_returned"] is not False
            and result["alloc_drift_bounded_5pct"]
            and result["rss_drift_bounded_5pct"]
        )
        result["passed"] = g6_pass
        result["g6_soak_clean_teardown"] = g6_pass
        result["g6"] = {
            "pass": g6_pass,
            "samples_total": result["samples_total"],
            "free_ranges_returned": result["free_ranges_returned"],
            "free_tokens_returned": result["free_tokens_returned"],
            "free_active_slots_returned": result["free_active_slots_returned"],
            "alloc_drift_bounded_5pct": result["alloc_drift_bounded_5pct"],
            "rss_drift_bounded_5pct": result["rss_drift_bounded_5pct"],
            "alloc_drift_pct": result["alloc_drift_pct"],
            "rss_drift_pct": result["rss_drift_pct"],
            "initial_free_ranges": result["initial_free_ranges"],
            "final_free_ranges": result["final_free_ranges"],
            "initial_free_tokens": result["initial_free_tokens"],
            "final_free_tokens": result["final_free_tokens"],
            "initial_free_active_slots": result["initial_free_active_slots"],
            "final_free_active_slots": result["final_free_active_slots"],
            "soak_seconds_actual": result["soak_seconds_actual"],
        }
    finally:
        try:
            atexit.unregister(llm.exit)
        except Exception:
            pass
        llm.exit()
        torch.cuda.empty_cache()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
