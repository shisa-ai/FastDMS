"""I1.2 multi-tenant compact DMS smoke at TP=1.

Sends N=16 concurrent requests with mixed prompt lengths (1k/4k/8k) at FP8 KV
+ compact DMS retention. Confirms (a) the scheduler does not deadlock on
compact-arena admission, (b) all requests complete with non-empty
completions, and (c) fragmentation is bounded (residual free range counts at
end ~ initial).
"""
from __future__ import annotations

import argparse
import atexit
import json
import sys
import time
from pathlib import Path

import torch

from fastdms import LLM, SamplingParams
from fastdms.engine.sequence import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.bench_vllm_turboquant import _make_prompts  # noqa: F401  (kept for reference)
import random


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("results/dms/llama32-1b-cr8-v5-correctmask/final"))
    parser.add_argument("--gen-len", type=int, default=64)
    parser.add_argument("--num-requests", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.20)
    parser.add_argument("--compact-kv-budget-bytes", type=int, default=2415919104)
    parser.add_argument("--compact-kv-target-live-tokens-per-seq", type=int, default=9216)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    # Mix of 1k / 4k / 8k prompts; cycle lengths so 16 requests have e.g. 6 1k
    # + 5 4k + 5 8k.
    lengths = [1024, 4096, 8192]
    pattern = []
    while len(pattern) < args.num_requests:
        for L in lengths:
            pattern.append(L)
            if len(pattern) >= args.num_requests:
                break

    # Build truly unique prompts (random tokens with a pinned vocab range) to
    # guarantee no prefix-cache hits. I1.1 documents that compact + prefix
    # cache is currently broken in the fastdms scheduler/allocate
    # interaction; this smoke deliberately holds prefix cache off so I1.2
    # measures fragmentation under genuine multi-tenant traffic.
    rng = random.Random(args.seed)
    prompts = []
    for idx, L in enumerate(pattern):
        rng_seq = random.Random(args.seed + idx * 1009 + 7)
        # Stay inside a safe vocab id range (pad/eos token id is rarely 12345
        # for Llama 3.2; use a wide range and step through unique tokens).
        prompts.append([rng_seq.randint(64, 100000) for _ in range(L)])

    max_model_len = max(pattern) + args.gen_len
    llm = LLM(
        str(args.model),
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=max_model_len,
        max_num_batched_tokens=max(pattern) * args.num_requests,
        max_num_seqs=args.num_requests,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_dtype="float8_e4m3fn",
        compact_kv_enabled=True,
        compact_kv_retention_mode="dms",
        compact_kv_budget_bytes=args.compact_kv_budget_bytes,
        compact_kv_target_live_tokens_per_seq=args.compact_kv_target_live_tokens_per_seq,
    )
    try:
        for prompt in prompts:
            seq = Sequence(prompt, SamplingParams(temperature=1.0, max_tokens=args.gen_len, ignore_eos=True))
            llm.scheduler.add(seq)

        completions: dict[int, list[int]] = {}
        steps = 0
        max_admission_pending = 0
        free_ranges_initial = len(llm.model_runner.compact_kv_manager.free_ranges)
        free_ranges_max = free_ranges_initial
        deadlock_step_budget = args.num_requests * (max(pattern) + args.gen_len) * 4

        while not llm.is_finished() and steps < deadlock_step_budget:
            seqs, is_prefill = llm.scheduler.schedule()
            tokens = llm.model_runner.call("run", seqs, is_prefill)
            llm.scheduler.postprocess(seqs, tokens, is_prefill)
            for sid in llm.scheduler.consume_freed_seq_ids():
                llm.model_runner.call("free_compact", [sid])
            for seq in seqs:
                if seq.is_finished:
                    completions[seq.seq_id] = list(seq.completion_token_ids)
            free_ranges_max = max(free_ranges_max, len(llm.model_runner.compact_kv_manager.free_ranges))
            steps += 1

        finished = sum(1 for c in completions.values() if c)
        non_empty_lengths = [len(c) for c in completions.values() if c]
        free_ranges_final = len(llm.model_runner.compact_kv_manager.free_ranges)
        result = {
            "schema_version": 1,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "num_requests": args.num_requests,
            "prompt_length_pattern": pattern,
            "completed_requests": finished,
            "deadlocked": steps >= deadlock_step_budget,
            "steps_taken": steps,
            "deadlock_step_budget": deadlock_step_budget,
            "completion_lengths_min": min(non_empty_lengths) if non_empty_lengths else None,
            "completion_lengths_max": max(non_empty_lengths) if non_empty_lengths else None,
            "completion_lengths_mean": (
                sum(non_empty_lengths) / len(non_empty_lengths) if non_empty_lengths else None
            ),
            "expected_completion_length": args.gen_len,
            "free_ranges_initial": free_ranges_initial,
            "free_ranges_max_during_run": free_ranges_max,
            "free_ranges_final": free_ranges_final,
            "fragmentation_bounded": free_ranges_max <= max(2, args.num_requests + 2),
            "elapsed_s": time.perf_counter() - started,
        }
    finally:
        try:
            atexit.unregister(llm.exit)
        except Exception:
            pass
        llm.exit()
        torch.cuda.empty_cache()

    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
