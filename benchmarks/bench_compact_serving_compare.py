from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from fastdms import LLM, SamplingParams
from smoke_compact_end_to_end import ensure_tiny_qwen3


CASES = ("dense", "compact_all", "compact_dms_stride")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare tiny FastDMS dense and compact serving paths.")
    parser.add_argument("--model-dir", type=Path, default=Path("results/fastdms/tiny-qwen3-random"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--case", choices=CASES, default="dense")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--prompt-len", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--compact-capacity-tokens", type=int, default=1024)
    parser.add_argument("--compact-retention-stride", type=int, default=2)
    parser.add_argument("--compact-retention-recent-tokens", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.02)
    return parser.parse_args()


def run_worker(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    prompt = list(range(4, 4 + args.prompt_len))
    prompts = [prompt[:] for _ in range(args.batch)]
    sampling_params = [
        SamplingParams(temperature=1.0, max_tokens=args.max_tokens, ignore_eos=True)
        for _ in range(args.batch)
    ]
    compact_enabled = args.case != "dense"
    compact_mode = "dms_stride" if args.case == "compact_dms_stride" else "all"
    llm_kwargs = dict(
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=64,
        max_num_batched_tokens=128,
        max_num_seqs=max(args.batch, 2),
        gpu_memory_utilization=args.gpu_memory_utilization,
        compact_kv_enabled=compact_enabled,
    )
    if compact_enabled:
        llm_kwargs.update(
            compact_kv_capacity_tokens=args.compact_capacity_tokens,
            compact_kv_retention_mode=compact_mode,
            compact_kv_retention_stride=args.compact_retention_stride,
            compact_kv_retention_recent_tokens=args.compact_retention_recent_tokens,
        )

    llm = LLM(str(args.model_dir), **llm_kwargs)
    warmup_params = SamplingParams(temperature=1.0, max_tokens=1, ignore_eos=True)
    llm.generate([prompt], [warmup_params], use_tqdm=False)
    torch.cuda.synchronize()
    started = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - started

    generated_tokens = sum(len(output["token_ids"]) for output in outputs)
    manager = llm.model_runner.compact_kv_manager
    result = {
        "case": args.case,
        "batch": args.batch,
        "prompt_len": args.prompt_len,
        "max_tokens": args.max_tokens,
        "generated_tokens": generated_tokens,
        "elapsed_s": elapsed_s,
        "generated_tok_s": generated_tokens / elapsed_s,
        "num_dense_kv_blocks": llm.model_runner.config.num_kvcache_blocks,
        "compact_enabled": compact_enabled,
        "compact_retention_mode": compact_mode if compact_enabled else None,
        "compact_retention_stride": args.compact_retention_stride if compact_enabled else None,
        "compact_retention_recent_tokens": args.compact_retention_recent_tokens if compact_enabled else None,
        "compact_capacity_tokens": None if manager is None else manager.capacity_tokens,
        "compact_free_tokens_after_generate": None if manager is None else manager.free_tokens,
        "active_compact_sequences_after_generate": None if manager is None else len(manager.seq_states),
    }
    print(json.dumps(result, sort_keys=True))


def run_parent(args: argparse.Namespace) -> None:
    model_created = ensure_tiny_qwen3(args.model_dir, args.seed)
    rows = []
    for case in CASES:
        cmd = [
            sys.executable,
            __file__,
            "--worker",
            "--case",
            case,
            "--model-dir",
            str(args.model_dir),
            "--seed",
            str(args.seed),
            "--batch",
            str(args.batch),
            "--prompt-len",
            str(args.prompt_len),
            "--max-tokens",
            str(args.max_tokens),
            "--compact-capacity-tokens",
            str(args.compact_capacity_tokens),
            "--compact-retention-stride",
            str(args.compact_retention_stride),
            "--compact-retention-recent-tokens",
            str(args.compact_retention_recent_tokens),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
        ]
        proc = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        rows.append(json.loads(proc.stdout.strip().splitlines()[-1]))

    result = {
        "artifact": "plan13-c12-nanovllm-tiny-serving-diagnostic",
        "model_dir": str(args.model_dir),
        "model_created": model_created,
        "note": (
            "Tiny random Qwen3 diagnostic only. Use this to compare FastDMS dense vs compact "
            "wiring, not as a quality or production-speed comparison to retained mini-sglang rows."
        ),
        "rows": rows,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.worker:
        run_worker(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
