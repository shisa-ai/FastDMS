from __future__ import annotations

import argparse
import atexit
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from fastdms import LLM, SamplingParams
from fastdms.engine.sequence import Sequence
from fastdms.utils.context import reset_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare FP8 KV logits against dense BF16 KV.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--max-num-batched-tokens", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.08)
    parser.add_argument("--max-mean-kld", type=float, default=0.05)
    parser.add_argument("--max-greedy-mismatches", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def close_llm(llm: LLM) -> None:
    try:
        atexit.unregister(llm.exit)
    except Exception:
        pass
    llm.exit()
    torch.cuda.empty_cache()


def run_case(args: argparse.Namespace, tokenizer: AutoTokenizer, kv_cache_dtype: str) -> dict:
    llm = LLM(
        str(args.model),
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_dtype=kv_cache_dtype,
    )
    prompt_ids = tokenizer.encode(args.prompt)
    seq = Sequence(prompt_ids, SamplingParams(temperature=1.0, max_tokens=args.max_tokens, ignore_eos=True))
    llm.scheduler.add(seq)
    logits_by_step: list[torch.Tensor] = []
    token_ids: list[int] = []

    try:
        while not llm.scheduler.is_finished():
            seqs, is_prefill = llm.scheduler.schedule()
            input_ids, positions = llm.model_runner.prepare_prefill(seqs) if is_prefill else (
                llm.model_runner.prepare_decode(seqs)
            )
            try:
                logits = llm.model_runner.run_model(input_ids, positions, is_prefill).detach().float().cpu()
            finally:
                reset_context()
            next_token = int(logits.argmax(dim=-1)[0].item())
            logits_by_step.append(logits[0])
            token_ids.append(next_token)
            llm.scheduler.postprocess(seqs, [next_token], is_prefill)
    finally:
        metadata = {
            "kv_cache_dtype_config": llm.model_runner.config.kv_cache_dtype,
            "kv_cache_dtype": str(llm.model_runner.kv_cache.dtype).removeprefix("torch."),
            "num_dense_kv_blocks": llm.model_runner.config.num_kvcache_blocks,
            "torch_allocated_gib": torch.cuda.memory_allocated() / 2**30,
            "torch_reserved_gib": torch.cuda.memory_reserved() / 2**30,
        }
        close_llm(llm)

    return {
        "prompt_len": len(prompt_ids),
        "token_ids": token_ids,
        "token_text": tokenizer.decode(token_ids),
        "logits": torch.stack(logits_by_step),
        **metadata,
    }


def compare_logits(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    diff = (reference - candidate).abs()
    ref_log_probs = F.log_softmax(reference, dim=-1)
    cand_log_probs = F.log_softmax(candidate, dim=-1)
    ref_probs = ref_log_probs.exp()
    per_step_kld = (ref_probs * (ref_log_probs - cand_log_probs)).sum(dim=-1)
    return {
        "max_abs_logit_diff": float(diff.max().item()),
        "mean_abs_logit_diff": float(diff.mean().item()),
        "mean_kld": float(per_step_kld.mean().item()),
        "max_kld": float(per_step_kld.max().item()),
        "per_step_kld": [float(x) for x in per_step_kld.tolist()],
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    bf16 = run_case(args, tokenizer, "bfloat16")
    fp8 = run_case(args, tokenizer, "float8_e4m3fn")
    logit_stats = compare_logits(bf16["logits"], fp8["logits"])
    greedy_mismatches = sum(int(a != b) for a, b in zip(bf16["token_ids"], fp8["token_ids"]))
    guard_pass = greedy_mismatches <= args.max_greedy_mismatches and logit_stats["mean_kld"] <= args.max_mean_kld

    result = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "model": str(args.model),
        "prompt": args.prompt,
        "prompt_len": bf16["prompt_len"],
        "max_tokens": args.max_tokens,
        "reference": {
            "kv_cache_dtype_config": bf16["kv_cache_dtype_config"],
            "kv_cache_dtype": bf16["kv_cache_dtype"],
            "token_ids": bf16["token_ids"],
            "token_text": bf16["token_text"],
            "num_dense_kv_blocks": bf16["num_dense_kv_blocks"],
        },
        "candidate": {
            "kv_cache_dtype_config": fp8["kv_cache_dtype_config"],
            "kv_cache_dtype": fp8["kv_cache_dtype"],
            "token_ids": fp8["token_ids"],
            "token_text": fp8["token_text"],
            "num_dense_kv_blocks": fp8["num_dense_kv_blocks"],
        },
        "greedy_mismatches": greedy_mismatches,
        "max_greedy_mismatches": args.max_greedy_mismatches,
        "max_mean_kld": args.max_mean_kld,
        "guard_pass": guard_pass,
        "elapsed_s": time.perf_counter() - started,
        **logit_stats,
    }
    result.pop("logits", None)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not guard_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
