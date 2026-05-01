"""A1.7 no-evict regression.

Runs compact_kv_retention_mode=all at ctx_len=8192+128 and confirms parity with
dense BF16 to within KLD tolerance. Acts as a safety net for D1 packed-payload
work: any regression in compact_pack_prefill, compact_decode_attention, or
compact_append_store will surface here independently of DMS-specific logic.
"""
from __future__ import annotations

import argparse
import atexit
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from fastdms import LLM, SamplingParams
from fastdms.engine.sequence import Sequence
from fastdms.utils.context import reset_context

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.bench_vllm_turboquant import _make_prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compact-no-evict vs dense parity guard.")
    parser.add_argument("--model", type=Path, default=Path("results/dms/llama32-1b-cr8-v5-correctmask/final"))
    parser.add_argument("--ctx-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.16)
    parser.add_argument("--max-mean-kld", type=float, default=0.05)
    parser.add_argument("--max-greedy-mismatches", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _close_llm(llm: LLM) -> None:
    try:
        atexit.unregister(llm.exit)
    except Exception:
        pass
    llm.exit()
    torch.cuda.empty_cache()


def run_path(args, prompt_ids: list[int], *, compact_mode: str | None, forced_tokens: list[int] | None = None) -> dict:
    kwargs = {
        "enforce_eager": True,
        "tensor_parallel_size": 1,
        "max_model_len": args.ctx_len + args.max_tokens,
        "max_num_batched_tokens": args.ctx_len,
        "max_num_seqs": 1,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    if compact_mode is not None:
        kwargs.update({
            "compact_kv_enabled": True,
            "compact_kv_retention_mode": compact_mode,
        })
    llm = LLM(str(args.model), **kwargs)
    seq = Sequence(prompt_ids, SamplingParams(temperature=1.0, max_tokens=args.max_tokens, ignore_eos=True))
    llm.scheduler.add(seq)
    logits_by_step: list[torch.Tensor] = []
    used_token_ids: list[int] = []
    greedy_token_ids: list[int] = []
    try:
        while not llm.scheduler.is_finished():
            seqs, is_prefill = llm.scheduler.schedule()
            input_ids, positions = (
                llm.model_runner.prepare_prefill(seqs) if is_prefill else llm.model_runner.prepare_decode(seqs)
            )
            try:
                logits = llm.model_runner.run_model(input_ids, positions, is_prefill).detach().float().cpu()
                if is_prefill and llm.model_runner.compact_kv_manager is not None:
                    llm.model_runner.pack_compact_prefill(seqs)
                if not is_prefill and llm.model_runner.compact_kv_manager is not None:
                    llm.model_runner.record_dms_decode_decisions(seqs)
            finally:
                reset_context()
            step_logits = logits[0] if is_prefill else logits[0]
            greedy = int(step_logits.argmax(dim=-1).item())
            forced = forced_tokens[len(used_token_ids)] if forced_tokens is not None else greedy
            used_token_ids.append(forced)
            greedy_token_ids.append(greedy)
            logits_by_step.append(step_logits)
            llm.scheduler.postprocess(seqs, [forced], is_prefill)
            for sid in llm.scheduler.consume_freed_seq_ids():
                if llm.model_runner.compact_kv_manager is not None:
                    llm.model_runner.call("free_compact", [sid])
    finally:
        _close_llm(llm)
    return {
        "used_token_ids": used_token_ids,
        "greedy_token_ids": greedy_token_ids,
        "logits": torch.stack(logits_by_step),
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    prompt_ids = _make_prompts(str(args.model), 1, args.ctx_len, args.seed, "wikitext2")[0]["prompt_token_ids"]

    reference = run_path(args, prompt_ids, compact_mode=None)
    candidate = run_path(args, prompt_ids, compact_mode="all", forced_tokens=reference["used_token_ids"])

    diff = (reference["logits"] - candidate["logits"]).abs()
    rp = F.log_softmax(reference["logits"], dim=-1)
    cp = F.log_softmax(candidate["logits"], dim=-1)
    klds = (rp.exp() * (rp - cp)).sum(dim=-1)
    mismatches = sum(int(a != b) for a, b in zip(reference["used_token_ids"], candidate["greedy_token_ids"]))
    mean_kld = float(klds.mean().item())
    max_kld = float(klds.max().item())
    guard_pass = mismatches <= args.max_greedy_mismatches and mean_kld <= args.max_mean_kld
    result = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "model": str(args.model),
        "ctx_len": args.ctx_len,
        "max_tokens": args.max_tokens,
        "max_abs_logit_diff": float(diff.max().item()),
        "mean_kld": mean_kld,
        "max_kld": max_kld,
        "greedy_mismatches": mismatches,
        "guard_pass": guard_pass,
        "elapsed_s": time.perf_counter() - started,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
