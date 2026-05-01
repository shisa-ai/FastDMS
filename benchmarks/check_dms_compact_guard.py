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
    parser = argparse.ArgumentParser(description="Compare compact DMS against a no-evict DMS trace.")
    parser.add_argument("--model", type=Path, default=Path("results/dms/llama32-1b-cr8-v5-correctmask/final"))
    parser.add_argument("--ctx-len", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--prompt-source", choices=("wikitext2", "wikitext2_wrap", "synthetic"), default="wikitext2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.16)
    parser.add_argument(
        "--kv-cache-dtype",
        choices=("auto", "float16", "bfloat16", "float32", "float8_e4m3fn", "float8_e5m2"),
        default="auto",
    )
    parser.add_argument("--compact-kv-capacity-tokens", type=int, default=0)
    parser.add_argument("--compact-kv-budget-bytes", type=int, default=None)
    parser.add_argument("--compact-kv-layer-major-metadata", action="store_true")
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help=(
            "Override ctx_len * concurrency so long-context guards can exercise "
            "scheduler chunked prefill."
        ),
    )
    parser.add_argument("--candidate-window-size", type=int)
    parser.add_argument("--reference-window-size", type=int, default=1_000_000)
    parser.add_argument("--max-mean-kld", type=float, default=0.10)
    parser.add_argument("--max-greedy-mismatches", type=int, default=1)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def close_llm(llm: LLM) -> None:
    try:
        atexit.unregister(llm.exit)
    except Exception:
        pass
    llm.exit()
    torch.cuda.empty_cache()


def json_safe_stats(stats: list[dict]) -> list[dict]:
    result = []
    for item in stats:
        converted = {}
        for key, value in item.items():
            converted[key] = int(value.item()) if torch.is_tensor(value) else value
        result.append(converted)
    return result


def run_case(
    args: argparse.Namespace,
    prompts: list[list[int]],
    *,
    window_size: int | None,
    forced_tokens: list[list[int]] | None = None,
) -> dict:
    concurrency = len(prompts)
    max_num_batched_tokens = args.max_num_batched_tokens or (args.ctx_len * concurrency)
    llm = LLM(
        str(args.model),
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=args.ctx_len + args.max_tokens,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=concurrency,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_dtype=args.kv_cache_dtype,
        compact_kv_enabled=True,
        compact_kv_capacity_tokens=args.compact_kv_capacity_tokens,
        compact_kv_budget_bytes=args.compact_kv_budget_bytes,
        compact_kv_layer_major_metadata=args.compact_kv_layer_major_metadata,
        compact_kv_retention_mode="dms",
        dms_window_size=window_size,
    )
    seqs_by_id: dict[int, Sequence] = {}
    for prompt_ids in prompts:
        seq = Sequence(prompt_ids, SamplingParams(temperature=1.0, max_tokens=args.max_tokens, ignore_eos=True))
        llm.scheduler.add(seq)
        seqs_by_id[seq.seq_id] = seq
    seq_order = list(seqs_by_id)
    seq_index = {seq_id: idx for idx, seq_id in enumerate(seq_order)}

    decode_step_idx: dict[int, int] = {sid: 0 for sid in seqs_by_id}
    used_token_ids: dict[int, list[int]] = {sid: [] for sid in seqs_by_id}
    greedy_token_ids: dict[int, list[int]] = {sid: [] for sid in seqs_by_id}
    logits_by_seq: dict[int, list[torch.Tensor]] = {sid: [] for sid in seqs_by_id}
    vocab_size = int(llm.model_runner.config.hf_config.vocab_size)

    try:
        while not llm.scheduler.is_finished():
            seqs, is_prefill = llm.scheduler.schedule()
            input_ids, positions = llm.model_runner.prepare_prefill(seqs) if is_prefill else (
                llm.model_runner.prepare_decode(seqs)
            )
            try:
                logits = llm.model_runner.run_model(input_ids, positions, is_prefill).detach().float().cpu()
                if is_prefill:
                    llm.model_runner.pack_compact_prefill(seqs)
                else:
                    llm.model_runner.record_dms_decode_decisions(seqs)
            finally:
                reset_context()
            tokens_to_postprocess: list[int] = []
            if is_prefill:
                # ParallelLMHead.forward already extracts last-token logits per
                # cu_seqlens_q in prefill, so logits has shape (num_seqs, vocab).
                for seq_idx, seq in enumerate(seqs):
                    will_finish_prefill = (
                        seq.num_cached_tokens + seq.num_scheduled_tokens >= seq.num_tokens
                    )
                    step_logits = logits[seq_idx][:vocab_size].contiguous()
                    greedy = int(step_logits.argmax(dim=-1).item())
                    if will_finish_prefill:
                        step_i = decode_step_idx[seq.seq_id]
                        forced = (
                            forced_tokens[seq_index[seq.seq_id]][step_i]
                            if forced_tokens is not None else greedy
                        )
                        used_token_ids[seq.seq_id].append(forced)
                        greedy_token_ids[seq.seq_id].append(greedy)
                        logits_by_seq[seq.seq_id].append(step_logits)
                        decode_step_idx[seq.seq_id] = step_i + 1
                    else:
                        forced = greedy
                    tokens_to_postprocess.append(forced)
            else:
                for seq_idx, seq in enumerate(seqs):
                    step_logits = logits[seq_idx][:vocab_size].contiguous()
                    greedy = int(step_logits.argmax(dim=-1).item())
                    step_i = decode_step_idx[seq.seq_id]
                    if forced_tokens is not None:
                        forced_list = forced_tokens[seq_index[seq.seq_id]]
                        forced = forced_list[step_i]
                    else:
                        forced = greedy
                    used_token_ids[seq.seq_id].append(forced)
                    greedy_token_ids[seq.seq_id].append(greedy)
                    logits_by_seq[seq.seq_id].append(step_logits)
                    decode_step_idx[seq.seq_id] = step_i + 1
                    tokens_to_postprocess.append(forced)
            llm.scheduler.postprocess(seqs, tokens_to_postprocess, is_prefill)
            for seq_id in llm.scheduler.consume_freed_seq_ids():
                llm.model_runner.call("free_compact", [seq_id])
    finally:
        metadata = {
            "window_size": llm.model_runner.dms_metadata.window_size if llm.model_runner.dms_metadata is not None else None,
            "compact_prefill_stats": llm.model_runner.last_compact_prefill_stats,
            "dms_decode_stats": json_safe_stats(llm.model_runner.last_dms_decode_stats),
            "num_dense_kv_blocks": llm.model_runner.config.num_kvcache_blocks,
            "compact_capacity_tokens": (
                llm.model_runner.compact_kv_manager.capacity_tokens
                if llm.model_runner.compact_kv_manager is not None else None
            ),
        }
        close_llm(llm)

    used_lists = [used_token_ids[sid] for sid in seq_order]
    greedy_lists = [greedy_token_ids[sid] for sid in seq_order]
    logits_stacks = [torch.stack(logits_by_seq[sid]) for sid in seq_order]
    return {
        "used_token_ids": used_lists,
        "greedy_token_ids": greedy_lists,
        "logits": torch.stack(logits_stacks, dim=0),  # (concurrency, steps, vocab)
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
        "per_step_kld": per_step_kld.flatten().tolist(),
    }


def decode_step_ppl(logits: torch.Tensor, target_token_ids: list[list[int]]) -> dict:
    log_probs = F.log_softmax(logits, dim=-1)
    nlls: list[float] = []
    for seq_idx, tokens in enumerate(target_token_ids):
        for step_idx, token_id in enumerate(tokens):
            nlls.append(float(-log_probs[seq_idx, step_idx, int(token_id)].item()))
    mean_nll = sum(nlls) / len(nlls)
    return {
        "mean_nll": mean_nll,
        "ppl": float(torch.exp(torch.tensor(mean_nll)).item()),
        "scored_steps": len(nlls),
        "note": "decode-step PPL over no-evict greedy tokens, not teacher-forced WikiText next-token PPL",
    }


def main() -> None:
    args = parse_args()
    max_num_batched_tokens = args.max_num_batched_tokens or (args.ctx_len * args.concurrency)
    if max_num_batched_tokens <= 0:
        raise ValueError("--max-num-batched-tokens must be positive")
    started = time.perf_counter()
    prompt_pack = _make_prompts(str(args.model), args.concurrency, args.ctx_len, args.seed, args.prompt_source)
    prompts = [p["prompt_token_ids"] for p in prompt_pack]

    reference = run_case(args, prompts, window_size=args.reference_window_size)
    candidate = run_case(
        args,
        prompts,
        window_size=args.candidate_window_size,
        forced_tokens=reference["used_token_ids"],
    )
    logit_stats = compare_logits(reference["logits"], candidate["logits"])
    reference_ppl = decode_step_ppl(reference["logits"], reference["used_token_ids"])
    candidate_ppl = decode_step_ppl(candidate["logits"], reference["used_token_ids"])
    greedy_mismatches = sum(
        int(a != b)
        for ref_seq, cand_seq in zip(reference["used_token_ids"], candidate["greedy_token_ids"])
        for a, b in zip(ref_seq, cand_seq)
    )
    guard_pass = greedy_mismatches <= args.max_greedy_mismatches and logit_stats["mean_kld"] <= args.max_mean_kld
    result = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "model": str(args.model),
        "ctx_len": args.ctx_len,
        "max_tokens": args.max_tokens,
        "concurrency": args.concurrency,
        "prompt_source": args.prompt_source,
        "prompt_len": len(prompts[0]),
        "kv_cache_dtype": args.kv_cache_dtype,
        "compact_kv_capacity_tokens": args.compact_kv_capacity_tokens,
        "compact_kv_budget_bytes": args.compact_kv_budget_bytes,
        "compact_kv_layer_major_metadata": args.compact_kv_layer_major_metadata,
        "max_num_batched_tokens": max_num_batched_tokens,
        "reference": {
            "mode": "compact_dms_noevict",
            "window_size": reference["window_size"],
            "token_ids": reference["used_token_ids"],
            "greedy_token_ids": reference["greedy_token_ids"],
            "compact_prefill_stats": reference["compact_prefill_stats"],
            "dms_decode_stats": reference["dms_decode_stats"],
        },
        "candidate": {
            "mode": "compact_dms",
            "window_size": candidate["window_size"],
            "used_token_ids": candidate["used_token_ids"],
            "greedy_token_ids": candidate["greedy_token_ids"],
            "compact_prefill_stats": candidate["compact_prefill_stats"],
            "dms_decode_stats": candidate["dms_decode_stats"],
        },
        "greedy_mismatches": greedy_mismatches,
        "max_greedy_mismatches": args.max_greedy_mismatches,
        "max_mean_kld": args.max_mean_kld,
        "guard_pass": guard_pass,
        "elapsed_s": time.perf_counter() - started,
        "reference_decode_step": reference_ppl,
        "candidate_decode_step": candidate_ppl,
        **logit_stats,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not guard_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
