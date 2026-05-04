"""A1.2 long-context dense guard.

Teacher-forced 32-token decode at the retained 8192+128 surface. Compares
FastDMS dense logits to a Hugging Face reference at every decode step and
records max-KLD plus greedy mismatch. Pass thresholds default to KLD<=1e-2 and
greedy mismatch<=1 (a single tail mismatch is allowed because nondeterministic
ops can flip a tied logit; this still catches RoPE / mask regressions).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from fastdms import LLM, SamplingParams
from fastdms.engine.sequence import Sequence
from fastdms.utils.context import reset_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Long-context dense parity guard.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ctx-len", type=int, default=8192)
    parser.add_argument("--decode-window", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.16)
    parser.add_argument(
        "--kv-cache-dtype",
        choices=("auto", "float16", "bfloat16", "float32", "float8_e4m3fn", "float8_e5m2"),
        default="auto",
    )
    parser.add_argument("--kld-pass", type=float, default=1e-2)
    parser.add_argument("--greedy-pass-mismatches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _make_prompt_ids(model: Path, ctx_len: int, seed: int) -> list[int]:
    """Produce a deterministic ctx_len-token prompt by tiling wikitext-2 tokens."""
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        joined = "\n".join(ex for ex in ds["text"] if ex.strip())
        ids = tokenizer.encode(joined)
    except Exception:
        torch.manual_seed(seed)
        ids = torch.randint(low=10, high=tokenizer.vocab_size - 10, size=(ctx_len * 4,)).tolist()
    if len(ids) < ctx_len:
        ids = (ids * (ctx_len // max(1, len(ids)) + 1))[:ctx_len]
    return ids[:ctx_len]


def nano_logits(args, prompt_ids: list[int], decode_ids: list[int]) -> torch.Tensor:
    """Return (decode_window, vocab) tensor of logits for each appended token."""
    llm = LLM(
        str(args.model),
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=args.ctx_len + args.decode_window + 1,
        max_num_batched_tokens=args.ctx_len + args.decode_window + 1,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_dtype=args.kv_cache_dtype,
    )
    seq = Sequence(prompt_ids, SamplingParams(temperature=1.0, max_tokens=args.decode_window, ignore_eos=True))
    llm.scheduler.add(seq)
    seqs, is_prefill = llm.scheduler.schedule()
    assert is_prefill
    input_ids, positions = llm.model_runner.prepare_prefill(seqs)
    prefill_logits = llm.model_runner.run_model(input_ids, positions, True)
    reset_context()
    out_logits = [prefill_logits[-1].detach().float().cpu()]
    for forced in decode_ids[:-1]:
        seq.append_token(int(forced))
        seqs2, is_prefill2 = [seq], False
        input_ids2, positions2 = llm.model_runner.prepare_decode(seqs2)
        decode_logits = llm.model_runner.run_model(input_ids2, positions2, False)
        reset_context()
        out_logits.append(decode_logits[-1].detach().float().cpu())
    return torch.stack(out_logits, dim=0)


def hf_logits(args, prompt_ids: list[int], decode_ids: list[int]) -> torch.Tensor:
    config = AutoConfig.from_pretrained(args.model)
    hf_model = AutoModelForCausalLM.from_pretrained(args.model, dtype=config.dtype).to("cuda").eval()
    full_ids = prompt_ids + decode_ids[:-1]
    inputs = torch.tensor([full_ids], device="cuda")
    with torch.inference_mode():
        all_logits = hf_model(inputs).logits[0]  # (T, V)
    out = all_logits[len(prompt_ids) - 1 : len(prompt_ids) - 1 + len(decode_ids)]
    return out.detach().float().cpu()


def kl_per_step(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    p = F.log_softmax(p_logits, dim=-1)
    q = F.log_softmax(q_logits, dim=-1)
    return F.kl_div(q, p, reduction="none", log_target=True).sum(dim=-1)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    prompt_ids = _make_prompt_ids(args.model, args.ctx_len, args.seed)

    # Build a teacher-forced decode sequence using HF greedy decoding.
    config = AutoConfig.from_pretrained(args.model)
    hf_model = AutoModelForCausalLM.from_pretrained(args.model, dtype=config.dtype).to("cuda").eval()
    decode_ids: list[int] = []
    cur = torch.tensor([prompt_ids], device="cuda")
    with torch.inference_mode():
        for _ in range(args.decode_window):
            logits = hf_model(cur).logits[0, -1]
            nxt = int(logits.argmax().item())
            decode_ids.append(nxt)
            cur = torch.cat([cur, torch.tensor([[nxt]], device="cuda")], dim=1)
    del hf_model
    torch.cuda.empty_cache()

    nano = nano_logits(args, prompt_ids, decode_ids)
    hf = hf_logits(args, prompt_ids, decode_ids)

    klds = kl_per_step(hf, nano).tolist()
    nano_argmax = nano.argmax(dim=-1).tolist()
    hf_argmax = hf.argmax(dim=-1).tolist()
    mismatches = sum(1 for n, h in zip(nano_argmax, hf_argmax) if n != h)

    diff = (nano - hf).abs()
    result = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "model": str(args.model),
        "ctx_len": args.ctx_len,
        "decode_window": args.decode_window,
        "kv_cache_dtype": args.kv_cache_dtype,
        "max_kld": float(max(klds)),
        "mean_kld": float(sum(klds) / max(1, len(klds))),
        "klds_per_step": klds,
        "greedy_mismatches": mismatches,
        "nano_argmax": nano_argmax,
        "hf_argmax": hf_argmax,
        "max_abs_logit_diff": float(diff.max().item()),
        "mean_abs_logit_diff": float(diff.mean().item()),
        "kld_pass_threshold": args.kld_pass,
        "greedy_pass_threshold": args.greedy_pass_mismatches,
        "passed": (max(klds) <= args.kld_pass) and (mismatches <= args.greedy_pass_mismatches),
        "elapsed_s": time.perf_counter() - started,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
