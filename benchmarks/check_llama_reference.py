from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from fastdms import LLM, SamplingParams
from fastdms.engine.sequence import Sequence
from fastdms.utils.context import reset_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare FastDMS Llama prefill logits to HF reference.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--max-num-batched-tokens", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.08)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def nano_prefill_logits(args: argparse.Namespace, tokenizer: AutoTokenizer) -> tuple[torch.Tensor, list[int], dict]:
    llm = LLM(
        str(args.model),
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    prompt_ids = tokenizer.encode(args.prompt)
    seq = Sequence(prompt_ids, SamplingParams(temperature=1.0, max_tokens=1, ignore_eos=True))
    llm.scheduler.add(seq)
    seqs, is_prefill = llm.scheduler.schedule()
    if not is_prefill:
        raise RuntimeError("expected first scheduled step to be prefill")
    input_ids, positions = llm.model_runner.prepare_prefill(seqs)
    try:
        logits = llm.model_runner.run_model(input_ids, positions, True).detach().float().cpu()
    finally:
        reset_context()
    metadata = {
        "model_class": type(llm.model_runner.model).__name__,
        "num_dense_kv_blocks": llm.model_runner.config.num_kvcache_blocks,
    }
    return logits, prompt_ids, metadata


def hf_prefill_logits(model: Path, prompt: str, tokenizer: AutoTokenizer) -> torch.Tensor:
    config = AutoConfig.from_pretrained(model)
    hf_model = AutoModelForCausalLM.from_pretrained(model, dtype=config.dtype).to("cuda").eval()
    hf_inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        return hf_model(**hf_inputs).logits[:, -1, :].detach().float().cpu()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    nano_logits, prompt_ids, nano_metadata = nano_prefill_logits(args, tokenizer)
    hf_logits = hf_prefill_logits(args.model, args.prompt, tokenizer)

    nano_next = int(nano_logits.argmax(dim=-1)[0].item())
    hf_next = int(hf_logits.argmax(dim=-1)[0].item())
    diff = (nano_logits - hf_logits).abs()
    result = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "model": str(args.model),
        "prompt": args.prompt,
        "prompt_len": len(prompt_ids),
        "nano_next_token": nano_next,
        "hf_next_token": hf_next,
        "nano_next_text": tokenizer.decode([nano_next]),
        "hf_next_text": tokenizer.decode([hf_next]),
        "greedy_token_match": nano_next == hf_next,
        "max_abs_logit_diff": float(diff.max().item()),
        "mean_abs_logit_diff": float(diff.mean().item()),
        "elapsed_s": time.perf_counter() - started,
        **nano_metadata,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not result["greedy_token_match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
