from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config
from transformers import Qwen3ForCausalLM as HFQwen3ForCausalLM

from fastdms import LLM, SamplingParams


def ensure_tiny_qwen3(model_dir: Path, seed: int) -> bool:
    if (model_dir / "config.json").exists() and (model_dir / "model.safetensors").exists():
        return False

    model_dir.mkdir(parents=True, exist_ok=True)
    vocab = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<unk>": 3,
        **{f"tok_{idx}": idx for idx in range(4, 64)},
    }
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    fast_tokenizer.save_pretrained(model_dir)

    torch.manual_seed(seed)
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        attention_bias=False,
        tie_word_embeddings=False,
        dtype="float16",
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = HFQwen3ForCausalLM(config).to(dtype=torch.float16)
    model.save_pretrained(model_dir, safe_serialization=True)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny FastDMS compact-KV end-to-end smoke.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("results/fastdms/tiny-qwen3-random"),
        help="Ignored local directory for the generated tiny random Qwen3 model.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--prompt-len", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=3)
    parser.add_argument("--compact-capacity-tokens", type=int, default=512)
    parser.add_argument("--compact-retention-mode", choices=("all", "dms_stride"), default="all")
    parser.add_argument("--compact-retention-stride", type=int, default=1)
    parser.add_argument("--compact-retention-recent-tokens", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_created = ensure_tiny_qwen3(args.model_dir, args.seed)
    prompt = list(range(4, 4 + args.prompt_len))
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )

    started = time.perf_counter()
    llm = LLM(
        str(args.model_dir),
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=2,
        gpu_memory_utilization=args.gpu_memory_utilization,
        compact_kv_enabled=True,
        compact_kv_capacity_tokens=args.compact_capacity_tokens,
        compact_kv_retention_mode=args.compact_retention_mode,
        compact_kv_retention_stride=args.compact_retention_stride,
        compact_kv_retention_recent_tokens=args.compact_retention_recent_tokens,
    )
    outputs = llm.generate([prompt], [sampling_params], use_tqdm=False)
    elapsed_s = time.perf_counter() - started

    manager = llm.model_runner.compact_kv_manager
    assert manager is not None
    active_compact_sequences = len(manager.seq_states)
    compact_free_tokens = manager.free_tokens
    generated_tokens = outputs[0]["token_ids"]
    ok = (
        len(generated_tokens) == args.max_tokens
        and active_compact_sequences == 0
        and compact_free_tokens == manager.capacity_tokens
    )
    result = {
        "ok": ok,
        "model_dir": str(args.model_dir),
        "model_created": model_created,
        "prompt_len": args.prompt_len,
        "generated_tokens": len(generated_tokens),
        "elapsed_s": elapsed_s,
        "compact_capacity_tokens": manager.capacity_tokens,
        "compact_retention_mode": args.compact_retention_mode,
        "compact_retention_recent_tokens": args.compact_retention_recent_tokens,
        "compact_retention_stride": args.compact_retention_stride,
        "compact_free_tokens_after_generate": compact_free_tokens,
        "active_compact_sequences_after_generate": active_compact_sequences,
        "num_dense_kv_blocks": llm.model_runner.config.num_kvcache_blocks,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
