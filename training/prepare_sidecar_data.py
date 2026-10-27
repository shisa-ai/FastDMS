"""Tokenize local UTF-8 training text into unpadded fixed-length examples."""
import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Nanbeige/Nanbeige4.3-33B")
    p.add_argument("--text", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--context-len", type=int, default=512)
    args = p.parse_args()
    if args.context_len < 2:
        p.error("context-len must be at least 2")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ids = tokenizer(Path(args.text).read_text(), add_special_tokens=False)["input_ids"]
    usable = len(ids) // args.context_len * args.context_len
    if usable < 2 * args.context_len:
        p.error("provide enough text for at least two complete examples")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tokens = torch.tensor(ids[:usable], dtype=torch.long).reshape(-1, args.context_len)
    torch.save(tokens, output)
    print(f"Saved {len(tokens)} examples to {output}; dropped {len(ids) - usable} trailing tokens")


if __name__ == "__main__":
    main()
