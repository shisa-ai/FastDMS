"""Bounded Nanbeige sidecar serving smoke; not a quality benchmark."""
import argparse
import json
from pathlib import Path
from fastdms import LLM, SamplingParams


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    llm = LLM(args.model, enforce_eager=True, max_model_len=256,
              max_num_batched_tokens=256, max_num_seqs=1,
              gpu_memory_utilization=0.85, compact_kv_enabled=True,
              compact_kv_capacity_tokens=4096, compact_kv_retention_mode="dms",
              dms_metadata_path=args.metadata)
    result = llm.generate(["Explain why the sky is blue in one sentence. " * 6],
                          SamplingParams(temperature=0, max_tokens=8), use_tqdm=False)
    assert len(result[0]["token_ids"]) > 0
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
