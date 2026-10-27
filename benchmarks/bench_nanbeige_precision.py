"""Matched single-request Nanbeige weight/KV precision benchmark."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import time

import torch
from transformers import AutoTokenizer
from fastdms import LLM, SamplingParams


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--metadata')
    p.add_argument('--kv-dtype', default='bfloat16', choices=['bfloat16', 'float8_e4m3fn'])
    p.add_argument('--output', required=True)
    p.add_argument('--lengths', nargs='+', type=int, default=[512, 4096])
    p.add_argument('--decode', type=int, default=64)
    p.add_argument('--repeats', type=int, default=3)
    args = p.parse_args()
    # Split timing requires one prefill step, not the compact fused generation loop.
    os.environ["FASTDMS_COMPACT_GREEDY_FAST_LOOP"] = "0"
    os.environ["FASTDMS_COMPACT_DECODE_CUDAGRAPH"] = "0"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    text = ('The sky appears blue because molecules scatter shorter wavelengths of sunlight more strongly. '
            'Explain the physical mechanism and distinguish scattering from absorption. ')
    piece = tokenizer.encode(text, add_special_tokens=False)
    maximum = max(args.lengths) + args.decode
    kwargs = dict(enforce_eager=True, max_model_len=maximum, max_num_batched_tokens=max(args.lengths),
                  max_num_seqs=1, gpu_memory_utilization=0.85, kv_cache_dtype=args.kv_dtype,
                  num_page_override=math.ceil(maximum / 256))
    if args.metadata:
        kwargs.update(compact_kv_enabled=True, compact_kv_retention_mode='dms',
                      compact_kv_capacity_tokens=8 * maximum, dms_metadata_path=args.metadata)
    start = time.perf_counter()
    llm = LLM(args.model, **kwargs)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - start
    model = llm.model_runner.model
    model_bytes = sum(t.numel() * t.element_size() for t in list(model.parameters()) + list(model.buffers()))
    rows = []
    for length in args.lengths:
        prompt = (piece * math.ceil(length / len(piece)))[:length]
        trials = []
        for trial in range(args.repeats + 1):
            llm.scheduler.block_manager.clear_prefix_cache()
            llm.add_request(prompt, SamplingParams(temperature=0, max_tokens=args.decode, ignore_eos=True))
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            llm.step()
            torch.cuda.synchronize()
            prefill = time.perf_counter() - start
            manager = llm.model_runner.compact_kv_manager
            live_cells = (sum(int(s.live_counts.sum().item()) for s in manager.seq_states.values())
                          if manager is not None else 52 * 8 * length)
            kv_bytes = sum(t.numel() * t.element_size()
                           for layer in model.model.layers
                           for t in ((layer.self_attn.attn.compact_k_cache, layer.self_attn.attn.compact_v_cache)
                                     if manager is not None else
                                     (layer.self_attn.attn.k_cache, layer.self_attn.attn.v_cache)))
            start = time.perf_counter()
            outputs = []
            while not llm.is_finished():
                output, _ = llm.step()
                outputs.extend(output)
            torch.cuda.synchronize()
            decode = time.perf_counter() - start
            free, total = torch.cuda.mem_get_info()
            row = dict(prefill_seconds=prefill, prefill_tok_s=length / prefill,
                       decode_seconds=decode, decode_tok_s=(args.decode - 1) / decode,
                       allocated_bytes=torch.cuda.memory_allocated(),
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       reserved_bytes=torch.cuda.memory_reserved(), device_used_bytes=total-free,
                       kv_allocated_bytes=kv_bytes, prefill_live_head_tokens=live_cells,
                       prefill_retention_ratio=live_cells / (52 * 8 * length),
                       live_kv_payload_bytes=live_cells * 128 * 2 * (2 if args.kv_dtype == 'bfloat16' else 1),
                       output_ids=outputs[0][1])
            if trial:
                trials.append(row)
        rows.append(dict(prompt_tokens=length, trials=trials,
                         median_prefill_tok_s=statistics.median(r['prefill_tok_s'] for r in trials),
                         median_decode_tok_s=statistics.median(r['decode_tok_s'] for r in trials)))
        print(json.dumps(rows[-1]), flush=True)
    result = dict(model=args.model, metadata=args.metadata, kv_dtype=args.kv_dtype,
                  gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
                  load_seconds=load_seconds, model_tensor_bytes=model_bytes,
                  eager=True, batch=1, decode_tokens=args.decode, rows=rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
