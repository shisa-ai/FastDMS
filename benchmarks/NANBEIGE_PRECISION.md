# Nanbeige BF16 / FP8 speed and memory comparison

Measured 2026-10-26 on NVIDIA RTX PRO 6000 Blackwell 96 GB GPUs, using the
`fastdms` Mamba environment (PyTorch 2.11.0+cu130). All eight configurations
completed. The regression suite passed 45 tests.

## Results at 4,096 input tokens

Batch size 1, 64 output tokens, eager execution, prefix cache cleared before each
trial. Speeds are medians of three trials after one warmup. Prefill includes the
first generated token; decode measures the remaining 63. Memory is the maximum
PyTorch allocated peak across those trials, excluding model-load transients.

| Model weights | Retention | KV storage | Prefill tok/s | Decode tok/s | Peak allocated GiB |
| --- | --- | --- | ---: | ---: | ---: |
| BF16 | Dense | BF16 | 4,421 | 23.29 | 63.64 |
| BF16 | Dense | FP8 | 4,245 | 21.18 | 63.21 |
| BF16 | DMS | BF16 | 4,197 | 22.88 | 63.06 |
| BF16 | DMS | FP8 | 4,094 | 22.97 | 62.93 |
| primitive-ai FP8 | Dense | BF16 | 6,949 | 39.58 | 35.50 |
| primitive-ai FP8 | Dense | FP8 | 6,836 | 29.42 | 35.07 |
| primitive-ai FP8 | DMS | BF16 | 6,686 | 37.08 | 34.92 |
| primitive-ai FP8 | DMS | FP8 | 6,571 | 35.94 | 34.80 |

With dense BF16 KV, FP8 weights improve prefill by 1.57x and decode by 1.70x;
peak allocated memory falls by 44.2%. Resident model tensors, including model
buffers, decrease from 61.94 to 33.34 GiB. Embeddings and the output head remain
BF16 in the primitive-ai checkpoint.

DMS is not faster at this length: with FP8 weights/BF16 KV it reduces peak memory
by about 0.58 GiB but lowers decode speed by 6.3%. The dense FP8 KV path converts
cache blocks to BF16 for FlashAttention on every decode step, explaining why
storage savings do not imply a speedup. Compact DMS attention consumes FP8 KV
without that dense conversion path.

## Results at 512 input tokens

| Model weights | Retention | KV storage | Prefill tok/s | Decode tok/s | Peak allocated GiB |
| --- | --- | --- | ---: | ---: | ---: |
| BF16 | Dense | BF16 | 4,523 | 23.63 | 62.96 |
| BF16 | Dense | FP8 | 4,817 | 21.24 | 62.53 |
| BF16 | DMS | BF16 | 3,814 | 23.14 | 62.17 |
| BF16 | DMS | FP8 | 3,647 | 23.13 | 62.14 |
| primitive-ai FP8 | Dense | BF16 | 7,186 | 40.74 | 34.50 |
| primitive-ai FP8 | Dense | FP8 | 7,149 | 29.85 | 34.07 |
| primitive-ai FP8 | DMS | BF16 | 5,338 | 36.47 | 33.71 |
| primitive-ai FP8 | DMS | FP8 | 5,303 | 36.05 | 33.68 |

Dense cache capacity is fixed at the larger benchmark limit (4,096 + 64 tokens,
rounded to 256-token pages), including for the 512-token row. DMS uses lazy
per-layer allocation. These are allocated-memory comparisons, not comparisons
of live KV bytes alone. Raw results also include reserved and device-used bytes;
those are not interchangeable with allocated memory.

## Reproduction

The implementation is `benchmarks/bench_nanbeige_precision.py`. Run each row in
a fresh process. Use distinct `FASTDMS_DIST_PORT` values for simultaneous runs.
The benchmark disables compact CUDA graphs and the fused greedy generation loop
so prefill/decode timing is split consistently.

```bash
mamba activate fastdms
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4
CUDA_VISIBLE_DEVICES=0 FASTDMS_DIST_PORT=2340 \
python -m benchmarks.bench_nanbeige_precision \
    --model primitive-ai/Nanbeige4.3-33B-FP8 \
    --kv-dtype bfloat16 \
    --output artifacts/nanbeige-precision/fp8-dense-bfloat16.json
```

For BF16 weights use `--model Nanbeige/Nanbeige4.3-33B`. For FP8 KV use
`--kv-dtype float8_e4m3fn`. For DMS add
`--metadata artifacts/nanbeige-smoke/trained512/dms_metadata.json`.
Defaults are `--lengths 512 4096 --decode 64 --repeats 3`.

Exact local model revisions:

- BF16: `c1093c501cc6460427c0ac634db7223d0450c0ee`.
- primitive-ai FP8: `0140c157bc44507929a742c220865b0870052552`.

Raw results: `artifacts/nanbeige-precision/{bf16,fp8}-{dense,dms}-{bfloat16,float8_e4m3fn}.json`.
BF16 ran on GPU 1, FP8 on GPU 0, concurrently. Both are the same GPU model;
clocks/power were not locked. Other workloads occupied GPUs 2 and 3. Treat small
percentage differences as provisional, not hardware-isolated performance claims.

## Scope and limitations

The FP8 loader preserves the checkpoint's channel-wise E4M3 weights and scales
and uses dynamic per-token FP8 activations. It does not dequantize/requantize the
checkpoint. Tests cover packed QKV values, scales, linear outputs and avoiding
accidental re-quantization. A prefix-cache scheduling defect discovered during
benchmark preparation was fixed and regression-tested.

These runs measure inference, not FP8 training. The DMS sidecar received only two
smoke-training updates on repository text. The prompt is repeated explanatory
text, not a task-quality suite. No claim is made about retained answer quality,
long-context accuracy, quantized-model sidecar transfer quality, CUDA-graph
throughput, concurrent serving, or maximum supported context. The sidecar's
fingerprint binds the BF16 base, not the FP8 checkpoint. Do not promote this
sidecar on speed/memory evidence alone.

For the measured workloads, FP8 weights with dense BF16 KV provide the best
speed/memory tradeoff. DMS and FP8 KV should be reconsidered at longer contexts
only after the sidecar passes quality checks.
