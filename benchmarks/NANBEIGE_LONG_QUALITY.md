# Nanbeige long-context DMS memory and KV quality

Measured 2026-10-26 on RTX PRO 6000 Blackwell 96 GB GPUs in Mamba `fastdms`.
All runs use primitive-ai/Nanbeige4.3-33B-FP8 revision
`0140c157bc44507929a742c220865b0870052552`. DMS uses the two-update smoke sidecar
at `artifacts/nanbeige-smoke/trained512/dms_metadata.json` (window 64).

**Result: memory savings are real, but this DMS sidecar fails long-context
retrieval with either BF16 or FP8 KV. Do not use it as a quality-qualified policy.**

## Performance and memory

Batch 1, eager execution, 32 output tokens; one warmup and two measured trials.
Median speeds below use a repeated explanatory-text prompt. Live KV payload is
measured immediately after prefill, before freeing the request. Total peak is
PyTorch allocated memory during generation; it includes temporary activations
but excludes model-loading transients and CUDA allocator reserves.

| Input tokens | DMS KV dtype | Live KV GiB | Dense same-dtype KV GiB (calculated) | KV saved GiB | Total peak GiB | Prefill tok/s | Decode tok/s |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32,768 | BF16 | 2.031 | 6.500 | 4.469 | 42.27 | 4,140 | 31.67 |
| 32,768 | FP8 | 1.015 | 3.250 | 2.235 | 41.25 | 3,909 | 32.37 |
| 131,072 | BF16 | 8.686 | 26.000 | 17.314 | 67.81 | 1,990 | 19.24 |
| 131,072 | FP8 | 4.343 | 13.000 | 8.657 | 63.46 | 1,993 | 21.09 |

Retention is 31.24% at 32K and 33.41% at 128K, identical for the two KV dtypes
on this prompt. Actual compact arena allocations are slightly larger than live
payload: 2.040/1.020 GiB at 32K and 8.695/4.348 GiB at 128K (BF16/FP8).
DMS FP8 KV saves 21.657 GiB at 128K relative to dense BF16 KV payload.
These are prompt-dependent retention measurements, not a guaranteed compression
ratio. The configured CR2 training label target does not constrain inference to
exactly 2x retention compression.

The first dense BF16 quality reference exhausted memory at 128K while converting
FP8 linear inputs, despite its model and cache fitting. `_fp8_linear` now bounds
activation conversion to 4,096-row chunks. Per-token scaling is unchanged. The
updated path passes a long-row regression and the full 46-test suite. Performance
was rerun after this fix: use `chunked-dms-*.json`, not the earlier unprefixed
artifacts. The earlier DMS 128K peaks were 79.08/74.73 GiB; the final measurements
above supersede them.

## Quality comparison

All four variants use the same FP8 weights and exact same token inputs. Dense
BF16 KV is the reference. Tests consist of three synthetic, training-disjoint
512-token contexts (science, code, Japanese) with fixed continuations and four
retrieval contexts (32K/128K, answer at 10%/80% of the context). The retrieval
answer is the six-digit code `739281`.

The harness feeds the same target token after each prediction: these are
**teacher-forced** metrics, not free-generation task scores. Prefill is dense
in all variants, so the first prediction is excluded from aggregate KL and
agreement below. Only subsequent predictions exercise retained/quantized KV.
The short-context aggregate has 79 decode positions; retrieval has 24.

KL divergence measures changes in the whole next-token distribution (lower is
closer to reference); top-1 agreement measures identical most-likely tokens.
Neither alone proves general answer quality. Aggregates are weighted by scored
decode positions, not by case.

| KV variation | Short-context mean KL | Short-context top-1 agreement | Retrieval mean KL | Retrieval top-1 agreement | Retrieval cases with every target token correct |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense BF16 | 0 | 100% | 0 | 100% | 4/4 |
| Dense FP8 | 0.00827 | 92.4% | 0.000192 | 100% | 4/4 |
| DMS BF16 | 0.09164 | 82.3% | 1.4481 | 62.5% | 0/4 |
| DMS FP8 | 0.08718 | 87.3% | 1.5092 | 62.5% | 0/4 |

DMS retrieval target-token accuracy, including the initial dense prediction,
was 57.1% for 32K/10% and 71.4% for the other three cases, with either KV dtype.
Dense variants achieved 100% on all four. DMS retrieval target negative
log-likelihood rose to 1.14–1.40 nats/token versus 0.008–0.028 for the reference.
No predetermined statistical quality threshold was used; failing exact recovery
of the code in all four cases is sufficient to reject this smoke sidecar for
this retrieval use. Small differences between DMS dtypes on the short suite do
not establish that FP8 improves quality.

The near-identical retrieval failures in DMS BF16 and DMS FP8 implicate eviction,
not FP8 cache storage alone. This does not isolate poor training from transfer
of a BF16-trained sidecar onto FP8 weights. Dense FP8 looks promising on this
bounded suite but is not certified across tasks or longer generations.

## Reproduction and artifacts

Performance command (repeat with both KV dtypes and both lengths):

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 FASTDMS_DIST_PORT=2360 \
mamba run -n fastdms python -m benchmarks.bench_nanbeige_precision \
  --model primitive-ai/Nanbeige4.3-33B-FP8 \
  --metadata artifacts/nanbeige-smoke/trained512/dms_metadata.json \
  --kv-dtype bfloat16 --lengths 131072 --decode 32 --repeats 2 \
  --output artifacts/nanbeige-long/chunked-dms-bfloat16-131072.json
```

Quality reference:

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 FASTDMS_DIST_PORT=2360 \
mamba run -n fastdms python -m benchmarks.check_nanbeige_kv_quality \
  --model primitive-ai/Nanbeige4.3-33B-FP8 \
  --reference artifacts/nanbeige-quality/reference --write-reference \
  --output artifacts/nanbeige-quality/dense-bfloat16.json
```

For candidates omit `--write-reference`, select `--kv-dtype float8_e4m3fn` for
FP8 KV, and add the sidecar `--metadata` argument for DMS. Use a different output
file for each configuration. References store float32 logits and a SHA-256 of
the prompt/target token IDs; candidates reject input identity mismatches.

Raw performance: `artifacts/nanbeige-long/chunked-dms-*.json`.
Raw quality: `artifacts/nanbeige-quality/*.json`, seven cases per variant.
The quality suite completed 28 case/configuration combinations. GPUs 2/3 ran
unrelated workloads; clocks were not locked. Small speed differences should not
be treated as hardware-isolated measurements.
