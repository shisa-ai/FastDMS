# Nanbeige sidecar smoke verification

Tested on 2026-10-26 with `Nanbeige/Nanbeige4.3-33B`, cached revision
`c1093c501cc6460427c0ac634db7223d0450c0ee`, RTX PRO 6000 Blackwell 96 GB GPUs,
and Mamba environment `fastdms` (PyTorch 2.11.0+cu130, Transformers 5.12.1).

- Two-GPU training at 128 tokens, window 32, CR2, two updates and accumulation 2:
  loss 0.693147 → 0.672862; rank-0 peak allocated memory 61.85 GiB.
- Two-GPU training at 512 tokens, window 64, CR2, two updates:
  loss 0.693147 → 0.669429; rank-0 peak allocated memory 61.92 GiB.
- Export: 2,982,304 parameters; BF16 weights `[52, 8, 7168]` and biases `[52, 8]`.
- hipEngine's actual schema-v2 loader accepted the trained export. Tests compare
  its CPU projection and future-attention oracle against the training code.
- FastDMS completed prefill and eight decode tokens with the 128-token-run
  sidecar, a prompt exceeding its window, and a 4096-slot compact arena.
- Repository suite: 43 tests passed, including CUDA compact-cache tests,
  non-layer-major int64-position regression, sidecar gradients/accumulation,
  frozen base weights, GQA group size 7, unchanged query channels, head sharding,
  and sidecar hash rejection. `pip check` and `git diff --check` passed.

Local outputs are under `artifacts/nanbeige-smoke/` (`trained`, `trained512`,
`inference.json`). Inputs were tokenized repository documentation, deliberately
for execution checks only. These sidecars are **not quality-qualified**. This
run does not establish long-context retention quality, AMD execution, GGUF
transfer quality, tensor-parallel end-to-end serving, or throughput benchmarks.

The inference smoke found and fixed an int64-to-int32 indexed assignment in
non-layer-major decode metadata. A separate arena-exhaustion attempt was a smoke
configuration error: capacity is shared across KV heads per layer, not a
per-head token limit. The successful run used 4096 slots.
