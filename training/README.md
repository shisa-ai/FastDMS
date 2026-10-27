# DMS Training

Training scripts for [Dynamic Memory Sparsification (DMS)](https://openreview.net/forum?id=8ZiElzQxf1) — learned per-head KV cache token eviction via logit distillation.

This is an independent implementation based on the paper description and the inference code in [nvidia/Qwen3-8B-DMS-8x](https://huggingface.co/nvidia/Qwen3-8B-DMS-8x). No official training code was released by the authors.

## Nanbeige 33B: two-GPU external sidecars

`training.train_sidecar` trains independent linear eviction heads without changing
base-model weights. It supports the Llama architecture, including
`Nanbeige/Nanbeige4.3-33B` (52 layers, 56 query heads, 8 KV heads, head dimension
128). This is a separate path from the borrowed-query-neuron trainer below.

Each GPU holds one BF16 model replica. Different examples run on each GPU; only
sidecar gradients are averaged. Two 96 GB GPUs are sufficient for the verified
short-context smoke configuration. Longer contexts require their own memory and
quality checks. The trainer computes future-attention labels from post-RoPE Q/K,
protects keys whose age is at most `window_size`, and trains linear heads on the
normalized input to each attention layer. The frozen model runs without an
autograd graph. Query chunks bound the temporary label-computation memory.

### Environment and launch

Run from the repository root:

```bash
mamba env create -f training/environment.yml
mamba activate fastdms
export PYTHONNOUSERSITE=1
python -m pip install -r training/requirements.lock
python -m pip check
python -m training.prepare_sidecar_data \
    --text /path/to/train.txt --output artifacts/nanbeige/train.pt \
    --context-len 512
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 \
python -m torch.distributed.run --standalone --nproc_per_node=2 \
    -m training.train_sidecar \
    --model Nanbeige/Nanbeige4.3-33B \
    --tokens artifacts/nanbeige/train.pt \
    --output-dir artifacts/nanbeige/sidecar \
    --context-len 512 --window-size 64 --target-cr 2 \
    --steps 100 --grad-accum 4
```

For an existing environment, use `mamba env update -f training/environment.yml`
and reinstall the lock file. The lock pins PyTorch 2.11.0/CUDA 13 and Transformers
5.12.1; it includes package hashes to respect hash-required pip installations.
Training does not require FlashAttention. FastDMS inference still requires a
FlashAttention build compatible with the installed PyTorch and GPU.
On this host, the Mamba environment uses the existing build at
`/home/lhl/flash-attention/build/lib.linux-x86_64-cpython-312` through
`fastdms-local-flash-attention.pth` in its site-packages directory. That local
build is not included in the portable training lock file.

`--steps` counts optimizer updates, each consuming `--grad-accum` examples per
GPU. Tokens must be an unpadded int64 tensor `[examples, context]`; keep validation
text separate. The preparation script drops the final incomplete chunk. The
small default run is a starting configuration, not a quality-qualified recipe.

### hipEngine compatibility

Exports contain `sidecar.safetensors`, `dms_metadata.json`, a model manifest, a
training-data manifest and `training_summary.json`. The sidecar has BF16 `weight`
`[layers, kv_heads, hidden_size]` and `bias` `[layers, kv_heads]` tensors. Metadata
uses hipEngine schema v2, `external_linear_sidecar_v1`, input stage
`post_attn_rmsnorm_pre_q_projection`, and training method
`future_attention_distillation_v1`. Ordinary query channels are preserved.

The exported policy uses threshold selection (`logit > 0`), supported by both
engines. The target compression ratio governs eligible-history labels; it does
not guarantee the same realized compression at inference. FastDMS also supports
hipEngine `exact_budget` prefill policies, ranking eligible history independently
per sequence and KV head while protecting the window. Decode remains
threshold-based. FastDMS accepts the export through `dms_metadata_path` with
`compact_kv_enabled=True` and `compact_kv_retention_mode="dms"`.

The artifact fingerprint hashes a manifest of the exact HF config and weight
shards. It is not a GGUF file hash. hipEngine may run a geometrically compatible
sidecar on another artifact, but its hash-mismatch warning means quality has not
been established for that artifact. A short smoke run and loader compatibility
do not qualify long-context quality, quantized-model transfer, or AMD serving.
Source hashes in the training summary identify uncommitted trainer changes in
addition to the repository commit recorded in metadata.

Run the consumer contract tests against a local hipEngine checkout:

```bash
HIPENGINE_ROOT=/path/to/hipEngine python -m pytest tests/test_sidecar.py -q
```

## Quick Start

Train a DMS checkpoint for Llama-3.2-1B with 8x target compression:

```bash
python -m training.dms_train \
    --model meta-llama/Llama-3.2-1B \
    --target-cr 8 \
    --context-len 4096 \
    --device cuda:0 \
    --output-dir results/dms/llama32-1b-cr8
```

This takes about **20 minutes on a single RTX PRO 6000**. The output directory will contain the trained checkpoint (full model weights with DMS eviction heads) plus `training_log.json` with DMS metadata needed for inference.

## How DMS Training Works

DMS retrofits a small eviction head into each attention layer by borrowing one dimension of the query projection. Training has two phases:

1. **Phase 1 — Neuron zeroing** (default: 2000 steps): Gradually zeros the borrowed query dimension so the model adapts to losing that dimension without quality collapse.

2. **Phase 2 — DMS retrofitting** (default: 100 steps per CR unit): Trains binary keep/evict decisions using Gumbel-sigmoid sampling with logit distillation against the original model. The compression ratio is annealed linearly from 1x to the target CR.

The training loss combines:
- **Distillation loss**: KL divergence between student and teacher logits
- **Compression loss**: penalty for retaining more tokens than the target CR allows

## Scripts

| Script | Purpose |
| --- | --- |
| `dms_train.py` | Main DMS training loop (two-phase: neuron zeroing → eviction retrofitting) |
| `dms_eval.py` | Evaluate a trained DMS checkpoint: PPL with hard eviction active |
| `train_predictors.py` | Train AQUA-KV cross-layer predictors (optional, not needed for FP8 DMS) |

### Supporting modules

| Module | Purpose |
| --- | --- |
| `predictors.py` | AQUA-KV predictor architecture |
| `quantizers.py` | HIGGS/EDEN quantizer implementations |
| `cache.py` | DMS-aware KV cache for HuggingFace models |
| `packed_cache.py` | Packed HIGGS KV cache layout |
| `data_utils.py` | WikiText-2 / calibration data loading |
| `model_utils.py` | Model loading and layer access utilities |
| `linear_utils.py` | Linear layer utilities for quantization |
| `edenn.py` | EDEN lattice codebook support |
| `dms_mask.py` | DMS attention mask construction |
| `grids/` | Pre-computed EDEN quantization grids |

## Evaluation

Evaluate a trained checkpoint with hard eviction:

```bash
python -m training.dms_eval \
    --model results/dms/llama32-1b-cr8/final \
    --n-chunks 4 \
    --seq-len 1024
```

This runs teacher-forced PPL evaluation on WikiText-2 with DMS eviction active and reports vanilla PPL, no-eviction PPL, and DMS-evicted PPL along with the effective compression ratio.

## Key Training Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--model` | (required) | HuggingFace model name or local path |
| `--target-cr` | `8` | Target compression ratio (e.g. 8 = keep 1/8 of tokens) |
| `--context-len` | `4096` | Training context length |
| `--window-size` | `256` | DMS sliding window — recent tokens protected from eviction |
| `--zeroing-steps` | `2000` | Phase 1 neuron zeroing steps |
| `--steps-per-cr` | `100` | Phase 2 steps per CR unit |
| `--lr` | `1e-5` | Learning rate |
| `--output-dir` | (required) | Where to save the trained checkpoint |
| `--device` | `cuda:1` | Training device |

## Validated Checkpoints

| Checkpoint | Base model | CR | Training time | GPU |
| --- | --- | ---: | --- | --- |
| [shisa-ai/Llama-3.2-1B-DMS-8x](https://huggingface.co/shisa-ai/Llama-3.2-1B-DMS-8x) | Llama-3.2-1B | 8x | ~20 min | RTX PRO 6000 |
| [nvidia/Qwen3-8B-DMS-8x](https://huggingface.co/nvidia/Qwen3-8B-DMS-8x) | Qwen3-8B | 8x | — | NVIDIA (official) |
