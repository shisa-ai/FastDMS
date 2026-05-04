# DMS Training

Training scripts for [Dynamic Memory Sparsification (DMS)](https://openreview.net/forum?id=8ZiElzQxf1) — learned per-head KV cache token eviction via logit distillation.

This is an independent implementation based on the paper description and the inference code in [nvidia/Qwen3-8B-DMS-8x](https://huggingface.co/nvidia/Qwen3-8B-DMS-8x). No official training code was released by the authors.

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
