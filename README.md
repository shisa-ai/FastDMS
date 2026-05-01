# FastDMS

A production-speed implementation of compact Dynamic Memory Sparsification (DMS) for KV cache compression.

FastDMS runs faster than vLLM with BF16, FP8, and TurboQuant KV cache settings while using dramatically less KV memory. On Llama-3.2-1B at `c=8`, the zero-BF16 default decodes `1.57x` faster than vLLM BF16 with `19.9x` less KV memory. On Qwen3-8B at `c=8`, it decodes `2.06x` faster with `6.35x` less KV memory.

## Install

```bash
pip install fastdms
```

Or from source:

```bash
git clone https://github.com/shisa-ai/FastDMS
cd FastDMS
pip install -e .
```

Requires CUDA, `torch>=2.4`, `triton>=3.0`, and `flash-attn`.

## Quick Start

```python
from fastdms import LLM, SamplingParams

llm = LLM("shisa-ai/Llama-3.2-1B-DMS-8x", enforce_eager=True)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
outputs = llm.generate(["Hello, FastDMS."], sampling_params)
print(outputs[0]["text"])
```

## Supported Models

DMS-trained checkpoints with per-head learned eviction heads. Validated:

- `shisa-ai/Llama-3.2-1B-DMS-8x` — promoted path, passes PLAN4 G1–G7 including a 60-minute c=8 soak
- `nvidia/Qwen3-8B-DMS-8x` — cross-model translation evidence

To train your own DMS checkpoints with the eviction-head retrofit recipe, see [github.com/shisa-ai/FastDMS](https://github.com/shisa-ai/FastDMS).

## How It Works

DMS trains a small per-head eviction head into an existing model via logit distillation. FastDMS implements a compact KV cache layout that actually reclaims the evicted slots — eviction signals turn into byte savings rather than just a quality-neutral mask. The result is FP8 compact KV with allocator-visible compression ratios up to `152x` smaller than vLLM BF16 KV at single-request c=1, scaling down to ~`20x` at c=8.

## Acknowledgements

FastDMS is built on top of [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) by Xingkai Yu. The clean, readable nano-vLLM codebase was used as the starting harness for the compact-DMS implementation. We're grateful for the upstream work that made rapid iteration on the KV-cache layout possible.
