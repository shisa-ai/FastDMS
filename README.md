# FastDMS

A production-speed implementation of compact Dynamic Memory Sparsification (DMS) for KV cache compression ([arXiv:2506.05345](https://arxiv.org/abs/2506.05345), [NeurIPS 2025 poster](https://openreview.net/forum?id=8ZiElzQxf1)).

FastDMS runs faster than vLLM (`v0.19.2rc1`, nightly) with BF16, FP8, and TurboQuant KV cache settings while using significantly less KV memory. On Llama-3.2-1B at `c=8`, the zero-BF16 default decodes `1.53x` faster than vLLM BF16 with `4.8x` less KV memory. On Qwen3-8B at `c=8`, it decodes `2.06x` faster with `6.35x` less KV memory. All vLLM baselines use exact-sized token pools (KV cache sized to the workload, not over-provisioned). All tests were run on NVIDIA RTX PRO 6000 (Blackwell, sm120).

Note, while the *speed* is faster than vLLM, this should be considered a **fast reference implementation** as only two DMS-trained checkpoints have been tested/validated:

- [shisa-ai/Llama-3.2-1B-DMS-8x](https://huggingface.co/shisa-ai/Llama-3.2-1B-DMS-8x) - primary testing model, including a 60-minute c=8 soak
- [nvidia/Qwen3-8B-DMS-8x](https://huggingface.co/nvidia/Qwen3-8B-DMS-8x) - cross-model translation evidence

To train your own DMS checkpoints with the eviction-head retrofit recipe, see the [`training/`](training/) folder. The in-repo trainer is a fast for single-GPU training. For larger models or multi-GPU training, start from NVIDIA's [Model-Optimizer DMS trainer](https://github.com/NVIDIA/Model-Optimizer/tree/main/experimental/dms).

## About DMS

DMS trains learned per-head token eviction via logit distillation. This takes only a very short time to train, with our Llama 3.2 1B DMS model taking only about 20 minutes on a single RTX PRO 6000.

FastDMS implements a compact KV cache layout that reclaims evicted slots. The result is FP8 compact KV with allocator-visible compression ratios of `5-8x` smaller than vLLM BF16 KV at 8K context, while also decoding `1.5-2x` faster.

## Initial Work

This project started as a research pack that reviewed a broad range of academic and "folk" KV-cache compression techniques. Among all the combinations tested, the strongest near-lossless research-stack result was **DMS + AQUA-KV + HIGGS 4-bit**: **25.6x theoretical KV compression at +0.09% PPL** on Llama-3.2-1B, trained in ~20 minutes on a single GPU:

| Configuration | PPL | Delta | KLD (nats/tok) | Compression |
|---|---:|---:|---:|---:|
| Vanilla Llama-3.2-1B | 9.226 | - | - | 1x |
| DMS (trained, eviction active) | 9.200 | -0.28% | 0.026 | 6.4x |
| DMS + AQUA | 9.205 | -0.23% | 0.026 | 6.4x |
| DMS + HIGGS 4-bit | 9.621 | +4.28% | 0.058 | 25.6x |
| **DMS + AQUA + HIGGS 4-bit** | **9.234** | **+0.09%** | **0.032** | **25.6x** |

A fair amount of effort was spent optimizing performance for this stack, but ultimately [HIGGS](https://arxiv.org/abs/2411.17525) was tabled due to our best efforts only hitting about 50% of BF16/FP8 prefill/decode speed. [AQUA-KV](https://arxiv.org/abs/2501.19392) was not required for best FP8+DMS quality. HIGGS+AQUA composes naturally with DMS, and is an obvious target for future kernel work.

Initial bring-up of DMS was done on a HuggingFace-based correctness/PPL harness (`training/dms_eval.py`) that ran full-forward evaluation at ~`18 tok/s`. That harness validated quality but didn't reclaim any memory (evicted tokens were masked in attention but their KV slots stayed allocated).

## Optimized Performance

FastDMS is able to bring DMS up to production speeds, beating vLLM's BF16 and FP8 KV cache speeds at both prefill and decode, and being nearly 40x faster than our initial HF implementation.

FastDMS is benchmarked on WikiText-2 with `ctx_len=8192`, `gen_len=128`, and post-warmup timing. vLLM baselines use exact-sized token pools — the KV cache is sized to fit exactly the sequences being tested (e.g. `10240` tokens for `c=1`, `67584` tokens for `c=8`), so reported KV memory reflects what the workload actually needs, not pre-allocated headroom. Ratio columns are against vLLM BF16 KV because many external KV-cache papers report against FP16/BF16 cache baselines; FP8 rows, however, are probably best for direct serving-engine comparison.

### shisa-ai/Llama-3.2-1B-DMS-8x

The zero-BF16 FastDMS default is `1.52x` / `1.53x` faster than vLLM BF16 decode at `c=1` / `c=8`, while using `5.6x` / `4.8x` less KV memory. Against vLLM FP8, it is `1.43x` / `1.25x` faster decode and `2.8x` / `2.4x` smaller in KV. vLLM's TurboQuant 4-bit is actually *slower* than BF16 (`0.73x` / `0.72x` decode) for only `2.2x` KV savings — and with [worse output quality](#shisa-aillama-32-1b-dms-8x-2). The default-off B46 c=1 speed profile reaches `2.30x` BF16 decode at the same compact-KV footprint, with `0.719 GiB` of int4 shadow storage.

| Path | c | Prefill tok/s | Prefill vs BF16 | Decode tok/s | Decode vs BF16 | KV / stage memory | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| vLLM BF16 | 1 | `123098.0` | `1.00x` | `459.4` | `1.00x` | `0.312 GiB` BF16 KV | dense BF16-KV baseline |
| vLLM FP8 | 1 | `119991.3` | `0.97x` | `489.4` | `1.07x` | `0.156 GiB` FP8 KV | dense FP8-KV baseline |
| vLLM TurboQuant `4bit_nc` | 1 | `126429.0` | `1.03x` | `333.4` | `0.73x` | `0.142 GiB` TQ4 KV | 4-bit KV baseline |
| FastDMS FP8 compact-DMS default | 1 | **`123194.6`** | **`1.00x`** | **`698.9`** | **`1.52x`** | **`0.056 GiB`** | promoted zero-BF16 row |
| FastDMS B46 int4 speed profile | 1 | `121489.9` | `0.99x` | **`1060.0`** | **`2.31x`** | `0.056 GiB` + `0.719 GiB` int4 shadow | default-off storage-for-speed |
| vLLM BF16 | 8 | `103668.5` | `1.00x` | `2357.5` | `1.00x` | `2.062 GiB` BF16 KV | dense BF16-KV baseline |
| vLLM FP8 | 8 | `102959.5` | `0.99x` | `2888.7` | `1.23x` | `1.031 GiB` FP8 KV | dense FP8-KV baseline |
| vLLM TurboQuant `4bit_nc` | 8 | `104409.9` | `1.01x` | `1696.0` | `0.72x` | `0.939 GiB` TQ4 KV | 4-bit KV baseline |
| FastDMS FP8 compact-DMS default | 8 | **`105531.7`** | **`1.02x`** | **`3606.9`** | **`1.53x`** | **`0.431 GiB`** | promoted zero-BF16 row |
| FastDMS B25 narrow int4 speed profile | 8 | `104753.7` | `1.01x` | `3640.7` | `1.54x` | `0.431 GiB` + `0.078 GiB` int4 shadow | default-off storage-for-speed |
| FastDMS BF16-attention speed control | 8 | `108070.5` | `1.04x` | **`3745.3`** | **`1.59x`** | `0.429 GiB` + `0.312 GiB` BF16 backing | explicit speed control |

### nvidia/Qwen3-8B-DMS-8x

FastDMS performs similarly well with Nvidia's Qwen3-8B example model. It is `1.54x` / `2.06x` faster than vLLM BF16 decode and `7.64x` / `6.35x` smaller than vLLM BF16 KV at `c=1` / `c=8`. Against vLLM FP8, it is `1.48x` / `1.57x` faster decode and `3.82x` / `3.17x` smaller in KV. Versus same-engine FastDMS dense FP8, it is `3.26x` / `3.46x` faster decode and `11.47x` / `9.52x` smaller in allocator-visible KV/stage memory.

| Path | c | Prefill tok/s | Prefill vs BF16 | Decode tok/s | Decode vs BF16 | KV / stage memory | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| vLLM BF16 | 1 | `17143.6` | `1.00x` | `89.44` | `1.00x` | `1.406 GiB` BF16 KV | dense BF16-KV baseline |
| vLLM FP8 | 1 | `16700.1` | `0.97x` | `93.12` | `1.04x` | `0.703 GiB` FP8 KV | dense FP8-KV baseline |
| FastDMS dense FP8 | 1 | `22125.7` | `1.29x` | `42.24` | `0.47x` | `0.703 GiB` FP8 KV + `1.406 GiB` BF16 staging | same-engine dense baseline |
| FastDMS compact DMS | 1 | **`21610.1`** | **`1.26x`** | **`137.76`** | **`1.54x`** | **`0.184 GiB`** compact+metadata | zero retained LM-head BF16 backing |
| vLLM BF16 | 8 | `15800.2` | `1.00x` | `444.59` | `1.00x` | `9.281 GiB` BF16 KV | dense BF16-KV baseline |
| vLLM FP8 | 8 | `15659.1` | `0.99x` | `583.39` | `1.31x` | `4.641 GiB` FP8 KV | dense FP8-KV baseline |
| FastDMS dense FP8 | 8 | `19502.4` | `1.23x` | `265.00` | `0.60x` | `4.641 GiB` FP8 KV + `9.281 GiB` BF16 staging | same-engine dense baseline |
| FastDMS compact DMS | 8 | **`19366.5`** | **`1.23x`** | **`917.90`** | **`2.06x`** | **`1.462 GiB`** compact+metadata | zero retained LM-head BF16 backing |

## Memory Savings vs vLLM

Compact DMS saves real allocator/device memory, not just theoretical KV bytes. The table below uses the same WikiText-2 `ctx_len=8192`, `gen_len=128` rows as the speed tables above. All vLLM baselines use exact-sized token pools matching the workload. KV/stage memory is the cache or cache-plus-staging footprint. vLLM BF16 means `dtype=bfloat16` with `kv_cache_dtype=auto`; vLLM FP8 means `kv_cache_dtype=fp8`.

| Model / compact-DMS row | c | vLLM BF16 KV → FastDMS KV | BF16 KV saved | vLLM FP8 KV → FastDMS KV | FP8 KV saved | vLLM TQ4 KV → FastDMS KV | TQ4 KV saved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama-3.2-1B FastDMS default | 1 | `0.312 → 0.056 GiB` | **`5.6x`** | `0.156 → 0.056 GiB` | **`2.8x`** | `0.142 → 0.056 GiB` | **`2.5x`** |
| Llama-3.2-1B FastDMS default | 8 | `2.062 → 0.431 GiB` | **`4.8x`** | `1.031 → 0.431 GiB` | **`2.4x`** | `0.939 → 0.431 GiB` | **`2.2x`** |
| Qwen3-8B FastDMS compact DMS | 1 | `1.406 → 0.184 GiB` | **`7.6x`** | `0.703 → 0.184 GiB` | **`3.8x`** | — | — |
| Qwen3-8B FastDMS compact DMS | 8 | `9.281 → 1.462 GiB` | **`6.3x`** | `4.641 → 1.462 GiB` | **`3.2x`** | — | — |

Per token, Qwen3 KV is larger: about `72 KiB/token` for FP8 KV (`36` layers × `8` KV heads × `128` head dim × K/V) versus Llama-3.2-1B’s `16 KiB/token`.

## Max Context

Max-context rows use each model's config-supported final window while leaving `128` generated tokens: Llama-3.2-1B at `130944+128` of `131072`, and Qwen3-8B at `40832+128` of `40960`. vLLM rows use `vllm-nightly` with post-warmup timing, wrapped WikiText-2 prompts, and a small KV-pool margin above the exact context budget to avoid exact-full-pool scheduler waits. vLLM FP8 max-context CUDA peaks are upper bounds from paired BF16/FP8 runs; KV memory is dtype-specific.

### shisa-ai/Llama-3.2-1B-DMS-8x

| Path | c | ctx+gen | Prefill tok/s | Decode tok/s | KV / compact+meta GiB | CUDA peak GiB | Decode vs BF16 | KV vs BF16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vLLM BF16 KV | 1 | `130944+128` | `22906.4` | `189.1` | `4.125` | `10.92` | `1.00x` | `1.00x` |
| vLLM FP8 KV | 1 | `130944+128` | `21639.8` | `265.9` | `2.063` | `<=10.92` | `1.41x` | `2.00x smaller` |
| FastDMS compact DMS | 1 | `130944+128` | **`29059.6`** | **`286.0`** | **`0.76669`** | `15.20` | **`1.51x`** | **`5.38x smaller`** |
| vLLM BF16 KV | 8 | `130944+128` | `20384.0` | `178.8` | `33.000` | `38.90` | `1.00x` | `1.00x` |
| vLLM FP8 KV | 8 | `130944+128` | `20448.6` | `492.2` | `16.500` | `<=38.90` | `2.75x` | `2.00x smaller` |
| FastDMS compact DMS | 8 | `130944+128` | **`27655.0`** | `273.6` | **`1.30995`** | **`12.17`** | **`1.53x`** | **`25.2x smaller`** |

FastDMS is faster than vLLM BF16 at both Llama max-context concurrencies and is dramatically smaller in KV. Against vLLM FP8, c=1 still wins decode (`1.08x`) while c=8 trades lower decode (`0.56x`) for `12.6x` lower KV memory and `3.2x` lower paired-run CUDA peak. The c=1 FastDMS CUDA peak is higher than vLLM BF16 because the max-context FastDMS row carries graph/runtime overhead even though compact KV is much smaller.

### nvidia/Qwen3-8B-DMS-8x

| Path | c | ctx+gen | Prefill tok/s | Decode tok/s | KV / compact+meta GiB | CUDA peak GiB | Decode vs BF16 | KV vs BF16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vLLM BF16 KV | 1 | `40832+128` | `9614.8` | `69.24` | `6.188` | `26.04` | `1.00x` | `1.00x` |
| vLLM FP8 KV | 1 | `40832+128` | `9321.5` | `79.46` | `3.094` | `<=26.04` | `1.15x` | `2.00x smaller` |
| FastDMS compact DMS | 1 | `40832+128` | **`12544.1`** | **`111.9`** | **`0.58313`** | **`14.64`** | **`1.62x`** | **`10.6x smaller`** |
| vLLM BF16 KV | 8 | `40832+128` | `9116.6` | `161.6` | `49.500` | `69.22` | `1.00x` | `1.00x` |
| vLLM FP8 KV | 8 | `40832+128` | `8996.9` | `286.3` | `24.750` | `<=69.22` | `1.77x` | `2.00x smaller` |
| FastDMS compact DMS | 8 | `40832+128` | **`10517.6`** | `79.3` | **`1.00698`** | **`20.69`** | `0.49x` | **`49.2x smaller`** |

FastDMS transfers cleanly at Qwen c=1 max context: `1.62x` BF16 decode, `10.6x` lower KV, and `1.78x` lower CUDA peak. Qwen c=8 max context is the one long-context throughput caveat: it is a large memory/capacity win (`49.2x` lower BF16 KV, `3.35x` lower CUDA peak), but decode is `0.49x` of vLLM BF16 and `0.28x` of vLLM FP8 at this context.

## Compression Quality

Of course, none of this matters if the compression tanks output quality. In theory, DMS eviction is applied *before* FP8 quantization, deciding which tokens to keep or evict, so the quality comparison for FastDMS compact-DMS *should* be the same versus FP8 quantization alone, but it's still worth double-checking quality.

We measure this by generating tokens with a compressed KV cache and comparing against an uncompressed reference, token by token. Lower KLD (KL divergence) is better - it means the compressed model's next-token probabilities are closer to the reference. Higher token match is better - it means greedy decoding produces the same output.

**How to read the columns:**

- **KLD vs ref** - KL divergence in nats/token between the compressed and reference logits. Measures how much the probability distribution over next tokens shifts due to compression. Lower is better; `0.000` means identical.
- **Token match** - percentage of greedy-decoded tokens that are identical to the reference. `96.9%` means ~2 out of 64 tokens differed.
- **Tokens scored** - how many decode steps could be compared. Once the candidate produces a different token than the reference, the sequences diverge and later steps aren't comparable. `33/60` means quality metrics only cover the first 33 tokens before divergence - the reported KLD and PPL are over that prefix, not the full generation. A higher ratio means the comparison is more complete.

**Test setup:** `ctx_len=1024`, `decode_len=16`, four prompts (60-64 total decode steps). vLLM rows compare against vLLM BF16 full-KV logits. FastDMS rows compare against FastDMS with eviction disabled (reference window of 1M tokens, effectively keeping the full KV cache).

### shisa-ai/Llama-3.2-1B-DMS-8x

| Path | Reference | KLD vs ref | Token match | PPL | Tokens scored |
| --- | --- | ---: | ---: | ---: | ---: |
| vLLM BF16 full KV | self | `0.000000` | `100.0%` | `2.3748` | `60/60` |
| vLLM FP8 KV | vLLM BF16 | `0.005110` | `92.2%` | `2.0893` | `33/60` |
| vLLM TurboQuant `4bit_nc` | vLLM BF16 | `0.012730` | `76.6%` | `1.9606` | `22/60` |
| FastDMS FP8 compact-DMS | FastDMS no-evict | `0.003009` | `96.9%` | `2.2810` | `64/64` |

### nvidia/Qwen3-8B-DMS-8x

| Path | Reference | KLD vs ref | Token match | PPL | Tokens scored |
| --- | --- | ---: | ---: | ---: | ---: |
| vLLM BF16 full KV | self | `0.000000` | `100.0%` | `1.6738` | `60/60` |
| vLLM FP8 KV | vLLM BF16 | `0.001042` | `70.3%` | `1.1971` | `32/60` |
| vLLM TurboQuant `4bit_nc` | vLLM BF16 | `0.006039` | `84.4%` | `1.4910` | `45/60` |
| FastDMS FP8 compact-DMS | FastDMS no-evict | `0.005284` | `95.3%` | `1.8301` | `64/64` |

FastDMS compact-DMS scores `64/64` tokens on both models - every decode step was comparable to the reference, and the KLD is lower than or comparable to vLLM's own FP8 and TurboQuant compression. Note that PPL values across rows are not directly comparable when `Tokens scored` differs, because each row's PPL is computed over a different-length prefix.

## Why a Standalone Engine Instead of a vLLM Plugin?

We investigated porting compact DMS directly into vLLM and concluded it is **major surgery**, not a plugin. DMS compact KV touches nearly every serving-engine subsystem:

| Subsystem | What changes for DMS |
| --- | --- |
| **PagedAttention / KV memory pool** | DMS needs per-layer, per-head variable token counts with partial block deallocation - not standard fixed-page blocks |
| **Prefill kernel** | Must stream surviving K/V into compact per-layer storage after DMS extraction, rather than writing dense KV pages |
| **Decode kernel** | Each decode step evaluates per-head keep/evict, manages a sliding retention window, and appends to compact storage |
| **Attention scoring** | Replaced entirely: split-K grouped compact decode attention over variable-length per-head live spans |
| **Scheduler / admission** | Must admit requests based on compact KV capacity, not dense full-sequence page count - this is the hardest boundary |
| **Prefix caching** | DMS eviction is per-sequence and per-head; shared prefix blocks need per-sequence eviction overlays or must be disabled |
| **Continuous batching** | Memory accounting must reflect actual surviving token count, not logical sequence length |

vLLM's TurboQuant backend provides a useful template (custom cache dtype, custom `KVCacheSpec`, backend-owned metadata builder, custom store/decode ops). But TurboQuant still uses vLLM's paged block table with one compressed slot per logical token. DMS needs per-layer, per-sequence, per-KV-head live spans and eviction state - the decode metadata is fundamentally different.

The critical gap is **scheduler/cache accounting**: until vLLM stops reserving dense full-sequence pages, a DMS backend is only a functional/speed experiment and not the compact-memory serving path that delivers the real value. That scheduler change touches `scheduler.py`, `kv_cache_manager.py`, `kv_cache_coordinator.py`, `single_type_kv_cache_manager.py`, and `block_pool.py` - the core of vLLM's memory management.

A proper vLLM port would need to:

1. Add DMS as a first-class cache dtype with config gates and hard-disable unsupported features (prefix cache, chunked prefill, spec decode, distributed)
2. Port DMS metadata loading and per-model borrowed-channel extraction (Llama-only first)
3. Build a custom `DMSCompactAttentionBackend` with sidecar compact arena
4. Port compact append-store, DMS expiry, fused DMS/RoPE/store, and split-K compact decode kernels
5. Replace sidecar-only accounting with native DMS reservation-cap admission in vLLM's cache manager
6. Only then re-enable chunked prefill, prefix cache overlays, and distributed execution

This is a viable path but a large, bounded engineering project. FastDMS exists mainly to show *why* this effort might be worthwhile and it proves that DMS can be served efficiently.

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

Requires recent CUDA, `torch`, `triton`, and `flash-attn`.

## Quick Start

```python
from fastdms import LLM, SamplingParams

llm = LLM("shisa-ai/Llama-3.2-1B-DMS-8x", enforce_eager=True)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
outputs = llm.generate(["Hello, FastDMS."], sampling_params)
print(outputs[0]["text"])
```

## Canonical Papers

- **DMS**: [Inference-Time Hyper-Scaling with KV Cache Compression](https://openreview.net/forum?id=8ZiElzQxf1) ([arXiv:2506.05345](https://arxiv.org/abs/2506.05345)). NeurIPS 2025.
  - NVIDIA reference implementation (discovered after independent re-implementation!): [Model-Optimizer/experimental/dms](https://github.com/NVIDIA/Model-Optimizer/tree/main/experimental/dms) (this implementation does dense KV during prefill, doesn't have DMS expiry for inference, and is not optimized for serving speed, but its trainer is the better starting point for multi-GPU or larger-model training)
- **AQUA-KV**: [Cache Me If You Must: Adaptive Key-Value Quantization for Large Language Models](https://arxiv.org/abs/2501.19392).
- **HIGGS**: [Pushing the Limits of Large Language Model Quantization via the Linearity Theorem](https://arxiv.org/abs/2411.17525).

## Acknowledgements

FastDMS is built on top of [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) by Xingkai Yu. The clean, readable nano-vLLM codebase was used as the starting harness for the compact-DMS implementation. We're grateful for the upstream work that made rapid iteration on the KV-cache layout possible.

## Citation

```bibtex
@misc{fastdms2026,
  title        = {FastDMS: Production-Speed Compact DMS for KV Cache Compression},
  author       = {{Leonard Lin}},
  year         = {2026},
  url          = {https://github.com/shisa-ai/FastDMS},
  note         = {Fast reference implementation of compact Dynamic Memory Sparsification with FP8 KV cache}
}
```

## License

[MIT](LICENSE)
