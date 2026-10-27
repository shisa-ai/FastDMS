# Nanbeige real-data DMS run

Status: 32K baseline trained for 20 epochs and development-evaluated; 64K
capture is running. Candidate comparison and fresh-data continuation remain open.

## 32K epoch-20 baseline results

Training BCE decreased from 0.4974 to 0.2659; heldout BCE from 0.4476 to
0.4210, with heldout classifier accuracy 81.32%. All four synthetic 32K/128K
retrieval cases recovered every target token under teacher forcing for both
BF16 and FP8 KV. Shared 64K real-data validation top-1 agreement with dense BF16
KV was 93.5–100% by category for DMS BF16 and 90.3–100% for DMS FP8.
Dense FP8 alone measured 93.5–100% on the same reference. These development
results do not consume the sealed final streams.

| Context | KV dtype | Live KV GiB | Peak allocated GiB | Decode tok/s |
| ---: | --- | ---: | ---: | ---: |
| 32K | BF16 | 4.063 | 44.30 | 30.64 |
| 32K | FP8 | 2.031 | 42.26 | 31.67 |
| 128K | BF16 | 13.813 | 72.94 | 16.79 |
| 128K | FP8 | 6.906 | 66.02 | 20.33 |

Performance uses one warmup and two measured runs, batch 1, 32 output tokens,
with the repeated performance prompt. Exact-budget CR2 protects the latest
8193 positions and retains half the older history. Raw results are
`artifacts/nanbeige-real/{validation-*,retrieval-*,perf-*}.json`; the dense FP8
control is `artifacts/nanbeige-real64/dense-fp8-validation.json`.


This run follows the external-sidecar recipe recovered from
`/home/lhl/hipEngine/docs/reference/DMS.md`: 32K source contexts, 20 epochs,
AdamW learning rate 0.001, batch 512, budget loss weight 0.1, gradient cap 1,
train-only per-head calibration, CR2 exact-budget prefill and window 8192.
The earlier hipEngine threshold-only and small-window candidates failed; its
qualified W8192 candidate used exact-budget selection. The older
`kvcache-quantization-research` runs trained borrowed-query heads and established
that actual masking semantics and inference quality must be checked, not
inferred from training loss.

## Data

`artifacts/nanbeige-real/data/manifest.json` records 60 streams of 32,768 tokens:
48 train (1,572,864 tokens), six validation, six sealed final. Sources are local
Shisa-v2.1 SFT conversations and Sansho cached FineWiki English/Japanese,
Stack-Edu Python, PG19 and Swallow Math v2 textbooks. All are re-tokenized with
the Nanbeige tokenizer. Conversation text uses its chat template.

The corpus contains 1,659 distinct documents. SHA-256 of document text assigns
splits (modulo 20: 0 final, 1 validation, remainder train). Exact duplicate text
is removed globally; document tails are discarded at stream boundaries.
Document hashes do not cross splits. This is exact-document separation, not a
claim of semantic or near-duplicate decontamination. Sources include no benchmark
prompts deliberately; the earlier synthetic checks are separate development
checks, not the final real-data holdout.

## Capture and fit

`training/capture_real_sidecar.py` uses the exact FastDMS FP8 serving model,
normalized pre-Q input and post-RoPE Q/K, without changing model weights.
Two workers divide whole streams across GPUs 0/1. Full 32K contexts are used;
query positions are subsampled at stride 16 and 1,024 eligible rows are sampled
per layer/stream. This is an approximate future-attention oracle, not the
stride-one oracle used in the final historical hipEngine run. The stride and
selected positions are recorded. Final streams are never captured for fitting.

`training/train_captured_sidecar.py` fits independent per-layer optimizers,
validates on document-disjoint streams every epoch, and saves optimizer state
and each epoch's weights. Train-only median calibration uses BF16-rounded
weights. Export records the actual FP8 model fingerprint and exact-budget
selection. Integration quality, including BF16 and FP8 KV, must pass before the
candidate can be called qualified. Training completion alone is not promotion.

Commands are run in Mamba `fastdms` with `PYTHONNOUSERSITE=1`. Data preparation:

```bash
python -m training.prepare_real_data --output artifacts/nanbeige-real/data
```

Capture, once per rank with distinct visible GPU and distributed port:

```bash
CUDA_VISIBLE_DEVICES=0 FASTDMS_DIST_PORT=2370 \
python -m training.capture_real_sidecar \
  --data artifacts/nanbeige-real/data --output artifacts/nanbeige-real/labels \
  --model /path/to/primitive-ai-FP8-snapshot --rank 0
```

Fit after both manifests contain all 54 non-final streams and 52 layers each:

```bash
python -m training.train_captured_sidecar \
  --labels artifacts/nanbeige-real/labels --output artifacts/nanbeige-real/training \
  --model /path/to/primitive-ai-FP8-snapshot --epochs 20
```
