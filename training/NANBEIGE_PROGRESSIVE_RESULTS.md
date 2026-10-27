# Nanbeige progressive fresh-data results

The continuation completed ten epochs from the selected 64K epoch-10 raw
checkpoint. Each epoch used 24 fresh 64K streams plus 24 replay streams;
validation stayed fixed. Total fresh tokens: 15,728,640. Total training tokens
processed across these epochs: 31,457,280. Capture plus fitting took about
3 hours 26 minutes. All ten completion records and source manifests were
verified; fresh exact-text document hashes do not overlap earlier corpora or
other refresh epochs.

Fresh activations were deleted after each successful epoch, after preserving
source selections, shard hashes, metrics and checkpoints. Original replay
captures remain. Fresh source token files and provenance remain under
`artifacts/nanbeige-progressive/epoch-*/`. The continuation began with weights
from epoch10 and fresh optimizer state; subsequent refresh epochs carried
optimizer state forward. Each epoch's export was calibrated on that epoch's
training rows only.

Heldout classification BCE changed from 0.420536 before continuation to
0.393684 at continuation epoch10; accuracy changed from 81.4026% to 82.5859%.

## Shared development set

Six 64K source categories, 31 decode predictions per case. All comparisons use
the same FP8 model weights and dense BF16 KV reference. KL is in nats.

| Candidate | KV dtype | Mean KL | Top-1 agreement | Maximum per-token KL |
| --- | --- | ---: | ---: | ---: |
| Selected 64K epoch10 | BF16 | .015563 | 97.85% | .2212 |
| Fresh continuation epoch10 | BF16 | .016123 | 96.24% | .1937 |
| Selected 64K epoch10 | FP8 | .022474 | 95.70% | .1925 |
| Fresh continuation epoch10 | FP8 | .027148 | 97.31% | .8485 |

Both candidates recover every target token in all four synthetic retrieval
cases (32K/128K, early/late needle), with both KV dtypes under teacher forcing.

## Previously sealed final set

Both candidate checkpoints were frozen before opening the six final 64K
streams. No checkpoint or policy was changed using final results. The final
set has 186 scored decode predictions. This is a bounded teacher-forced
comparison, not free-generation task accuracy.

| Configuration | Mean KL | Top-1 agreement | Maximum per-token KL |
| --- | ---: | ---: | ---: |
| Dense BF16 KV reference | 0 | 100% | 0 |
| Dense FP8 KV | .081011 | 94.09% | 8.9380 |
| Selected baseline, DMS BF16 | .023749 | 94.09% | .6982 |
| Continuation, DMS BF16 | .029896 | 93.55% | 1.6364 |
| Selected baseline, DMS FP8 | .062752 | 91.94% | 4.6649 |
| Continuation, DMS FP8 | .040502 | 90.32% | .8821 |

The continuation reduced heldout classifier loss. Serving metrics are mixed:
final BF16 mean KL increased and prediction agreement decreased; final FP8 mean
and maximum KL decreased while prediction agreement decreased. These results
do not support replacing the selected baseline with the continuation as an
across-metric improvement. Both artifacts are retained; neither is labelled
universally quality-qualified.

## Continuation performance

Batch 1, eager, one warmup and two measured trials, 32 output tokens. Same
performance prompt and exact-budget CR2/window8192 policy as the baseline.

| Context | KV dtype | Live KV GiB | Peak allocated GiB | Decode tok/s |
| ---: | --- | ---: | ---: | ---: |
| 32K | BF16 | 4.063 | 44.30 | 30.47 |
| 32K | FP8 | 2.031 | 42.26 | 31.84 |
| 128K | BF16 | 13.813 | 72.94 | 16.80 |
| 128K | FP8 | 6.906 | 66.02 | 20.35 |

## Artifacts

- Baseline: `artifacts/nanbeige-real64/epoch10-export/final/`.
- Continuation: `artifacts/nanbeige-progressive/epoch-10/training/final/`.
- Per-epoch source and capture manifests, resume checkpoints and classifier
  histories: `artifacts/nanbeige-progressive/epoch-*/`.
- Development quality/performance: `artifacts/nanbeige-progressive/{validation,retrieval,perf}-*.json`.
- Final-set results and reference logits: `artifacts/nanbeige-final/`.
- Driver: `training/progressive_sidecar.py`; data builder:
  `training/prepare_real_data.py`; capture: `training/capture_real_sidecar.py`.

Training was at 32K and 64K. 128K serving was tested, but no 128K training run
was performed. Capture uses query stride16 and 1024 sampled eligible rows per
layer/stream, recorded in manifests. Source separation checks exact text hashes;
it is not a near-duplicate decontamination claim.
