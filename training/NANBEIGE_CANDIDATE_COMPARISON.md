# Nanbeige context/depth comparison

All candidates use exact-budget CR2, window 8192 and the same primitive-ai FP8
model. The shared validation set has six real-data 64K contexts and 31 scored
decode predictions per context. Final holdouts remain sealed.

| Candidate | BF16 mean KL | BF16 agreement | FP8 mean KL | FP8 agreement | Worst KL BF16 / FP8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32K epoch20 | .015638 | 96.24% | .030867 | 95.16% | .242 / 1.162 |
| 64K epoch10 | .015563 | 97.85% | .022474 | 95.70% | .221 / .193 |
| 64K epoch20 | .015922 | 96.77% | .020007 | 95.70% | .290 / .305 |

All candidates recovered every target token on all four synthetic retrieval
cases under teacher forcing, with both KV dtypes. At 128K, BF16 decode was
16.79/16.81/16.87 tok/s and FP8 decode 20.33/20.39/20.47 tok/s in table order.
Exact-budget live KV is 13.813 GiB BF16 or 6.906 GiB FP8 at 128K.

Selection: 64K epoch10, based on higher BF16 prediction agreement and lower
worst per-token KL for both KV dtypes. Epoch20 has lower mean FP8 KL but a
higher maximum. No final-set result was used in selection.

Raw results: `artifacts/nanbeige-real/{validation,retrieval,perf}-*.json` and
`artifacts/nanbeige-real64/epoch{10,20}-{validation,retrieval,perf}-*.json`.
Selected raw checkpoint: `artifacts/nanbeige-real64/training/epoch-10.pt`.
Selected calibrated export: `artifacts/nanbeige-real64/epoch10-export/final/`.
