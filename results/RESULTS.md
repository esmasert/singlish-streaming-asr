# Evaluation Results

## Final fine-tuned model

Evaluation used cache-aware streaming inference with a 560 ms chunk,
attention context `[70, 6]`, and normalized WER/CER scoring.

| Metric | Result |
|---|---:|
| Evaluation files | 346 |
| Audio duration | 2.443 hours |
| Normalized WER | **18.13%** |
| Normalized CER | **11.58%** |
| Word correctness | 85.94% |
| Character correctness | 91.90% |

## Results by dataset

| Dataset | Files | WER | CER |
|---|---:|---:|---:|
| tier0 | 58 | 16.93% | 10.26% |
| tier1 | 143 | 16.76% | 10.65% |
| tier2 | 145 | 19.87% | 12.93% |

## Single-stream latency

Latency was profiled using batch size 1. The batch-64 evaluator throughput
numbers are not presented as conversational-agent latency.

| Metric | Result |
|---|---:|
| Compute RTF | 0.0736 |
| End-to-end RTF | 0.0769 |
| Chunk compute p50 | 40.59 ms |
| Chunk compute p95 | 45.49 ms |
| First partial p50 | 1161.63 ms |
| First partial p95 | 1725.06 ms |
| Finalization p50 | 301.44 ms |
| Finalization p95 | 596.27 ms |

The full per-sample predictions and task transcripts are intentionally not
committed to the public repository.
