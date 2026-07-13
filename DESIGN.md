# Design Document: Streaming Singlish ASR for Telephony

## 1. Problem and design goals

The task is to adapt an automatic speech recognition system to Singlish and telephony-style audio while retaining true streaming behaviour. A valid system must emit partial hypotheses while audio is arriving, maintain state across the utterance, and operate with latency suitable for a conversational agent. Calling an offline model independently on fixed chunks would not satisfy the requirement because each chunk would lose linguistic and acoustic context.

The main design goals were therefore:

1. Use a model with native cache-aware streaming inference.
2. Improve recognition of Singlish vocabulary, discourse markers, pronunciation, and code-switching patterns.
3. Retain enough general English performance to avoid catastrophic forgetting.
4. Evaluate accuracy and latency with a methodology that reflects a live, single-user stream.
5. Provide a runnable demo that exposes partial hypotheses, final output, WER/CER, and latency measurements.

The target was <=15% WER. The final overall result was 18.13%, so the accuracy target was not reached. However, the system met the streaming requirement and showed strong real-time compute headroom.

## 2. System approach

### 2.1 Model selection

I selected NVIDIA Nemotron Speech Streaming EN 0.6B. The key reason was architectural fit rather than only offline accuracy: the model supports cache-aware RNNT streaming, incremental hypotheses, configurable chunk sizes, and persistent encoder/decoder state. This makes it suitable for a live conversational agent and avoids approximating streaming with repeated batch calls.

The deployed path is:

```text
WAV/PCM -> ordered chunks -> cache-aware encoder/RNNT -> partial hypotheses -> final hypothesis
```

The streaming session retains the encoder cache, time cache, cache lengths, previous RNNT hypotheses, and previous prediction outputs. Each new chunk advances the same stream rather than starting a new transcription request.

### 2.2 Data preparation and fine-tuning

The adaptation data combined Singlish speech with English replay data. The English component was included to reduce catastrophic forgetting while the Singlish component focused the model on local pronunciation, conversational fillers, discourse particles, and domain-specific phrasing.

Data preparation included:

- transcript normalization and validation;
- Whisper-based sanity checks using normalized WER/CER signals;
- removal of likely non-speech, silence, hallucinated, or severely mismatched samples;
- decontamination against the provided evaluation set;
- clean 16 kHz examples and telephony-processed variants produced through a GSM-style channel and returned to the model input rate;
- duration and manifest checks before training.

The training objective remained the model's native streaming ASR objective. The final checkpoint used for reporting was exported to a `.nemo` model and evaluated with the same cache-aware inference path used by the demo.

### 2.3 Demo architecture

The demo separates inference and presentation:

- **FastAPI WebSocket backend:** loads the base or fine-tuned model, owns model state, processes sequential chunks, and returns transcript and timing events.
- **Gradio frontend:** uploads WAV files, controls model/chunk size, displays partial text, calculates normalized metrics, renders a coloured word-error alignment, and exports per-step traces.

Two UI modes are available. Real-time playback follows the audio clock. Fast evaluation removes sleeping between chunks but preserves the exact streaming order and caches. The latter is useful for rapid testing without converting the system into batch ASR.

## 3. Evaluation methodology

### 3.1 Accuracy

The headline evaluation used a filtered/decontaminated manifest of 346 files (2.443 hours) across Tier 0, Tier 1, and Tier 2. Audio was evaluated as mono 16 kHz input with a 560 ms streaming chunk and attention context `[70, 6]`.

Reference and hypothesis text were normalized consistently using Unicode NFKC normalization, case folding, removal of markup/apostrophes/punctuation, preservation of letters and numbers, and whitespace collapse. WER and CER were calculated with `jiwer`.

The full accuracy evaluator used an effective batch size of 64 for throughput. Batching did not reset streaming state within an utterance; each sample still advanced through ordered chunks with caches. Full-set batch throughput is not used as the latency claim.

### 3.2 Latency

Conversational latency was profiled separately with batch size 1 on 30 utterances. Reported measurements include:

- model-compute RTF;
- end-to-end RTF;
- per-chunk compute p50/p95;
- first non-empty partial p50/p95;
- final tail after the audio ends p50/p95.

This separation prevents the misleading practice of presenting batch-64 throughput as single-user real-time latency.

## 4. Results

### 4.1 Accuracy

| Evaluation group | Files | Normalized WER | Normalized CER |
|---|---:|---:|---:|
| Overall | 346 | **18.13%** | **11.58%** |
| Tier 0 | 58 | 16.93% | 10.26% |
| Tier 1 | 143 | 16.76% | 10.65% |
| Tier 2 | 145 | 19.87% | 12.93% |

Tier 2 remained the most difficult group. The <=15% overall target was not met. The result should therefore be interpreted as a working streaming adaptation with measurable domain performance, not as completion of the target accuracy threshold.

### 4.2 Single-stream latency

| Metric | Result |
|---|---:|
| Compute RTF | **0.0736** |
| End-to-end RTF | **0.0769** |
| Chunk compute p50 | 40.59 ms |
| Chunk compute p95 | 45.49 ms |
| First partial p50 | 1161.63 ms |
| First partial p95 | 1725.06 ms |
| Finalization p50 | 301.44 ms |
| Finalization p95 | 596.27 ms |
| Chunks over 560 ms compute budget | 0% |

An end-to-end RTF of 0.0769 means the profiled inference path uses substantially less compute time than the incoming audio duration. The first-partial metric also contains the acoustic evidence needed before a non-empty hypothesis is available; it is therefore larger than the model's per-step compute time.

## 5. Trade-offs and design decisions

### Chunk size

A smaller chunk can update the interface more frequently, but it increases the number of streaming steps, cache operations, and frontend/backend messages. Larger chunks reduce step overhead but delay the first opportunity to emit text. I selected 560 ms as the main reporting point because it was stable, computationally efficient, and directly aligned with the main evaluator configuration. The demo also exposes 80, 160, and 1120 ms for qualitative comparison.

### Domain adaptation versus English retention

Increasing the proportion or training weight of Singlish data should improve the harder Singlish tiers, but it risks degrading general English. English replay data was therefore mixed into adaptation. The remaining gap suggests that the next run needs a better replay balance or a staged continuation schedule rather than simply more aggressive Singlish-only training.

### Accuracy evaluation versus live serving

Batching is useful for evaluating hundreds of utterances efficiently, but it can obscure deployment latency. For this reason, accuracy was computed over the full set while live latency was reported from batch-size-1 streams. The local Apple MPS demo is intended for interaction and visualisation; the submitted headline numbers come from the controlled A100 evaluator.

### Reproducibility versus repository size

The repository includes the runnable backend/frontend and public aggregate outputs, but not the task audio, per-sample transcripts, or fine-tuned checkpoint. Those artifacts are excluded because of access restrictions, licensing, privacy, and size. The README documents checkpoint placement, and a recorded walkthrough can demonstrate the fully configured environment.

## 6. Limitations and next steps

The most important limitation is that 18.13% WER remains above the requested threshold. Further work should prioritise:

1. More balanced English replay to protect Tier 0 while retaining Singlish gains.
2. Additional high-quality Tier 2 coverage and targeted transcript cleanup.
3. Low-learning-rate continuation from the strongest checkpoint rather than restarting from an over-specialised mixture.
4. Streaming decoder and endpointing tuning, including the effect of chunk size on partial stability.
5. Evaluation on native 8 kHz telephony captures, packet loss, codec variation, and background noise.
6. Concurrency, memory, and quantisation tests for production deployment.

The current system is a valid cache-aware streaming implementation with strong compute latency, a transparent evaluation methodology, and a practical demo. Its remaining challenge is accuracy rather than streaming feasibility.
