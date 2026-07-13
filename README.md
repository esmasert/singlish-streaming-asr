# Streaming Singlish ASR for Telephony

A cache-aware, low-latency Singlish automatic speech recognition demo built on **NVIDIA Nemotron Speech Streaming EN 0.6B**. The system emits partial hypotheses while audio is arriving; it does not run an offline ASR model independently on fixed chunks.

## Headline results

Evaluation used the filtered/decontaminated 346-file task manifest, 560 ms streaming chunks, attention context `[70, 6]`, and normalized WER/CER scoring.

| Metric | Result |
|---|---:|
| Evaluation audio | 2.443 hours / 346 files |
| Normalized WER | **18.13%** |
| Normalized CER | **11.58%** |
| Tier 0 WER | 16.93% |
| Tier 1 WER | 16.76% |
| Tier 2 WER | 19.87% |
| Single-stream compute RTF | **0.0736** |
| Single-stream end-to-end RTF | **0.0769** |
| Chunk compute p50 / p95 | 40.59 / 45.49 ms |
| First partial p50 / p95 | 1161.63 / 1725.06 ms |
| Finalization p50 / p95 | 301.44 / 596.27 ms |

The <=15% overall WER target was **not reached**. The final system nevertheless demonstrates true streaming behaviour with comfortable real-time compute headroom. See [results/RESULTS.md](results/RESULTS.md) and [DESIGN.md](DESIGN.md).

## What is included

- FastAPI WebSocket inference backend
- Gradio live demo with partial and final transcripts
- Base and fine-tuned model selection
- 80, 160, 560, and 1120 ms chunk controls
- Real-time playback and fast cache-aware evaluation modes
- WER, CER, correctness, RTF, first-partial, and finalization metrics
- Coloured word-error alignment for substitutions, deletions, and insertions
- Folder-level streaming evaluation with CSV/JSON export
- Public result summaries without task transcripts or private paths

## Architecture

```text
┌──────────────────────────────┐
│ Input audio                  │
│ 16 kHz mono WAV              │
└──────────────┬───────────────┘
               │
               v
┌──────────────────────────────┐
│ Gradio frontend              │
│ - file upload / playback     │
│ - model selection            │
│ - chunk pacing               │
└──────────────┬───────────────┘
               │ WebSocket
               v
┌──────────────────────────────┐
│ FastAPI streaming backend    │
│ - chunk receiver             │
│ - session management         │
│ - timing collection          │
└──────────────┬───────────────┘
               │ sequential audio chunks
               v
┌──────────────────────────────┐
│ Nemotron streaming ASR       │
│ - cache-aware RNNT inference │
│ - encoder state persistence  │
│ - decoder state persistence  │
└──────────────┬───────────────┘
               │ partial + final hypotheses
               v
┌──────────────────────────────┐
│ Evaluation + visualization   │
│ - WER / CER                  │
│ - coloured alignment         │
│ - latency cards              │
│ - CSV / JSON export          │
└──────────────────────────────┘
```

The backend preserves streaming caches across successive chunks. `Fast evaluation` removes artificial wall-clock sleeping, but it does not turn the system into independent batch inference.

## Repository structure

```text
.
├── backend/                  # model loading, streaming engine, WebSocket API
├── frontend/                 # Gradio app, client, metrics, reference lookup
├── models/README.md          # checkpoint placement instructions
├── results/                  # public WER and latency summaries
├── sample_audio/README.md    # input format and data note
├── scripts/                  # launch and health-check helpers
├── .env.example
├── requirements-mac.txt
├── requirements-runpod.txt
├── DESIGN.md
└── README.md
```

## Model and data requirements

The task audio and fine-tuned `.nemo` checkpoint are not committed because of data access, model licensing, and file-size constraints.

The base model is configured as:

```text
nvidia/nemotron-speech-streaming-en-0.6b
```

For the fine-tuned model, place the exported checkpoint anywhere locally and set its path in `.env`:

```bash
FINETUNED_MODEL_PATH=/absolute/path/to/finetuned_model.nemo
```

The demo accepts **mono, 16 kHz WAV** input. A ground-truth JSONL manifest is optional; without one, transcription and latency still work, but WER/CER are not calculated.

## Installation

Python 3.11 is recommended.

### Apple Silicon / macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-mac.txt
cp .env.example .env
```

Edit `.env`:

```bash
DEVICE=mps
GROUND_TRUTH_JSONL=/absolute/path/to/optional_manifest.jsonl
FINETUNED_MODEL_PATH=/absolute/path/to/finetuned_model.nemo
```

### NVIDIA GPU / RunPod

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-runpod.txt
cp .env.example .env
```

Set:

```bash
DEVICE=cuda
```

NeMo/CUDA installation can be heavy. A recorded walkthrough is therefore a useful companion to the repository.

## Run the demo

Use two terminals.

### Terminal 1: inference backend

```bash
source .venv/bin/activate
set -a
source .env
set +a

export PYTORCH_ENABLE_MPS_FALLBACK=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

python -m uvicorn backend.server:app \
  --host 127.0.0.1 \
  --port 8011 \
  --workers 1
```

Check the backend:

```bash
curl http://127.0.0.1:8011/health
```

### Terminal 2: Gradio UI

```bash
source .venv/bin/activate
./scripts/start_mac.sh
```

Open:

```text
http://127.0.0.1:7870
```

The first request loads and warms the selected model, so run one short WAV before a live presentation.

## Demo modes

**Real-time playback (1x)** follows the audio clock and shows partial hypotheses as a user would experience them.

**Fast evaluation** sends the same ordered chunks without waiting between them. Streaming caches are still preserved, so this is suitable for faster local testing.

The 560 ms setting is the primary reported operating point. Smaller chunks can produce earlier updates but increase the number of model steps and compute overhead.

## Evaluation methodology

1. Convert/evaluate audio as 16 kHz mono WAV.
2. Feed chunks sequentially through the cache-aware streaming API.
3. Preserve encoder, decoder, and previous-hypothesis state across the utterance.
4. Normalize references and hypotheses with Unicode normalization, case folding, apostrophe removal, punctuation removal, and whitespace collapse.
5. Compute WER/CER with `jiwer`.
6. Report full-set accuracy from the evaluator and latency from a separate batch-size-1 single-stream profile.

The full evaluator used batch size 64 for efficient accuracy evaluation. It also profiled 30 utterances with batch size 1 for conversational latency. Batch-throughput RTF is intentionally not presented as live-agent latency.

## Results

Public machine-readable summaries:

- [results/overall_summary.csv](results/overall_summary.csv)
- [results/by_dataset.csv](results/by_dataset.csv)
- [results/latency_summary.csv](results/latency_summary.csv)
- [results/RESULTS.md](results/RESULTS.md)

Full per-sample predictions and task transcripts are intentionally excluded from the public repository.

## Approach summary

The fine-tuning corpus combined curated Singlish speech with English replay data to reduce catastrophic forgetting. Training data preparation included transcript sanity checking, filtering of likely hallucination/non-speech examples, test-set decontamination, and both clean and telephony-processed audio variants. The model was selected because it natively supports cache-aware streaming RNNT inference and partial hypotheses.

## Limitations and next steps

- Overall WER remains above the 15% target, especially on Tier 2.
- The public result set is the filtered/decontaminated 346-file manifest; figures from different manifests should not be compared directly.
- The fine-tuned checkpoint is not included in GitHub.
- The strongest next experiment is more balanced English replay plus Singlish coverage, followed by low-learning-rate continuation and streaming-decoder tuning.
- Production work should include real 8 kHz telephony capture, endpointing/VAD integration, concurrency tests, and deployment-specific quantization.

## Design document

A concise discussion of the approach, trade-offs, evaluation methodology, and results is available in [DESIGN.md](DESIGN.md). A PDF version can be included under `docs/` for submission.
