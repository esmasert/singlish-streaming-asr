# Model checkpoint

The fine-tuned NeMo checkpoint is not committed to this repository because of
its file size and the licensing requirements of the underlying model.

Set the local checkpoint path in the `.env` file:

FINETUNED_MODEL_PATH=/absolute/path/to/finetuned_model.nemo

The base model is loaded from NVIDIA NGC / Hugging Face using:

nvidia/nemotron-speech-streaming-en-0.6b
