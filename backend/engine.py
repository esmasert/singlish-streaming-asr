from __future__ import annotations

import copy
import os
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

CHUNK_TO_RIGHT_CONTEXT = {80: 0, 160: 1, 560: 6, 1120: 13}
SUPPORTED_CHUNKS = tuple(CHUNK_TO_RIGHT_CONTEXT)
TARGET_SAMPLE_RATE = 16_000
ATTENTION_LEFT = int(os.getenv("ATTENTION_LEFT", "70"))


def _extract_text(hypothesis: Any) -> str:
    if hypothesis is None:
        return ""
    if isinstance(hypothesis, str):
        return hypothesis
    text = getattr(hypothesis, "text", None)
    if isinstance(text, str):
        return text
    if isinstance(hypothesis, (list, tuple)) and hypothesis:
        return _extract_text(hypothesis[0])
    return str(hypothesis)


def _resolve_finetuned_path(raw_path: str | None) -> Path:
    if not raw_path:
        raise RuntimeError(
            "FINETUNED_MODEL_PATH is not set. Point it to a .ckpt or .nemo model."
        )
    path = Path(raw_path).expanduser().resolve()
    if path.is_file() and path.suffix.lower() in {".ckpt", ".nemo"}:
        return path
    if path.is_dir():
        candidates = [*path.rglob("*.ckpt"), *path.rglob("*.nemo")]
        if candidates:
            return max(candidates, key=lambda item: item.stat().st_mtime)
    raise FileNotFoundError(f"Could not find a .ckpt or .nemo model at {path}.")


def _configure_decoding_for_streaming(model: Any) -> None:
    """Match the evaluator: preserve strategy; disable fused/CUDA-graph decoding."""
    if not hasattr(model, "change_decoding_strategy") or not hasattr(model.cfg, "decoding"):
        return
    from omegaconf import OmegaConf, open_dict

    decoding_cfg = OmegaConf.create(
        OmegaConf.to_container(model.cfg.decoding, resolve=False)
    )
    with open_dict(decoding_cfg):
        try:
            decoding_cfg.fused_batch_size = -1
        except Exception:
            pass
        try:
            decoding_cfg.preserve_alignments = False
        except Exception:
            pass
        try:
            decoding_cfg.greedy.use_cuda_graph_decoder = False
        except Exception:
            pass
    model.change_decoding_strategy(decoding_cfg)


def _strip_uniform_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    for prefix in ("model.", "module."):
        if state_dict and all(key.startswith(prefix) for key in state_dict):
            return {key[len(prefix):]: value for key, value in state_dict.items()}
    return state_dict


class ModelRegistry:
    """Lazy-loads base and fine-tuned models and serializes sessions per model."""

    def __init__(self) -> None:
        self.base_name = os.getenv(
            "BASE_MODEL_NAME", "nvidia/nemotron-speech-streaming-en-0.6b"
        )
        self.finetuned_path = os.getenv("FINETUNED_MODEL_PATH")
        self.device_name = os.getenv("DEVICE", "auto")
        self._models: dict[str, Any] = {}
        self._load_lock = threading.Lock()
        self.session_locks = {
            "base": threading.Lock(),
            "finetuned": threading.Lock(),
        }

    def _device(self):
        import torch

        if self.device_name != "auto":
            return torch.device(self.device_name)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def get(self, model_key: str):
        if model_key not in ("base", "finetuned"):
            raise ValueError(f"Unknown model key: {model_key}")
        if model_key in self._models:
            return self._models[model_key]

        with self._load_lock:
            if model_key in self._models:
                return self._models[model_key]

            import torch
            import nemo.collections.asr as nemo_asr

            torch.set_grad_enabled(False)
            if torch.cuda.is_available():
                torch.set_float32_matmul_precision("high")
                torch.backends.cuda.matmul.allow_tf32 = True

            device = self._device()
            if model_key == "base":
                model = nemo_asr.models.ASRModel.from_pretrained(
                    model_name=self.base_name,
                    map_location="cpu",
                )
            else:
                model_path = _resolve_finetuned_path(self.finetuned_path)
                if model_path.suffix.lower() == ".ckpt":
                    model = nemo_asr.models.ASRModel.from_pretrained(
                        model_name=self.base_name,
                        map_location="cpu",
                    )
                    try:
                        checkpoint = torch.load(
                            str(model_path),
                            map_location="cpu",
                            weights_only=False,
                            mmap=True,
                        )
                    except TypeError:
                        checkpoint = torch.load(str(model_path), map_location="cpu")
                    state_dict = checkpoint.get("state_dict", checkpoint)
                    try:
                        model.load_state_dict(state_dict, strict=True)
                    except RuntimeError:
                        model.load_state_dict(
                            _strip_uniform_prefix(state_dict), strict=True
                        )
                    del checkpoint, state_dict
                else:
                    model = nemo_asr.models.ASRModel.restore_from(
                        restore_path=str(model_path),
                        map_location="cpu",
                    )

            model = model.to(device)
            model.eval()
            _configure_decoding_for_streaming(model)
            self._models[model_key] = model
            return model

    def loaded(self) -> list[str]:
        return sorted(self._models)


REGISTRY = ModelRegistry()


def _cuda_sync(device: Any) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _autocast_context(device: Any):
    import torch

    amp_dtype = os.getenv("AMP_DTYPE", "bfloat16").lower()
    if device.type != "cuda" or amp_dtype == "none":
        return nullcontext()
    dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)


def _drop_extra_pre_encoded(model: Any, step_number: int) -> int:
    if step_number == 0:
        return 0
    return int(model.encoder.streaming_cfg.drop_extra_pre_encoded)


@dataclass
class StepResult:
    text: str
    compute_ms: float
    audio_received_ms: float
    audio_available_ms: float
    virtual_realtime_ms: float
    tail_after_audio_ms: float
    step: int
    is_final: bool


class FileStreamingSession:
    """Exact batch=1 evaluator path for uploaded WAV files."""

    def __init__(self, model: Any, chunk_ms: int) -> None:
        if chunk_ms not in CHUNK_TO_RIGHT_CONTEXT:
            raise ValueError(f"chunk_ms must be one of {SUPPORTED_CHUNKS}")
        self.model = model
        self.chunk_ms = int(chunk_ms)
        self.chunk_samples = TARGET_SAMPLE_RATE * self.chunk_ms // 1000
        self.device = next(model.parameters()).device
        right_context = CHUNK_TO_RIGHT_CONTEXT[self.chunk_ms]
        model.encoder.set_default_att_context_size([ATTENTION_LEFT, right_context])

    def process_wav(self, wav_path: str | Path) -> Iterator[StepResult]:
        import soundfile as sf
        import torch
        from nemo.collections.asr.parts.utils.streaming_utils import (
            CacheAwareStreamingAudioBuffer,
        )

        info = sf.info(str(wav_path))
        if info.samplerate != TARGET_SAMPLE_RATE:
            raise ValueError(
                f"Expected {TARGET_SAMPLE_RATE} Hz, received {info.samplerate} Hz."
            )
        if info.channels != 1:
            raise ValueError(f"Expected mono audio, received {info.channels} channels.")
        duration_ms = info.frames / info.samplerate * 1000.0

        streaming_buffer = CacheAwareStreamingAudioBuffer(
            model=self.model,
            online_normalization=False,
            pad_and_drop_preencoded=False,
        )
        streaming_buffer.append_audio_file(str(wav_path), stream_id=-1)

        (
            cache_last_channel,
            cache_last_time,
            cache_last_channel_len,
        ) = self.model.encoder.get_initial_cache_state(batch_size=1)
        previous_hypotheses = None
        pred_out_stream = None
        virtual_realtime_ms = 0.0

        try:
            # Look one chunk ahead so the actual final iterator item is always
            # passed to conformer_stream_step(..., keep_all_outputs=True).
            # CacheAwareStreamingAudioBuffer.is_buffer_empty() can fail to mark
            # the final item for some 80 ms streams.
            buffer_iterator = iter(streaming_buffer)
            try:
                current_item = next(buffer_iterator)
            except StopIteration:
                return

            step_number = 0
            while True:
                try:
                    next_item = next(buffer_iterator)
                    is_final = False
                except StopIteration:
                    next_item = None
                    is_final = True

                chunk_audio, chunk_lengths = current_item
                chunk_audio = chunk_audio.to(dtype=torch.float32)

                _cuda_sync(self.device)
                started = time.perf_counter()
                with torch.inference_mode():
                    with _autocast_context(self.device):
                        (
                            pred_out_stream,
                            transcribed_texts,
                            cache_last_channel,
                            cache_last_time,
                            cache_last_channel_len,
                            previous_hypotheses,
                        ) = self.model.conformer_stream_step(
                            processed_signal=chunk_audio,
                            processed_signal_length=chunk_lengths,
                            cache_last_channel=cache_last_channel,
                            cache_last_time=cache_last_time,
                            cache_last_channel_len=cache_last_channel_len,
                            keep_all_outputs=is_final,
                            previous_hypotheses=previous_hypotheses,
                            previous_pred_out=pred_out_stream,
                            drop_extra_pre_encoded=_drop_extra_pre_encoded(
                                self.model, step_number
                            ),
                            return_transcription=True,
                        )
                _cuda_sync(self.device)
                compute_ms = (time.perf_counter() - started) * 1000.0

                audio_available_ms = (step_number + 1) * self.chunk_ms
                virtual_realtime_ms = (
                    max(virtual_realtime_ms, float(audio_available_ms)) + compute_ms
                )
                text = _extract_text(transcribed_texts).strip()

                yield StepResult(
                    text=text,
                    compute_ms=compute_ms,
                    audio_received_ms=min(float(audio_available_ms), duration_ms),
                    audio_available_ms=float(audio_available_ms),
                    virtual_realtime_ms=virtual_realtime_ms,
                    tail_after_audio_ms=max(0.0, virtual_realtime_ms - duration_ms),
                    step=step_number + 1,
                    is_final=is_final,
                )

                if is_final:
                    break

                current_item = next_item
                step_number += 1
        finally:
            streaming_buffer.reset_buffer()

class PCMStreamingSession:
    """Frame-based path retained for the FastRTC live microphone page."""

    def __init__(self, model: Any, chunk_ms: int) -> None:
        if chunk_ms not in CHUNK_TO_RIGHT_CONTEXT:
            raise ValueError(f"chunk_ms must be one of {SUPPORTED_CHUNKS}")

        import torch
        from omegaconf import OmegaConf

        self.torch = torch
        self.model = model
        self.chunk_ms = int(chunk_ms)
        self.chunk_samples = TARGET_SAMPLE_RATE * self.chunk_ms // 1000
        self.device = next(model.parameters()).device
        self.step_number = 0
        self.audio_samples_received = 0

        right_context = CHUNK_TO_RIGHT_CONTEXT[self.chunk_ms]
        model.encoder.set_default_att_context_size([ATTENTION_LEFT, right_context])

        cfg = copy.deepcopy(model._cfg)
        OmegaConf.set_struct(cfg.preprocessor, False)
        cfg.preprocessor.dither = 0.0
        cfg.preprocessor.pad_to = 0
        if hasattr(cfg.preprocessor, "normalize"):
            cfg.preprocessor.normalize = "None"
        self.preprocessor = model.from_config_dict(cfg.preprocessor).to(self.device)
        self.preprocessor.eval()

        (
            self.cache_last_channel,
            self.cache_last_time,
            self.cache_last_channel_len,
        ) = model.encoder.get_initial_cache_state(batch_size=1)

        cache_size = model.encoder.streaming_cfg.pre_encode_cache_size
        if isinstance(cache_size, (list, tuple)):
            cache_size = cache_size[1] if len(cache_size) > 1 else cache_size[0]
        features = int(model.cfg.preprocessor.features)
        self.cache_pre_encode = torch.zeros(
            (1, features, int(cache_size)),
            dtype=torch.float32,
            device=self.device,
        )
        self.previous_hypotheses = None
        self.previous_pred_out = None

    def process_pcm16(self, pcm16: np.ndarray, is_final: bool = False) -> StepResult:
        torch = self.torch
        pcm16 = np.asarray(pcm16, dtype=np.int16).reshape(-1)
        actual_samples = int(pcm16.size)
        if actual_samples == 0:
            pcm16 = np.zeros(self.chunk_samples, dtype=np.int16)
        elif actual_samples < self.chunk_samples:
            pcm16 = np.pad(pcm16, (0, self.chunk_samples - actual_samples))
        elif actual_samples > self.chunk_samples:
            raise ValueError(
                f"Chunk has {actual_samples} samples; expected at most {self.chunk_samples}."
            )

        self.audio_samples_received += actual_samples
        audio = (
            torch.from_numpy(pcm16.astype(np.float32) / 32768.0)
            .unsqueeze(0)
            .to(self.device)
        )
        audio_len = torch.tensor([audio.shape[-1]], dtype=torch.long, device=self.device)

        _cuda_sync(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=audio,
                length=audio_len,
            )
            processed_signal = torch.cat(
                [self.cache_pre_encode, processed_signal], dim=-1
            )
            processed_signal_length = (
                processed_signal_length + self.cache_pre_encode.shape[-1]
            )
            self.cache_pre_encode = processed_signal[
                :, :, -self.cache_pre_encode.shape[-1]:
            ].detach()

            (
                self.previous_pred_out,
                hypotheses,
                self.cache_last_channel,
                self.cache_last_time,
                self.cache_last_channel_len,
                self.previous_hypotheses,
                *_rest,
            ) = self.model.conformer_stream_step(
                processed_signal=processed_signal,
                processed_signal_length=processed_signal_length,
                cache_last_channel=self.cache_last_channel,
                cache_last_time=self.cache_last_time,
                cache_last_channel_len=self.cache_last_channel_len,
                keep_all_outputs=bool(is_final),
                previous_hypotheses=self.previous_hypotheses,
                previous_pred_out=self.previous_pred_out,
                drop_extra_pre_encoded=_drop_extra_pre_encoded(
                    self.model, self.step_number
                ),
                return_transcription=True,
            )
        _cuda_sync(self.device)
        compute_ms = (time.perf_counter() - started) * 1000.0
        self.step_number += 1
        audio_ms = self.audio_samples_received / TARGET_SAMPLE_RATE * 1000.0

        return StepResult(
            text=_extract_text(hypotheses).strip(),
            compute_ms=compute_ms,
            audio_received_ms=audio_ms,
            audio_available_ms=audio_ms,
            virtual_realtime_ms=audio_ms + compute_ms,
            tail_after_audio_ms=0.0,
            step=self.step_number,
            is_final=bool(is_final),
        )
