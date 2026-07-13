from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import tempfile
import time
from typing import Any, Iterator

import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .engine import (
    FileStreamingSession,
    PCMStreamingSession,
    REGISTRY,
    SUPPORTED_CHUNKS,
    TARGET_SAMPLE_RATE,
)

LOGGER = logging.getLogger(__name__)

app = FastAPI(title="Nemotron Streaming ASR Backend", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "version": "evaluator-aligned-file-streaming-v1",
        "loaded_models": REGISTRY.loaded(),
        "device": REGISTRY.device_name,
        "supported_chunks_ms": list(SUPPORTED_CHUNKS),
        "sample_rate": TARGET_SAMPLE_RATE,
    }


async def _send_error(ws: WebSocket, message: str) -> None:
    await ws.send_json({"type": "error", "message": message})


def _next_item(iterator: Iterator[Any]) -> tuple[bool, Any | None]:
    try:
        return True, next(iterator)
    except StopIteration:
        return False, None


async def _run_file_mode(
    ws: WebSocket,
    session: FileStreamingSession,
    started_at: float,
) -> None:
    while True:
        payload = await ws.receive_json()
        message_type = payload.get("type")
        if message_type == "ping":
            await ws.send_json({
                "type": "pong",
                "id": payload.get("id"),
                "at": time.time(),
            })
            continue
        if message_type != "audio_file":
            raise ValueError("File mode expects ping or audio_file messages.")

        pcm_bytes = base64.b64decode(payload.get("pcm16_b64", ""))
        pcm16 = np.frombuffer(pcm_bytes, dtype="<i2").copy()
        if pcm16.size == 0:
            raise ValueError("Uploaded WAV contains no audio samples.")

        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                temporary_path = handle.name
            sf.write(
                temporary_path,
                pcm16,
                TARGET_SAMPLE_RATE,
                subtype="PCM_16",
                format="WAV",
            )

            iterator = iter(session.process_wav(temporary_path))
            first_partial_ms: float | None = None
            last_text = ""

            while True:
                has_item, result = await asyncio.to_thread(_next_item, iterator)
                if not has_item:
                    break
                if result.text:
                    last_text = result.text
                    if first_partial_ms is None:
                        first_partial_ms = result.virtual_realtime_ms

                await ws.send_json({
                    "type": "final" if result.is_final else "partial",
                    "sequence": result.step,
                    "text": result.text or last_text,
                    "compute_ms": result.compute_ms,
                    "audio_received_ms": result.audio_received_ms,
                    "audio_available_ms": result.audio_available_ms,
                    "virtual_realtime_ms": result.virtual_realtime_ms,
                    "tail_after_audio_ms": result.tail_after_audio_ms,
                    "finalization_ms": (
                        result.tail_after_audio_ms if result.is_final else None
                    ),
                    "server_elapsed_ms": (time.perf_counter() - started_at) * 1000.0,
                    "first_partial_ms": first_partial_ms,
                    "is_final": result.is_final,
                })
                if result.is_final:
                    break
        finally:
            if temporary_path:
                try:
                    os.unlink(temporary_path)
                except FileNotFoundError:
                    pass
        break


async def _run_pcm_mode(
    ws: WebSocket,
    session: PCMStreamingSession,
    started_at: float,
) -> None:
    first_nonempty_at: float | None = None
    last_text = ""
    while True:
        payload = await ws.receive_json()
        message_type = payload.get("type")
        if message_type == "ping":
            await ws.send_json({"type": "pong", "id": payload.get("id"), "at": time.time()})
            continue
        if message_type == "audio":
            pcm16 = np.frombuffer(
                base64.b64decode(payload.get("pcm16_b64", "")),
                dtype="<i2",
            ).copy()
            is_final = bool(payload.get("is_final", False))
            result = await asyncio.to_thread(session.process_pcm16, pcm16, is_final)
        elif message_type == "finish":
            result = await asyncio.to_thread(
                session.process_pcm16,
                np.zeros(session.chunk_samples, dtype=np.int16),
                True,
            )
            is_final = True
        else:
            raise ValueError(f"Unknown message type: {message_type}")

        now = time.perf_counter()
        if result.text and first_nonempty_at is None:
            first_nonempty_at = now
        if result.text:
            last_text = result.text
        await ws.send_json({
            "type": "final" if is_final else "partial",
            "sequence": int(payload.get("sequence", result.step)),
            "text": result.text or last_text,
            "compute_ms": result.compute_ms,
            "audio_received_ms": result.audio_received_ms,
            "audio_available_ms": result.audio_available_ms,
            "virtual_realtime_ms": result.virtual_realtime_ms,
            "tail_after_audio_ms": result.tail_after_audio_ms,
            "finalization_ms": result.tail_after_audio_ms if is_final else None,
            "server_elapsed_ms": (now - started_at) * 1000.0,
            "first_partial_ms": (
                (first_nonempty_at - started_at) * 1000.0
                if first_nonempty_at is not None else None
            ),
            "is_final": is_final,
        })
        if is_final:
            break


@app.websocket("/v1/ws/transcribe")
async def transcribe_socket(ws: WebSocket) -> None:
    await ws.accept()
    model_lock = None
    lock_acquired = False
    try:
        start = await ws.receive_json()
        if start.get("type") != "start":
            raise ValueError("First message must have type='start'.")

        model_key = str(start.get("model", "base"))
        chunk_ms = int(start.get("chunk_ms", 560))
        sample_rate = int(start.get("sample_rate", TARGET_SAMPLE_RATE))
        mode = str(start.get("mode", "pcm"))
        if sample_rate != TARGET_SAMPLE_RATE:
            raise ValueError("Backend accepts 16 kHz mono PCM16 only.")
        if chunk_ms not in SUPPORTED_CHUNKS:
            raise ValueError(f"Unsupported chunk_ms={chunk_ms}.")
        if mode not in {"file", "pcm"}:
            raise ValueError("mode must be 'file' or 'pcm'.")

        model_lock = REGISTRY.session_locks[model_key]
        acquired = await asyncio.to_thread(model_lock.acquire, True, 300)
        if not acquired:
            raise TimeoutError("Timed out waiting for the selected model.")
        lock_acquired = True

        model = await asyncio.to_thread(REGISTRY.get, model_key)
        if mode == "file":
            session = await asyncio.to_thread(FileStreamingSession, model, chunk_ms)
        else:
            session = await asyncio.to_thread(PCMStreamingSession, model, chunk_ms)
        started_at = time.perf_counter()

        await ws.send_json({
            "type": "ready",
            "mode": mode,
            "model": model_key,
            "chunk_ms": chunk_ms,
            "chunk_samples": session.chunk_samples,
            "sample_rate": TARGET_SAMPLE_RATE,
        })

        if mode == "file":
            await _run_file_mode(ws, session, started_at)
        else:
            await _run_pcm_mode(ws, session, started_at)

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        LOGGER.exception("WebSocket transcription failed")
        try:
            await _send_error(ws, f"{type(exc).__name__}: {exc}")
        except Exception:
            pass
    finally:
        if lock_acquired and model_lock is not None:
            model_lock.release()
        try:
            await ws.close()
        except Exception:
            pass
