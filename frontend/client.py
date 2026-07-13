from __future__ import annotations

import base64
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import numpy as np
import soundfile as sf
import websocket
from scipy.signal import resample_poly

TARGET_SR = 16_000


@dataclass
class AudioData:
    pcm16: np.ndarray
    duration_s: float
    source_sample_rate: int


def load_audio(path: str | Path) -> AudioData:
    audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
    audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        divisor = math.gcd(int(sr), TARGET_SR)
        audio = resample_poly(audio, TARGET_SR // divisor, int(sr) // divisor)
    audio = np.nan_to_num(audio)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).round().astype("<i2")
    return AudioData(
        pcm16=pcm16,
        duration_s=len(pcm16) / TARGET_SR,
        source_sample_rate=int(sr),
    )


def _json_receive(ws) -> dict:
    raw = ws.recv()
    if raw in (None, "", b""):
        raise RuntimeError(
            "Inference backend closed the WebSocket before sending a final result."
        )
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Inference backend returned invalid JSON: {raw[:200]!r}"
        ) from exc
    if payload.get("type") == "error":
        raise RuntimeError(payload.get("message", "Unknown backend error"))
    return payload


def _measure_network_rtt(ws, count: int = 5) -> list[float]:
    values: list[float] = []
    for index in range(count):
        started = time.perf_counter()
        ws.send(json.dumps({"type": "ping", "id": index}))
        response = _json_receive(ws)
        if response.get("type") != "pong":
            raise RuntimeError(f"Unexpected ping response: {response}")
        values.append((time.perf_counter() - started) * 1000.0)
    return values


def stream_wav(
    *,
    wav_path: str | Path,
    backend_ws_url: str,
    model_key: str,
    chunk_ms: int,
    pace_realtime: bool,
    connect_timeout_s: float = 300.0,
) -> Generator[dict, None, None]:
    audio = load_audio(wav_path)
    connection_started = time.perf_counter()
    ws = websocket.create_connection(
        backend_ws_url,
        timeout=connect_timeout_s,
        enable_multithread=True,
    )
    try:
        ws.send(json.dumps({
            "type": "start",
            "mode": "file",
            "model": model_key,
            "chunk_ms": int(chunk_ms),
            "sample_rate": TARGET_SR,
            "audio_name": Path(wav_path).name,
        }))
        ready = _json_receive(ws)
        if ready.get("type") != "ready":
            raise RuntimeError(f"Unexpected backend response: {ready}")

        network_rtts = _measure_network_rtt(ws)
        sorted_rtts = sorted(network_rtts)
        rtt_p95 = sorted_rtts[min(len(sorted_rtts) - 1, round(0.95 * (len(sorted_rtts) - 1)))]
        stream_started = time.perf_counter()
        yield {
            "type": "ready",
            "audio_duration_s": audio.duration_s,
            "source_sample_rate": audio.source_sample_rate,
            "backend_ready_ms": (stream_started - connection_started) * 1000.0,
            "network_rtt_ms": statistics.median(network_rtts),
            "network_rtt_p95_ms": rtt_p95,
        }

        ws.send(json.dumps({
            "type": "audio_file",
            "pcm16_b64": base64.b64encode(audio.pcm16.tobytes()).decode("ascii"),
            "duration_s": audio.duration_s,
            "audio_name": Path(wav_path).name,
        }))

        while True:
            response = _json_receive(ws)
            if response.get("type") not in {"partial", "final"}:
                continue

            if pace_realtime:
                target_elapsed_s = float(
                    response.get("audio_available_ms")
                    or response.get("audio_received_ms")
                    or 0.0
                ) / 1000.0
                delay = target_elapsed_s - (time.perf_counter() - stream_started)
                if delay > 0:
                    time.sleep(delay)

            received_at = time.perf_counter()
            response["round_trip_ms"] = rtt_p95
            response["client_elapsed_ms"] = (
                received_at - stream_started
            ) * 1000.0
            response["audio_duration_s"] = audio.duration_s
            response["source_sample_rate"] = audio.source_sample_rate
            yield response
            if response.get("type") == "final" or response.get("is_final"):
                break
    finally:
        try:
            ws.close()
        except Exception:
            pass
