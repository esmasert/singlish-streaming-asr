from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from typing import Any

import gradio as gr
import numpy as np
import websockets
from fastrtc import AdditionalOutputs, AsyncStreamHandler, Stream, wait_for_item

TARGET_SR = 16_000
DEFAULT_WS = os.getenv("NEMOTRON_WS_URL", "ws://127.0.0.1:8011/v1/ws/transcribe")


class RemoteNemotronHandler(AsyncStreamHandler):
    """Forwards real microphone frames to the cache-aware GPU WebSocket session."""

    def __init__(self) -> None:
        super().__init__(
            expected_layout="mono",
            input_sample_rate=TARGET_SR,
            output_sample_rate=TARGET_SR,
        )
        self.audio_queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
        self.output_queue: asyncio.Queue[Any] = asyncio.Queue()
        self.buffer = np.zeros(0, dtype="<i2")
        self.quit = asyncio.Event()
        self.sequence = 0
        self.started_at = 0.0
        self.latest_text = ""
        self.chunk_samples = TARGET_SR * 560 // 1000
        self.ws = None

    def copy(self) -> "RemoteNemotronHandler":
        return RemoteNemotronHandler()

    async def start_up(self) -> None:
        await self.wait_for_args()
        backend_url = str(self.latest_args[1] or DEFAULT_WS).strip()
        model_key = str(self.latest_args[2] or "finetuned")
        chunk_ms = int(self.latest_args[3] or 560)
        self.chunk_samples = TARGET_SR * chunk_ms // 1000
        self.started_at = time.perf_counter()

        try:
            async with websockets.connect(
                backend_url,
                open_timeout=300,
                close_timeout=5,
                max_size=8 * 1024 * 1024,
            ) as ws:
                self.ws = ws
                await ws.send(json.dumps({
                    "type": "start",
                    "model": model_key,
                    "chunk_ms": chunk_ms,
                    "sample_rate": TARGET_SR,
                    "audio_name": "live_microphone",
                }))
                ready = json.loads(await ws.recv())
                if ready.get("type") == "error":
                    raise RuntimeError(ready.get("message"))
                await self.output_queue.put(AdditionalOutputs(
                    "*Microphone connected. Start speaking…*",
                    {"state": "ready", "model": model_key, "chunk_ms": chunk_ms},
                ))

                while not self.quit.is_set():
                    chunk = await self.audio_queue.get()
                    if chunk is None:
                        break
                    self.sequence += 1
                    sent_at = time.perf_counter()
                    await ws.send(json.dumps({
                        "type": "audio",
                        "sequence": self.sequence,
                        "is_final": False,
                        "pcm16_b64": base64.b64encode(chunk.tobytes()).decode("ascii"),
                    }))
                    response = json.loads(await ws.recv())
                    if response.get("type") == "error":
                        raise RuntimeError(response.get("message"))
                    if response.get("text"):
                        self.latest_text = response["text"]
                    rtt = (time.perf_counter() - sent_at) * 1000.0
                    await self.output_queue.put(AdditionalOutputs(
                        self.latest_text or "*Listening…*",
                        {
                            "state": "streaming",
                            "step": self.sequence,
                            "audio_received_ms": response.get("audio_received_ms"),
                            "compute_ms": response.get("compute_ms"),
                            "round_trip_ms": rtt,
                            "first_partial_ms": response.get("first_partial_ms"),
                        },
                    ))

                # Flush any remaining microphone audio as the final chunk.
                if self.buffer.size:
                    self.sequence += 1
                    await ws.send(json.dumps({
                        "type": "audio",
                        "sequence": self.sequence,
                        "is_final": True,
                        "pcm16_b64": base64.b64encode(self.buffer.tobytes()).decode("ascii"),
                    }))
                else:
                    await ws.send(json.dumps({"type": "finish", "sequence": self.sequence + 1}))
                response = json.loads(await ws.recv())
                if response.get("text"):
                    self.latest_text = response["text"]
                await self.output_queue.put(AdditionalOutputs(
                    self.latest_text or "*No speech detected.*",
                    {
                        "state": "final",
                        "steps": self.sequence,
                        "elapsed_ms": (time.perf_counter() - self.started_at) * 1000.0,
                        "compute_ms": response.get("compute_ms"),
                        "first_partial_ms": response.get("first_partial_ms"),
                    },
                ))
        except Exception as exc:
            await self.output_queue.put(AdditionalOutputs(
                self.latest_text,
                {"state": "error", "message": f"{type(exc).__name__}: {exc}"},
            ))

    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        _sample_rate, array = frame
        array = np.asarray(array).reshape(-1)
        if np.issubdtype(array.dtype, np.floating):
            array = (np.clip(array, -1.0, 1.0) * 32767.0).round().astype("<i2")
        else:
            array = array.astype("<i2", copy=False)
        self.buffer = np.concatenate([self.buffer, array])
        while self.buffer.size >= self.chunk_samples:
            chunk = self.buffer[: self.chunk_samples].copy()
            self.buffer = self.buffer[self.chunk_samples :]
            await self.audio_queue.put(chunk)

    async def emit(self):
        return await wait_for_item(self.output_queue, 0.05)

    async def shutdown(self) -> None:
        if not self.quit.is_set():
            self.quit.set()
            await self.audio_queue.put(None)


def build_stream() -> Stream:
    backend = gr.Textbox(label="GPU WebSocket", value=DEFAULT_WS)
    model = gr.Dropdown(
        label="Model",
        choices=[("Nemotron Base", "base"), ("Fine-tuned Nemotron", "finetuned")],
        value="finetuned",
    )
    chunk = gr.Dropdown(label="Chunk (ms)", choices=[80, 160, 560, 1120], value=560)
    transcript = gr.Markdown("*Press Record and speak.*")
    stats = gr.JSON(label="Live streaming trace")

    return Stream(
        handler=RemoteNemotronHandler(),
        modality="audio",
        mode="send",
        additional_inputs=[backend, model, chunk],
        additional_outputs=[transcript, stats],
        additional_outputs_handler=lambda _old_text, _old_stats, new_text, new_stats: (
            new_text,
            new_stats,
        ),
        concurrency_limit=1,
        time_limit=300,
        ui_args={
            "title": "Nemotron Live Microphone",
            "subtitle": "FastRTC WebRTC → cache-aware Nemotron GPU stream",
            "pulse_color": "rgb(228, 140, 88)",
            "icon_button_color": "rgb(217, 122, 67)",
        },
    )


def main() -> None:
    stream = build_stream()
    stream.ui.launch(
        server_name="127.0.0.1",
        server_port=int(os.getenv("FASTRTC_PORT", "7861")),
        inbrowser=False,
    )


if __name__ == "__main__":
    main()
