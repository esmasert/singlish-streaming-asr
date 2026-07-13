from __future__ import annotations

import html
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import gradio as gr
import pandas as pd

from .client import stream_wav
from .ground_truth import GroundTruthIndex
from .metrics import latency_summary, text_metrics, word_error_alignment

DEFAULT_GT = os.getenv("GROUND_TRUTH_JSONL", "")
DEFAULT_WS = os.getenv("NEMOTRON_WS_URL", "ws://127.0.0.1:8011/v1/ws/transcribe")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", Path(__file__).resolve().parents[1] / "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STEP_COLUMNS = [
    "step", "audio_received_ms", "audio_available_ms", "compute_ms",
    "round_trip_ms", "server_elapsed_ms", "client_elapsed_ms",
    "first_partial_ms", "tail_after_audio_ms", "is_final", "text",
]
RESULT_COLUMNS = [
    "dataset", "file", "model", "chunk_ms", "duration_s", "transcript",
    "reference", "wer_pct", "cer_pct", "wer_corr_pct", "cer_corr_pct",
    "backend_ready_ms", "first_partial_ms", "finalization_ms", "compute_p95_ms",
    "round_trip_p95_ms", "rtf_compute", "rtf_wall", "status",
]

CSS = r"""
:root {
  --bg-main:#fff3e8;
  --bg-soft:#fde7d3;
  --bg-soft-2:#fff8f2;
  --panel:#fffaf6;
  --panel-2:#fff5ed;
  --border:#e7b88f;
  --accent:#d97a43;
  --accent-2:#f2a56e;
  --accent-3:#fff0e2;
  --ink:#3a2418;
  --muted:#6f5443;
  --ok:#2f7d4d;
  --warn:#9a6500;
  --err:#b33a3a;
}

.gradio-container {
  max-width: 1500px !important;
  margin: 0 auto !important;
  background:
    radial-gradient(circle at 10% 10%, rgba(255, 210, 170, 0.40), transparent 30%),
    radial-gradient(circle at 90% 15%, rgba(255, 190, 145, 0.28), transparent 26%),
    linear-gradient(145deg, #fff6ee 0%, #fde7d3 55%, #fff3e8 100%) !important;
  color: var(--ink) !important;
}

.hero {
  padding: 28px 30px;
  border: 1px solid var(--border);
  border-radius: 24px;
  background: linear-gradient(135deg, #fff7f0 0%, #fde3cc 100%);
  box-shadow: 0 18px 45px rgba(166, 105, 53, 0.12);
  margin-bottom: 18px;
  overflow: hidden;
  position: relative;
}

.hero:after {
  content:'';
  position:absolute;
  width:250px;
  height:250px;
  right:-70px;
  top:-110px;
  border-radius:999px;
  background: rgba(242, 165, 110, 0.18);
  filter: blur(2px);
}

.hero h1 {
  font-size: clamp(30px,4vw,54px);
  margin: 0 0 6px;
  letter-spacing: -1.4px;
  color: var(--ink) !important;
}

.hero p {
  color: var(--muted) !important;
  font-size: 16px;
  max-width: 900px;
  margin: 0;
}

.badge {
  display:inline-flex;
  align-items:center;
  gap:7px;
  padding:7px 11px;
  border-radius:999px;
  border:1px solid #efc49d;
  background:#fff4e9;
  color: var(--ink);
  font-size:12px;
  margin:12px 6px 0 0;
}

.dot {
  width:8px;
  height:8px;
  border-radius:50%;
  background:#e28a52;
  box-shadow:0 0 10px rgba(226, 138, 82, 0.5);
}

.panel {
  border:1px solid var(--border) !important;
  border-radius:20px !important;
  background:var(--panel) !important;
  box-shadow:0 12px 30px rgba(140, 97, 58, 0.10);
}

.transcript {
  min-height:210px;
  border-radius:18px;
  padding:22px !important;
  background: linear-gradient(145deg, #fffaf6, #fff3ea);
  border:1px solid #ecc7a7;
  color: var(--ink) !important;
  font-size:23px;
  line-height:1.5;
}

.metric-grid {
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(135px,1fr));
  gap:10px;
}

.metric {
  padding:14px 15px;
  border-radius:16px;
  background: var(--panel-2);
  border:1px solid #edd0b4;
  min-height:84px;
}

.metric .label {
  color:#7e5c46;
  font-size:11px;
  text-transform:uppercase;
  letter-spacing:.08em;
}

.metric .value {
  color:var(--ink);
  font-size:23px;
  font-weight:750;
  margin-top:8px;
}

.metric .sub {
  color:#8d6a52;
  font-size:11px;
  margin-top:3px;
}


.wer-analysis {
  margin-top:12px;
  padding:18px 20px;
  border:1px solid #e7b88f;
  border-radius:18px;
  background:linear-gradient(145deg, #fffaf6, #fff2e7);
  box-shadow:0 10px 24px rgba(140, 97, 58, 0.08);
}

.wer-analysis-head {
  display:flex;
  flex-wrap:wrap;
  align-items:flex-start;
  justify-content:space-between;
  gap:10px;
  margin-bottom:14px;
}

.wer-analysis-title {
  font-size: 16px;
  font-weight:800;
  color:var(--ink) !important;
}

.wer-analysis-note {
  margin-top:3px;
  color:#7e5c46 !important;
  font-size: 11px;
}

.wer-summary {
  display:flex;
  flex-wrap:wrap;
  gap:7px;
}

.wer-chip {
  display:inline-flex;
  align-items:center;
  gap:5px;
  padding: 5px 9px;
  border-radius:999px;
  border:1px solid #ead1bb;
  background:#fffaf6;
  font-size: 11px;
  font-weight:700;
  color:var(--ink) !important;
}

.wer-alignment {
  font-size: 15px;
  line-height: 1.8;
  word-break:break-word;
}

.wer-token {
  display: inline;
  align-items:baseline;
  gap: 2px;
  padding: 2px 4px;
  margin: 1px 1px;
  border-radius:7px;
  white-space:nowrap;
}

.gradio-container .wer-ok {
  color:var(--ink) !important;
}

.gradio-container .wer-sub {
  color:#247447 !important;
  background:#eaf7ef;
  border:1px solid #bfe2cb;
  font-weight:750;
}

.gradio-container .wer-sub-ref {
  color:#7a5140 !important;
  font-size: .82em;
  font-weight:600;
}

.gradio-container .wer-del {
  color:#b33a3a !important;
  background:#fdeaea;
  border:1px solid #f1bcbc;
  text-decoration:line-through;
  text-decoration-thickness:2px;
}

.gradio-container .wer-ins {
  color:#315fc4 !important;
  background:#eaf1ff;
  border:1px solid #bfd0f7;
  font-weight:750;
}

.gradio-container .wer-placeholder {
  color:#80604d !important;
  font-size:13px;
  padding:4px 0;
}

.wer-legend {
  display:flex;
  flex-wrap:wrap;
  gap:10px;
  margin-top:14px;
  padding-top:12px;
  border-top:1px dashed #e8c7aa;
  font-size: 11px;
}

.wer-legend-item {
  display:inline-flex;
  align-items:center;
  gap:6px;
  color:#6f5443 !important;
}

.wer-legend-dot {
  width:9px;
  height:9px;
  border-radius:3px;
}

.wer-dot-ok { background:#3a2418; }
.wer-dot-sub { background:#3c9a63; }
.wer-dot-del { background:#d95b5b; }
.wer-dot-ins { background:#527fd7; }
.status-ok { color: var(--ok); }
.status-warn { color: var(--warn); }
.status-err { color: var(--err); }

label, .gr-form label, .gr-block label, .gr-markdown, .gradio-container p,
.gradio-container h1, .gradio-container h2, .gradio-container h3,
.gradio-container h4, .gradio-container span, .gradio-container div {
  color: var(--ink);
}

input, textarea, select {
  background: #fffdfb !important;
  color: var(--ink) !important;
  border: 1px solid var(--border) !important;
}

.gr-button, .gr-button-primary {
  background: linear-gradient(135deg, #e48c58, #f2a56e) !important;
  color: #ffffff !important;
  border: none !important;
}

.gr-button:hover, .gr-button-primary:hover {
  filter: brightness(0.98);
}

table, .dataframe, .gr-dataframe {
  color: var(--ink) !important;
  background: #fffaf6 !important;
}


#single-wer-analysis {
  position: relative !important;
  left: -130px !important;
  width: calc(100% + 130px) !important;
  max-width: none !important;
  overflow: visible !important;
}

#single-wer-analysis .wer-analysis {
  width: 100% !important;
  margin-left: 0 !important;
  box-sizing: border-box !important;
}

footer { display:none !important; }
"""


def _fmt(value: Any, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def _metrics_html(text_result=None, latency: dict | None = None, partial: bool = False) -> str:
    latency = latency or {}
    cards: list[tuple[str, str, str]] = []
    if text_result is not None:
        cards += [
            ("WER", _fmt(text_result.wer * 100, 2, "%"), f"{text_result.word_errors} word errors"),
            ("WER Corr", _fmt(text_result.wer_corr * 100, 2, "%"), "word hits / reference words"),
            ("CER", _fmt(text_result.cer * 100, 2, "%"), f"{text_result.char_errors} char errors"),
            ("CER Corr", _fmt(text_result.cer_corr * 100, 2, "%"), "character hits / reference characters"),
        ]
    cards += [
        ("Backend ready", _fmt(latency.get("backend_ready_ms"), 1, " ms"), "connect + session setup"),
        ("First partial", _fmt(latency.get("first_partial_ms"), 1, " ms"), "first non-empty hypothesis"),
        ("Finalization", _fmt(latency.get("finalization_ms"), 1, " ms"), "last chunk → final"),
        ("RTF compute", _fmt(latency.get("rtf_compute"), 3), "Device compute / audio"),
        ("RTF wall", _fmt(latency.get("rtf_wall"), 3), "end-to-end / audio"),
        ("Compute p95", _fmt(latency.get("compute_p95_ms"), 1, " ms"), "per streaming step"),
        ("RTT p95", _fmt(latency.get("round_trip_p95_ms"), 1, " ms"), "Client ↔ inference backend"),
    ]
    if partial:
        cards.insert(0, ("Streaming", "LIVE", "cache-aware partials"))
    return '<div class="metric-grid">' + ''.join(
        f'<div class="metric"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>'
        for label, value, sub in cards
    ) + '</div>'



def _coloured_analysis_html(
    reference: str | None = None,
    hypothesis: str | None = None,
    text_result=None,
    partial: bool = False,
) -> str:
    if partial:
        body = (
            '<div class="wer-placeholder">'
            'Streaming in progress — coloured WER analysis will appear after the final transcript.'
            '</div>'
        )
        summary = ""
    elif not reference or not str(reference).strip():
        body = (
            '<div class="wer-placeholder">'
            'No ground truth is available, so word-level error analysis cannot be calculated.'
            '</div>'
        )
        summary = ""
    elif hypothesis is None:
        body = (
            '<div class="wer-placeholder">'
            'Run streaming to generate the coloured word-error alignment.'
            '</div>'
        )
        summary = ""
    else:
        operations = word_error_alignment(reference, hypothesis)
        rendered: list[str] = []

        for operation in operations:
            op_type = operation["type"]
            ref_word = html.escape(operation.get("reference", ""))
            hyp_word = html.escape(operation.get("hypothesis", ""))

            if op_type == "equal":
                rendered.append(
                    f'<span class="wer-token wer-ok">{hyp_word}</span>'
                )
            elif op_type == "substitute":
                rendered.append(
                    '<span class="wer-token wer-sub" '
                    'title="Substitution: prediction (reference)">'
                    f'{hyp_word}<span class="wer-sub-ref">({ref_word})</span>'
                    '</span>'
                )
            elif op_type == "delete":
                rendered.append(
                    f'<span class="wer-token wer-del" '
                    f'title="Deletion">{ref_word}</span>'
                )
            elif op_type == "insert":
                rendered.append(
                    f'<span class="wer-token wer-ins" '
                    f'title="Insertion">{hyp_word}</span>'
                )

        body = (
            '<div class="wer-alignment">'
            + (
                " ".join(rendered)
                if rendered
                else '<span class="wer-placeholder">Empty transcript.</span>'
            )
            + '</div>'
        )

        if text_result is not None:
            summary = (
                '<div class="wer-summary">'
                f'<span class="wer-chip">Correct&nbsp; {text_result.word_hits}</span>'
                f'<span class="wer-chip">Sub&nbsp; {text_result.word_substitutions}</span>'
                f'<span class="wer-chip">Del&nbsp; {text_result.word_deletions}</span>'
                f'<span class="wer-chip">Ins&nbsp; {text_result.word_insertions}</span>'
                f'<span class="wer-chip">WER&nbsp; {text_result.wer * 100:.2f}%</span>'
                '</div>'
            )
        else:
            summary = ""

    return (
        '<section class="wer-analysis">'
        '<div class="wer-analysis-head">'
        '<div><div class="wer-analysis-title">Coloured WER Analysis</div>'
        '<div class="wer-analysis-note">'
        'Normalized word alignment used by the WER calculation'
        '</div></div>'
        f'{summary}'
        '</div>'
        f'{body}'
        '<div class="wer-legend">'
        '<span class="wer-legend-item">'
        '<span class="wer-legend-dot wer-dot-ok"></span>Correct</span>'
        '<span class="wer-legend-item">'
        '<span class="wer-legend-dot wer-dot-sub"></span>'
        'Substitution: prediction (reference)</span>'
        '<span class="wer-legend-item">'
        '<span class="wer-legend-dot wer-dot-del"></span>Deletion</span>'
        '<span class="wer-legend-item">'
        '<span class="wer-legend-dot wer-dot-ins"></span>Insertion</span>'
        '</div>'
        '</section>'
    )


def _status(message: str, kind: str = "ok") -> str:
    icon = {"ok": "●", "warn": "◆", "err": "✕"}.get(kind, "●")
    return f'<div class="status-{kind}"><b>{icon}</b> {message}</div>'


def _get_gt(path: str) -> GroundTruthIndex:
    return GroundTruthIndex(path)


def lookup_reference(audio_path: str | None, gt_path: str):
    if not audio_path:
        return "No file selected", "", _status(
            "Ground truth will be matched automatically after a WAV is selected.",
            "warn",
        )

    filename = Path(audio_path).name
    try:
        record = _get_gt(gt_path).lookup(audio_path)
    except Exception as exc:
        return filename, "", _status(str(exc), "err")

    if record is None:
        return filename, "", _status(f"Ground truth not found for: {filename}", "warn")

    return (
        filename,
        record.text,
        _status(
            f"Ground truth matched · {record.dataset_name} · "
            f"{Path(record.audio_path).name}"
        ),
    )


def _write_json(payload: dict, prefix: str) -> str:
    filename = OUTPUT_DIR / f"{prefix}_{int(time.time() * 1000)}.json"
    filename.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(filename)


def run_single(
    wav_path: str | None,
    reference: str,
    backend_url: str,
    model_key: str,
    chunk_ms: int,
    realtime: bool,
):
    if not wav_path:
        yield "", reference, _metrics_html(), _coloured_analysis_html(reference), pd.DataFrame(columns=STEP_COLUMNS), _status("Upload a WAV first.", "err"), None
        return

    step_rows: list[dict] = []
    final_text = ""
    duration_s = 0.0
    stream_started: float | None = None
    backend_ready_ms: float | None = None
    yield "*Preparing the model…*", reference, _metrics_html(partial=True), _coloured_analysis_html(reference, partial=True), pd.DataFrame(columns=STEP_COLUMNS), _status("Establishing the streaming connection…", "warn"), None

    try:
        for event in stream_wav(
            wav_path=wav_path,
            backend_ws_url=backend_url.strip(),
            model_key=model_key,
            chunk_ms=int(chunk_ms),
            pace_realtime=bool(realtime),
        ):
            if event["type"] == "ready":
                duration_s = float(event["audio_duration_s"])
                backend_ready_ms = event.get("backend_ready_ms")
                stream_started = time.perf_counter()
                continue
            final_text = event.get("text", final_text) or final_text
            row = {column: event.get(column) for column in STEP_COLUMNS}
            row["step"] = event.get("sequence")
            step_rows.append(row)
            elapsed = 0.0 if stream_started is None else time.perf_counter() - stream_started
            live_latency = latency_summary(step_rows, duration_s, elapsed)
            live_latency["backend_ready_ms"] = backend_ready_ms
            yield (
                final_text or "*No speech detected yet…*",
                reference,
                _metrics_html(latency=live_latency, partial=not event.get("is_final")),
                _coloured_analysis_html(reference, final_text, partial=True),
                pd.DataFrame(step_rows, columns=STEP_COLUMNS),
                _status(f"Chunk {len(step_rows)} processed · {event.get('audio_received_ms', 0):.0f} ms audio"),
                None,
            )

        wall_time_s = 0.0 if stream_started is None else time.perf_counter() - stream_started
        latency = latency_summary(step_rows, duration_s, wall_time_s)
        latency["backend_ready_ms"] = backend_ready_ms
        tm = text_metrics(reference, final_text)
        payload = {
            "file": str(wav_path),
            "model": model_key,
            "chunk_ms": int(chunk_ms),
            "reference": reference,
            "transcript": final_text,
            "text_metrics": tm.to_dict() if tm else None,
            "latency": latency,
            "steps": step_rows,
        }
        export_path = _write_json(payload, "single_result")
        gt_note = " · WER/CER calculated" if tm else " · no ground truth; latency only"
        yield (
            final_text or "*Empty transcript*",
            reference,
            _metrics_html(tm, latency),
            _coloured_analysis_html(reference, final_text, tm),
            pd.DataFrame(step_rows, columns=STEP_COLUMNS),
            _status(f"Completed{gt_note}"),
            export_path,
        )
    except Exception as exc:
        yield (
            final_text,
            reference,
            _metrics_html(),
            _coloured_analysis_html(reference, final_text),
            pd.DataFrame(step_rows, columns=STEP_COLUMNS),
            _status(f"{type(exc).__name__}: {exc}", "err"),
            None,
        )


def _aggregate_rows(rows: list[dict]) -> dict:
    valid = [r for r in rows if r.get("_text_metrics") is not None]
    ref_words = sum(r["_text_metrics"].reference_words for r in valid)
    word_errors = sum(r["_text_metrics"].word_errors for r in valid)
    ref_chars = sum(len(r["_text_metrics"].reference_normalized.replace(" ", "")) for r in valid)
    char_errors = sum(r["_text_metrics"].char_errors for r in valid)
    duration = sum(float(r.get("duration_s") or 0) for r in rows)
    compute = sum(float(r.get("_total_compute_s") or 0) for r in rows)
    wall = sum(float(r.get("_wall_time_s") or 0) for r in rows)
    return {
        "files": len(rows),
        "scored_files": len(valid),
        "wer": word_errors / ref_words if ref_words else None,
        "cer": char_errors / ref_chars if ref_chars else None,
        "rtf_compute": compute / duration if duration else None,
        "rtf_wall": wall / duration if duration else None,
        "audio_duration_s": duration,
    }


def _aggregate_html(rows: list[dict]) -> str:
    agg = _aggregate_rows(rows)
    pseudo = None
    if agg["wer"] is not None:
        class P: pass
        pseudo = P()
        pseudo.wer = agg["wer"]
        pseudo.cer = agg["cer"]
        pseudo.wer_corr = 1 - agg["wer"]
        pseudo.cer_corr = 1 - agg["cer"]
        pseudo.word_errors = sum(r["_text_metrics"].word_errors for r in rows if r.get("_text_metrics"))
        pseudo.char_errors = sum(r["_text_metrics"].char_errors for r in rows if r.get("_text_metrics"))
    latency = {"rtf_compute": agg["rtf_compute"], "rtf_wall": agg["rtf_wall"]}
    extra = (
        f'<div class="badge"><span class="dot"></span>{agg["files"]} files · '
        f'{agg["scored_files"]} scored · {_fmt(agg["audio_duration_s"],1," s")} audio</div>'
    )
    return extra + _metrics_html(pseudo, latency)


def run_folder(
    files: list[str] | None,
    gt_path: str,
    backend_url: str,
    model_key: str,
    chunk_ms: int,
    fast_mode: bool,
):
    if not files:
        yield "", _aggregate_html([]), pd.DataFrame(columns=RESULT_COLUMNS), _status("Select a WAV folder.", "err"), None, None
        return
    wavs = sorted(str(p) for p in files if str(p).lower().endswith(".wav"))
    if not wavs:
        yield "", _aggregate_html([]), pd.DataFrame(columns=RESULT_COLUMNS), _status("No WAV files were found in the selected folder.", "err"), None, None
        return

    try:
        gt = _get_gt(gt_path)
    except Exception as exc:
        yield "", _aggregate_html([]), pd.DataFrame(columns=RESULT_COLUMNS), _status(str(exc), "err"), None, None
        return

    rows: list[dict] = []
    for file_index, wav in enumerate(wavs, start=1):
        record = gt.lookup(wav)
        reference = record.text if record else ""
        dataset = record.dataset_name if record else "unknown"
        step_rows: list[dict] = []
        final_text = ""
        duration_s = 0.0
        started = time.perf_counter()
        stream_started: float | None = None
        backend_ready_ms: float | None = None
        try:
            for event in stream_wav(
                wav_path=wav,
                backend_ws_url=backend_url.strip(),
                model_key=model_key,
                chunk_ms=int(chunk_ms),
                pace_realtime=not bool(fast_mode),
            ):
                if event["type"] == "ready":
                    duration_s = float(event["audio_duration_s"])
                    backend_ready_ms = event.get("backend_ready_ms")
                    stream_started = time.perf_counter()
                    continue
                final_text = event.get("text", final_text) or final_text
                step_rows.append({
                    "compute_ms": event.get("compute_ms"),
                    "round_trip_ms": event.get("round_trip_ms"),
                    "first_partial_ms": event.get("first_partial_ms"),
                })
                yield (
                    f"**{file_index}/{len(wavs)} · {Path(wav).name}**\n\n{final_text or '*listening…*'}",
                    _aggregate_html(rows),
                    pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows], columns=RESULT_COLUMNS),
                    _status(f"Streaming {file_index}/{len(wavs)} · chunk {len(step_rows)}", "warn"),
                    None,
                    None,
                )
            wall = time.perf_counter() - (stream_started or started)
            latency = latency_summary(step_rows, duration_s, wall)
            latency["backend_ready_ms"] = backend_ready_ms
            tm = text_metrics(reference, final_text)
            row = {
                "dataset": dataset,
                "file": Path(wav).name,
                "model": model_key,
                "chunk_ms": int(chunk_ms),
                "duration_s": duration_s,
                "transcript": final_text,
                "reference": reference,
                "wer_pct": tm.wer * 100 if tm else None,
                "cer_pct": tm.cer * 100 if tm else None,
                "wer_corr_pct": tm.wer_corr * 100 if tm else None,
                "cer_corr_pct": tm.cer_corr * 100 if tm else None,
                "backend_ready_ms": latency.get("backend_ready_ms"),
                "first_partial_ms": latency.get("first_partial_ms"),
                "finalization_ms": latency.get("finalization_ms"),
                "compute_p95_ms": latency.get("compute_p95_ms"),
                "round_trip_p95_ms": latency.get("round_trip_p95_ms"),
                "rtf_compute": latency.get("rtf_compute"),
                "rtf_wall": latency.get("rtf_wall"),
                "status": "ok" if tm else "no_gt",
                "_text_metrics": tm,
                "_total_compute_s": sum(float(s.get("compute_ms") or 0) for s in step_rows) / 1000.0,
                "_wall_time_s": wall,
            }
            rows.append(row)
        except Exception as exc:
            rows.append({
                "dataset": dataset, "file": Path(wav).name, "model": model_key,
                "chunk_ms": int(chunk_ms), "duration_s": duration_s,
                "transcript": final_text, "reference": reference,
                "status": f"error: {exc}", "_text_metrics": None,
                "_total_compute_s": 0.0, "_wall_time_s": time.perf_counter() - started,
            })

    public_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
    df = pd.DataFrame(public_rows, columns=RESULT_COLUMNS)
    stamp = int(time.time() * 1000)
    csv_path = OUTPUT_DIR / f"folder_results_{stamp}.csv"
    json_path = OUTPUT_DIR / f"folder_results_{stamp}.json"
    df.to_csv(csv_path, index=False)
    payload = {
        "aggregate": _aggregate_rows(rows),
        "results": public_rows,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    yield (
        f"**Completed · {len(rows)} WAV files**",
        _aggregate_html(rows),
        df,
        _status("Folder evaluation completed."),
        str(csv_path),
        str(json_path),
    )


def build_demo() -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="cyan",
        neutral_hue="slate",
        radius_size="lg",
    )
    with gr.Blocks(theme=theme, css=CSS, title="Nemotron Streaming ASR Lab") as demo:
        gr.HTML(
            """
            <section class="hero">
              <h1>Nemotron Streaming ASR Lab</h1>
              <p>Compare the base and fine-tuned Nemotron models with true cache-aware streaming. Listen to the WAV while partial transcripts appear, and review WER, CER, accuracy, and latency metrics on the same screen.</p>
              <span class="badge"><span class="dot"></span>16 kHz mono stream</span>
              <span class="badge">Cache-aware RNNT</span><span class="badge">Mac UI + local inference backend</span>
            </section>
            """
        )

        with gr.Row():
            with gr.Column(scale=2, elem_classes="panel"):
                backend_url = gr.Textbox(label="Inference WebSocket", value=DEFAULT_WS)
            with gr.Column(scale=1, elem_classes="panel"):
                model = gr.Dropdown(
                    label="Model",
                    choices=[("Nemotron Base", "base"), ("Fine-tuned Nemotron", "finetuned")],
                    value="finetuned",
                )
            with gr.Column(scale=1, elem_classes="panel"):
                chunk = gr.Dropdown(
                    label="Streaming chunk",
                    choices=[80, 160, 560, 1120],
                    value=560,
                )
            with gr.Column(scale=2, elem_classes="panel"):
                gt_path = gr.Textbox(label="Ground-truth JSONL", value=DEFAULT_GT)

        with gr.Tabs():
            with gr.Tab("Single WAV · live demo"):
                with gr.Row():
                    with gr.Column(scale=5):
                        audio = gr.Audio(
                            label="Upload WAV · player remains available during streaming",
                            type="filepath",
                            sources=["upload"],
                            waveform_options=gr.WaveformOptions(
                                waveform_color="#7c89ff",
                                waveform_progress_color="#40e7d2",
                                show_recording_waveform=True,
                            ),
                        )
                        selected_file = gr.Textbox(
                            label="Selected file",
                            value="No file selected",
                            interactive=False,
                            container=True,
                        )
                        reference = gr.Textbox(
                            label="Ground truth / reference",
                            lines=4,
                            placeholder="Matched automatically by WAV filename; editable.",
                        )
                        with gr.Row():
                            realtime = gr.Radio(
                                label="Streaming mode",
                                choices=[
                                    ("Real-time playback (1×)", 1),
                                    ("Fast evaluation (no waiting)", 0),
                                ],
                                value=1,
                                info=(
                                    "Real-time mode follows the audio clock. Fast evaluation removes "
                                    "the waiting between chunks, but still uses cache-aware streaming."
                                ),
                            )
                            run_button = gr.Button("▶ Run streaming", variant="primary", size="lg")
                    with gr.Column(scale=7):
                        transcript = gr.Markdown("*Upload a WAV to begin.*", elem_classes="transcript")
                        single_metrics = gr.HTML(_metrics_html())
                        single_analysis = gr.HTML(
                            _coloured_analysis_html(),
                            elem_id="single-wer-analysis",
                        )
                        single_status = gr.HTML(_status("Ready."))
                steps = gr.Dataframe(
                    headers=STEP_COLUMNS,
                    datatype=["number", "number", "number", "number", "number", "number", "number", "number", "number", "bool", "str"],
                    label="Per-chunk streaming trace",
                    interactive=False,
                    wrap=True,
                )
                single_export = gr.File(label="Download JSON result")

                audio.change(
                    lookup_reference,
                    inputs=[audio, gt_path],
                    outputs=[selected_file, reference, single_status],
                    show_progress="hidden",
                )
                run_button.click(
                    run_single,
                    inputs=[audio, reference, backend_url, model, chunk, realtime],
                    outputs=[transcript, reference, single_metrics, single_analysis, steps, single_status, single_export],
                    concurrency_limit=1,
                )

            with gr.Tab("Folder · streaming evaluation"):
                gr.Markdown(
                    "Upload a directory. **Fast streaming evaluation** still sends sequential chunks and keeps model caches; it only removes wall-clock sleeping."
                )
                folder_files = gr.File(
                    label="WAV folder",
                    file_count="directory",
                    file_types=[".wav"],
                    type="filepath",
                )
                with gr.Row():
                    fast_mode = gr.Checkbox(label="Fast evaluation (no real-time sleep)", value=True)
                    folder_button = gr.Button("Evaluate folder", variant="primary", size="lg")
                folder_live = gr.Markdown("*No job running.*", elem_classes="transcript")
                folder_metrics = gr.HTML(_aggregate_html([]))
                folder_status = gr.HTML(_status("Ready."))
                results = gr.Dataframe(
                    headers=RESULT_COLUMNS,
                    label="Per-file results",
                    interactive=False,
                    wrap=True,
                )
                with gr.Row():
                    csv_export = gr.File(label="CSV export")
                    json_export = gr.File(label="JSON export")
                folder_button.click(
                    run_folder,
                    inputs=[folder_files, gt_path, backend_url, model, chunk, fast_mode],
                    outputs=[folder_live, folder_metrics, results, folder_status, csv_export, json_export],
                    concurrency_limit=1,
                )

        gr.Markdown(
            "**Metric definitions:** normalized lowercase/punctuation-stripped WER; CER excludes spaces; "
            "WER Corr = word hits / reference words; CER Corr = character hits / reference characters; compute RTF excludes network and real-time sleeping; wall RTF includes both."
        )
    return demo


def main() -> None:
    demo = build_demo()
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=int(os.getenv("GRADIO_PORT", "7860")),
        show_api=False,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
