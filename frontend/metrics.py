from __future__ import annotations

import math
import re
import statistics
import unicodedata
from dataclasses import asdict, dataclass
from typing import Iterable

import jiwer


def normalize_text(text: str | None) -> str:
    """Exactly matches the checkpoint evaluator's normalized WER text."""
    value = unicodedata.normalize("NFKC", str(text or "")).casefold().strip()
    value = re.sub(r"<[^>]+>", " ", value)
    value = value.replace("’", "'").replace("`", "'").replace("'", "")
    output: list[str] = []
    for char in value:
        category = unicodedata.category(char)
        if char.isspace():
            output.append(" ")
        elif category.startswith(("L", "N", "M")):
            output.append(char)
        else:
            output.append(" ")
    return re.sub(r"\s+", " ", "".join(output)).strip()


@dataclass
class TextMetrics:
    reference_normalized: str
    hypothesis_normalized: str
    reference_words: int
    hypothesis_words: int
    word_errors: int
    char_errors: int
    word_hits: int
    word_substitutions: int
    word_deletions: int
    word_insertions: int
    char_hits: int
    char_substitutions: int
    char_deletions: int
    char_insertions: int
    wer: float
    cer: float
    wer_corr: float
    cer_corr: float
    exact_match: bool

    def to_dict(self) -> dict:
        return asdict(self)


def text_metrics(reference: str | None, hypothesis: str | None) -> TextMetrics | None:
    if reference is None or not str(reference).strip():
        return None
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    word = jiwer.process_words(ref, hyp)
    char = jiwer.process_characters(ref, hyp)
    ref_word_count = int(word.hits + word.substitutions + word.deletions)
    ref_char_count = int(char.hits + char.substitutions + char.deletions)
    word_errors = int(word.substitutions + word.deletions + word.insertions)
    char_errors = int(char.substitutions + char.deletions + char.insertions)
    return TextMetrics(
        reference_normalized=ref,
        hypothesis_normalized=hyp,
        reference_words=ref_word_count,
        hypothesis_words=len(hyp.split()) if hyp else 0,
        word_errors=word_errors,
        char_errors=char_errors,
        word_hits=int(word.hits),
        word_substitutions=int(word.substitutions),
        word_deletions=int(word.deletions),
        word_insertions=int(word.insertions),
        char_hits=int(char.hits),
        char_substitutions=int(char.substitutions),
        char_deletions=int(char.deletions),
        char_insertions=int(char.insertions),
        wer=float(word.wer),
        cer=float(char.cer),
        wer_corr=(float(word.hits) / ref_word_count) if ref_word_count else 0.0,
        cer_corr=(float(char.hits) / ref_char_count) if ref_char_count else 0.0,
        exact_match=ref == hyp,
    )



def word_error_alignment(
    reference: str | None,
    hypothesis: str | None,
) -> list[dict[str, str]]:
    # Return normalized word operations using the same jiwer alignment as WER.
    if reference is None or not str(reference).strip():
        return []

    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    output = jiwer.process_words(ref, hyp)

    reference_words = ref.split()
    hypothesis_words = hyp.split()
    chunks = output.alignments[0] if output.alignments else []
    operations: list[dict[str, str]] = []

    for chunk in chunks:
        op = str(chunk.type)
        ref_chunk = reference_words[chunk.ref_start_idx:chunk.ref_end_idx]
        hyp_chunk = hypothesis_words[chunk.hyp_start_idx:chunk.hyp_end_idx]

        if op == "equal":
            for ref_word, hyp_word in zip(ref_chunk, hyp_chunk):
                operations.append({
                    "type": "equal",
                    "reference": ref_word,
                    "hypothesis": hyp_word,
                })
            continue

        if op == "substitute":
            paired = min(len(ref_chunk), len(hyp_chunk))

            for index in range(paired):
                operations.append({
                    "type": "substitute",
                    "reference": ref_chunk[index],
                    "hypothesis": hyp_chunk[index],
                })

            for ref_word in ref_chunk[paired:]:
                operations.append({
                    "type": "delete",
                    "reference": ref_word,
                    "hypothesis": "",
                })

            for hyp_word in hyp_chunk[paired:]:
                operations.append({
                    "type": "insert",
                    "reference": "",
                    "hypothesis": hyp_word,
                })
            continue

        if op == "delete":
            for ref_word in ref_chunk:
                operations.append({
                    "type": "delete",
                    "reference": ref_word,
                    "hypothesis": "",
                })
            continue

        if op == "insert":
            for hyp_word in hyp_chunk:
                operations.append({
                    "type": "insert",
                    "reference": "",
                    "hypothesis": hyp_word,
                })

    return operations


def percentile(values: Iterable[float], q: float) -> float | None:
    clean = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return clean[lo]
    return clean[lo] * (hi - pos) + clean[hi] * (pos - lo)


def latency_summary(step_rows: list[dict], audio_duration_s: float, wall_time_s: float) -> dict:
    compute = [float(r["compute_ms"]) for r in step_rows if r.get("compute_ms") is not None]
    round_trip = [float(r["round_trip_ms"]) for r in step_rows if r.get("round_trip_ms") is not None]
    final_row = step_rows[-1] if step_rows else {}
    first_partial = next(
        (float(r["first_partial_ms"]) for r in step_rows if r.get("first_partial_ms") is not None),
        None,
    )
    total_compute_s = sum(compute) / 1000.0
    return {
        "audio_duration_s": audio_duration_s,
        "wall_time_s": wall_time_s,
        "rtf_compute": total_compute_s / audio_duration_s if audio_duration_s else None,
        "rtf_wall": wall_time_s / audio_duration_s if audio_duration_s else None,
        "first_partial_ms": first_partial,
        "finalization_ms": final_row.get("finalization_ms") or final_row.get("tail_after_audio_ms"),
        "compute_mean_ms": statistics.fmean(compute) if compute else None,
        "compute_median_ms": statistics.median(compute) if compute else None,
        "compute_p95_ms": percentile(compute, 0.95),
        "compute_max_ms": max(compute) if compute else None,
        "round_trip_mean_ms": statistics.fmean(round_trip) if round_trip else None,
        "round_trip_median_ms": statistics.median(round_trip) if round_trip else None,
        "round_trip_p95_ms": percentile(round_trip, 0.95),
        "round_trip_max_ms": max(round_trip) if round_trip else None,
        "num_stream_steps": len(step_rows),
    }
