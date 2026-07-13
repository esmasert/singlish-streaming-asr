from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

AUDIO_KEYS = (
    "audio_filepath", "audio", "wav", "wav_path", "path", "file", "filename"
)
TEXT_KEYS = (
    "text", "transcript", "reference", "sentence", "gt", "ground_truth"
)
DATASET_KEYS = ("dataset_name", "dataset", "tier", "split")


def _first(row: dict[str, Any], keys: tuple[str, ...]):
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return value
    return None


@dataclass(frozen=True)
class GroundTruthRecord:
    audio_path: str
    text: str
    dataset_name: str
    raw: dict[str, Any]


class GroundTruthIndex:
    def __init__(self, jsonl_path: str | Path):
        self.path = Path(jsonl_path).expanduser()
        self.by_basename: dict[str, list[GroundTruthRecord]] = {}
        self.by_stem: dict[str, list[GroundTruthRecord]] = {}
        self.records: list[GroundTruthRecord] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"Ground-truth manifest not found: {self.path}")
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at line {line_number}: {exc}") from exc
                audio = _first(row, AUDIO_KEYS)
                text = _first(row, TEXT_KEYS)
                if audio is None or text is None:
                    continue
                dataset = str(_first(row, DATASET_KEYS) or self._infer_dataset(str(audio)))
                record = GroundTruthRecord(str(audio), str(text), dataset, row)
                self.records.append(record)
                basename = Path(str(audio)).name.lower()
                stem = Path(str(audio)).stem.lower()
                self.by_basename.setdefault(basename, []).append(record)
                self.by_stem.setdefault(stem, []).append(record)

    @staticmethod
    def _infer_dataset(path: str) -> str:
        lower = path.lower().replace("\\", "/")
        for tier in ("tier0", "tier1", "tier2"):
            if f"/{tier}/" in f"/{lower}":
                return tier
        return "unknown"

    def lookup(self, uploaded_path: str | Path | None) -> GroundTruthRecord | None:
        if not uploaded_path:
            return None
        path = Path(str(uploaded_path))
        candidates = self.by_basename.get(path.name.lower(), [])
        if not candidates:
            candidates = self.by_stem.get(path.stem.lower(), [])
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        # Prefer a manifest path whose suffix best matches the selected path.
        selected_parts = [p.lower() for p in path.parts]
        def suffix_score(record: GroundTruthRecord) -> int:
            parts = [p.lower() for p in Path(record.audio_path).parts]
            score = 0
            for a, b in zip(reversed(parts), reversed(selected_parts)):
                if a != b:
                    break
                score += 1
            return score
        return max(candidates, key=suffix_score)
