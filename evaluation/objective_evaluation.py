#!/usr/bin/env python
"""
Compute the objective metrics from Bhandari et al. (2024) for a directory of
generated MIDI files. The script:

* Loads the MidiCaps metadata from `captions/all_captions.json`
* Keeps only entries that belong to the requested split (default: test)
* Resolves the ground-truth MIDI path as well as the generated MIDI path
* Measures compression ratio, CLAP similarity, tempo bin, tempo bin (±1 bin),
  key accuracy, key accuracy with relative-major/minor tolerance
* Adds an instrument-control metric that compares predicted instrument
  families with the reference instrument classes stored in MidiCaps

Example usage:
python evaluation/objective_evaluation.py \
    --predictions_dir output/generated_midis \
    --captions captions/all_captions.json \
    --split test \
    --report_file artifacts/text2midi_objective_eval.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import zlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pretty_midi
from tqdm import tqdm

from data.instruments_mapping import INSTRUMENT_CLASSES
from utils import midi_to_wav

try:
    from laion_clap import CLAP_Module

    _CLAP_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    CLAP_Module = None
    _CLAP_AVAILABLE = False

try:
    from music21 import converter, key as m21_key

    _MUSIC21_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    converter = None
    m21_key = None
    _MUSIC21_AVAILABLE = False


LOGGER = logging.getLogger("objective_eval")


TEMPO_BIN_EDGES = [40, 60, 70, 90, 110, 140, 160, 210]
PROGRAM_TO_CLASS: Dict[int, str] = {}
for class_name, programs in INSTRUMENT_CLASSES.items():
    for prog in programs:
        PROGRAM_TO_CLASS[prog] = class_name


@dataclass
class MidiFeatures:
    """Container describing the features extracted from a MIDI file."""

    compression_ratio: float
    tempo_bpm: Optional[float]
    tempo_bin: Optional[int]
    key_readable: Optional[str]
    key_tuple: Optional[Tuple[int, str]]
    instrument_classes: Set[str]


def tempo_to_bin(tempo_bpm: Optional[float]) -> Optional[int]:
    if tempo_bpm is None:
        return None
    for idx, edge in enumerate(TEMPO_BIN_EDGES):
        if tempo_bpm < edge:
            return idx
    return len(TEMPO_BIN_EDGES)


def detect_key_label(key_name: Optional[str]) -> Optional[Tuple[int, str]]:
    """Convert a key label (e.g. 'Bb minor') into (pitch_class, mode)."""
    if key_name is None or not _MUSIC21_AVAILABLE:
        return None
    try:
        tonic, mode = key_name.split()
    except ValueError:
        parts = key_name.split()
        if not parts:
            return None
        tonic = parts[0]
        mode = parts[1] if len(parts) > 1 else "major"
    try:
        key_obj = m21_key.Key(tonic, mode)
    except Exception:  # pragma: no cover - best-effort parsing
        return None
    return key_obj.tonic.pitchClass, key_obj.mode


def _format_key_string(key_obj) -> Optional[str]:
    """Return a human-readable key string from a music21 key object."""
    if key_obj is None:
        return None
    tonic_name = key_obj.tonic.name.replace("-", "b")
    mode = key_obj.mode or "major"
    return f"{tonic_name} {mode}"


def _detect_key_from_midi(midi_path: str) -> Tuple[Optional[str], Optional[Tuple[int, str]]]:
    """Estimate the musical key of a MIDI file using music21."""
    if not _MUSIC21_AVAILABLE:
        return None, None
    try:
        score = converter.parse(midi_path)
        analysis_result = score.analyze("KrumhanslSchmuckler")
        readable = _format_key_string(analysis_result)
        return readable, (analysis_result.tonic.pitchClass, analysis_result.mode)
    except Exception as exc:  # pragma: no cover - analyzer best effort
        LOGGER.debug("Failed to analyze key for %s: %s", midi_path, exc)
        return None, None


def _instrument_classes_from_midi(pm: pretty_midi.PrettyMIDI) -> Set[str]:
    classes: Set[str] = set()
    for instrument in pm.instruments:
        if instrument.is_drum:
            classes.add("Percussive")
            continue
        class_name = PROGRAM_TO_CLASS.get(instrument.program)
        if class_name:
            classes.add(class_name)
    return classes


def _compression_ratio_from_midi(pm: pretty_midi.PrettyMIDI) -> float:
    """Estimate compression ratio by zlib compressing note tuples."""
    signatures: List[str] = []
    for instrument in pm.instruments:
        inst_id = -1 if instrument.is_drum else instrument.program
        for note in instrument.notes:
            start = round(note.start, 3)
            duration = round(note.end - note.start, 3)
            signatures.append(
                f"{inst_id}:{note.pitch}:{start}:{duration}:{note.velocity}"
            )
    if not signatures:
        signatures = ["EMPTY"]
    serialized = "|".join(signatures)
    compressed = zlib.compress(serialized.encode("utf-8"))
    if len(compressed) == 0:
        return 1.0
    return len(serialized) / len(compressed)


def extract_midi_features(midi_path: str) -> Optional[MidiFeatures]:
    """Load a MIDI file and compute the objective features."""
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception as exc:
        LOGGER.warning("Failed to parse %s: %s", midi_path, exc)
        return None

    tempo_changes = pm.get_tempo_changes()[1]
    tempo_bpm = float(np.median(tempo_changes)) if len(tempo_changes) else None
    key_readable, key_tuple = _detect_key_from_midi(midi_path)
    features = MidiFeatures(
        compression_ratio=_compression_ratio_from_midi(pm),
        tempo_bpm=tempo_bpm,
        tempo_bin=tempo_to_bin(tempo_bpm),
        key_readable=key_readable,
        key_tuple=key_tuple or detect_key_label(key_readable),
        instrument_classes=_instrument_classes_from_midi(pm),
    )
    return features


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


class ClapScorer:
    """Wrap CLAP model loading and scoring."""

    def __init__(self, device: Optional[str], cache_dir: Path):
        self.device = device or ("cuda:0" if shutil.which("nvidia-smi") else "cpu")
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = _CLAP_AVAILABLE and shutil.which("fluidsynth")
        self._model: Optional[CLAP_Module] = None

        if not self.enabled:
            if not _CLAP_AVAILABLE:
                LOGGER.warning("laion_clap is not installed. Skipping CLAP scores.")
            elif not shutil.which("fluidsynth"):
                LOGGER.warning("fluidsynth binary not found. Skipping CLAP scores.")
            return

        # Check if soundfont exists before attempting download
        import yaml
        from utils.midi_to_wav import get_soundfont, download_and_extract_soundfont
        
        # Load config to find soundfont dir
        config_path = Path("configs/config.yaml")
        soundfont_dir = "soundfont" # Default
        if config_path.exists():
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                if "soundfont_path" in config:
                    soundfont_dir = config["soundfont_path"]
        
        self.soundfont_path = get_soundfont(soundfont_dir)
        if not self.soundfont_path:
             # If not found, try downloading to that dir
             self.soundfont_path = download_and_extract_soundfont(soundfont_dir)
        
        try:
            self._model = CLAP_Module(enable_fusion=False, device=self.device)
            self._model.load_ckpt(model_id=1, verbose=False)
        except Exception as exc:  # pragma: no cover - heavy dependency
            LOGGER.warning("CLAP model failed to load (%s). CLAP will be skipped.", exc)
            self.enabled = False
            self._model = None

    def _render_wav(self, midi_path: str, tag: str) -> Optional[Path]:
        wav_path = self.cache_dir / f"{tag}_{Path(midi_path).stem}.wav"
        midi_to_wav.save_wav(midi_path, str(wav_path), self.soundfont_path)
        if wav_path.exists():
            return wav_path
        return None

    def score(self, midi_path: str, caption: str, tag: str) -> Optional[float]:
        if not self.enabled or not self._model:
            return None
        wav_path = self._render_wav(midi_path, tag)
        if wav_path is None:
            return None
        audio_embedding = self._model.get_audio_embedding_from_filelist([str(wav_path)])
        text_embedding = self._model.get_text_embedding([caption])
        return cosine_similarity(audio_embedding[0], text_embedding[0])


class MetricTracker:
    """Utility to accumulate scalar metrics and accuracies."""

    def __init__(self) -> None:
        self.scalars: Dict[str, List[float]] = defaultdict(list)
        self.accuracies: Dict[str, List[int]] = defaultdict(list)
        self.instrument_precision: List[float] = []
        self.instrument_recall: List[float] = []
        self.instrument_f1: List[float] = []

    def add_scalar(self, name: str, value: Optional[float]) -> None:
        if value is not None and np.isfinite(value):
            self.scalars[name].append(float(value))

    def add_accuracy(self, name: str, hit: Optional[bool]) -> None:
        if hit is not None:
            self.accuracies[name].append(int(bool(hit)))

    def add_instrument_scores(self, precision: float, recall: float, f1_score: float) -> None:
        self.instrument_precision.append(precision)
        self.instrument_recall.append(recall)
        self.instrument_f1.append(f1_score)

    def summarize(self) -> Dict[str, float]:
        summary: Dict[str, float] = {}
        for name, values in self.scalars.items():
            if values:
                summary[name] = float(np.mean(values))
        for name, hits in self.accuracies.items():
            if hits:
                summary[name] = float(np.mean(hits))
        if self.instrument_precision:
            summary["instrument_precision"] = float(np.mean(self.instrument_precision))
        if self.instrument_recall:
            summary["instrument_recall"] = float(np.mean(self.instrument_recall))
        if self.instrument_f1:
            summary["instrument_f1"] = float(np.mean(self.instrument_f1))
        return summary


def _resolve_prediction_path(predictions_dir: Path, location: str) -> Optional[Path]:
    rel_path = predictions_dir / location
    if rel_path.exists():
        return rel_path
    alt_path = predictions_dir / Path(location).name
    if alt_path.exists():
        return alt_path
    return None


def _resolve_ground_truth_path(ground_truth_root: Path, location: str) -> Path:
    return ground_truth_root / location


def _load_metadata_from_json(json_path: Path) -> List[Dict]:
    """Load metadata directly from the generated JSON manifest."""
    examples: List[Dict] = []
    with json_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            examples.append(json.loads(line))
    return examples


def evaluate(
    generated_json: Path,
    ground_truth_root: Path,
    *,
    clap_device: Optional[str],
    report_file: Optional[Path],
) -> Dict[str, float]:
    metadata = _load_metadata_from_json(generated_json)
    if not metadata:
        raise ValueError(f"No samples found in {generated_json}")

    tracker = MetricTracker()
    clap_cache = Path(tempfile.mkdtemp(prefix="clap_audio_"))
    clap_scorer = ClapScorer(device=clap_device, cache_dir=clap_cache)

    processed = 0
    missing_predictions = 0
    sample_rows: List[Dict] = []

    try:
        for entry in tqdm(metadata, desc="Evaluating"):
            location = entry["location"]
            caption = entry.get("caption", "")
            
            # Use the predicted path directly from the JSON
            pred_path_str = entry.get("predicted_midi_path")
            if not pred_path_str:
                missing_predictions += 1
                continue
            
            # Resolve path relative to the JSON file's directory (which is the output_dir)
            # This assumes the JSON is at the root of the output directory
            pred_path = generated_json.parent / pred_path_str
            
            if not pred_path.exists():
                LOGGER.warning("Predicted file missing: %s", pred_path)
                missing_predictions += 1
                continue

            gt_path = _resolve_ground_truth_path(ground_truth_root, location)
            if not gt_path.exists():
                LOGGER.warning("Ground truth file missing for %s", location)
                continue

            pred_features = extract_midi_features(str(pred_path))
            gt_features = extract_midi_features(str(gt_path))
            if pred_features is None or gt_features is None:
                continue

            tracker.add_scalar("compression_ratio", pred_features.compression_ratio)

            tracker.add_accuracy(
                "tempo_bin",
                pred_features.tempo_bin is not None
                and gt_features.tempo_bin is not None
                and pred_features.tempo_bin == gt_features.tempo_bin,
            )
            tracker.add_accuracy(
                "tempo_bin_tolerance",
                pred_features.tempo_bin is not None
                and gt_features.tempo_bin is not None
                and abs(pred_features.tempo_bin - gt_features.tempo_bin) <= 1,
            )
            key_match_exact, key_match_rel = _keys_match(
                pred_features.key_tuple, gt_features.key_tuple
            )
            tracker.add_accuracy("key_accuracy", key_match_exact)
            tracker.add_accuracy("key_accuracy_rel", key_match_rel)

            clap_score = clap_scorer.score(str(pred_path), caption, tag="pred")
            tracker.add_scalar("clap_score", clap_score)

            precision, recall, f1_score = _instrument_scores(
                pred_features.instrument_classes, entry.get("instrument_classes", [])
            )
            tracker.add_instrument_scores(precision, recall, f1_score)

            sample_rows.append(
                {
                    "location": location,
                    "caption": caption,
                    "compression_ratio": pred_features.compression_ratio,
                    "clap_score": clap_score,
                    "tempo_bin_match": tracker.accuracies["tempo_bin"][-1]
                    if tracker.accuracies["tempo_bin"]
                    else None,
                    "tempo_bin_tolerance": tracker.accuracies["tempo_bin_tolerance"][-1]
                    if tracker.accuracies["tempo_bin_tolerance"]
                    else None,
                    "key_accuracy": tracker.accuracies["key_accuracy"][-1]
                    if tracker.accuracies["key_accuracy"]
                    else None,
                    "key_accuracy_rel": tracker.accuracies["key_accuracy_rel"][-1]
                    if tracker.accuracies["key_accuracy_rel"]
                    else None,
                    "instrument_precision": precision,
                    "instrument_recall": recall,
                    "instrument_f1": f1_score,
                }
            )

            processed += 1
            
    finally:
        shutil.rmtree(clap_cache, ignore_errors=True)

    summary = tracker.summarize()
    summary["num_evaluated"] = processed
    summary["num_missing_predictions"] = missing_predictions

    if report_file:
        report = {"summary": summary, "samples": sample_rows}
        with report_file.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        LOGGER.info("Wrote detailed report to %s", report_file)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Objective evaluation for text2midi outputs.")
    parser.add_argument(
        "--generated_json",
        type=Path,
        required=True,
        help="Path to the generated samples JSON manifest (produced by generate_eval_midis.py).",
    )
    parser.add_argument(
        "--ground_truth_root",
        type=Path,
        default=Path("data/midicaps"),
        help="Root directory that mirrors the MidiCaps structure.",
    )
    parser.add_argument(
        "--clap_device",
        type=str,
        default=None,
        help="Torch device string for CLAP (default: cuda:0 if available).",
    )
    parser.add_argument(
        "--report_file",
        type=Path,
        default=None,
        help="Optional JSON file where per-sample metrics are saved.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    summary = evaluate(
        generated_json=args.generated_json,
        ground_truth_root=args.ground_truth_root,
        clap_device=args.clap_device,
        report_file=args.report_file,
    )

    LOGGER.info("Objective evaluation summary:\n%s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
