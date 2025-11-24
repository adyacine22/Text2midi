import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set

from symusic import Score

# Make project root importable so we can reuse mappings
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.instruments_mapping import INSTRUMENT_CLASSES, INSTRUMENTS


# ---------- Music analysis helpers ----------

MAJOR_KEYS = ["C", "G", "D", "A", "E", "B", "F#", "C#", "F", "Bb", "Eb", "Ab"]
MINOR_KEYS = ["A", "E", "B", "F#", "C#", "G#", "D#", "A#", "D", "G", "C", "F"]
MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


def key_from_notes(pitch_classes: Iterable[int]) -> str:
    pcs_list = [pc % 12 for pc in pitch_classes]
    if not pcs_list:
        return "unknown"
    hist = [0] * 12
    for pc in pcs_list:
        hist[pc] += 1

    best_score, best_key, best_mode = -1e9, "C", "major"
    for tonic in range(12):
        rotated = hist[tonic:] + hist[:tonic]
        maj_score = sum(x * y for x, y in zip(rotated, MAJOR_PROFILE))
        min_score = sum(x * y for x, y in zip(rotated, MINOR_PROFILE))
        if maj_score > best_score:
            best_score, best_key, best_mode = maj_score, MAJOR_KEYS[tonic], "major"
        if min_score > best_score:
            best_score, best_key, best_mode = min_score, MINOR_KEYS[tonic], "minor"
    return f"{best_key} {best_mode}"


def tempo_word(bpm: float) -> str:
    if bpm < 76:
        return "slow"
    if bpm < 110:
        return "moderate"
    if bpm < 150:
        return "fast"
    return "very fast"


def program_to_name(program: int, is_drum: bool) -> str:
    """Convert MIDI program number to instrument name."""
    if is_drum:
        return INSTRUMENTS.get(-1, "Drums")
    return INSTRUMENTS.get(program, f"Program {program}")


def program_to_classes(program: int, is_drum: bool) -> List[str]:
    """Get instrument class for a program (for metadata)."""
    if is_drum:
        return ["Drums / Percussion"]
    classes = []
    for name, plist in INSTRUMENT_CLASSES.items():
        if program in plist:
            classes.append(name)
    return classes

def number_of_tracks_to_adjective(n_tracks: int) -> List[str]:
    if n_tracks <= 2:
        return ["a simple", "a minimal"]
    elif n_tracks <= 5:
        return ["a moderately complex", "a layered", "a textured"]
    elif n_tracks <= 10:
        return ["a rich", "a layered", "a detailed",]
    else:
        return ["a highly complex ", "an elaborate", "a densely arranged",]


def analyze_midi(path: Path) -> Dict:
    score = Score(str(path))
    pcs: List[int] = []
    classes_set: Set[str] = set()
    programs_set: Set[str] = set()  # NEW: Track specific instrument programs
    
    for tr in score.tracks:
        if tr.is_drum:
            classes_set.update(program_to_classes(-1, True))
            programs_set.add("Drums")
        else:
            classes_set.update(program_to_classes(tr.program, False))
            programs_set.add(program_to_name(tr.program, False))  # NEW: Add program name
        
        for note in tr.notes:
            pcs.append(note.pitch)

    key = key_from_notes(pcs)
    tempo_val = score.tempos[0].qpm if score.tempos else 120
    ts = score.time_signatures[0] if score.time_signatures else None
    time_sig = f"{ts.numerator}/{ts.denominator}" if ts else "4/4"

    return {
        "key": key,
        "tempo": tempo_val,
        "time_signature": time_sig,
        "instrument_classes": sorted(classes_set),  # Keep for metadata
        "instrument_programs": sorted(programs_set),  # NEW: Specific instruments for caption
        "adjectives_tracks": number_of_tracks_to_adjective(len(score.tracks)),
    }


# ---------- Caption generation helpers ----------



# Generic Introduction phrases 


INTRODUCTION_PHRASES = [
    "piece",
    "composition",
    "arrangement",
    "work",
    "musical piece"
]

INSTRUMENT_PHRASES = [
    "featuring {instruments}",
    "with {instruments} leading the ensemble",
    "built around {instruments}",
    "where {instruments} carry the texture",
    "layering {instruments} for the core sound",
    "centered on {instruments}",
    "highlighting {instruments}",
]

STRUCTURE_PHRASES = [
    "set in {key} and a {time_signature} meter",
    "written in {key} with a {time_signature} signature",
    "framed in {key} and {time_signature}",
    "shaped around {key} and {time_signature}",
    "composed in {key} with a {time_signature} time signature",
]

TEMPO_PHRASES = [
    "at a {tempo_word} pace (~{bpm} BPM)",
    "moving at about {bpm} BPM ({tempo_word})",
    "flowing near {bpm} BPM with a {tempo_word} feel",
    "keeping a {tempo_word} motion around {bpm} BPM",
    "progressing at {bpm} BPM, giving it a {tempo_word} character",
    "driven by a {tempo_word} tempo of approximately {bpm} BPM",
    "paced at {bpm} BPM, resulting in a {tempo_word} vibe",
]


def pick_phrase(choices: List[str], **kwargs) -> str:
    return random.choice(choices).format(**kwargs)


def caption_from_meta(meta: Dict) -> str:
    # Use specific instrument programs instead of classes
    instruments = ", ".join(meta["instrument_programs"]) or "varied orchestral sections"
    return " ".join(
            

        [   "A",
            pick_phrase(meta["adjectives_tracks"]),
            pick_phrase(INTRODUCTION_PHRASES),
            pick_phrase(INSTRUMENT_PHRASES, instruments=instruments),
            pick_phrase(STRUCTURE_PHRASES, key=meta["key"], time_signature=meta["time_signature"]),
            pick_phrase(TEMPO_PHRASES, tempo_word=tempo_word(meta["tempo"]), bpm=int(meta["tempo"])),
        ]
    )


# ---------- Main pipeline ----------


def build_records(
    midi_root: Path,
    seed: int,
) -> List[Dict]:
    midi_paths = sorted(midi_root.rglob("*.mid"))
    print(f"Found {len(midi_paths)} MIDI files under {midi_root}")
    random.seed(seed)
    records: List[Dict] = []

    for p in midi_paths:
        # Store location relative to the SymphonyNet root (no leading "data/...")
        rel = str(p.relative_to(midi_root))
        print(f"Processing {rel}...")
        meta = analyze_midi(p)
        record = {
            "location": rel,
            "caption": caption_from_meta(meta),
            "instrument_classes": meta["instrument_classes"],
            "key": meta["key"],
            "split": "train",  # placeholder, filled after shuffle
            "stage": "pretrain",
            "dataset": "symphonynet",
        }
        records.append(record)

    random.shuffle(records)
    n = len(records)
    n_train = int(n * 0.90)
    n_val = int(n * 0.05)
    print(f"Total records: {n} -> train: {n_train}, val: {n_val}, test: {n - n_train - n_val}")
    for i, rec in enumerate(records):
        if i < n_train:
            rec["split"] = "train"
        elif i < n_train + n_val:
            rec["split"] = "validation"
        else:
            rec["split"] = "test"
    return records


def write_jsonl(records: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {len(records)} records to {output_path}")
    with output_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SymphonyNet captions with modular templates.")
    parser.add_argument("--root", default="data/symphonynet", help="Path to SymphonyNet MIDI root.")
    parser.add_argument("--output", default="captions/symphonynet_captions.json", help="Output JSONL path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling/splitting.")
    args = parser.parse_args()

    midi_root = PROJECT_ROOT / args.root
    records = build_records(midi_root, seed=args.seed)
    write_jsonl(records, PROJECT_ROOT / args.output)
    print("Done.")


if __name__ == "__main__":
    main()
