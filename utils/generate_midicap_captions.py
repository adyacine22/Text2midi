import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set

import json

# Ensure project root is on sys.path so that `data` can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.instruments_mapping import INSTRUMENT_CLASSES


def program_to_classes(program: int) -> List[str]:
    """Return all high-level instrument classes for a MIDI program number."""
    classes = []
    for cls_name, programs in INSTRUMENT_CLASSES.items():
        if program in programs:
            classes.append(cls_name)
    return classes


def infer_dataset_from_location(location: str) -> str:
    """
    Infer dataset name from the MIDI location.

    Assumes a path like "lmd_full/1/xxx.mid" and returns the first component
    ("lmd_full" in this example).
    """
    return location.split("/", 1)[0] if "/" in location else "unknown"


def build_records(
    captions_path: str,
    stage: str = "finetune",
    val_fraction: float = 0.05,
    seed: int = 42,
) -> List[Dict]:
    """
    Load captions.json and build a list of records with the desired schema:
    location, caption, instrument_classes, key, split, stage, dataset.

    - split: "test" if test_set is True, else "train" (later some train -> "validation")
    - stage: fixed to the provided value (default "finetune")
    - dataset: inferred from the first component of location path
    """
    random.seed(seed)

    records: List[Dict] = []
    train_indices: List[int] = []

    with open(captions_path, "r") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            location = item["location"]
            caption = item["caption"]
            key = item.get("key")

            # Instrument classes from instrument_numbers_sorted
            programs: Iterable[int] = item.get("instrument_numbers_sorted", [])
            inst_classes_set: Set[str] = set()
            for p in programs:
                for cls_name in program_to_classes(p):
                    inst_classes_set.add(cls_name)
            instrument_classes = sorted(inst_classes_set)

            test_flag = bool(item.get("test_set"))
            split = "test" if test_flag else "train"

            dataset = infer_dataset_from_location(location)

            record = {
                "location": location,
                "caption": caption,
                "instrument_classes": instrument_classes,
                "key": key,
                "split": split,
                "stage": stage,
                "dataset": dataset,
            }
            records.append(record)

            if not test_flag:
                train_indices.append(idx)

    # Convert a fraction of train examples to validation
    if train_indices and 0.0 < val_fraction < 1.0:
        num_val = int(len(train_indices) * val_fraction)
        if num_val > 0:
            val_indices = set(random.sample(train_indices, num_val))
            for idx in val_indices:
                records[idx]["split"] = "validation"

    return records


def write_records(records: List[Dict], output_path: str) -> None:
    """Write records to a JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate captions_final.json from captions.json "
        "with instrument_classes, split, stage and dataset fields."
    )
    parser.add_argument(
        "--input",
        default="captions/captions.json",
        help="Path to input captions JSONL file.",
    )
    parser.add_argument(
        "--output",
        default="captions/captions_final.json",
        help="Path to output captions JSONL file.",
    )
    parser.add_argument(
        "--stage",
        default="finetune",
        help='Stage value to assign (e.g. "pretrain" or "finetune").',
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.05,
        help="Fraction of train examples to use for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for validation split.",
    )

    args = parser.parse_args()

    records = build_records(
        captions_path=args.input,
        stage=args.stage,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    write_records(records, args.output)


if __name__ == "__main__":
    main()
