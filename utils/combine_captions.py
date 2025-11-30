#!/usr/bin/env python3
"""
Combine multiple caption JSONL files into a single unified dataset.

Usage:
    python utils/combine_captions.py --output captions/all_captions.json

This will combine:
    - captions/symphonynet_captions.json
    - captions/midicaps_captions.json
    
Into a single file with proper train/val/test splits maintained.
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict


def load_jsonl(path: Path) -> List[Dict]:
    """Load records from a JSONL file."""
    records = []
    if not path.exists():
        print(f"Warning: {path} does not exist, skipping...")
        return records
    
    with path.open('r') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    
    print(f"Loaded {len(records)} records from {path}")
    return records


def save_jsonl(records: List[Dict], path: Path) -> None:
    """Save records to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with path.open('w') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(records)} records to {path}")


def print_statistics(records: List[Dict], name: str = "Dataset") -> None:
    """Print statistics about the dataset."""
    print(f"\n{name} Statistics:")
    print("=" * 60)
    
    # Count by split
    splits = {}
    for rec in records:
        split = rec.get('split', 'unknown')
        splits[split] = splits.get(split, 0) + 1
    
    print(f"Total records: {len(records)}")
    for split, count in sorted(splits.items()):
        percentage = (count / len(records) * 100) if records else 0
        print(f"  {split}: {count} ({percentage:.1f}%)")
    
    # Count by dataset
    datasets = {}
    for rec in records:
        dataset = rec.get('dataset', 'unknown')
        datasets[dataset] = datasets.get(dataset, 0) + 1
    
    print(f"\nBy dataset:")
    for dataset, count in sorted(datasets.items()):
        percentage = (count / len(records) * 100) if records else 0
        print(f"  {dataset}: {count} ({percentage:.1f}%)")
    
    # Count by stage
    stages = {}
    for rec in records:
        stage = rec.get('stage', 'unknown')
        stages[stage] = stages.get(stage, 0) + 1
    
    print(f"\nBy stage:")
    for stage, count in sorted(stages.items()):
        percentage = (count / len(records) * 100) if records else 0
        print(f"  {stage}: {count} ({percentage:.1f}%)")


def combine_captions(
    input_files: List[Path],
    output_file: Path,
    shuffle: bool = True,
    seed: int = 42
) -> None:
    """Combine multiple caption files into one."""
    
    all_records = []
    
    # Load all files
    for input_file in input_files:
        records = load_jsonl(input_file)
        all_records.extend(records)
    
    # Optional shuffle
    if shuffle:
        random.seed(seed)
        random.shuffle(all_records)
        print(f"\nShuffled {len(all_records)} records with seed={seed}")
    
    # Print statistics
    print_statistics(all_records, "Combined Dataset")
    
    # Save combined file
    save_jsonl(all_records, output_file)
    
    print(f"\n✅ Successfully combined {len(input_files)} files into {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Combine multiple caption JSONL files into one."
    )
    parser.add_argument(
        '--inputs',
        nargs='+',
        default=['captions/symphonynet_captions.json', 'captions/midicaps_captions.json'],
        help='Input JSONL files to combine (default: symphonynet and midicaps)'
    )
    parser.add_argument(
        '--output',
        default='captions/all_captions.json',
        help='Output combined JSONL file (default: captions/all_captions.json)'
    )
    parser.add_argument(
        '--no-shuffle',
        action='store_true',
        help='Do not shuffle the combined records'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for shuffling (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    input_files = [Path(f) for f in args.inputs]
    output_file = Path(args.output)
    
    # Combine
    combine_captions(
        input_files=input_files,
        output_file=output_file,
        shuffle=not args.no_shuffle,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
