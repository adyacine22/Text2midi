"""Unified music dataset classes with inheritance-based architecture.

This module provides:
- BaseMusicDataset: Shared logic for all music datasets
- TrainingMusicDataset: For training with MIDI loading and caching
- InferenceMusicDataset: For inference with minimal processing
- MusicCollator: Flexible collate function for batching
- create_dataloader: Factory function for creating dataloaders
"""

import os
import sys
import re
import random
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any
from copy import deepcopy

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from transformers import T5Tokenizer
from spacy.lang.en import English

# Add parent directory to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

try:
    from utils.instruments_mapping import INSTRUMENT_CLASSES
except ModuleNotFoundError:
    from .utils.instruments_mapping import INSTRUMENT_CLASSES


class BaseMusicDataset(Dataset, ABC):
    """Abstract base class for music datasets with shared functionality."""
    
    def __init__(
        self,
        configs: Dict,
        captions: List[Dict],
        remi_tokenizer=None,
        mode: str = "train",
        split: Optional[str] = None,
        shuffle: bool = False
    ):
        """Initialize base dataset.
        
        Args:
            configs: Configuration dictionary
            captions: List of caption dictionaries
            remi_tokenizer: REMI tokenizer instance (optional for inference)
            mode: Dataset mode ('train', 'val', 'test', 'inference')
            split: Explicit split filter (overrides mode-based filtering)
            shuffle: Whether to shuffle captions
        """
        self.mode = mode
        self.configs = configs
        self.remi_tokenizer = remi_tokenizer
        
        # Filter captions by split
        self.captions = self._filter_split(captions, split, mode)
        
        if shuffle:
            random.shuffle(self.captions)
        
        # Setup dataset paths from config
        self.dataset_paths = self._load_dataset_paths(configs)
        
        # Build instrument mapping
        self.instrument_class_to_id, self.no_instr_token_id = self._build_instrument_mapping(configs)
        
    def _filter_split(self, captions: List[Dict], split: Optional[str], mode: str) -> List[Dict]:
        """Filter captions by split based on mode or explicit split."""
        # If explicit split provided, use it
        if split:
            return [c for c in captions if c.get("split") == split]
        
        # Otherwise, filter based on mode
        split_filter = None
        if any("split" in c for c in captions):
            if mode in ("train", "training"):
                split_filter = {"train"}
            elif mode in ("val", "validation"):
                split_filter = {"validation"}
            elif mode in ("test", "eval", "evaluation", "inference"):
                split_filter = {"test"}
        
        if split_filter:
            return [c for c in captions if c.get("split") in split_filter]
        
        return list(captions)
    
    def _load_dataset_paths(self, configs: Dict) -> Dict[str, str]:
        """Load dataset paths from config."""
        paths = {}
        if "raw_data" in configs and "raw_data_folders" in configs["raw_data"]:
            for name, data in configs["raw_data"]["raw_data_folders"].items():
                if isinstance(data, dict) and "folder_path" in data:
                    paths[name] = data["folder_path"]
        return paths
    
    def _build_instrument_mapping(self, configs: Dict) -> tuple:
        """Build instrument class to ID mapping."""
        instrument_vocab = configs.get("model", {}).get("text2midi_model", {}).get(
            "instrument_classes", list(INSTRUMENT_CLASSES.keys())
        )
        instrument_class_to_id = {
            name.lower(): idx for idx, name in enumerate(instrument_vocab)
        }
        no_instr_token_id = len(instrument_class_to_id)
        return instrument_class_to_id, no_instr_token_id
    
    def _resolve_midi_path(self, location: str, dataset_name: Optional[str] = None) -> str:
        """Resolve MIDI file path from location and dataset name."""
        if os.path.isabs(location):
            return location
        
        # Try to find base path from dataset name
        base = None
        if dataset_name and dataset_name in self.dataset_paths:
            base = self.dataset_paths[dataset_name]
        
        # Handle common aliases
        if base is None and dataset_name:
            alias_map = {
                "lmd_full": "midicaps",
                "midicaps": "midicaps",
            }
            alias_key = alias_map.get(dataset_name)
            if alias_key and alias_key in self.dataset_paths:
                base = self.dataset_paths[alias_key]
        
        # Fallback to default
        if base is None:
            base = self.dataset_paths.get("midicaps") or next(
                iter(self.dataset_paths.values()), "."
            )
        
        return os.path.join(base, location)
    
    def _extract_instruments(self, item: Dict) -> List[int]:
        """Extract instrument IDs from caption metadata."""
        raw_inst = item.get("instrument_classes") or item.get("instruments")
        
        inst_list = []
        if isinstance(raw_inst, str):
            inst_list = [s.strip().lower() for s in raw_inst.split(",") if s]
        elif isinstance(raw_inst, (list, tuple)):
            inst_list = [str(s).strip().lower() for s in raw_inst if s]
        
        inst_ids = [
            self.instrument_class_to_id[name]
            for name in inst_list
            if name in self.instrument_class_to_id
        ]
        
        return inst_ids if inst_ids else []
    
    def __len__(self) -> int:
        return len(self.captions)
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Optional[Dict]:
        """Get item at index. Must be implemented by subclass."""
        raise NotImplementedError


class TrainingMusicDataset(BaseMusicDataset):
    """Dataset for training - loads and tokenizes MIDI files."""
    
    def __init__(
        self,
        configs: Dict,
        captions: List[Dict],
        remi_tokenizer,
        mode: str = "train",
        split: Optional[str] = None,
        shuffle: bool = True,
        use_augmentation: bool = True,
        use_cache: bool = True,
        limit: Optional[int] = None
    ):
        """Initialize training dataset.
        
        Args:
            configs: Configuration dictionary
            captions: List of caption dictionaries
            remi_tokenizer: REMI tokenizer instance
            mode: Dataset mode
            split: Explicit split filter
            shuffle: Whether to shuffle captions
            use_augmentation: Whether to use caption augmentation
            use_cache: Whether to cache tokenized MIDI
            limit: Maximum number of samples to use
        """
        super().__init__(configs, captions, remi_tokenizer, mode, split, shuffle)
        
        # Apply limit after filtering and shuffling
        if limit and limit > 0:
            self.captions = self.captions[:limit]
        
        self.use_augmentation = use_augmentation
        self.use_cache = use_cache
        self.midi_cache = {} if use_cache else None
        
        # Get max sequence length from config
        self.max_seq_len = configs["model"]["text2midi_model"]["decoder_max_sequence_length"]
        
        # Setup sentence splitter for augmentation
        if use_augmentation:
            self.nlp = English()
            self.nlp.add_pipe("sentencizer")
        
        print(f"TrainingMusicDataset: {len(self.captions)} samples, augmentation={use_augmentation}, cache={use_cache}")
        
        # Pre-calculate pitch token mappings for augmentation
        self.pitch_ids_map = {} # id -> pitch_value (0-127)
        self.pitch_val_to_id = {} # pitch_value -> id
        
        if self.remi_tokenizer:
            # We only care about melodic pitches p-0 to p-127
            for i in range(128):
                token = f"p-{i}"
                if hasattr(self.remi_tokenizer, "token_to_id") and token in self.remi_tokenizer.token_to_id:
                    tid = self.remi_tokenizer.token_to_id[token]
                    self.pitch_ids_map[tid] = i
                    self.pitch_val_to_id[i] = tid
    
    def __getitem__(self, idx: int) -> Optional[Dict]:
        """Get training item with MIDI tokens and caption."""
        item = self.captions[idx]
        
        # 1. Load and tokenize MIDI
        midi_path = self._resolve_midi_path(item["location"], item.get("dataset"))
        
        if not os.path.exists(midi_path):
            return None
        
        # Check cache
        if self.use_cache and midi_path in self.midi_cache:
            tokenized_midi = self.midi_cache[midi_path]
        else:
            try:
                tokens = self.remi_tokenizer(midi_path)
                if len(tokens.ids) == 0:
                    tokenized_midi = [
                        self.remi_tokenizer["BOS_None"],
                        self.remi_tokenizer["EOS_None"],
                    ]
                else:
                    tokenized_midi = (
                        [self.remi_tokenizer["BOS_None"]]
                        + tokens.ids
                        + [self.remi_tokenizer["EOS_None"]]
                    )
                
                if self.use_cache:
                    self.midi_cache[midi_path] = tokenized_midi
            except Exception as e:
                # Skip problematic MIDI files
                return None
        
        # 2. Process caption (with optional augmentation)
        caption = item["caption"]
        if self.use_augmentation:
            # Caption augmentation (dropout sentences)
            if random.random() > 0.5:
                caption = self._augment_caption(caption)
            
            # MIDI augmentation (pitch shift)
            # Apply with 50% probability
            if random.random() > 0.5:
                tokenized_midi = self._augment_midi(tokenized_midi)
                # If we shifted pitch, we MUST mask the key in the caption to avoid mismatch
                caption = self._mask_key_info(caption)
        
        # 3. Extract instruments
        inst_ids = self._extract_instruments(item)
        
        # Return dict - collate will handle tokenization and padding
        return {
            "caption": caption,
            "midi_tokens": tokenized_midi,
            "inst_ids": inst_ids,
            "metadata": item
        }
    
    def _augment_caption(self, caption: str) -> str:
        """Randomly drop sentences from caption for augmentation."""
        sentences = list(self.nlp(caption).sents)
        sent_length = len(sentences)
        
        if sent_length <= 1:
            return caption
        
        # Drop 20-50% of sentences
        if sent_length < 4:
            how_many_to_drop = int(
                np.floor((20 + random.random() * 30) / 100 * sent_length)
            )
        else:
            how_many_to_drop = int(
                np.ceil((20 + random.random() * 30) / 100 * sent_length)
            )
        
        if how_many_to_drop >= sent_length:
            return caption
        
        which_to_drop = np.random.choice(sent_length, how_many_to_drop, replace=False)
        new_sentences = [
            sentences[i] for i in range(sent_length) if i not in which_to_drop
        ]
        
        return " ".join([s.text for s in new_sentences])

    
    def _augment_midi(self, tokens: List[int]) -> List[int]:
        """Augment MIDI tokens with pitch shift."""
        # Shift range: -5 to +6 semitones
        shift = random.randint(-5, 6)
        if shift == 0:
            return tokens
            
        new_tokens = []
        for tid in tokens:
            if tid in self.pitch_ids_map:
                old_pitch = self.pitch_ids_map[tid]
                new_pitch = old_pitch + shift
                # Clamp to valid range [0, 127]
                new_pitch = max(0, min(127, new_pitch))
                if new_pitch in self.pitch_val_to_id:
                    new_tokens.append(self.pitch_val_to_id[new_pitch])
                else:
                    # Should not happen if map is complete, but fallback to original
                    new_tokens.append(tid)
            else:
                new_tokens.append(tid)
        
        return new_tokens
    
    def _mask_key_info(self, caption: str) -> str:
        """Mask key information in caption to prevent mismatch with pitch-shifted MIDI."""
        # Regex to find key declarations
        # Matches: "in X major", "key of X minor", "centered in X major", etc.
        key_pattern = r"(?i)\b(in|key of|centered in|composed in|written in|framed in|set in)\s+([A-G][#b]?)\s+(major|minor)"
        
        # Remove entirely (replace with empty string)
        return re.sub(key_pattern, "", caption)

class InferenceMusicDataset(BaseMusicDataset):
    """Dataset for inference - returns metadata only, no MIDI loading."""
    
    def __init__(
        self,
        configs: Dict,
        captions: List[Dict],
        remi_tokenizer=None,
        mode: str = "inference",
        split: Optional[str] = None,
        limit: Optional[int] = None
    ):
        """Initialize inference dataset.
        
        Args:
            configs: Configuration dictionary
            captions: List of caption dictionaries
            remi_tokenizer: REMI tokenizer (not used, for compatibility)
            mode: Dataset mode
            split: Explicit split filter
            limit: Maximum number of samples
        """
        super().__init__(configs, captions, remi_tokenizer, mode, split, shuffle=False)
        
        if limit:
            self.captions = self.captions[:limit]
        
        print(f"InferenceMusicDataset: {len(self.captions)} samples")
    
    def __getitem__(self, idx: int) -> Dict:
        """Get inference item with caption and metadata only."""
        item = self.captions[idx]
        
        # Extract instruments
        inst_ids = self._extract_instruments(item)
        
        # Return minimal data - collate will handle text tokenization
        return {
            "caption": item["caption"],
            "inst_ids": inst_ids,
            "metadata": item
        }


class MusicCollator:
    """Flexible collate function for batching music data."""
    
    def __init__(
        self,
        text_tokenizer,
        inst_pad_id: int,
        mode: str = "train",
        max_seq_len: int = 2048,
        text_max_len: int = 128
    ):
        """Initialize collator.
        
        Args:
            text_tokenizer: T5 tokenizer for text
            inst_pad_id: Padding ID for instruments
            mode: Collation mode ('train' or 'inference')
            max_seq_len: Maximum MIDI sequence length
            text_max_len: Maximum text length
        """
        self.text_tokenizer = text_tokenizer
        self.inst_pad_id = inst_pad_id
        self.mode = mode
        self.max_seq_len = max_seq_len
        self.text_max_len = text_max_len
    
    def __call__(self, batch: List[Optional[Dict]]) -> Optional[Dict]:
        """Collate batch of items."""
        # Filter out None items
        batch = [item for item in batch if item is not None]
        if not batch:
            return None
        
        # 1. Tokenize text (dynamic padding per batch)
        captions = [item["caption"] for item in batch]
        text_encoded = self.text_tokenizer(
            captions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.text_max_len
        )
        
        # 2. Collate instruments (dynamic padding)
        inst_ids_batch, inst_mask_batch = self._collate_instruments(batch)
        
        # Build result
        result = {
            "input_ids": text_encoded.input_ids,
            "attention_mask": text_encoded.attention_mask,
            "inst_ids": inst_ids_batch,
            "inst_mask": inst_mask_batch,
        }
        
        # 3. Training-specific: collate MIDI tokens
        if self.mode == "train" and "midi_tokens" in batch[0]:
            midi_labels = self._collate_midi_tokens(batch)
            result["labels"] = midi_labels
        
        # 4. Inference-specific: preserve metadata
        if self.mode == "inference":
            result["raw_items"] = [item["metadata"] for item in batch]
            result["ids"] = [item["metadata"].get("id", item["metadata"].get("md5_id")) for item in batch]
            result["locations"] = [item["metadata"]["location"] for item in batch]
            result["captions"] = captions
        
        return result
    
    def _collate_instruments(self, batch: List[Dict]) -> tuple:
        """Collate instrument IDs with dynamic padding."""
        all_inst_ids = [item["inst_ids"] for item in batch]
        
        # Handle empty instrument lists
        if not any(all_inst_ids):
            # All empty - return padding tokens
            return (
                torch.full((len(batch), 1), self.inst_pad_id, dtype=torch.long),
                torch.zeros((len(batch), 1), dtype=torch.long)
            )
        
        max_inst_len = max(len(ids) if ids else 1 for ids in all_inst_ids)
        
        # Pad to batch max
        padded_inst = torch.full((len(batch), max_inst_len), self.inst_pad_id, dtype=torch.long)
        inst_mask = torch.zeros((len(batch), max_inst_len), dtype=torch.long)
        
        for i, inst_ids in enumerate(all_inst_ids):
            if inst_ids:
                padded_inst[i, :len(inst_ids)] = torch.tensor(inst_ids, dtype=torch.long)
                inst_mask[i, :len(inst_ids)] = 1
        
        return padded_inst, inst_mask
    
    def _collate_midi_tokens(self, batch: List[Dict]) -> torch.Tensor:
        """Collate MIDI tokens with padding."""
        all_tokens = [item["midi_tokens"] for item in batch]
        
        # Dynamic padding to batch max (capped at global max)
        max_len = min(max(len(t) for t in all_tokens), self.max_seq_len)
        
        labels = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, tokens in enumerate(all_tokens):
            length = min(len(tokens), max_len)
            labels[i, :length] = torch.tensor(tokens[:length], dtype=torch.long)
        
        return labels


def create_dataloader(
    configs: Dict,
    captions: List[Dict],
    remi_tokenizer,
    mode: str = "train",
    batch_size: int = 32,
    num_workers: int = 4,
    **kwargs
) -> DataLoader:
    """Factory function to create appropriate dataloader.
    
    Args:
        configs: Configuration dictionary
        captions: List of caption dictionaries
        remi_tokenizer: REMI tokenizer instance
        mode: Mode ('train', 'val', 'test', 'inference')
        batch_size: Batch size
        num_workers: Number of worker processes
        **kwargs: Additional arguments (split, limit, shuffle, etc.)
    
    Returns:
        DataLoader instance
    """
    # Choose dataset class based on mode
    if mode in ["train", "training", "val", "validation"]:
        dataset = TrainingMusicDataset(
            configs=configs,
            captions=captions,
            remi_tokenizer=remi_tokenizer,
            mode=mode,
            split=kwargs.get("split"),
            shuffle=kwargs.get("shuffle", mode == "train"),
            use_augmentation=kwargs.get("use_augmentation", mode == "train"),
            use_cache=kwargs.get("use_cache", True),
            limit=kwargs.get("limit")
        )
    else:  # inference/test
        dataset = InferenceMusicDataset(
            configs=configs,
            captions=captions,
            remi_tokenizer=remi_tokenizer,
            mode="inference",
            split=kwargs.get("split"),
            limit=kwargs.get("limit")
        )
    
    # Create collator
    text_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    max_seq_len = configs["model"]["text2midi_model"]["decoder_max_sequence_length"]
    
    collator = MusicCollator(
        text_tokenizer=text_tokenizer,
        inst_pad_id=dataset.no_instr_token_id,
        mode="train" if mode in ["train", "training", "val", "validation"] else "inference",
        max_seq_len=max_seq_len,
        text_max_len=kwargs.get("text_max_len", 128)
    )
    
    # Create dataloader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == "train" and kwargs.get("shuffle", True)),
        num_workers=num_workers,
        collate_fn=collator,
        drop_last=(mode == "train"),
        pin_memory=True,
        prefetch_factor=kwargs.get("prefetch_factor", 2) if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )
