import yaml
import jsonlines
import glob
import random
import os
import sys
import pickle
import json
import argparse
import numpy as np
from copy import deepcopy
from torch.utils.data import Dataset
import torch
from torch.nn import functional as F
from transformers import T5Tokenizer
from spacy.lang.en import English

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


class Text2MusicDataset(Dataset):
    def __init__(self, configs, captions, remi_tokenizer, mode="train", shuffle=False):
        self.mode = mode
        # Optional split filtering if captions include a "split" field
        split_filter = None
        if any("split" in c for c in captions):
            if mode in ("train", "training"):
                split_filter = {"train"}
            elif mode in ("val", "validation"):
                split_filter = {"validation"}
            elif mode in ("test", "eval", "evaluation"):
                split_filter = {"test"}

        if split_filter:
            self.captions = [c for c in captions if c.get("split") in split_filter]
        else:
            self.captions = list(captions)

        if shuffle:
            random.shuffle(self.captions)

        # Resolve dataset roots from config (supports multiple datasets)
        self.dataset_paths = {
            name: data.get("folder_path")
            for name, data in configs["raw_data"]["raw_data_folders"].items()
            if isinstance(data, dict) and "folder_path" in data
        }
        # Fallback root if no dataset is specified in captions
        self.default_dataset_path = self.dataset_paths.get("midicaps") or next(
            iter(self.dataset_paths.values()), "."
        )

        # Artifact folder
        self.artifact_folder = configs["artifact_folder"]
        # Load encoder tokenizer json file dictionary
        # tokenizer_filepath = os.path.join(self.artifact_folder, "vocab.pkl")
        # Load the pickled tokenizer dictionary
        # with open(tokenizer_filepath, 'rb') as f:
        #     self.tokenizer = pickle.load(f)

        self.remi_tokenizer = remi_tokenizer

        # Load the sentencizer
        self.nlp = English()
        self.nlp.add_pipe("sentencizer")

        # Load the FLAN-T5 tokenizer and encoder
        self.t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

        # Get the maximum sequence length
        self.decoder_max_sequence_length = configs["model"]["text2midi_model"][
            "decoder_max_sequence_length"
        ]

        # Print length of dataset
        print("Length of dataset: ", len(self.captions))

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        caption = self.captions[idx]["caption"]
        location = self.captions[idx]["location"]
        dataset_name = self.captions[idx].get("dataset")

        if os.path.isabs(location):
            midi_filepath = location
        else:
            base = self.dataset_paths.get(dataset_name) if dataset_name else None
            # Handle common aliases in caption files (e.g., lmd_full -> lmd)
            if base is None and dataset_name:
                alias_map = {
                    "lmd_full": "lmd",
                    "midicaps": "midicaps",
                }
                alias_key = alias_map.get(dataset_name)
                if alias_key:
                    base = self.dataset_paths.get(alias_key)
            if base is None:
                base = self.default_dataset_path
            midi_filepath = os.path.join(base, location)
        # print(f'midi filepath: {midi_filepath}')
        # Read the MIDI file
        tokens = self.remi_tokenizer(midi_filepath)

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

        # Drop a random number of sentences from the caption
        do_drop = random.random() > 0.5
        if do_drop:
            sentences = list(self.nlp(caption).sents)
            sent_length = len(sentences)
            if sent_length < 4:
                how_many_to_drop = int(
                    np.floor((20 + random.random() * 30) / 100 * sent_length)
                )  # between 20 and 50 percent of sentences
            else:
                how_many_to_drop = int(
                    np.ceil((20 + random.random() * 30) / 100 * sent_length)
                )  # between 20 and 50 percent of sentences
            which_to_drop = np.random.choice(
                sent_length, how_many_to_drop, replace=False
            )
            new_sentences = [
                sentences[i]
                for i in range(sent_length)
                if i not in which_to_drop.tolist()
            ]
            new_sentences = " ".join(
                [new_sentences[i].text for i in range(len(new_sentences))]
            )  # combine sentences back with a space
        else:
            new_sentences = caption

        # Tokenize the caption
        inputs = self.t5_tokenizer(
            new_sentences, return_tensors="pt", padding=True, truncation=True
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Convert the tokenized MIDI file to a tensor and pad it to the maximum sequence length
        if len(tokenized_midi) < self.decoder_max_sequence_length:
            labels = F.pad(
                torch.tensor(tokenized_midi),
                (0, self.decoder_max_sequence_length - len(tokenized_midi)),
            ).to(torch.int64)
        else:
            labels = torch.tensor(
                tokenized_midi[0 : self.decoder_max_sequence_length]
            ).to(torch.int64)

        return input_ids, attention_mask, labels


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.normpath("../configs/config.yaml"),
        help="Path to the config file",
    )
    args = parser.parse_args()

    # Load config file first to get proper paths
    with open(args.config, "r") as f:
        configs = yaml.safe_load(f)

    tokenizer_filepath = os.path.join(configs["artifact_folder"], "vocab_remi.pkl")
    # Load the tokenizer dictionary
    with open(tokenizer_filepath, "rb") as f:
        tokenizer = pickle.load(f)
    bos_token_number = tokenizer["PAD_None"]
    print(f"bos_token_number: {bos_token_number}")

    caption_dataset_path = configs["raw_data"]["caption_dataset_path"]
    # Load the caption dataset
    with jsonlines.open(caption_dataset_path) as reader:
        captions = list(reader)

    # Load the dataset
    dataset = Text2MusicDataset(
        configs, captions, remi_tokenizer=tokenizer, mode="train", shuffle=True
    )
    a, b, c = dataset[0]
    print(type(a))
    generated_midi = tokenizer.decode(c)
    print(type(generated_midi))
    generated_midi.dump_midi("decoded_midi.mid")
