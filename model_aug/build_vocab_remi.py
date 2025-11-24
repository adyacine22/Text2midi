import argparse
import os
import pickle

import yaml
from miditok import REMI, TokenizerConfig  # base REMI, optional fallback

from remi_z_tokenizer import RemiZTokenizer, run_inline_test

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    default=os.path.normpath("configs/config.yaml"),
    help="Path to the config file",
)
parser.add_argument(
    "--tokenizer",
    choices=["remi", "remi_z"],
    default="remi_z",
    help="Tokenizer to instantiate and persist",
)
parser.add_argument(
    "--include-time-signature",
    action="store_true",
    help="Include time signature tokens (REMI-z only)",
)
parser.add_argument(
    "--include-tempo",
    action="store_true",
    help="Include tempo tokens (REMI-z only)",
)
parser.add_argument(
    "--run-inline-test",
    action="store_true",
    help="Run a small REMI-z smoke test instead of building from data",
)
args = parser.parse_args()

# Load config file
with open(args.config, "r") as f:
    configs = yaml.safe_load(f)

artifact_folder = configs["artifact_folder"]
tokenizer_filename = configs.get("tokenizer_file", "vocab_remi_z.pkl")

# Our parameters
# Load parameters from config
tokenizer_config = configs.get("tokenizer", {})

# Helper to convert string keys "0,1" to tuple keys (0, 1) for beat_res
raw_beat_res = tokenizer_config.get("beat_res", {(0, 1): 12, (1, 2): 4, (2, 4): 2, (4, 8): 1})
beat_res = {}
for k, v in raw_beat_res.items():
    if isinstance(k, str):
        try:
            # Parse "0,1" -> (0, 1)
            parts = k.split(',')
            key = (int(parts[0]), int(parts[1]))
            beat_res[key] = v
        except:
            print(f"Warning: Could not parse beat_res key {k}, skipping")
    else:
        beat_res[k] = v

TOKENIZER_PARAMS = {
    "pitch_range": tuple(tokenizer_config.get("pitch_range", (21, 109))),
    "beat_res": beat_res,
    "num_velocities": tokenizer_config.get("num_velocities", 32),
    "special_tokens": tokenizer_config.get("special_tokens", ["PAD", "BOS", "EOS", "MASK"]),
    "use_chords": tokenizer_config.get("use_chords", False),
    "use_rests": tokenizer_config.get("use_rests", False),
    "use_tempos": tokenizer_config.get("use_tempos", True),
    "use_time_signatures": tokenizer_config.get("use_time_signatures", True),
    "use_programs": tokenizer_config.get("use_programs", True),
    "num_tempos": tokenizer_config.get("num_tempos", 32),
    "tempo_range": tuple(tokenizer_config.get("tempo_range", (40, 250))),
}
config = TokenizerConfig(**TOKENIZER_PARAMS)

if args.tokenizer == "remi_z":
    tokenizer = RemiZTokenizer(
        config,
        include_time_signature=args.include_time_signature,
        include_tempo=args.include_tempo,
    )
    if args.run_inline_test:
        run_inline_test()
else:
    tokenizer = REMI(config)

# Persist the tokenizer for downstream training
vocab_path = os.path.join(artifact_folder, tokenizer_filename)
with open(vocab_path, "wb") as f:
    pickle.dump(tokenizer, f)

print(f"Using tokenizer: {args.tokenizer}")
print(f"Vocabulary length: {tokenizer.vocab_size}")
print(f"Vocabulary saved to {vocab_path}")
