"""Main training script for Text2MIDI model.

This is the entry point for training. It uses the modular training components
from the training/ module for clean separation of concerns.

Usage:
    python model_aug/train.py --stage finetune
    python model_aug/train.py --resume checkpoints/epoch_10
    python model_aug/train.py --stage finetune --lr 5e-5
    python model_aug/train.py --resume checkpoints/epoch_62 --stage finetune --reset_optimizer
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path to allow for absolute imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training import Trainer, ConfigManager


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Text2MIDI model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="pretrain",
        choices=["pretrain", "finetune"],
        help="Training stage: 'pretrain' (70 epochs no inst + 30 epochs inst) or 'finetune' (30 epochs inst)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume from"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate from config (e.g., 1e-4, 5e-5)"
    )
    parser.add_argument(
        "--reset_optimizer",
        action="store_true",
        help="Reset optimizer state when resuming from a checkpoint. Useful for starting a new finetuning phase."
    )
    return parser.parse_args()


def main():
    """Main training entry point."""
    # Parse arguments
    args = parse_args()
    
    # Resolve config path
    config_path = Path(__file__).parent.parent / args.config
    
    # Initialize config manager
    config_manager = ConfigManager(str(config_path))
    
    # Override learning rate if provided
    if args.lr is not None:
        print(f"Overriding learning rate: {config_manager.learning_rate:.2e} -> {args.lr:.2e}")
        config_manager.override_learning_rate(args.lr)
    
    # Initialize and run trainer
    trainer = Trainer(config_manager, args)
    trainer.train()


if __name__ == "__main__":
    main()
