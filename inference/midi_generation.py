"""MIDI generation from text using legacy or new model."""

import argparse
import logging
import json
import random
import sys
from pathlib import Path
from datetime import datetime

# Add project root to python path to allow imports from inference package
sys.path.append(str(Path(__file__).parent.parent))

from inference.legacy_model_helper import LegacyModelHelper
from inference.new_model_helper import NewModelHelper
from utils.instruments_mapping import INSTRUMENT_CLASSES


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=level,
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def interactive_mode(model_type: str):
    """Run interactive mode for gathering user input."""
    print("\n" + "=" * 60)
    print("Interactive MIDI Generation")
    print("=" * 60)
    
    prompt = ""
    allowed_instruments = []
    
    while True:
        print("\nSelect input method:")
        print("1. Manual Entry")
        print("2. Random Example from Dataset")
        choice = input("Enter choice (1/2): ").strip()
        
        if choice == "1":
            prompt = input("\nEnter text caption: ").strip()
            while not prompt:
                print("Caption cannot be empty.")
                prompt = input("Enter text caption: ").strip()
            break
            
        elif choice == "2":
            captions_path = Path("captions/all_captions.json")
            if not captions_path.exists():
                print(f"Error: Captions file not found at {captions_path}")
                continue
                
            print("Loading captions...")
            try:
                # Read file line by line to avoid loading everything into memory if possible, 
                # but for random selection we might need to seek or read all.
                # Given the file size (~100MB), reading lines is fine.
                with open(captions_path, 'r') as f:
                    lines = f.readlines()
                
                if not lines:
                    print("Error: Captions file is empty.")
                    continue
                    
                random_line = random.choice(lines)
                data = json.loads(random_line)
                
                prompt = data.get("caption", "")
                instruments = data.get("instrument_classes", [])
                
                print(f"\nSelected Caption: {prompt}")
                print(f"Original Instruments: {', '.join(instruments) if instruments else 'None'}")
                
                confirm = input("Use this caption? (y/n): ").strip().lower()
                if confirm == 'y':
                    # Ask if user wants to use original instruments
                    if model_type == "new" and instruments:
                        use_orig_inst = input("Use original instruments for conditioning? (y/n): ").strip().lower()
                        if use_orig_inst == 'y':
                            allowed_instruments = instruments
                    break
            except Exception as e:
                print(f"Error reading captions: {e}")
        else:
            print("Invalid choice. Please try again.")

    # Instrument selection for new model if not already set
    if model_type == "new" and not allowed_instruments:
        print("\nInstrument Conditioning (Optional)")
        print("Available instruments:")
        inst_list = list(INSTRUMENT_CLASSES.keys())
        for i, inst in enumerate(inst_list):
            print(f"{i+1}. {inst}")
            
        print("\nEnter comma-separated numbers to select instruments (e.g., '1, 5')")
        print("Or press Enter to skip (no specific instrument conditioning)")
        
        inst_choice = input("Selection: ").strip()
        if inst_choice:
            try:
                indices = [int(x.strip()) - 1 for x in inst_choice.split(",") if x.strip()]
                selected = []
                for idx in indices:
                    if 0 <= idx < len(inst_list):
                        selected.append(inst_list[idx])
                
                if selected:
                    allowed_instruments = selected
                    print(f"Selected: {', '.join(allowed_instruments)}")
                else:
                    print("No valid instruments selected.")
            except ValueError:
                print("Invalid input format. Skipping instrument conditioning.")
    
    return prompt, allowed_instruments


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate MIDI from text description",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["legacy", "new"],
        default="new",
        help="Model type to use"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text description of the music to generate (required unless --interactive)"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="generated_midi",
        help="Output directory for generated MIDI files"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (uses default if not specified)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (new model only, uses default if not specified)"
    )
    
    parser.add_argument(
        "--max-len",
        type=int,
        default=1024,
        help="Maximum generation length"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (default: 0.9 for legacy, 0.7 for new)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Device to use for inference"
    )
    
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Custom output filename (without extension)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def main():
    """Main generation function."""
    args = parse_args()
    
    # Validate args
    if not args.interactive and not args.prompt:
        print("Error: --prompt is required unless --interactive is specified.")
        return 1
        
    setup_logging(args.verbose)
    
    logger = logging.getLogger(__name__)
    
    # Print header
    logger.info("=" * 60)
    logger.info("MIDI Generation from Text")
    logger.info("=" * 60)
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Device: {args.device}")
    logger.info("")
    
    # Interactive mode
    prompt = args.prompt
    allowed_instruments = []
    
    if args.interactive:
        prompt, allowed_instruments = interactive_mode(args.model_type)
        logger.info(f"Prompt: {prompt}")
        if allowed_instruments:
            logger.info(f"Instruments: {allowed_instruments}")
    
    # Set default temperature based on model type
    temperature = args.temperature
    if temperature is None:
        temperature = 0.9 if args.model_type == "legacy" else 0.7
    
    # Initialize helper
    logger.info("Initializing model helper...")
    if args.model_type == "legacy":
        helper = LegacyModelHelper(device=args.device)
    else:
        helper = NewModelHelper(device=args.device)
    
    try:
        # Load model
        logger.info("")
        if args.model_type == "legacy":
            helper.load_model(checkpoint_path=args.checkpoint)
        else:
            helper.load_model(
                checkpoint_path=args.checkpoint,
                config_path=args.config
            )
        
        # Generate
        logger.info("")
        
        gen_kwargs = {
            "prompt": prompt,
            "max_len": args.max_len,
            "temperature": temperature
        }
        
        if args.model_type == "new" and allowed_instruments:
            gen_kwargs["allowed_instruments"] = allowed_instruments
            
        output = helper.generate(**gen_kwargs)
        
        # Save
        logger.info("")
        output_dir = Path(args.output_dir)
        
        if args.output_name:
            output_filename = f"{args.output_name}.mid"
        else:
            # Use timestamp-based naming
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"output_{timestamp}.mid"
        
        output_path = output_dir / output_filename
        helper.save_midi(output, output_path)
        
        # Success message
        logger.info("")
        logger.info("=" * 60)
        logger.info("✓ Generation completed successfully!")
        logger.info(f"Output: {output_path.absolute()}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        return 1
    
    finally:
        # Cleanup
        logger.info("")
        logger.info("Cleaning up resources...")
        helper.cleanup()
    
    return 0


if __name__ == "__main__":
    exit(main())
