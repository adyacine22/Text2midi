"""
Test script for fine-tuning Text2MIDI model with a small subset of captions
This script allows you to quickly test the training pipeline without processing the entire dataset
"""

import os
import sys
import torch.nn as nn
import torch.optim as optim
import yaml
import pickle
import jsonlines
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add model directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))
from data_loader_remi import Text2MusicDataset
from transformer_model import Transformer


def test_finetuning(num_samples=100, num_epochs=2, batch_size=2):
    """
    Test fine-tuning with a small subset of data

    Args:
        num_samples: Number of caption samples to use for testing (default: 100)
        num_epochs: Number of epochs to train (default: 2)
        batch_size: Batch size for training (default: 2)
    """

    print(f"\n{'='*60}")
    print(f"Testing Fine-tuning with {num_samples} samples")
    print(f"{'='*60}\n")

    # Load config
    config_file = "configs/config.yaml"
    with open(config_file, "r") as f:
        configs = yaml.safe_load(f)

    # Device setup
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}\n")

    # Load tokenizer
    artifact_folder = configs["artifact_folder"]
    tokenizer_filepath = os.path.join(artifact_folder, "vocab_remi.pkl")

    print(f"Loading tokenizer from: {tokenizer_filepath}")
    with open(tokenizer_filepath, "rb") as f:
        tokenizer = pickle.load(f)

    vocab_size = len(tokenizer)
    print(f"Vocabulary size: {vocab_size}\n")

    # Load captions
    caption_dataset_path = configs["raw_data"]["caption_dataset_path"]
    print(f"Loading captions from: {caption_dataset_path}")

    with jsonlines.open(caption_dataset_path) as reader:
        all_captions = list(reader)

    print(f"Total captions available: {len(all_captions)}")

    # Use only a subset for testing
    captions = all_captions[:num_samples]
    print(f"Using {len(captions)} captions for testing\n")

    # Create dataset
    print("Creating dataset...")
    dataset = Text2MusicDataset(
        configs, captions, remi_tokenizer=tokenizer, mode="train", shuffle=True
    )

    # Collate function
    def collate_fn(batch):
        input_ids = [item[0].squeeze(0) for item in batch]
        input_ids = nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=0
        )
        attention_mask = [item[1].squeeze(0) for item in batch]
        attention_mask = nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        labels = [item[2].squeeze(0) for item in batch]
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
        return input_ids, attention_mask, labels

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for testing to avoid multiprocessing issues
        collate_fn=collate_fn,
        drop_last=True,
    )

    print(f"DataLoader created with batch size: {batch_size}")
    print(f"Number of batches: {len(dataloader)}\n")

    # Model parameters from config
    d_model = configs["model"]["text2midi_model"]["decoder_d_model"]
    nhead = configs["model"]["text2midi_model"]["decoder_num_heads"]
    num_layers = configs["model"]["text2midi_model"]["decoder_num_layers"]
    max_len = configs["model"]["text2midi_model"]["decoder_max_sequence_length"]
    dim_feedforward = configs["model"]["text2midi_model"]["decoder_intermediate_size"]
    use_moe = configs["model"]["text2midi_model"]["use_moe"]
    num_experts = configs["model"]["text2midi_model"]["num_experts"]

    print("Model Configuration:")
    print(f"  - Layers: {num_layers}")
    print(f"  - Attention heads: {nhead}")
    print(f"  - Model dimension: {d_model}")
    print(f"  - Feed-forward dimension: {dim_feedforward}")
    print(f"  - Max sequence length: {max_len}")
    print(f"  - Use MoE: {use_moe}\n")

    # Create model
    print("Creating model...")
    model = Transformer(
        vocab_size,
        d_model,
        nhead,
        max_len,
        num_layers,
        dim_feedforward,
        use_moe,
        num_experts,
        device=device,
    )

    # Load pretrained weights (optional - comment out to train from scratch)
    pretrained_path = os.path.join(artifact_folder, "pytorch_model.bin")
    if os.path.exists(pretrained_path):
        print(f"Loading pretrained weights from: {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        print("Pretrained weights loaded successfully!")
    else:
        print(f"No pretrained weights found at {pretrained_path}")
        print("Starting training from scratch...")

    model = model.to(device)
    model.train()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}\n")

    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    print(f"{'='*60}")
    print(f"Starting Training for {num_epochs} epochs")
    print(f"{'='*60}\n")

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        epoch_losses = []

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            encoder_input, attention_mask, tgt = batch
            encoder_input = encoder_input.to(device)
            attention_mask = attention_mask.to(device)
            tgt = tgt.to(device)

            # Prepare input and target
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            # Forward pass
            optimizer.zero_grad()

            if use_moe:
                outputs, aux_loss = model(encoder_input, attention_mask, tgt_input)
            else:
                outputs = model(encoder_input, attention_mask, tgt_input)
                aux_loss = 0

            # Calculate loss
            loss = criterion(outputs.view(-1, outputs.size(-1)), tgt_output.reshape(-1))
            loss += aux_loss

            # Backward pass
            loss.backward()
            optimizer.step()

            # Track loss
            loss_value = loss.item()
            total_loss += loss_value
            epoch_losses.append(loss_value)

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "loss": f"{loss_value:.4f}",
                    "avg_loss": f"{total_loss/(batch_idx+1):.4f}",
                }
            )

        avg_loss = total_loss / len(dataloader)
        print(f"\nEpoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}\n")

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}\n")

    # Save the fine-tuned model
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)

    model_save_path = os.path.join(output_dir, "test_finetuned_model.bin")
    torch.save(model.state_dict(), model_save_path)
    print(f"Fine-tuned model saved to: {model_save_path}")

    # Test generation
    print("\n" + "=" * 60)
    print("Testing Generation")
    print("=" * 60 + "\n")

    model.eval()

    from transformers import T5Tokenizer

    t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

    test_prompt = "A melodic piano piece in C major with a relaxing atmosphere"
    print(f"Test prompt: {test_prompt}\n")

    inputs = t5_tokenizer(
        test_prompt, return_tensors="pt", padding=True, truncation=True
    )
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    print("Generating MIDI...")
    with torch.no_grad():
        output = model.generate(input_ids, attention_mask, max_len=500, temperature=1.0)

    output_list = output[0].tolist()
    generated_midi = tokenizer.decode(output_list)

    test_output_path = os.path.join(output_dir, "test_generated.mid")
    generated_midi.dump_midi(test_output_path)
    print(f"Generated MIDI saved to: {test_output_path}\n")

    print("Test complete! ✓")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test fine-tuning with a subset of captions"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of samples to use (default: 100)",
    )
    parser.add_argument(
        "--epochs", type=int, default=2, help="Number of epochs (default: 2)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size (default: 2)"
    )

    args = parser.parse_args()

    test_finetuning(
        num_samples=args.samples, num_epochs=args.epochs, batch_size=args.batch_size
    )
