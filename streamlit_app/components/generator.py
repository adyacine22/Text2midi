"""MIDI generation component"""

import os
import torch
import streamlit as st
from datetime import datetime
from pathlib import Path
import torch.nn as nn


def generate_midi_from_prompt(
    prompt,
    model,
    remi_tokenizer,
    t5_tokenizer,
    device,
    max_length=1000,
    temperature=1.0,
    top_k=0,
    top_p=0.0,
):
    """
    Generate MIDI from text prompt

    Args:
        prompt: Text description
        model: Text2MIDI model
        remi_tokenizer: REMI tokenizer for MIDI
        t5_tokenizer: T5 tokenizer for text
        device: Device to run inference on
        max_length: Maximum sequence length
        temperature: Sampling temperature
        top_k: Top-k sampling (0 = disabled)
        top_p: Nucleus sampling (0.0 = disabled)

    Returns:
        tuple: (midi_object, generation_time, num_tokens)
    """
    try:
        start_time = datetime.now()

        # Tokenize text input
        inputs = t5_tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True
        )

        # Prepare inputs
        input_ids = nn.utils.rnn.pad_sequence(
            [inputs.input_ids.squeeze(0)], batch_first=True, padding_value=0
        ).to(device)

        attention_mask = nn.utils.rnn.pad_sequence(
            [inputs.attention_mask.squeeze(0)], batch_first=True, padding_value=0
        ).to(device)

        # Generate MIDI tokens
        with torch.no_grad():
            output = model.generate(
                input_ids,
                attention_mask,
                max_len=max_length,
                temperature=temperature,
            )

        # Decode to MIDI
        output_list = output[0].tolist()
        generated_midi = remi_tokenizer.decode(output_list)

        # Calculate generation time
        end_time = datetime.now()
        generation_time = (end_time - start_time).total_seconds()

        return generated_midi, generation_time, len(output_list)

    except Exception as e:
        st.error(f"Generation failed: {str(e)}")
        raise e


def save_midi_file(midi_object, output_dir, filename_prefix="generated"):
    """
    Save MIDI object to file

    Args:
        midi_object: MIDI object from tokenizer
        output_dir: Directory to save file
        filename_prefix: Prefix for filename

    Returns:
        str: Path to saved MIDI file
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.mid"
        filepath = os.path.join(output_dir, filename)

        # Save MIDI file
        midi_object.dump_midi(filepath)

        return filepath

    except Exception as e:
        st.error(f"Failed to save MIDI file: {str(e)}")
        raise e


def get_generation_summary(prompt, generation_time, num_tokens, midi_path):
    """
    Create a summary of the generation

    Args:
        prompt: Original prompt
        generation_time: Time taken to generate
        num_tokens: Number of tokens generated
        midi_path: Path to MIDI file

    Returns:
        dict: Generation summary
    """
    from utils.midi_processor import get_file_size

    summary = {
        "prompt": prompt,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "generation_time": f"{generation_time:.2f}s",
        "num_tokens": num_tokens,
        "midi_path": midi_path,
        "file_size": get_file_size(midi_path),
    }

    return summary
