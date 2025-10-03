"""Model loading and initialization utilities"""

import os
import sys
import pickle
import torch
import streamlit as st
from pathlib import Path

# Add parent directories to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "model"))

from transformer_model import Transformer
from transformers import T5Tokenizer


def get_device():
    """Detect and return the best available device"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


@st.cache_resource
def load_text2midi_model():
    """
    Load and cache the Text2MIDI model

    Returns:
        tuple: (model, remi_tokenizer, t5_tokenizer, device, model_info)
    """
    try:
        device = get_device()

        # Paths
        artifact_folder = ROOT_DIR / "artifacts"
        tokenizer_path = artifact_folder / "vocab_remi.pkl"
        model_path = artifact_folder / "pytorch_model.bin"

        # Check if files exist
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"REMI tokenizer not found at {tokenizer_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found at {model_path}")

        # Load REMI tokenizer
        with open(tokenizer_path, "rb") as f:
            remi_tokenizer = pickle.load(f)

        vocab_size = len(remi_tokenizer)

        # Model parameters (from config)
        d_model = 768
        nhead = 8
        num_layers = 18
        max_len = 2048
        dim_feedforward = 1024
        use_moe = False
        num_experts = 8

        # Create model
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

        # Load weights
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        model.eval()
        model = model.to(device)

        # Load T5 tokenizer for text encoding
        t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

        # Model info
        total_params = sum(p.numel() for p in model.parameters())
        model_info = {
            "architecture": f"Transformer ({num_layers} layers, {nhead} heads)",
            "parameters": f"{total_params:,}",
            "vocab_size": vocab_size,
            "context_length": max_len,
            "device": device.upper(),
            "d_model": d_model,
            "dim_feedforward": dim_feedforward,
        }

        return model, remi_tokenizer, t5_tokenizer, device, model_info

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        raise e


def get_model_info_display(model_info):
    """
    Format model information for display

    Args:
        model_info: Dictionary containing model information

    Returns:
        str: Formatted model information
    """
    info_text = f"""
    **Model Details:**
    - Architecture: {model_info['architecture']}
    - Parameters: {model_info['parameters']}
    - Vocabulary Size: {model_info['vocab_size']} tokens (REMI)
    - Context Length: {model_info['context_length']} tokens
    - Model Dimension: {model_info['d_model']}
    - Feed-forward Dimension: {model_info['dim_feedforward']}
    
    **System Information:**
    - Device: {model_info['device']}
    - PyTorch Version: {torch.__version__}
    """
    return info_text
