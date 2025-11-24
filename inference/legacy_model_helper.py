"""Legacy model helper for MIDI generation."""

import pickle
import logging
from pathlib import Path
from typing import Optional, Any
import torch
from transformers import T5Tokenizer

from inference.base_helper import BaseModelHelper
from model_legacy.transformer_model import Transformer

logger = logging.getLogger(__name__)


class LegacyModelHelper(BaseModelHelper):
    """Helper for legacy REMI model."""
    
    def load_model(self, checkpoint_path: Optional[str] = None, **kwargs):
        """Load legacy model and tokenizers.
        
        Args:
            checkpoint_path: Path to model checkpoint (default: artifacts/pytorch_model.bin)
        """
        logger.info("Loading legacy model...")
        logger.info(f"Device: {self.device}")
        
        # Default paths
        if checkpoint_path is None:
            checkpoint_path = "artifacts/pytorch_model.bin"
        tokenizer_path = "artifacts/vocab_remi.pkl"
        
        # Load REMI tokenizer
        logger.info(f"Loading REMI tokenizer from {tokenizer_path}")
        with open(tokenizer_path, "rb") as f:
            self.remi_tokenizer = pickle.load(f)
        vocab_size = len(self.remi_tokenizer)
        logger.info(f"Vocabulary size: {vocab_size}")
        
        # Load model with legacy architecture
        logger.info("Initializing model with architecture:")
        logger.info("  d_model=768, nhead=8, max_len=2048")
        logger.info("  num_layers=18, dim_feedforward=1024")
        logger.info("  use_moe=False, num_experts=8")
        
        self.model = Transformer(
            n_vocab=vocab_size,
            d_model=768,
            nhead=8,
            max_len=2048,
            num_decoder_layers=18,
            dim_feedforward=1024,
            use_moe=False,
            num_experts=8,
            device=self.device
        )
        
        # Load checkpoint
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # Disable gradients
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Load text tokenizer
        logger.info("Loading T5 tokenizer")
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
        
        logger.info("✓ Legacy model loaded successfully")
    
    def generate(self, prompt: str, max_len: int = 1024, temperature: float = 0.9, **kwargs) -> Any:
        """Generate MIDI from text prompt.
        
        Args:
            prompt: Text description of the music
            max_len: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            Generated token IDs (list)
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        logger.info(f"Generating MIDI for prompt: '{prompt}'")
        logger.info(f"Parameters: max_len={max_len}, temperature={temperature}")
        
        # Tokenize input
        logger.info("Tokenizing input...")
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        
        # Generate
        logger.info("Generating tokens...")
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                attention_mask,
                max_len=max_len,
                temperature=temperature
            )
        
        # Extract result
        output_list = output[0].cpu().tolist()
        logger.info(f"✓ Generated {len(output_list)} tokens")
        
        return output_list
    
    def save_midi(self, output: Any, output_path: Path):
        """Save generated output as MIDI file.
        
        Args:
            output: Generated token IDs (list)
            output_path: Path to save MIDI file
        """
        if self.remi_tokenizer is None:
            raise RuntimeError("REMI tokenizer not loaded")
        
        logger.info("Converting tokens to MIDI...")
        generated_midi = self.remi_tokenizer.decode(output)
        
        logger.info(f"Saving MIDI to {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        generated_midi.dump_midi(str(output_path))
        
        logger.info(f"✓ MIDI saved successfully: {output_path}")
