"""New model helper for MIDI generation with REMI-z tokenization."""

import pickle
import logging
import yaml
import sys
from pathlib import Path
from typing import Optional, Any
import torch
from transformers import T5Tokenizer
from miditok import TokenizerConfig

from inference.base_helper import BaseModelHelper
from model_aug.transformer_model import Transformer
from model_aug.remi_z_tokenizer import RemiZTokenizer
from utils.instruments_mapping import INSTRUMENT_CLASSES

logger = logging.getLogger(__name__)


class NewModelHelper(BaseModelHelper):
    """Helper for new REMI-z model with instrument conditioning."""
    
    def load_model(self, checkpoint_path: Optional[str] = None, config_path: Optional[str] = None, **kwargs):
        """Load new model and tokenizers.
        
        Args:
            checkpoint_path: Path to model checkpoint (default: checkpoints/epoch_18/pytorch_model.bin)
            config_path: Path to config file (default: configs/config.yaml)
        """
        logger.info("Loading new REMI-z model...")
        logger.info(f"Device: {self.device}")
        
        # Default paths
        if checkpoint_path is None:
            checkpoint_path = "checkpoints/epoch_18/pytorch_model.bin"
        if config_path is None:
            config_path = "configs/config.yaml"
        tokenizer_path = "artifacts/vocab_remi_z.pkl"
        
        # Load config
        logger.info(f"Loading configuration from {config_path}")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Load REMI-z tokenizer
        logger.info(f"Loading REMI-z tokenizer from {tokenizer_path}")
        
        # Import for pickle
        from model_aug import remi_z_tokenizer
        sys.modules['remi_z_tokenizer'] = remi_z_tokenizer
        
        with open(tokenizer_path, "rb") as f:
            self.remi_tokenizer = pickle.load(f)
        
        # Rebuild tokenizer with fresh instance
        if isinstance(self.remi_tokenizer, RemiZTokenizer):
            saved_vocab = self.remi_tokenizer.vocab
            saved_token_to_id = self.remi_tokenizer.token_to_id
        else:
            saved_vocab = getattr(self.remi_tokenizer, "vocab", None)
            saved_token_to_id = getattr(self.remi_tokenizer, "token_to_id", None)
        
        # Load tokenizer config from config.yaml
        if config and "tokenizer" in config:
            tok_config = config["tokenizer"]
            beat_res_yaml = tok_config.get("beat_res", {})
            beat_res = {}
            for key_str, val in beat_res_yaml.items():
                parts = [int(x) for x in key_str.split(",")]
                beat_res[tuple(parts)] = val
            
            cfg = TokenizerConfig(
                pitch_range=tuple(tok_config.get("pitch_range", [0, 127])),
                beat_res=beat_res if beat_res else {(0, 1): 12, (1, 2): 4, (2, 4): 2, (4, 8): 1},
                num_velocities=tok_config.get("num_velocities", 32),
                special_tokens=tok_config.get("special_tokens", ["PAD", "BOS", "EOS", "MASK"]),
                use_chords=tok_config.get("use_chords", False),
                use_rests=tok_config.get("use_rests", False),
                use_tempos=tok_config.get("use_tempos", False),
                use_time_signatures=tok_config.get("use_time_signatures", False),
                use_programs=tok_config.get("use_programs", True),
                num_tempos=tok_config.get("num_tempos", 32),
                tempo_range=tuple(tok_config.get("tempo_range", [40, 250])),
            )
            logger.info("Loaded tokenizer config from config.yaml")
        else:
            cfg = TokenizerConfig(
                pitch_range=(0, 127),
                beat_res={(0, 1): 12, (1, 2): 4, (2, 4): 2, (4, 8): 1},
                num_velocities=32,
                special_tokens=["PAD", "BOS", "EOS", "MASK"],
                use_chords=False,
                use_rests=False,
                use_tempos=False,
                use_time_signatures=False,
                use_programs=True,
                num_tempos=32,
                tempo_range=(40, 250),
            )
            logger.info("Using default tokenizer config")
        
        fresh_tokenizer = RemiZTokenizer(cfg)
        
        if saved_vocab is not None and saved_token_to_id is not None:
            fresh_tokenizer.vocab = list(saved_vocab)
            fresh_tokenizer.token_to_id = dict(saved_token_to_id)
            fresh_tokenizer.id_to_token = {v: k for k, v in saved_token_to_id.items()}
        
        self.remi_tokenizer = fresh_tokenizer
        vocab_size = len(self.remi_tokenizer)
        logger.info(f"Vocabulary size: {vocab_size}")
        
        # Load model architecture from config
        if config and "model" in config and "text2midi_model" in config["model"]:
            model_config = config["model"]["text2midi_model"]
            d_model = model_config.get("decoder_d_model", 768)
            nhead = model_config.get("decoder_num_heads", 8)
            num_decoder_layers = model_config.get("decoder_num_layers", 18)
            dim_feedforward = model_config.get("decoder_intermediate_size", 1024)
            max_len = model_config.get("decoder_max_sequence_length", 2048)
            use_instrument_conditioning = model_config.get("use_instrument_conditioning", True)
            
            logger.info("Model architecture from config:")
            logger.info(f"  d_model={d_model}, nhead={nhead}, max_len={max_len}")
            logger.info(f"  num_layers={num_decoder_layers}, dim_feedforward={dim_feedforward}")
            logger.info(f"  use_instrument_conditioning={use_instrument_conditioning}")
        else:
            d_model = 768
            nhead = 8
            num_decoder_layers = 18
            dim_feedforward = 1024
            max_len = 2048
            use_instrument_conditioning = True
            logger.warning("Using default model architecture")
        
        # Instrument classes
        instrument_class_names = list(INSTRUMENT_CLASSES.keys())
        instrument_vocab_size = len(instrument_class_names)
        instrument_pad_id = instrument_vocab_size
        
        self.instrument_class_to_id = {
            name.lower(): idx for idx, name in enumerate(instrument_class_names)
        }
        self.instrument_pad_id = instrument_pad_id
        
        logger.info(f"Instrument classes: {instrument_vocab_size}")
        
        # Initialize model
        self.model = Transformer(
            n_vocab=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            max_len=max_len,
            use_instrument_conditioning=use_instrument_conditioning,
            instrument_vocab_size=instrument_vocab_size,
            no_instr_token_id=instrument_pad_id,
            device=self.device,
            tokenizer=self.remi_tokenizer,
        )
        
        # Load checkpoint
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        
        # Disable gradients
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Load text tokenizer
        logger.info("Loading T5 tokenizer")
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
        
        logger.info("✓ New REMI-z model loaded successfully")
    
    def generate(self, prompt: str, max_len: int = 1024, temperature: float = 0.7, **kwargs) -> Any:
        """Generate MIDI from text prompt.
        
        Args:
            prompt: Text description of the music
            max_len: Maximum generation length
            temperature: Sampling temperature
            **kwargs: Additional arguments, including 'allowed_instruments' (list of str)
            
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
        
        # Prepare instrument conditioning
        inst = None
        inst_mask = None
        allowed_instruments = kwargs.get("allowed_instruments")
        
        if allowed_instruments:
            logger.info(f"Conditioning on instruments: {allowed_instruments}")
            inst_ids = []
            for name in allowed_instruments:
                name_lower = name.lower()
                if name_lower in self.instrument_class_to_id:
                    inst_ids.append(self.instrument_class_to_id[name_lower])
                else:
                    logger.warning(f"Instrument '{name}' not found in vocabulary, skipping.")
            
            if inst_ids:
                # Create tensor (batch_size=1, num_instruments)
                inst = torch.tensor(inst_ids, dtype=torch.long, device=self.device).unsqueeze(0)
                inst_mask = torch.ones_like(inst)
        
        # Generate
        logger.info("Generating tokens...")
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                attention_mask,
                max_len=max_len,
                temperature=temperature,
                inst=inst,
                inst_mask=inst_mask
            )
        
        # Extract result
        output_list = output[0].cpu().tolist()
        logger.info(f"✓ Generated {len(output_list)} tokens")
        
        # Convert to token strings
        tokens = [self.remi_tokenizer.id_to_token.get(t_id, "PAD_None") for t_id in output_list]
        
        return tokens
    
    def save_midi(self, output: Any, output_path: Path):
        """Save generated output as MIDI file.
        
        Args:
            output: Generated tokens (list of strings)
            output_path: Path to save MIDI file
        """
        if self.remi_tokenizer is None:
            raise RuntimeError("REMI tokenizer not loaded")
        
        logger.info("Converting tokens to MIDI...")
        midi_score = self.remi_tokenizer.tokens_to_score(output)
        
        logger.info(f"Saving MIDI to {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        midi_score.dump_midi(str(output_path))
        
        logger.info(f"✓ MIDI saved successfully: {output_path}")
