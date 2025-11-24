"""Base helper class for MIDI generation inference."""

from abc import ABC, abstractmethod
from pathlib import Path
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class BaseModelHelper(ABC):
    """Abstract base class for model helpers."""
    
    def __init__(self, device: str = "auto"):
        """Initialize the helper.
        
        Args:
            device: Device to use ('cuda', 'cpu', 'mps', or 'auto')
        """
        self.device = self._resolve_device(device)
        self.model = None
        self.tokenizer = None
        self.remi_tokenizer = None
        
    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device."""
        import torch
        
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    @abstractmethod
    def load_model(self, checkpoint_path: Optional[str] = None, **kwargs):
        """Load model and tokenizers.
        
        Args:
            checkpoint_path: Optional path to model checkpoint
            **kwargs: Additional model-specific arguments
        """
        pass
    
    @abstractmethod
    def generate(self, prompt: str, max_len: int = 1024, temperature: float = 0.9, **kwargs) -> Any:
        """Generate MIDI from text prompt.
        
        Args:
            prompt: Text description of the music
            max_len: Maximum generation length
            temperature: Sampling temperature
            **kwargs: Additional generation arguments
            
        Returns:
            Generated token IDs or MIDI object
        """
        pass
    
    @abstractmethod
    def save_midi(self, output: Any, output_path: Path):
        """Save generated output as MIDI file.
        
        Args:
            output: Generated output from generate()
            output_path: Path to save MIDI file
        """
        pass
    
    def cleanup(self):
        """Clean up resources."""
        import torch
        import gc
        
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        if self.remi_tokenizer is not None:
            del self.remi_tokenizer
            
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()
            
        gc.collect()
        logger.info("Resources cleaned up")
