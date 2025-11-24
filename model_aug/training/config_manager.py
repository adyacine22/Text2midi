"""Configuration manager for training."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigManager:
    """Manages configuration loading and provides easy access to config values."""
    
    def __init__(self, config_path: str):
        """Initialize config manager.
        
        Args:
            config_path: Path to YAML config file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Cache frequently accessed sections
        self._training_config = self.config.get("training", {}).get("text2midi_model", {})
        self._model_config = self.config.get("model", {}).get("text2midi_model", {})
        self._data_config = self.config.get("raw_data", {})
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)
    
    def override_learning_rate(self, lr: float):
        """Override the learning rate.
        
        Args:
            lr: New learning rate value
        """
        if "training" not in self.config:
            self.config["training"] = {}
        if "text2midi_model" not in self.config["training"]:
            self.config["training"]["text2midi_model"] = {}
        
        self.config["training"]["text2midi_model"]["learning_rate"] = lr
        self._training_config = self.config["training"]["text2midi_model"]
    
    # Training configuration properties
    @property
    def batch_size(self) -> int:
        """Per-device training batch size."""
        return self._training_config.get("per_device_train_batch_size", 32)
    
    @property
    def learning_rate(self) -> float:
        """Learning rate."""
        return self._training_config.get("learning_rate", 1e-4)
    
    @property
    def epochs(self) -> int:
        """Number of training epochs."""
        return self._training_config.get("epochs", 20)
    
    @property
    def gradient_accumulation_steps(self) -> int:
        """Gradient accumulation steps."""
        return self._training_config.get("gradient_accumulation_steps", 4)
    
    @property
    def weight_decay(self) -> float:
        """Weight decay."""
        return self._training_config.get("weight_decay", 0.01)
    
    @property
    def num_warmup_steps(self) -> int:
        """Number of warmup steps."""
        return self._training_config.get("num_warmup_steps", 500)
    
    @property
    def lr_scheduler_type(self) -> str:
        """Learning rate scheduler type."""
        return self._training_config.get("lr_scheduler_type", "linear")
    
    @property
    def output_dir(self) -> str:
        """Output directory for checkpoints."""
        return self._training_config.get("output_dir", "checkpoints")
    
    @property
    def with_tracking(self) -> bool:
        """Whether to use TensorBoard tracking."""
        return self._training_config.get("with_tracking", True)
    
    @property
    def checkpoint_strategy(self) -> Dict[str, Any]:
        """Checkpoint strategy configuration."""
        return self._training_config.get("checkpoint_strategy", {
            "keep_best": True,
            "keep_latest": True,
            "keep_last_n": 3,
            "save_every": 5,
            "use_validation": False,
            "save_metadata": True,
            "cleanup_old": True
        })
    
    # Model configuration properties
    @property
    def d_model(self) -> int:
        """Model dimension."""
        return self._model_config.get("decoder_d_model", 768)
    
    @property
    def num_heads(self) -> int:
        """Number of attention heads."""
        return self._model_config.get("decoder_num_heads", 8)
    
    @property
    def num_layers(self) -> int:
        """Number of decoder layers."""
        return self._model_config.get("decoder_num_layers", 18)
    
    @property
    def dim_feedforward(self) -> int:
        """Feedforward dimension."""
        return self._model_config.get("decoder_intermediate_size", 1024)
    
    @property
    def max_sequence_length(self) -> int:
        """Maximum sequence length."""
        return self._model_config.get("decoder_max_sequence_length", 2048)
    
    @property
    def use_instrument_conditioning(self) -> bool:
        """Whether to use instrument conditioning."""
        return self._model_config.get("use_instrument_conditioning", True)
    
    # Data configuration properties
    @property
    def caption_dataset_path(self) -> str:
        """Path to caption dataset."""
        return self._data_config.get("caption_dataset_path", "./captions/all_captions.json")
    
    @property
    def artifact_folder(self) -> str:
        """Artifact folder path."""
        return self.config.get("artifact_folder", "artifacts")
    
    @property
    def tokenizer_file(self) -> str:
        """Tokenizer file name."""
        return self.config.get("tokenizer_file", "vocab_remi_z.pkl")
    
    # Helper methods
    def get_training_config(self) -> Dict:
        """Get full training configuration."""
        return self._training_config
    
    def get_model_config(self) -> Dict:
        """Get full model configuration."""
        return self._model_config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value by key.
        
        Args:
            key: Config key (supports nested keys with dots, e.g., 'training.batch_size')
            default: Default value if key not found
            
        Returns:
            Config value or default
        """
        keys = key.split(".")
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value if value is not None else default
