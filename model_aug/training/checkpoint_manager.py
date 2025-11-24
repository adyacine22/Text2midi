"""Checkpoint manager for training."""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Set
import logging

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpoint saving, loading, and cleanup."""
    
    def __init__(self, config_manager, accelerator, output_dir: str):
        """Initialize checkpoint manager.
        
        Args:
            config_manager: ConfigManager instance
            accelerator: Accelerate Accelerator instance
            output_dir: Output directory for checkpoints
        """
        self.config = config_manager
        self.accelerator = accelerator
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint strategy
        strategy = self.config.checkpoint_strategy
        self.keep_best = strategy.get("keep_best", True)
        self.keep_latest = strategy.get("keep_latest", True)
        self.keep_last_n = strategy.get("keep_last_n", 3)
        self.save_every = strategy.get("save_every", 5)
        self.use_validation = strategy.get("use_validation", False)
        self.save_metadata = strategy.get("save_metadata", True)
        self.cleanup_old = strategy.get("cleanup_old", True)
        
        # Track best loss
        self.best_loss = float('inf')
    
    def save_if_needed(
        self,
        epoch: int,
        step: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        learning_rate: float = 0.0
    ) -> None:
        """Save checkpoints based on strategy.
        
        Args:
            epoch: Current epoch (0-indexed)
            step: Current training step
            train_loss: Training loss
            val_loss: Validation loss (optional)
            learning_rate: Current learning rate
        """
        if not self.accelerator.is_main_process:
            return
        
        # Determine metric for "best" model
        metric_for_best = val_loss if val_loss is not None else train_loss
        
        checkpoints_saved = []
        
        # 1. Save latest (always)
        if self.keep_latest:
            self._save_checkpoint(
                "latest", epoch, step, train_loss, val_loss, learning_rate
            )
            checkpoints_saved.append("latest")
        
        # 2. Save best (if improved)
        if self.keep_best and metric_for_best < self.best_loss:
            self.best_loss = metric_for_best
            self._save_checkpoint(
                "best", epoch, step, train_loss, val_loss, learning_rate
            )
            checkpoints_saved.append(f"best (loss={metric_for_best:.4f})")
        
        # 3. Save periodic checkpoints
        if (epoch + 1) % self.save_every == 0:
            checkpoint_name = f"epoch_{epoch + 1}"
            self._save_checkpoint(
                checkpoint_name, epoch, step, train_loss, val_loss, learning_rate
            )
            checkpoints_saved.append(checkpoint_name)
            
            # Cleanup old checkpoints
            if self.cleanup_old:
                self._cleanup_old_checkpoints()
        
        # Log what was saved
        if checkpoints_saved:
            logger.info(f"✓ Saved checkpoints: {', '.join(checkpoints_saved)}")
    
    def _save_checkpoint(
        self,
        name: str,
        epoch: int,
        step: int,
        train_loss: float,
        val_loss: Optional[float],
        learning_rate: float
    ) -> None:
        """Save a single checkpoint.
        
        Args:
            name: Checkpoint name (e.g., 'best', 'latest', 'epoch_10')
            epoch: Current epoch
            step: Current step
            train_loss: Training loss
            val_loss: Validation loss
            learning_rate: Learning rate
        """
        checkpoint_dir = self.output_dir / name
        
        # Save model state via Accelerator
        self.accelerator.save_state(str(checkpoint_dir))
        
        # Save metadata
        if self.save_metadata:
            self._save_metadata(
                checkpoint_dir, epoch, step, train_loss, val_loss, learning_rate
            )
    
    def _save_metadata(
        self,
        checkpoint_dir: Path,
        epoch: int,
        step: int,
        train_loss: float,
        val_loss: Optional[float],
        learning_rate: float
    ) -> None:
        """Save checkpoint metadata.
        
        Args:
            checkpoint_dir: Checkpoint directory
            epoch: Current epoch
            step: Current step
            train_loss: Training loss
            val_loss: Validation loss
            learning_rate: Learning rate
        """
        metadata = {
            "epoch": epoch + 1,
            "step": step,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6) if val_loss is not None else None,
            "learning_rate": learning_rate,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "batch_size": self.config.batch_size,
                "d_model": self.config.d_model,
                "num_layers": self.config.num_layers,
                "num_heads": self.config.num_heads,
            }
        }
        
        metadata_path = checkpoint_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    
    def _cleanup_old_checkpoints(self, keep_dirs: Optional[Set[str]] = None) -> None:
        """Remove old periodic checkpoints, keeping only the last N.
        
        Args:
            keep_dirs: Set of directory names to always keep
        """
        if keep_dirs is None:
            keep_dirs = {"best", "latest"}
        
        # Find all epoch checkpoints
        epoch_checkpoints = sorted(
            [d for d in self.output_dir.glob("epoch_*") if d.is_dir()],
            key=lambda x: int(x.name.split("_")[1])
        )
        
        # Remove old ones (keep last N)
        if len(epoch_checkpoints) > self.keep_last_n:
            for old_checkpoint in epoch_checkpoints[:-self.keep_last_n]:
                if old_checkpoint.name not in keep_dirs:
                    shutil.rmtree(old_checkpoint)
                    logger.info(f"Removed old checkpoint: {old_checkpoint.name}")
    
    def load_checkpoint(self, checkpoint_path: str, model, optimizer, lr_scheduler) -> int:
        """Load checkpoint and return starting epoch.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            model: Model to load state into
            optimizer: Optimizer to load state into
            lr_scheduler: LR scheduler to load state into
            
        Returns:
            Starting epoch number
        """
        checkpoint_dir = Path(checkpoint_path)
        
        if not checkpoint_dir.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return 0
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        # Load model state with strict=False to accommodate architectural changes
        unwrapped_model = self.accelerator.unwrap_model(model)
        model_path = checkpoint_dir / "pytorch_model.bin"
        
        if model_path.exists():
            import torch
            model_state = torch.load(model_path, map_location="cpu")
            incompatible = unwrapped_model.load_state_dict(model_state, strict=False)
            
            if incompatible.missing_keys:
                logger.warning(f"Missing keys: {incompatible.missing_keys}")
            if incompatible.unexpected_keys:
                logger.warning(f"Unexpected keys: {incompatible.unexpected_keys}")
            
            # Only load optimizer/scheduler if no missing keys and objects are provided
            if not incompatible.missing_keys and optimizer is not None and lr_scheduler is not None:
                self._load_optimizer_state(checkpoint_dir, optimizer, lr_scheduler)
        
        # Load RNG states
        self._load_rng_states(checkpoint_dir)
        
        # Determine starting epoch from directory name
        starting_epoch = 0
        if "epoch_" in checkpoint_dir.name:
            try:
                starting_epoch = int(checkpoint_dir.name.split("_")[-1])
            except ValueError:
                pass
        
        return starting_epoch
    
    def _load_optimizer_state(self, checkpoint_dir: Path, optimizer, lr_scheduler) -> None:
        """Load optimizer and scheduler state."""
        import torch
        
        try:
            optimizer_path = checkpoint_dir / "optimizer.bin"
            if optimizer_path.exists():
                optimizer.load_state_dict(
                    torch.load(optimizer_path, map_location=self.accelerator.device)
                )
            
            scheduler_path = checkpoint_dir / "scheduler.bin"
            if scheduler_path.exists():
                lr_scheduler.load_state_dict(torch.load(scheduler_path))
            
            # Load scaler if using mixed precision
            if self.accelerator.scaler is not None:
                scaler_path = checkpoint_dir / "scaler.pt"
                if scaler_path.exists():
                    scaler_state = torch.load(scaler_path)
                    self.accelerator.scaler.load_state_dict(scaler_state)
        except Exception as e:
            logger.warning(f"Could not load optimizer/scheduler state: {e}")
    
    def _load_rng_states(self, checkpoint_dir: Path) -> None:
        """Load random number generator states."""
        import torch
        
        try:
            rng_state_file = checkpoint_dir / f"random_states_{self.accelerator.process_index}.pkl"
            if rng_state_file.exists():
                rng_states = torch.load(rng_state_file, map_location='cpu')
                
                if "cpu" in rng_states:
                    torch.set_rng_state(rng_states["cpu"])
                    if torch.cuda.is_available() and "cuda" in rng_states and rng_states["cuda"]:
                        torch.cuda.set_rng_state_all(rng_states["cuda"])
        except Exception as e:
            logger.warning(f"Could not load RNG state: {e}")
