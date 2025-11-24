"""Main trainer class for model training."""

import os
import math
import time
import random
import pickle
import json
import jsonlines
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from transformers import get_scheduler
from torch.utils.tensorboard import SummaryWriter
from accelerate import DistributedDataParallelKwargs, Accelerator

from .config_manager import ConfigManager
from .checkpoint_manager import CheckpointManager
from music_dataset import create_dataloader
from transformer_model import Transformer
from utils.instruments_mapping import INSTRUMENT_CLASSES

logger = logging.getLogger(__name__)


class Trainer:
    """Main trainer class for model training."""
    
    def __init__(self, config_manager: ConfigManager, args):
        """Initialize trainer.
        
        Args:
            config_manager: ConfigManager instance
            args: Command-line arguments
        """
        self.config = config_manager
        self.args = args
        
        # Will be initialized in setup()
        self.accelerator = None
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.train_dataloader = None
        self.checkpoint_manager = None
        self.tb_writer = None
        self.criterion = None
        
        # Training state
        self.num_epochs = 0
        self.max_train_steps = 0
        self.starting_epoch = 0
        self.stage = args.stage if hasattr(args, 'stage') and args.stage in ['pretrain', 'finetune'] else 'pretrain'
        self.epochs_no_inst = 0
        self.epochs_inst = 0
        self.current_use_inst_conditioning = True
        self.instrument_dropout_schedule = self.config.get("training.text2midi_model.instrument_dropout_schedule", [])
        
        # Setup everything
        self.setup()
    
    def setup(self):
        """Setup all components for training."""
        # 1. Setup accelerator
        self._setup_accelerator()
        
        # 2. Setup logging
        self._setup_logging()
        
        # 3. Setup output directory
        self._setup_output_dir()
        
        # 4. Setup TensorBoard
        self._setup_tensorboard()
        
        # 5. Load tokenizer and data
        self._setup_data()
        
        # 6. Setup model
        self._setup_model()
        
        # 7. Setup optimizer and scheduler
        self._setup_optimizer()
        
        # 8. Prepare with accelerator
        self._prepare_for_training()
        
        # 9. Setup checkpoint manager
        self._setup_checkpoint_manager()
        
        # 10. Setup criterion
        self.criterion = nn.CrossEntropyLoss()
        
        # 11. Resume from checkpoint if specified
        if self.args.resume:
            self._resume_from_checkpoint()
        
        # 12. Log training info
        self._log_training_info()
    
    def _setup_accelerator(self):
        """Setup Accelerate accelerator."""
        # Automatically select best mixed precision
        if torch.cuda.is_available():
            if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
                mixed_precision = "bf16"
                logger.info("Using BF16 mixed precision (GPU supports it)")
            else:
                mixed_precision = "fp16"
                logger.info("Using FP16 mixed precision (BF16 not supported)")
        else:
            mixed_precision = "no"
            logger.info("No GPU available, using FP32")
            
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
        )
        logger.info(f"Accelerator setup complete. Device: {self.accelerator.device}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        if self.accelerator.is_main_process:
            logger.info(f"Accelerator state: {self.accelerator.state}")
    
    def _setup_output_dir(self):
        """Create output directory."""
        if self.accelerator.is_main_process:
            os.makedirs(self.config.output_dir, exist_ok=True)
            os.makedirs(os.path.join(self.config.output_dir, "outputs"), exist_ok=True)
    
    def _setup_tensorboard(self):
        """Setup TensorBoard writer."""
        if self.config.with_tracking and self.accelerator.is_main_process:
            tb_dir = os.path.join(self.config.output_dir, "tb")
            self.tb_writer = SummaryWriter(log_dir=tb_dir)
    
    def _setup_data(self):
        """Load tokenizer and create dataloader."""
        # Load tokenizer
        tokenizer_path = os.path.join(self.config.artifact_folder, self.config.tokenizer_file)
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        
        # Robust fix for unpickling:
        # 1. Add project root to sys.path to ensure 'model_aug' is a discoverable package.
        # 2. Import the tokenizer module.
        # 3. Alias it in sys.modules so pickle can find it by its original name ('remi_z_tokenizer').
        import sys
        project_root = Path(__file__).resolve().parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        try:
            import model_aug.remi_z_tokenizer
            sys.modules['remi_z_tokenizer'] = model_aug.remi_z_tokenizer
        except ImportError:
            logger.error("CRITICAL: Could not import 'model_aug.remi_z_tokenizer'. This is required for unpickling the tokenizer. Ensure the file exists and the path is correct.")
            raise

        with open(tokenizer_path, "rb") as f:
            self.tokenizer = pickle.load(f)
        
        # Load captions
        logger.info(f"Loading captions from {self.config.caption_dataset_path}")
        with jsonlines.open(self.config.caption_dataset_path) as reader:
            captions = list(reader)
        
        # Filter by stage if specified
        if self.args.stage:
            before = len(captions)
            captions = [c for c in captions if c.get("stage") == self.args.stage]
            logger.info(f"Stage filter '{self.args.stage}': {before} -> {len(captions)} samples")
        
        # Create dataloader
        with self.accelerator.main_process_first():
            self.train_dataloader = create_dataloader(
                configs=self.config.config,  # Pass full config dict
                captions=captions,
                remi_tokenizer=self.tokenizer,
                mode="train",
                batch_size=self.config.batch_size,
                num_workers=self.config.get("training.num_workers", 8),
                shuffle=self.config.get("training.shuffle", True),
                use_augmentation=self.config.get("training.use_augmentation", True),
                use_cache=self.config.get("training.use_cache", True),
                prefetch_factor=self.config.get("training.prefetch_factor", 4)
            )
            
            # Setup validation dataloader if requested
            if self.config.checkpoint_strategy.get("use_validation", False):
                logger.info("Setting up validation dataloader...")
                validation_samples = self.config.get("training.text2midi_model.validation_samples")
                self.val_dataloader = create_dataloader(
                    configs=self.config.config,
                    captions=captions,
                    remi_tokenizer=self.tokenizer,
                    mode="validation",
                    batch_size=self.config.batch_size,
                    num_workers=max(1, self.config.get("training.num_workers", 8) // 2),  # Half the workers for validation
                    shuffle=False,
                    use_augmentation=False,
                    use_cache=True,
                    prefetch_factor=max(1, self.config.get("training.prefetch_factor", 4) // 2),  # Half the prefetch
                    limit=validation_samples
                )
                logger.info(f"Validation samples: {len(self.val_dataloader.dataset)}")
            else:
                self.val_dataloader = None
    
    def _setup_model(self):
        """Initialize model."""
        logger.info("Initializing model")
        
        # Get instrument classes
        instrument_classes = self.config.get_model_config().get(
            "instrument_classes", list(INSTRUMENT_CLASSES.keys())
        )
        instrument_vocab_size = len(instrument_classes) if instrument_classes else 0
        
        self.model = Transformer(
            n_vocab=len(self.tokenizer),
            d_model=self.config.d_model,
            nhead=self.config.num_heads,
            num_decoder_layers=self.config.num_layers,
            dim_feedforward=self.config.dim_feedforward,
            max_len=self.config.max_sequence_length,
            use_moe=self.config.get("model.text2midi_model.use_moe", False),
            num_experts=self.config.get("model.text2midi_model.num_experts", 4),
            device="cpu",  # Will be moved by accelerator
            instrument_vocab_size=instrument_vocab_size or 129,
            use_instrument_conditioning=self.config.use_instrument_conditioning,
            no_instr_token_id=instrument_vocab_size if instrument_vocab_size else 129,
            tokenizer=self.tokenizer,
        )
        
        # Optional: torch.compile
        if self.config.get("training.text2midi_model.use_torch_compile", False):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                logger.info("Compiled model with torch.compile")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")
    
    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        # Optimizer
        adamw_kwargs = {
            "lr": self.config.learning_rate,
            "weight_decay": self.config.weight_decay
        }
        
        if self.config.get("training.text2midi_model.use_fused_adamw", False) and torch.cuda.is_available():
            try:
                adamw_kwargs["fused"] = True
                self.optimizer = optim.AdamW(self.model.parameters(), **adamw_kwargs)
                logger.info("Using fused AdamW optimizer")
            except (TypeError, RuntimeError) as e:
                logger.warning(f"Fused AdamW not available: {e}")
                adamw_kwargs.pop("fused", None)
                self.optimizer = optim.AdamW(self.model.parameters(), **adamw_kwargs)
        else:
            self.optimizer = optim.AdamW(self.model.parameters(), **adamw_kwargs)
        
        # Learning rate scheduler
        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.config.gradient_accumulation_steps
        )
        
        # Calculate epochs based on stage
        strategies = self.config.get("training.text2midi_model.strategies", {})
        stage_config = strategies.get(self.stage, {})
        
        self.epochs_no_inst = stage_config.get("epochs_no_inst", 0)
        self.epochs_inst = stage_config.get("epochs_inst", self.config.epochs)
        total_epochs = self.epochs_no_inst + self.epochs_inst
        
        logger.info(f"Training stage: {self.stage}")
        logger.info(f"Epochs without instrument conditioning: {self.epochs_no_inst}")
        logger.info(f"Epochs with instrument conditioning: {self.epochs_inst}")
        logger.info(f"Total epochs: {total_epochs}")
        
        max_train_steps = self.config.get("training.text2midi_model.max_train_steps")
        if max_train_steps is None or str(max_train_steps).lower() == 'none':
            max_train_steps = total_epochs * num_update_steps_per_epoch
        
        self.max_train_steps = max_train_steps
        self.num_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
        
        # Get minimum learning rate (default to 10% of max LR for cosine)
        min_lr_ratio = self.config.get("training.text2midi_model.min_lr_ratio", 0.1)
        
        # Create scheduler
        if self.config.lr_scheduler_type == "cosine":
            from transformers import get_cosine_schedule_with_warmup
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.num_warmup_steps,
                num_training_steps=max_train_steps,
                num_cycles=0.5,  # Standard cosine (goes from max to min once)
                last_epoch=-1
            )
            logger.info(f"Using cosine scheduler with min LR = {self.config.learning_rate * min_lr_ratio:.2e}")
        else:
            self.lr_scheduler = get_scheduler(
                name=self.config.lr_scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=self.config.num_warmup_steps,
                num_training_steps=max_train_steps,
            )
    
    def _prepare_for_training(self):
        """Prepare model, optimizer, scheduler, and dataloader with accelerator."""
        if self.val_dataloader:
            self.model, self.optimizer, self.lr_scheduler, self.train_dataloader, self.val_dataloader = \
                self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler, self.train_dataloader, self.val_dataloader
                )
        else:
            self.model, self.optimizer, self.lr_scheduler, self.train_dataloader = \
                self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler, self.train_dataloader
                )
    
    def _setup_checkpoint_manager(self):
        """Setup checkpoint manager."""
        self.checkpoint_manager = CheckpointManager(
            self.config,
            self.accelerator,
            self.config.output_dir
        )
    
       
    def _resume_from_checkpoint(self):
        """Resume training from checkpoint."""
        # Determine if we should load the optimizer and scheduler state.
        # By default, we load them for an accurate resume.
        # If --reset_optimizer is passed, we only load model weights.
        load_optimizer = not self.args.reset_optimizer
        
        if load_optimizer:
            logger.info("Resuming with optimizer and scheduler state.")
        else:
            logger.info("Resuming with --reset_optimizer: Only model weights will be loaded. Optimizer and scheduler will be re-initialized.")
            
        self.starting_epoch = self.checkpoint_manager.load_checkpoint(
            self.args.resume,
            self.model,
            self.optimizer if load_optimizer else None,
            self.lr_scheduler if load_optimizer else None
        )
        
        # If we reset the optimizer, we should also restart the epoch count for the new training phase.
        if not load_optimizer:
            logger.info(f"Optimizer was reset. Overriding starting epoch from checkpoint ({self.starting_epoch}) to 0.")
            self.starting_epoch = 0
            
        logger.info(f"Resumed from checkpoint. Training will start at epoch: {self.starting_epoch}")

    def _log_training_info(self):
        """Log training configuration."""
        total_batch_size = (
            self.config.batch_size
            * self.accelerator.num_processes
            * self.config.gradient_accumulation_steps
        )
        
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataloader.dataset)}")
        logger.info(f"  Num Epochs = {self.num_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.config.batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.config.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.max_train_steps}")
    
    def train(self):
        """Main training loop."""
        progress_bar = tqdm(
            range(self.max_train_steps),
            disable=not self.accelerator.is_local_main_process,
            mininterval=1.0,  # Update display at most once per second
            maxinterval=10.0,  # Force update at least every 10 seconds
            smoothing=0.3,  # Smooth the rate estimate (higher = smoother)
            unit=' step',  # Custom unit name
            unit_scale=False,  # Disable auto-scaling to force s/it format
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
        )
        completed_steps = self.starting_epoch * math.ceil(
            len(self.train_dataloader) / self.config.gradient_accumulation_steps
        )
        progress_bar.update(completed_steps)
        
        for epoch in range(self.starting_epoch, self.num_epochs):
            total_loss, completed_steps = self.train_epoch(epoch, progress_bar, completed_steps)
            
            # Handle checkpointing and logging
            if self.accelerator.is_main_process:
                self._handle_epoch_end(epoch, total_loss, completed_steps)
            
            if completed_steps >= self.max_train_steps:
                break
        
        # Cleanup
        if self.tb_writer:
            self.tb_writer.close()
        
        logger.info("Training complete!")
    
        logger.info("Training complete!")
    
    def _get_current_dropout(self, epoch: int) -> float:
        """Get dropout probability for current epoch based on schedule."""
        prob = 0.0
        # Default to 0 if no schedule
        if not self.instrument_dropout_schedule:
            return self.config.get("training.text2midi_model.conditioning_dropout", 0.0)
            
        for threshold, value in self.instrument_dropout_schedule:
            if epoch >= threshold:
                prob = value
            else:
                # Schedule is assumed to be sorted
                break
        return prob
    
    def train_epoch(self, epoch: int, progress_bar, completed_steps: int):
        """Train for one epoch.
        
        Args:
            epoch: Current epoch number
            progress_bar: Progress bar
            completed_steps: Number of completed steps
            
        Returns:
            Tuple of (total_loss, completed_steps)
        """
        self.model.train()
        total_loss = 0
        
        # Determine if instrument conditioning should be enabled
        use_inst_conditioning = epoch >= self.epochs_no_inst
        self.current_use_inst_conditioning = use_inst_conditioning

        if epoch == 0 or epoch == self.epochs_no_inst:
            logger.info(f"Epoch {epoch}: Instrument conditioning {'ENABLED' if use_inst_conditioning else 'DISABLED'}")
            
        current_dropout_p = self._get_current_dropout(epoch)
        logger.info(f"Epoch {epoch}: Current instrument dropout probability: {current_dropout_p}")
        
        for step, batch in enumerate(self.train_dataloader):
            if batch is None:
                continue
            
            with self.accelerator.accumulate(self.model):
                encoder_input, attention_mask, tgt, inst_ids, inst_mask = batch.values() if isinstance(batch, dict) else batch
                # Note: MusicCollator returns a dict, but if it returns tuple, handle it.
                # MusicCollator returns dict: input_ids, attention_mask, inst_ids, inst_mask, labels
                if isinstance(batch, dict):
                    encoder_input = batch["input_ids"]
                    attention_mask = batch["attention_mask"]
                    inst_ids = batch["inst_ids"]
                    inst_mask = batch["inst_mask"]
                    tgt = batch["labels"]
                
                # Apply dynamic instrument dropout
                if use_inst_conditioning and current_dropout_p > 0:
                    # Sample mask for batch: (batch_size, 1)
                    # True means KEEP the instrument (did not drop)
                    # Logic: rand > dropout_p means we keep it. 
                    # e.g. dropout=0.8 -> rand > 0.8 happens 20% of time -> keep 20% of time.
                    # Wait, user said: "dropout_p = 0.8 (model mostly ignores instruments)"
                    # So we want to DROP with prob 0.8.
                    # keep = rand > dropout_p
                    keep = torch.rand(inst_mask.shape[0], 1, device=inst_mask.device) > current_dropout_p
                    
                    # If we don't keep, we replace with no_instr_token_id
                    if not keep.all():
                        no_instr_id = self.model.module.no_instr_token_id if hasattr(self.model, "module") else self.model.no_instr_token_id
                        inst_ids = inst_ids.masked_fill(~keep, no_instr_id)
                        inst_mask = inst_mask * keep.long()

                # Disable instrument conditioning if in no_inst phase
                if not use_inst_conditioning:
                    inst_ids = None
                    inst_mask = None
                
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                model_output = self.model(
                    encoder_input, attention_mask, tgt_input,
                    inst=inst_ids, inst_mask=inst_mask
                )
                
                if isinstance(model_output, tuple):
                    outputs, aux_loss = model_output
                else:
                    outputs = model_output
                    aux_loss = 0
                
                loss = self.criterion(outputs.view(-1, outputs.size(-1)), tgt_output.reshape(-1))
                if isinstance(aux_loss, torch.Tensor):
                    loss += aux_loss
                
                total_loss += loss.detach().float()
                self.accelerator.backward(loss)
                
                # Gradient clipping to prevent NaN
                max_grad_norm = self.config.get("training.text2midi_model.max_grad_norm", 1.0)
                if max_grad_norm > 0:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
            
            if self.accelerator.sync_gradients:
                progress_bar.set_postfix({"Loss": loss.item()})
                progress_bar.update(1)
                completed_steps += 1
                
                if self.accelerator.is_main_process and self.tb_writer:
                    # Basic loss
                    self.tb_writer.add_scalar("train/loss_step", loss.item(), completed_steps)
                    
                    # Learning rate
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.tb_writer.add_scalar("train/learning_rate", current_lr, completed_steps)
                    
                    # Perplexity (exp(loss))
                    perplexity = torch.exp(loss).item()
                    self.tb_writer.add_scalar("train/perplexity", perplexity, completed_steps)
                    
                    # Gradient norms
                    total_norm = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
                    self.tb_writer.add_scalar("train/grad_norm", total_norm, completed_steps)
                    
                    # Token accuracy (top-1)
                    with torch.no_grad():
                        predictions = outputs.argmax(dim=-1)
                        correct = (predictions == tgt_output).float()
                        accuracy = correct.mean().item()
                        self.tb_writer.add_scalar("train/token_accuracy", accuracy, completed_steps)
                    
                    # Instrument dropout probability
                    self.tb_writer.add_scalar("train/instrument_dropout", current_dropout_p, completed_steps)
            
            if completed_steps >= self.max_train_steps:
                break
        
        return total_loss, completed_steps
    
    def validate_model(self):
        """Validate the model on validation set."""
        if not self.val_dataloader:
            return None
            
        logger.info("Running validation...")
        self.model.eval()
        total_val_loss = 0
        total_val_accuracy = 0
        num_batches = 0
        use_inst_conditioning = getattr(self, "current_use_inst_conditioning", True)
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation", disable=not self.accelerator.is_local_main_process):
                if batch is None:
                    continue
                
                if isinstance(batch, dict):
                    encoder_input = batch["input_ids"]
                    attention_mask = batch["attention_mask"]
                    inst_ids = batch["inst_ids"]
                    inst_mask = batch["inst_mask"]
                    tgt = batch["labels"]
                else:
                    # Fallback if collator changes
                    encoder_input, attention_mask, tgt, inst_ids, inst_mask = batch
                
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                if not use_inst_conditioning:
                    inst_ids = None
                    inst_mask = None
                
                model_output = self.model(
                    encoder_input, attention_mask, tgt_input,
                    inst=inst_ids, inst_mask=inst_mask
                )
                
                if isinstance(model_output, tuple):
                    outputs, aux_loss = model_output
                else:
                    outputs = model_output
                    aux_loss = 0
                
                loss = self.criterion(outputs.view(-1, outputs.size(-1)), tgt_output.reshape(-1))
                if isinstance(aux_loss, torch.Tensor):
                    loss += aux_loss
                
                total_val_loss += loss.detach().float()
                
                # Calculate accuracy
                predictions = outputs.argmax(dim=-1)
                correct = (predictions == tgt_output).float()
                accuracy = correct.mean()
                total_val_accuracy += accuracy
                
                num_batches += 1
        
        avg_val_loss = total_val_loss / num_batches if num_batches > 0 else float('inf')
        avg_val_accuracy = total_val_accuracy / num_batches if num_batches > 0 else 0.0
        
        # Return both metrics
        return {
            'loss': avg_val_loss.item() if isinstance(avg_val_loss, torch.Tensor) else float(avg_val_loss),
            'accuracy': avg_val_accuracy.item() if isinstance(avg_val_accuracy, torch.Tensor) else float(avg_val_accuracy)
        }
    
    def _handle_epoch_end(self, epoch: int, total_loss, completed_steps: int):
        """Handle end of epoch logging and checkpointing.
        
        Args:
            epoch: Current epoch
            total_loss: Total loss for epoch
            completed_steps: Number of completed steps
        """
        if isinstance(total_loss, torch.Tensor):
            total_loss_value = total_loss.item()
        else:
            total_loss_value = float(total_loss)
        epoch_loss = total_loss_value / max(len(self.train_dataloader), 1)
        current_lr = self.optimizer.param_groups[0]["lr"]
        
        # Validation
        val_metrics = None
        if self.val_dataloader:
            val_metrics = self.validate_model()
        
        # Log results
        result = {
            "epoch": epoch + 1,
            "step": completed_steps,
            "train_loss": round(epoch_loss, 4),
            "val_loss": round(val_metrics['loss'], 4) if val_metrics else None,
            "val_accuracy": round(val_metrics['accuracy'], 4) if val_metrics else None
        }
        
        result_string = f"Epoch: {epoch + 1}, Loss Train: {result['train_loss']}"
        if val_metrics:
            result_string += f", Loss Val: {result['val_loss']}, Acc Val: {result['val_accuracy']:.2%}"
        result_string += "\n"
        
        self.accelerator.print(result_string)
        
        if self.tb_writer:
            self.tb_writer.add_scalar("epoch/train_loss", epoch_loss, epoch)
            self.tb_writer.add_scalar("epoch/learning_rate", current_lr, epoch)
            self.tb_writer.add_scalar("epoch/perplexity", np.exp(epoch_loss), epoch)
            
            if val_metrics:
                self.tb_writer.add_scalar("epoch/val_loss", val_metrics['loss'], epoch)
                self.tb_writer.add_scalar("epoch/val_accuracy", val_metrics['accuracy'], epoch)
                self.tb_writer.add_scalar("epoch/val_perplexity", np.exp(val_metrics['loss']), epoch)
        
        with open(f"{self.config.output_dir}/summary.jsonl", "a") as f:
            f.write(json.dumps(result) + "\n\n")
        
        logger.info(result)
        
        # Save checkpoints
        self.checkpoint_manager.save_if_needed(
            epoch=epoch,
            step=completed_steps,
            train_loss=epoch_loss,
            val_loss=val_metrics['loss'] if val_metrics else None,
            learning_rate=current_lr
        )
