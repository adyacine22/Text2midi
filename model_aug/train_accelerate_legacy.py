import os
import torch.nn as nn
import torch.optim as optim
import yaml
import math
import time
from transformers import get_scheduler
import pickle
import numpy as np
import json
import jsonlines
from tqdm import tqdm
import torch
import torch.nn.functional as F
from accelerate import DistributedDataParallelKwargs, Accelerator
from accelerate.logging import get_logger
from music_dataset import create_dataloader
from transformer_model import Transformer
from torch.utils.data import DataLoader
import logging
import argparse
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from utils.instruments_mapping import INSTRUMENT_CLASSES

logger = get_logger(__name__)

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, default=None, help="Filter captions by stage (e.g., pretrain or finetune)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()

def load_configs(config_file):
    """Loads configurations from a YAML file."""
    with open(config_file, "r") as f:
        configs = yaml.safe_load(f)
    return configs

def setup_accelerator(configs):
    """Sets up the Accelerator for distributed training."""
    gradient_accumulation_steps = configs["training"]["text2midi_model"]["gradient_accumulation_steps"]
    mixed_precision = "fp16" if torch.cuda.is_available() else "no"
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    return accelerator

def get_dataloader(configs, captions, tokenizer, stage_filter):
    """Creates the DataLoader for training using unified dataset."""
    if stage_filter:
        before = len(captions)
        captions = [c for c in captions if c.get("stage") == stage_filter]
        print(f"Stage filter '{stage_filter}': {before} -> {len(captions)} samples")

    return create_dataloader(
        configs=configs,
        captions=captions,
        remi_tokenizer=tokenizer,
        mode="train",
        batch_size=configs["training"]["text2midi_model"]["per_device_train_batch_size"],
        num_workers=8,
        shuffle=True,
        use_augmentation=True,
        use_cache=True,
        prefetch_factor=4
    )

def get_model(configs, vocab_size, tokenizer=None):
    """Creates the Transformer model."""
    model_configs = configs["model"]["text2midi_model"]
    instrument_classes = model_configs.get("instrument_classes", list(INSTRUMENT_CLASSES.keys()))
    instrument_vocab_size = len(instrument_classes) if instrument_classes else 0

    model = Transformer(
        vocab_size,
        model_configs["decoder_d_model"],
        model_configs["decoder_num_heads"],
        model_configs["decoder_max_sequence_length"],
        model_configs["decoder_num_layers"],
        model_configs["decoder_intermediate_size"],
        use_moe=model_configs.get("use_moe", False),
        num_experts=model_configs.get("num_experts", 4),
        device="cpu", # Device will be handled by Accelerator
        instrument_vocab_size=instrument_vocab_size or 129,
        use_instrument_conditioning=model_configs.get("use_instrument_conditioning", False),
        no_instr_token_id=instrument_vocab_size if instrument_vocab_size else 129,
        tokenizer=tokenizer,
    )

    if configs["training"]["text2midi_model"].get("use_torch_compile", False):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("Compiled model with torch.compile (mode=reduce-overhead).")
        except Exception as e:
            print(f"torch.compile failed, continuing without compilation: {e}")
    
    return model

def get_optimizer_and_scheduler(model, configs, num_update_steps_per_epoch):
    """Creates the optimizer and learning rate scheduler."""
    train_configs = configs["training"]["text2midi_model"]
    adamw_kwargs = {"lr": train_configs["learning_rate"], "weight_decay": train_configs["weight_decay"]}
    
    if train_configs.get("use_fused_adamw", False) and torch.cuda.is_available():
        try:
            adamw_kwargs["fused"] = True
            optimizer = optim.AdamW(model.parameters(), **adamw_kwargs)
            print("Using fused AdamW optimizer.")
        except (TypeError, RuntimeError) as e:
            print(f"Fused AdamW requested but not available ({e}); falling back to standard AdamW.")
            adamw_kwargs.pop("fused", None)
            optimizer = optim.AdamW(model.parameters(), **adamw_kwargs)
    else:
        optimizer = optim.AdamW(model.parameters(), **adamw_kwargs)

    max_train_steps = train_configs.get("max_train_steps")
    if max_train_steps is None or str(max_train_steps).lower() == 'none':
        max_train_steps = train_configs["epochs"] * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=train_configs["lr_scheduler_type"],
        optimizer=optimizer,
        num_warmup_steps=train_configs["num_warmup_steps"],
        num_training_steps=max_train_steps,
    )
    
    return optimizer, lr_scheduler, max_train_steps


def save_checkpoint_with_metadata(accelerator, checkpoint_dir, epoch, step, train_loss, val_loss, learning_rate, configs):
    """Save checkpoint with metadata."""
    from datetime import datetime
    
    # Save model state
    accelerator.save_state(checkpoint_dir)
    
    # Save metadata
    if accelerator.is_main_process:
        metadata = {
            "epoch": epoch + 1,
            "step": step,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6) if val_loss is not None else None,
            "learning_rate": learning_rate,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "batch_size": configs["training"]["text2midi_model"]["per_device_train_batch_size"],
                "d_model": configs["model"]["text2midi_model"]["decoder_d_model"],
                "num_layers": configs["model"]["text2midi_model"]["decoder_num_layers"],
                "num_heads": configs["model"]["text2midi_model"]["decoder_num_heads"],
            }
        }
        
        metadata_path = Path(checkpoint_dir) / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)


def cleanup_old_checkpoints(output_dir, keep_last_n=3, keep_dirs=None):
    """Remove old periodic checkpoints, keeping only the last N."""
    if keep_dirs is None:
        keep_dirs = {"best", "latest"}
    
    output_path = Path(output_dir)
    
    # Find all epoch checkpoints
    epoch_checkpoints = sorted(
        [d for d in output_path.glob("epoch_*") if d.is_dir()],
        key=lambda x: int(x.name.split("_")[1])
    )
    
    # Remove old ones (keep last N)
    if len(epoch_checkpoints) > keep_last_n:
        import shutil
        for old_checkpoint in epoch_checkpoints[:-keep_last_n]:
            if old_checkpoint.name not in keep_dirs:
                shutil.rmtree(old_checkpoint)
                logger.info(f"Removed old checkpoint: {old_checkpoint.name}")


def validate_model(model, val_dataloader, criterion, accelerator):
    """Run validation and return average loss."""
    model.eval()
    total_val_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_dataloader:
            if batch is None:
                continue
            
            encoder_input, attention_mask, tgt, inst_ids, inst_mask = batch
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            model_output = model(encoder_input, attention_mask, tgt_input, inst=inst_ids, inst_mask=inst_mask)
            
            if isinstance(model_output, tuple):
                outputs, aux_loss = model_output
            else:
                outputs = model_output
                aux_loss = 0
            
            loss = criterion(outputs.view(-1, outputs.size(-1)), tgt_output.reshape(-1))
            if isinstance(aux_loss, torch.Tensor):
                loss += aux_loss
            
            total_val_loss += loss.detach().float()
            num_batches += 1
    
    model.train()
    
    if num_batches == 0:
        return None
    
    avg_val_loss = total_val_loss / num_batches
    return avg_val_loss.item()


def train_epoch(epoch, model, dataloader, criterion, optimizer, lr_scheduler, accelerator, max_train_steps, progress_bar, completed_steps, writer):
    """Runs a single training epoch."""
    model.train()
    total_loss = 0
    fetch_start = time.perf_counter()

    for step, batch in enumerate(dataloader):
        if batch is None:
            continue
        
        fetch_end = time.perf_counter()
        fetch_time = fetch_end - fetch_start
        
        with accelerator.accumulate(model):
            transfer_start = time.perf_counter()
            encoder_input, attention_mask, tgt, inst_ids, inst_mask = batch
            
            transfer_time = time.perf_counter() - transfer_start
            compute_start = time.perf_counter()

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            model_output = model(encoder_input, attention_mask, tgt_input, inst=inst_ids, inst_mask=inst_mask)
            
            if isinstance(model_output, tuple): # Handling MoE output
                outputs, aux_loss = model_output
            else:
                outputs = model_output
                aux_loss = 0

            loss = criterion(outputs.view(-1, outputs.size(-1)), tgt_output.reshape(-1))
            if isinstance(aux_loss, torch.Tensor):
                loss += aux_loss
            
            total_loss += loss.detach().float()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            compute_time = time.perf_counter() - compute_start

        if accelerator.sync_gradients:
            progress_bar.set_postfix({"Loss": loss.item()})
            progress_bar.update(1)
            completed_steps += 1
            if accelerator.is_main_process and writer:
                writer.add_scalar("train/loss_step", loss.item(), completed_steps)

        if accelerator.is_main_process and step % 50 == 0:
            accelerator.print(f"Step {step}: fetch {fetch_time*1000:.1f} ms, transfer {transfer_time*1000:.1f} ms, compute {compute_time*1000:.1f} ms")
        
        fetch_start = time.perf_counter()
        if completed_steps >= max_train_steps:
            break
            
    return total_loss, completed_steps


def main():
    """Main training script."""
    args = parse_args()
    config_file = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
    configs = load_configs(config_file)
    
    accelerator = setup_accelerator(configs)
    device = accelerator.device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    output_dir = configs["training"]["text2midi_model"]["output_dir"]
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "outputs"), exist_ok=True)

    tb_writer = (
        SummaryWriter(log_dir=os.path.join(output_dir, "tb")) if configs["training"]["text2midi_model"]["with_tracking"] and accelerator.is_main_process else None
    )
    
    # Load tokenizer and captions
    tokenizer_filepath = os.path.join(configs["artifact_folder"], configs.get("tokenizer_file", "vocab_remi_z.pkl"))
    with open(tokenizer_filepath, "rb") as f:
        tokenizer = pickle.load(f)
    
    with jsonlines.open(configs["raw_data"]["caption_dataset_path"]) as reader:
        captions = list(reader)

    with accelerator.main_process_first():
        dataloader = get_dataloader(configs, captions, tokenizer, args.stage)

    model = get_model(configs, len(tokenizer), tokenizer=tokenizer)
    
    num_update_steps_per_epoch = math.ceil(len(dataloader) / configs["training"]["text2midi_model"]["gradient_accumulation_steps"])
    optimizer, lr_scheduler, max_train_steps = get_optimizer_and_scheduler(model, configs, num_update_steps_per_epoch)
    
    model, optimizer, lr_scheduler, dataloader = accelerator.prepare(model, optimizer, lr_scheduler, dataloader)
    
    criterion = nn.CrossEntropyLoss()

    num_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    total_batch_size = (
        configs["training"]["text2midi_model"]["per_device_train_batch_size"]
        * accelerator.num_processes
        * configs["training"]["text2midi_model"]["gradient_accumulation_steps"]
    )
    
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataloader.dataset)}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {configs['training']['text2midi_model']['per_device_train_batch_size']}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {configs['training']['text2midi_model']['gradient_accumulation_steps']}")
    logger.info(f"  Total optimization steps = {max_train_steps}")


    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    best_loss = np.inf

    if args.resume:
        accelerator.print(f"Resuming from {args.resume}")

        has_missing_keys = False
        # Manually load model state with strict=False to accommodate architectural changes
        unwrapped_model = accelerator.unwrap_model(model)
        try:
            model_state = torch.load(os.path.join(args.resume, "pytorch_model.bin"), map_location="cpu")
            incompatible = unwrapped_model.load_state_dict(model_state, strict=False)
            if incompatible.missing_keys:
                has_missing_keys = True
                accelerator.print(f"Partially loaded model weights. Missing keys: {incompatible.missing_keys}")
            if incompatible.unexpected_keys:
                accelerator.print(f"Partially loaded model weights. Unexpected keys: {incompatible.unexpected_keys}")
        except FileNotFoundError:
            accelerator.print("Could not find pytorch_model.bin, skipping model state loading.")

        if not has_missing_keys:
            # Manually load other states
            try:
                optimizer.load_state_dict(torch.load(os.path.join(args.resume, "optimizer.bin"), map_location=accelerator.device))
                lr_scheduler.load_state_dict(torch.load(os.path.join(args.resume, "scheduler.bin")))
                if accelerator.scaler is not None and os.path.exists(os.path.join(args.resume, "scaler.pt")):
                    scaler_state = torch.load(os.path.join(args.resume, "scaler.pt"))
                    accelerator.scaler.load_state_dict(scaler_state)
            except (FileNotFoundError, ValueError) as e:
                accelerator.print(f"Could not load optimizer/scheduler state, re-initializing: {e}")
        else:
            accelerator.print("Model architecture changed. Re-initializing optimizer and scheduler.")

        # Always load RNG states for reproducibility
        try:
            # Adjust random state loading for the current process
            rng_state_file = os.path.join(args.resume, f"random_states_{accelerator.process_index}.pkl")
            if os.path.exists(rng_state_file):
                rng_states = torch.load(rng_state_file, map_location='cpu')
                if "cpu" in rng_states:  # New format
                    torch.set_rng_state(rng_states["cpu"])
                    if torch.cuda.is_available() and "cuda" in rng_states and rng_states["cuda"]:
                        torch.cuda.set_rng_state_all(rng_states["cuda"])
                else:  # Old format
                    if "torch_manual_seed" in rng_states:
                        torch.set_rng_state(rng_states["torch_manual_seed"])
                    if torch.cuda.is_available() and "torch_cuda_manual_seed" in rng_states and rng_states["torch_cuda_manual_seed"]:
                        torch.cuda.set_rng_state_all(rng_states["torch_cuda_manual_seed"])
        except (FileNotFoundError, KeyError) as e:
            accelerator.print(f"Could not load RNG state: {e}")

        path = os.path.basename(os.path.normpath(args.resume))
        if "epoch_" in path:
            try:
                starting_epoch = int(path.split("_")[-1])
                completed_steps = starting_epoch * num_update_steps_per_epoch
                progress_bar.update(completed_steps)
            except ValueError:
                pass

    for epoch in range(starting_epoch, num_epochs):
        total_loss, completed_steps = train_epoch(
            epoch, model, dataloader, criterion, optimizer, lr_scheduler, accelerator,
            max_train_steps, progress_bar, completed_steps, tb_writer
        )

        if accelerator.is_main_process:
            epoch_loss = total_loss.item() / len(dataloader)
            result = {"epoch": epoch + 1, "step": completed_steps, "train_loss": round(epoch_loss, 4)}
            
            # Get checkpoint strategy from config
            checkpoint_strategy = configs["training"]["text2midi_model"].get("checkpoint_strategy", {})
            keep_best = checkpoint_strategy.get("keep_best", True)
            keep_latest = checkpoint_strategy.get("keep_latest", True)
            keep_last_n = checkpoint_strategy.get("keep_last_n", 3)
            save_every = checkpoint_strategy.get("save_every", 5)
            use_validation = checkpoint_strategy.get("use_validation", False)
            save_metadata = checkpoint_strategy.get("save_metadata", True)
            cleanup_old = checkpoint_strategy.get("cleanup_old", True)
            
            # Validation (if enabled and validation data available)
            val_loss = None
            if use_validation:
                # TODO: Add validation dataloader creation in main()
                # val_loss = validate_model(model, val_dataloader, criterion, accelerator)
                logger.warning("Validation enabled but not implemented yet. Using train loss.")
            
            # Determine metric for "best" model
            metric_for_best = val_loss if val_loss is not None else epoch_loss
            current_lr = optimizer.param_groups[0]["lr"]
            
            # Log results
            result_string = f"Epoch: {epoch + 1}, Loss Train: {result['train_loss']}"
            if val_loss is not None:
                result["val_loss"] = round(val_loss, 4)
                result_string += f", Val: {result['val_loss']}"
            result_string += "\n"
            
            accelerator.print(result_string)
            if tb_writer:
                tb_writer.add_scalar("train/loss_epoch", result["train_loss"], epoch + 1)
                if val_loss is not None:
                    tb_writer.add_scalar("val/loss_epoch", val_loss, epoch + 1)
            
            with open(f"{output_dir}/summary.jsonl", "a") as f:
                f.write(json.dumps(result) + "\n\n")
            logger.info(result)
            
            # Smart checkpoint saving
            checkpoints_saved = []
            
            # 1. Save latest (always)
            if keep_latest:
                checkpoint_dir = f"{output_dir}/latest"
                if save_metadata:
                    save_checkpoint_with_metadata(
                        accelerator, checkpoint_dir, epoch, completed_steps,
                        epoch_loss, val_loss, current_lr, configs
                    )
                else:
                    accelerator.save_state(checkpoint_dir)
                checkpoints_saved.append("latest")
            
            # 2. Save best (if improved)
            if keep_best and metric_for_best < best_loss:
                best_loss = metric_for_best
                checkpoint_dir = f"{output_dir}/best"
                if save_metadata:
                    save_checkpoint_with_metadata(
                        accelerator, checkpoint_dir, epoch, completed_steps,
                        epoch_loss, val_loss, current_lr, configs
                    )
                else:
                    accelerator.save_state(checkpoint_dir)
                checkpoints_saved.append(f"best (loss={metric_for_best:.4f})")
            
            # 3. Save periodic checkpoints
            if (epoch + 1) % save_every == 0:
                checkpoint_dir = f"{output_dir}/epoch_{epoch + 1}"
                if save_metadata:
                    save_checkpoint_with_metadata(
                        accelerator, checkpoint_dir, epoch, completed_steps,
                        epoch_loss, val_loss, current_lr, configs
                    )
                else:
                    accelerator.save_state(checkpoint_dir)
                checkpoints_saved.append(f"epoch_{epoch + 1}")
                
                # Cleanup old checkpoints
                if cleanup_old:
                    cleanup_old_checkpoints(output_dir, keep_last_n=keep_last_n)
            
            # Log what was saved
            if checkpoints_saved:
                logger.info(f"✓ Saved checkpoints: {', '.join(checkpoints_saved)}")


        if completed_steps >= max_train_steps:
            break
            
    if tb_writer:
        tb_writer.close()

if __name__ == "__main__":
    main()
