"""
Simple training script for Text2Terrain model.

Fireworks AI compatible training with minimal dependencies.
"""

import argparse
import os
import json
from pathlib import Path
from typing import Dict, Any
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from .model import Text2TerrainModel
from .datamodule import TerrainDataModule


def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    scaler,
    device: str,
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch."""
    
    model.train()
    total_loss = 0.0
    total_module_loss = 0.0
    total_param_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        module_targets = batch["module_targets"].to(device)
        param_targets = batch["param_targets"].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with autocast():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                module_targets=module_targets,
                param_targets=param_targets
            )
            
            loss = outputs["loss"]
            module_loss = outputs["module_loss"]
            param_loss = outputs["param_loss"]
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Accumulate losses
        total_loss += loss.item()
        total_module_loss += module_loss.item()
        total_param_loss += param_loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "mod": f"{module_loss.item():.4f}",
            "param": f"{param_loss.item():.4f}"
        })
    
    return {
        "train_loss": total_loss / num_batches,
        "train_module_loss": total_module_loss / num_batches,
        "train_param_loss": total_param_loss / num_batches
    }


def validate_epoch(
    model: nn.Module,
    dataloader,
    device: str
) -> Dict[str, float]:
    """Validate for one epoch."""
    
    model.eval()
    total_loss = 0.0
    total_module_loss = 0.0
    total_param_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            # Move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            module_targets = batch["module_targets"].to(device)
            param_targets = batch["param_targets"].to(device)
            
            # Forward pass
            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    module_targets=module_targets,
                    param_targets=param_targets
                )
                
                loss = outputs["loss"]
                module_loss = outputs["module_loss"]
                param_loss = outputs["param_loss"]
            
            # Accumulate losses
            total_loss += loss.item()
            total_module_loss += module_loss.item()
            total_param_loss += param_loss.item()
            num_batches += 1
    
    return {
        "val_loss": total_loss / num_batches,
        "val_module_loss": total_module_loss / num_batches,
        "val_param_loss": total_param_loss / num_batches
    }


def save_checkpoint(
    model: nn.Module,
    optimizer,
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_dir: Path
):
    """Save model checkpoint."""
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save LoRA weights and heads
    model_path = checkpoint_dir / f"model_epoch_{epoch:03d}.pt"
    model.save_lora_weights(str(model_path))
    
    # Save optimizer state
    optimizer_path = checkpoint_dir / f"optimizer_epoch_{epoch:03d}.pt"
    torch.save(optimizer.state_dict(), optimizer_path)
    
    # Save metrics
    metrics_path = checkpoint_dir / f"metrics_epoch_{epoch:03d}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Checkpoint saved: {model_path}")


def train_model(
    data_dir: str,
    output_dir: str,
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.1",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 5e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 100,
    save_every: int = 1,
    eval_every: int = 1,
    lora_r: int = 8,
    lora_alpha: int = 16,
    seed: int = 42
):
    """Main training function."""
    
    # Set random seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Setup directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_path / "checkpoints"
    
    # Initialize data module
    print("Setting up data...")
    datamodule = TerrainDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=4
    )
    datamodule.setup()
    
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    
    # Initialize model
    print(f"Initializing model: {model_name}")
    model = Text2TerrainModel(
        model_name=model_name,
        num_modules=datamodule.get_num_modules(),
        num_parameters=datamodule.get_num_parameters(),
        lora_r=lora_r,
        lora_alpha=lora_alpha
    )
    
    # Move to device
    model = model.to(device)
    
    # Setup optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Training loop
    print(f"Starting training for {epochs} epochs...")
    training_metrics = []
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler, device, epoch
        )
        
        # Validate
        if epoch % eval_every == 0:
            val_metrics = validate_epoch(model, val_loader, device)
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics, "epoch": epoch}
            training_metrics.append(epoch_metrics)
            
            # Print results
            print(f"Epoch {epoch} Results:")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"  Module Loss: {val_metrics['val_module_loss']:.4f}")
            print(f"  Param Loss: {val_metrics['val_param_loss']:.4f}")
        else:
            epoch_metrics = {**train_metrics, "epoch": epoch}
            training_metrics.append(epoch_metrics)
        
        # Save checkpoint
        if epoch % save_every == 0:
            save_checkpoint(model, optimizer, epoch, epoch_metrics, checkpoint_dir)
    
    # Save final model
    final_model_path = output_path / "final_model.pt"
    model.save_lora_weights(str(final_model_path))
    
    # Save training history
    history_path = output_path / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(training_metrics, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"Final model saved: {final_model_path}")
    print(f"Training history: {history_path}")
    
    return str(final_model_path)


def main():
    """CLI entry point."""
    
    parser = argparse.ArgumentParser(description="Train Text2Terrain model")
    parser.add_argument("--data-path", required=True, help="Path to processed data directory")
    parser.add_argument("--output-path", required=True, help="Output directory for model")
    parser.add_argument("--checkpoint-path", help="Checkpoint directory (optional)")
    parser.add_argument("--model-name", default="mistralai/Mistral-7B-Instruct-v0.1", 
                       help="Base model name")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--wandb-project", help="W&B project name (optional)")
    
    args = parser.parse_args()
    
    # Optional W&B logging
    if args.wandb_project:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                config=vars(args)
            )
            print(f"W&B logging enabled: {args.wandb_project}")
        except ImportError:
            print("W&B not available, continuing without logging")
    
    # Use checkpoint path if provided, otherwise use output path
    output_dir = args.checkpoint_path if args.checkpoint_path else args.output_path
    
    # Train model
    final_model_path = train_model(
        data_dir=args.data_path,
        output_dir=output_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        seed=args.seed
    )
    
    print(f"\nModel training completed successfully!")
    print(f"Final model: {final_model_path}")


if __name__ == "__main__":
    main()