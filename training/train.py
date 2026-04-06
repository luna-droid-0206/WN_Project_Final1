"""
Main training script for MCS selection models.
Supports all baseline models and the attention model.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to Python path for absolute imports when running as script
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.amp import GradScaler, autocast
from tqdm import tqdm

# Import local modules
from config import (
    set_global_seed,
    GLOBAL_SEED,
    TrainingConfig,
    DataConfig,
    AttentionModelConfig,
    DNNConfig,
    CNNConfig,
    LSTMConfig
)
from utils import count_parameters, measure_inference_latency
from models.baselines import get_model  # Import model factory


def setup_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int = 4,
    seed: int = GLOBAL_SEED
) -> Dict[str, DataLoader]:
    """Create dataloaders for train/val/test splits."""
    from data_generator import create_dataloader

    loaders = {}
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        shuffle = (split == "train")
        loader = create_dataloader(
            split_dir,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            seed=seed
        )
        loaders[split] = loader

    return loaders


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: Optional[GradScaler] = None,
    gradient_clip: float = 1.0,
    use_amp: bool = False
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)

    for batch_idx, (channels, labels) in enumerate(pbar):
        channels = channels.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            with autocast('cuda'):
                outputs = model(channels)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(channels)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100.*correct/total:.2f}%"
        })

    metrics = {
        "loss": total_loss / len(train_loader),
        "accuracy": correct / total
    }

    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    split_name: str = "val"
) -> Dict[str, float]:
    """Evaluate on a dataset split."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    correct_top3 = 0

    pbar = tqdm(val_loader, desc=f"[{split_name.capitalize()}]", leave=False)

    for channels, labels in pbar:
        channels = channels.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(channels)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Top-3 accuracy
        _, top3_pred = torch.topk(outputs, 3, dim=1)
        correct_top3 += torch.sum(top3_pred == labels.unsqueeze(1)).item()

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100.*correct/total:.2f}%"
        })

    metrics = {
        "loss": total_loss / len(val_loader),
        "accuracy": correct / total,
        "top3_accuracy": correct_top3 / total
    }

    return metrics


def train_model(
    model_name: str,
    data_dir: str,
    output_dir: str,
    config: Dict[str, Any],
    device: torch.device,
    n_epochs: int = TrainingConfig.NUM_EPOCHS,
    batch_size: int = TrainingConfig.BATCH_SIZE,
    learning_rate: float = TrainingConfig.LEARNING_RATE,
    weight_decay: float = TrainingConfig.WEIGHT_DECAY,
    patience: int = TrainingConfig.PATIENCE,
    lr_patience: int = TrainingConfig.LR_PATIENCE,
    lr_factor: float = TrainingConfig.LR_FACTOR,
    gradient_clip: float = TrainingConfig.GRADIENT_CLIP,
    use_amp: bool = TrainingConfig.USE_AMP,
    num_workers: int = 4,
    save_every: int = 5,
    resume_from: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Full training loop with checkpointing and early stopping.

    Args:
        model_name: Name of model (from MODEL_REGISTRY)
        data_dir: Path to data/ with train/val/test subdirs
        output_dir: Directory to save checkpoints and logs
        config: Model-specific configuration dict
        device: torch.device
        (other args: training hyperparameters)

    Returns:
        Dictionary with training results and best metrics
    """
    print(f"\n{'='*60}")
    print(f"Training: {model_name.upper()}")
    print(f"{'='*60}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Max epochs: {n_epochs}")
    print(f"{'='*60}\n")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "checkpoints").mkdir(exist_ok=True)

    # Save run configuration
    run_config = {
        "model_name": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "data_dir": data_dir,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "n_epochs": n_epochs,
        "patience": patience,
        "seed": GLOBAL_SEED,
        "config": config,
        "device": str(device)
    }
    with open(output_path / "config.json", 'w') as f:
        json.dump(run_config, f, indent=2)

    # Prepare data
    print("Loading data...")
    loaders = setup_dataloaders(data_dir, batch_size, num_workers, GLOBAL_SEED)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    # Build model
    print(f"\nBuilding {model_name} model...")
    model = get_model(model_name, **config).to(device)

    # Count parameters
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")

    if n_params > 5_000_000:
        print("[Warning] Model has >5M parameters. Consider simplifying for faster inference.")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=lr_factor,
        patience=lr_patience
        # verbose=True
    )

    # Mixed precision scaler
    scaler = GradScaler('cuda') if use_amp and device.type == 'cuda' else None
    if scaler:
        print("Using Automatic Mixed Precision (AMP)")

    # Training history
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "lr": []
    }

    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    # Resume from checkpoint?
    start_epoch = 0
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        history = checkpoint['history']
        best_val_loss = checkpoint.get('best_val_loss', best_val_loss)
        best_val_acc = checkpoint.get('best_val_acc', best_val_acc)
        best_epoch = checkpoint.get('best_epoch', best_epoch)
        print(f"Resumed from epoch {start_epoch}")

    # Main training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")

    for epoch in range(start_epoch, n_epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            scaler=scaler, gradient_clip=gradient_clip, use_amp=use_amp
        )

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device, split_name="val")

        # Update learning rate
        scheduler.step(val_metrics['loss'])
        current_lr = optimizer.param_groups[0]['lr']

        # Record history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['lr'].append(current_lr)

        # Print epoch summary
        print(f"Epoch {epoch+1:03d}/{n_epochs}: "
              f"Train Loss: {train_metrics['loss']:.4f}, Acc: {100.*train_metrics['accuracy']:.2f}% | "
              f"Val Loss: {val_metrics['loss']:.4f}, Acc: {100.*val_metrics['accuracy']:.2f}% | "
              f"LR: {current_lr:.2e}")

        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint_path = output_path / "checkpoints" / f"epoch_{epoch+1:03d}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'history': history,
                'best_val_loss': best_val_loss,
                'best_val_acc': best_val_acc,
                'best_epoch': best_epoch
            }, checkpoint_path)

        # Check for best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_val_loss = val_metrics['loss']
            best_epoch = epoch
            patience_counter = 0

            # Save best model
            best_path = output_path / "checkpoints" / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_metrics': val_metrics,
                'history': history
            }, best_path)
            print(f"  [New best!] Val accuracy: {100.*best_val_acc:.2f}% (saved)")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"\n[Early stopping] No improvement for {patience} epochs. Best val accuracy: {100.*best_val_acc:.2f}% at epoch {best_epoch+1}")
            break

    # Save final training history
    with open(output_path / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best validation accuracy: {100.*best_val_acc:.2f}% at epoch {best_epoch+1}")
    print(f"Checkpoints saved in: {output_path / 'checkpoints'}")
    print("="*60)

    return {
        "best_val_accuracy": best_val_acc,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "final_epoch": epoch,
        "history": history,
        "checkpoint_dir": str(output_path / "checkpoints")
    }


def main():
    parser = argparse.ArgumentParser(description="Train MCS selection model")
    parser.add_argument("--model", type=str, required=True,
                       choices=["attention", "dnn", "cnn", "lstm", "cnn_lstm"])
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--output_dir", type=str, default="results/")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    parser.add_argument("--no_amp", action="store_true", help="Disable Automatic Mixed Precision")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--num_workers", type=int, default=4)

    # Model-specific config (could be loaded from YAML, but keeping simple)
    parser.add_argument("--config_file", type=str, default=None,
                       help="JSON file with model hyperparameters")

    args = parser.parse_args()

    # Set seed
    set_global_seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Load model config
    if args.config_file:
        with open(args.config_file, 'r') as f:
            model_config = json.load(f)
    else:
        # Use default configs
        if args.model == "attention":
            model_config = {
                "d_model": 128,
                "n_heads": 4,
                "n_layers": 2,
                "dropout": 0.1
            }
        elif args.model == "dnn":
            model_config = {"hidden_dims": [512, 256, 128, 64], "dropout": 0.2}
        elif args.model == "cnn":
            model_config = {"channels": [32, 64, 128], "dropout": 0.2}
        elif args.model == "lstm":
            model_config = {"hidden_size": 128, "num_layers": 2, "dropout": 0.2}
        elif args.model == "cnn_lstm":
            model_config = {
                "cnn_config": {"channels": [32, 64, 128], "dropout": 0.2},
                "lstm_config": {"hidden_size": 128, "num_layers": 2, "dropout": 0.2}
            }

    # Output directory per model
    output_dir = os.path.join(args.output_dir, args.model)

    # Train
    try:
        results = train_model(
            model_name=args.model,
            data_dir=args.data_dir,
            output_dir=output_dir,
            config=model_config,
            device=device,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            use_amp=not args.no_amp,
            resume_from=args.resume,
            num_workers=args.num_workers
        )

        # Save final metrics
        from utils import save_metrics
        metrics_to_save = {
            "model_name": args.model,
            "seed": args.seed,
            "hyperparameters": model_config,
            "training": {
                "total_epochs": args.epochs,
                "batch_size": args.batch_size,
                "initial_lr": args.lr,
                "best_epoch": results["best_epoch"] + 1,
                "best_val_accuracy": float(results["best_val_accuracy"]),
                "best_val_loss": float(results["best_val_loss"])
            },
            "checkpoint_dir": results["checkpoint_dir"]
        }
        save_metrics(metrics_to_save, filename=f"{args.model}_training_metrics.json")

        print("\nTraining script completed successfully.")

    except Exception as e:
        print(f"\n[Error] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    set_global_seed(GLOBAL_SEED)
    main()