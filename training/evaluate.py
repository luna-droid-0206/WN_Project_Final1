"""
Evaluation script for trained MCS models.
Loads model checkpoint and evaluates on test set, computing all metrics.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Add project root to Python path for absolute imports when running as script
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from training.config import set_global_seed, GLOBAL_SEED, DataConfig
from training.utils import count_parameters, measure_inference_latency, save_metrics
from models.baselines import get_model  # Attention and baseline model factory


def load_model(
    checkpoint_path: str,
    model_name: str,
    model_config: Dict[str, Any],
    device: torch.device
) -> torch.nn.Module:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = get_model(model_name, **model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded checkpoint from: {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Val accuracy: {checkpoint.get('val_metrics', {}).get('accuracy', 'N/A'):.4f}")

    return model


def load_dataset(
    data_dir: str,
    batch_size: int = 256,
    num_workers: int = 4,
    seed: int = GLOBAL_SEED
) -> DataLoader:
    """Load test dataset."""
    test_dir = os.path.join(data_dir, "test")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    from data_generator import create_dataloader
    test_loader = create_dataloader(
        test_dir,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=seed
    )
    return test_loader


def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    compute_latency: bool = True,
    latency_trials: int = 100
) -> Dict[str, Any]:
    """
    Comprehensive evaluation on test set.

    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test set
        device: torch.device
        compute_latency: If True, measure inference latency
        latency_trials: Number of timing iterations

    Returns:
        Dictionary with all metrics
    """
    model.eval()
    model.to(device)

    # Get a sample input shape for latency measurement
    sample_batch = next(iter(test_loader))[0]
    input_shape = sample_batch.shape  # [batch, seq_len, n_subcarriers, 2]

    # Collect predictions and labels
    all_preds = []
    all_labels = []
    all_logits = []

    print("Evaluating on test set...")
    with torch.no_grad():
        for channels, labels in tqdm(test_loader):
            channels = channels.to(device, non_blocking=True)
            logits = model(channels)
            preds = torch.argmax(logits, dim=1)

            all_logits.append(logits.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_logits = torch.cat(all_logits, dim=0)

    # Compute accuracy metrics
    accuracy = (all_preds == all_labels).float().mean().item()
    correct_per_class = torch.bincount(all_labels[all_preds == all_labels], minlength=DataConfig.N_MCS_CLASSES)
    total_per_class = torch.bincount(all_labels, minlength=DataConfig.N_MCS_CLASSES)
    class_accuracy = correct_per_class.float() / (total_per_class + 1e-8)

    # Top-3 accuracy
    _, top3_preds = torch.topk(all_logits, 3, dim=1)
    top3_correct = torch.sum(top3_preds == all_labels.unsqueeze(1)).item()
    top3_accuracy = top3_correct / len(all_labels)

    # Confusion matrix
    confusion_matrix = torch.zeros(DataConfig.N_MCS_CLASSES, DataConfig.N_MCS_CLASSES, dtype=torch.int64)
    for t, p in zip(all_labels, all_preds):
        confusion_matrix[t.long(), p.long()] += 1

    # Model statistics
    n_params = count_parameters(model)
    model_size_mb = sum(p.element_size() * p.nelement() for p in model.parameters()) / 1024**2

    metrics = {
        "accuracy": accuracy,
        "top3_accuracy": top3_accuracy,
        "class_accuracy": class_accuracy.tolist(),
        "confusion_matrix": confusion_matrix.numpy().tolist(),
        "per_class_true": total_per_class.tolist(),
        "n_parameters": n_params,
        "model_size_mb": model_size_mb
    }

    # Measure inference latency (if requested)
    if compute_latency:
        print("\nMeasuring inference latency...")
        latency_ms = measure_inference_latency(
            model,
            input_shape,
            n_repeats=latency_trials,
            device=device
        )
        metrics["inference_latency_ms"] = latency_ms
        print(f"  Average latency: {latency_ms:.2f} ms")
    else:
        metrics["inference_latency_ms"] = None

    return metrics, all_preds, all_labels


def estimate_throughput(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    mcs_table: Dict[int, Tuple[str, float]]
) -> float:
    """
    Estimate normalized throughput relative to oracle.
    Throughput = sum(selected_MCS_rate) / sum(oracle_MCS_rate) averaged across test set.

    Simpler approach: use class index as proxy for rate (higher index = higher rate).
    This gives a normalized ratio (0-1).
    """
    model.eval()

    selected_rates = []
    oracle_rates = []

    with torch.no_grad():
        for channels, labels in tqdm(test_loader, desc="Throughput estimation"):
            channels = channels.to(device)
            labels = labels.to(device)

            # Model selection
            logits = model(channels)
            preds = torch.argmax(logits, dim=1)

            # Use MCS index as proxy for rate (higher index = higher spectral efficiency)
            # Clip to avoid 0 (no transmission)
            selected_rate = torch.clamp(preds.float(), min=1.0)
            oracle_rate = torch.clamp(labels.float(), min=1.0)

            selected_rates.append(selected_rate.cpu())
            oracle_rates.append(oracle_rate.cpu())

    selected_rates = torch.cat(selected_rates)
    oracle_rates = torch.cat(oracle_rates)

    # Normalized throughput ratio
    throughput_ratio = (selected_rates.sum() / oracle_rates.sum()).item()

    return throughput_ratio


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained MCS selection model on test set")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt file)")
    parser.add_argument("--data_dir", type=str, default="data/", help="Directory with test split")
    parser.add_argument("--model_name", type=str, required=True, choices=["attention", "dnn", "cnn", "lstm", "cnn_lstm"])
    parser.add_argument("--config_file", type=str, default=None, help="JSON file with model config (if not in checkpoint)")
    parser.add_argument("--output", type=str, default=None, help="Output path for metrics JSON")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_latency", action="store_true", help="Skip latency measurement")
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)

    args = parser.parse_args()

    set_global_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Extract model config
    if 'config' in checkpoint:
        model_config = checkpoint['config']
    elif args.config_file:
        with open(args.config_file, 'r') as f:
            model_config = json.load(f)
    else:
        # Minimal config for backward compatibility
        model_config = {}

    # Load model
    model = load_model(args.checkpoint, args.model_name, model_config, device)

    # Load test data
    test_loader = load_dataset(args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

    # Evaluate
    metrics, preds, labels = evaluate_model(
        model,
        test_loader,
        device,
        compute_latency=not args.no_latency
    )

    # Estimate throughput
    print("\nEstimating throughput ratio...")
    throughput_ratio = estimate_throughput(model, test_loader, device, {})
    metrics["throughput_ratio"] = throughput_ratio
    print(f"  Throughput ratio (model/oracle): {throughput_ratio:.4f}")

    # Save metrics
    from training.config import OracleConfig
    metrics_to_save = {
        "model_name": args.model_name,
        "checkpoint_path": args.checkpoint,
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "seed": args.seed,
        "hyperparameters": model_config,
        "dataset": {
            "test_samples": len(test_loader.dataset),
        },
        "metrics": {
            "test_accuracy": metrics["accuracy"],
            "test_top3_accuracy": metrics["top3_accuracy"],
            "throughput_ratio": throughput_ratio,
            "inference_latency_ms": metrics.get("inference_latency_ms"),
            "model_size_params": metrics["n_parameters"],
            "model_size_mb": metrics["model_size_mb"]
        },
        "per_class_accuracy": metrics["class_accuracy"]
    }

    output_path = args.output or f"results/metrics/{args.model_name}_test_metrics.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)

    print(f"\nMetrics saved to: {output_path}")

    # Print summary
    print("\n" + "="*60)
    print("Evaluation Summary")
    print("="*60)
    print(f"Model: {args.model_name}")
    print(f"Test accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Test top-3 accuracy: {metrics['top3_accuracy']*100:.2f}%")
    print(f"Throughput ratio: {throughput_ratio:.4f}")
    print(f"Inference latency: {metrics.get('inference_latency_ms', 'N/A'):.2f} ms" if metrics.get('inference_latency_ms') else "Latency: not measured")
    print(f"Parameters: {metrics['n_parameters']:,}")
    print(f"Model size: {metrics['model_size_mb']:.2f} MB")
    print("="*60)


if __name__ == "__main__":
    set_global_seed(GLOBAL_SEED)
    main()