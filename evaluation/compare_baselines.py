"""
Comparison script for all baseline models.
Loads metrics from each model's results and generates comprehensive comparison plots.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys

# Add project root to Python path for absolute imports when running as script
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import local modules
from training.config import set_global_seed, GLOBAL_SEED, DataConfig
from evaluation.metrics import load_metrics_file
from evaluation.visualization import (
    plot_throughput_vs_snr,
    plot_class_accuracy_comparison,
    plot_training_history
)

set_global_seed(GLOBAL_SEED)


def find_training_metrics(results_dir: str, model_name: str) -> dict:
    """Find the best training metrics for a given model."""
    model_dir = Path(results_dir) / model_name
    if not model_dir.exists():
        return None

    # Look for training_history.json
    history_path = model_dir / "training_history.json"
    if history_path.exists():
        with open(history_path, 'r') as f:
            return json.load(f)

    # Fallback: look for latest metrics file
    metrics_dir = Path("../results/metrics")
    pattern = f"{model_name}_*metrics.json"
    candidates = sorted(metrics_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    if candidates:
        return load_metrics_file(str(candidates[-1]))

    return None


def load_all_model_metrics(
    results_dir: str,
    model_names: List[str]
) -> Dict[str, dict]:
    """Load final test metrics for all models, with fallback to training metrics."""
    metrics_dir = Path("../results/metrics")
    all_metrics = {}

    for model_name in model_names:
        # Try to find the latest test metrics file first (preferred)
        test_candidates = sorted(
            metrics_dir.glob(f"{model_name}_test_metrics.json"),
            key=lambda p: p.stat().st_mtime
        )

        if test_candidates:
            metrics = load_metrics_file(str(test_candidates[-1]))
            all_metrics[model_name] = metrics
            print(f"[Info] Loaded test metrics for {model_name}")
        else:
            # Fallback: look for any metrics file (e.g., training_metrics.json)
            train_candidates = sorted(
                metrics_dir.glob(f"{model_name}_*metrics.json"),
                key=lambda p: p.stat().st_mtime
            )
            if train_candidates:
                metrics = load_metrics_file(str(train_candidates[-1]))
                all_metrics[model_name] = metrics
                print(f"[Warning] No test metrics for {model_name}, using training metrics instead")
            else:
                print(f"[Error] No metrics found for {model_name}")

    return all_metrics


def generate_comparison_plots(
    all_metrics: Dict[str, dict],
    class_accuracies: Dict[str, List[float]],
    output_dir: str = "results/plots/"
):
    """Generate all comparison plots."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model_names = list(all_metrics.keys())

    # 1. Throughput comparison bar chart
    print("\nGenerating throughput comparison...")
    plot_throughput_vs_snr(
        list(all_metrics.values()),
        model_names,
        save_path=str(Path(output_dir) / "throughput_comparison.png")
    )

    # 2. Per-class accuracy comparison (only if data available)
    if class_accuracies:
        print("Generating per-class accuracy comparison...")
        plot_class_accuracy_comparison(
            class_accuracies,
            save_path=str(Path(output_dir) / "class_accuracy_comparison.png")
        )
    else:
        print("Skipping per-class accuracy plot: no per-class data available (run evaluate.py first)")

    # 3. Training curves (if available)
    print("Checking training history...")
    histories = {}
    for model_name in model_names:
        model_results_dir = Path("results") / model_name
        history_path = model_results_dir / "training_history.json"
        if history_path.exists():
            with open(history_path, 'r') as f:
                histories[model_name] = json.load(f)

    if histories:
        print("Generating training history plots...")
        for model_name, history in histories.items():
            plot_training_history(
                history,
                save_path=str(Path(output_dir) / f"training_history_{model_name}.png"),
                title=f"{model_name.upper()} - Training History"
            )
    else:
        print("No training history found for any model")


def print_comparison_table(all_metrics: Dict[str, dict]):
    """Print nice table comparing models."""
    print("\n" + "="*80)
    print("MODEL COMPARISON TABLE")
    print("="*80)
    print(f"{'Model':<15} | {'Accuracy':<12} | {'Top-3 Acc':<12} | {'Throughput':<12} | {'Latency':<12} | {'Params':<12}")
    print("-"*80)

    for model_name, metrics in all_metrics.items():
        acc = metrics.get('test_accuracy', 0) * 100
        top3 = metrics.get('test_top3_accuracy', 0) * 100
        throughput = metrics.get('throughput_ratio', 0)
        latency = metrics.get('inference_latency_ms', 0)
        params = metrics.get('n_parameters', 0)

        print(f"{model_name:<15} | "
              f"{acc:>10.2f}%   | "
              f"{top3:>10.2f}%   | "
              f"{throughput:>10.4f}    | "
              f"{latency:>10.2f} ms  | "
              f"{params:>11,}")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Compare all baseline models")
    parser.add_argument("--results_dir", type=str, default="results/",
                       help="Directory containing model results")
    parser.add_argument("--models", type=str, nargs='+',
                       default=["attention", "dnn", "cnn", "lstm", "cnn_lstm"],
                       help="List of model names to compare")
    parser.add_argument("--output_dir", type=str, default="results/plots/",
                       help="Directory to save comparison plots")
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)

    args = parser.parse_args()

    set_global_seed(args.seed)

    print("="*80)
    print("Loading metrics for all models...")
    print("="*80)

    # Load metrics for all models
    all_metrics = load_all_model_metrics(args.results_dir, args.models)

    if not all_metrics:
        print("[Error] No metrics found for any model. Have you trained them yet?")
        return

    # Extract per-class accuracies
    class_accuracies = {}
    for model_name, metrics in all_metrics.items():
        if 'per_class_accuracy' in metrics:
            # Filter out non-class entries (like macro_avg)
            class_accs = []
            for i in range(DataConfig.N_MCS_CLASSES):
                if i < len(metrics['per_class_accuracy']):
                    class_accs.append(metrics['per_class_accuracy'][i])
                else:
                    class_accs.append(0.0)
            class_accuracies[model_name] = class_accs

    # Print comparison table
    print_comparison_table(all_metrics)

    # Generate plots
    generate_comparison_plots(all_metrics, class_accuracies, args.output_dir)

    # Note: If per-class accuracy data is missing, the per-class plot will be skipped

    # Save overall comparison JSON
    comparison_summary = {
        "models": list(all_metrics.keys()),
        "metrics": all_metrics,
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    }

    summary_path = Path(args.output_dir) / "comparison_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(comparison_summary, f, indent=2)

    print(f"\nComparison summary saved to: {summary_path}")
    print("\nAll plots saved to:", args.output_dir)
    print("\nDone!")


if __name__ == "__main__":
    main()