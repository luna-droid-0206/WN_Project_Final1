#!/usr/bin/env python
"""
Complete pipeline: Generate data, train all models, evaluate, and compare.
This script runs the full project workflow end-to-end.
"""

import os
import subprocess
import sys
from pathlib import Path

from training.config import set_global_seed, GLOBAL_SEED

set_global_seed(GLOBAL_SEED)

print("="*80)
print("MCS SELECTION PROJECT - FULL PIPELINE")
print("="*80+"\n")

# Configuration
N_SAMPLES = 100000
BATCH_SIZE = 128
EPOCHS = 50  # Reduced for quick demo; use 100 for final
MODELS = ["attention", "dnn", "cnn", "lstm", "cnn_lstm"]
DATA_DIR = "data/"
RESULTS_DIR = "results/"
PLOTS_DIR = "results/plots/"

# Step 1: Generate dataset (if not already exists)
if not os.path.exists(DATA_DIR) or not os.path.exists(os.path.join(DATA_DIR, "train")):
    print("[Step 1/6] Generating dataset...")
    print("-"*80)
    cmd = [
        sys.executable,
        "training/data_generator.py",
        "--n_samples", str(N_SAMPLES),
        "--output", DATA_DIR,
        "--channel_type", "rayleigh",
        "--doppler_hz", "100.0",
        "--delay_spread_s", "0.0",
        "--seed", str(GLOBAL_SEED)
    ]
    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        print("[Error] Dataset generation failed!")
        sys.exit(1)
    print("[✓] Dataset generated\n")
else:
    print("[Step 1/6] Dataset already exists, skipping generation.\n")

# Step 2-6: Train and evaluate each model
for model_name in MODELS:
    print(f"[{['2','3','4','5','6'][MODELS.index(model_name)]}/{6}] Processing model: {model_name.upper()}")
    print("-"*80)

    # Train (if checkpoint doesn't exist)
    checkpoint_path = Path(RESULTS_DIR) / model_name / "checkpoints" / "best_model.pt"
    if checkpoint_path.exists():
        print(f"  Checkpoint exists at {checkpoint_path}, skipping training.")
    else:
        print(f"  Training {model_name}...")
        cmd = [
            sys.executable,
            "training/train.py",
            "--model", model_name,
            "--data_dir", DATA_DIR,
            "--output_dir", RESULTS_DIR,
            "--epochs", str(EPOCHS),
            "--batch_size", str(BATCH_SIZE),
            "--seed", str(GLOBAL_SEED)
        ]
        result = subprocess.run(cmd, check=True)
        if result.returncode != 0:
            print(f"[Error] Training {model_name} failed!")
            continue
        print(f"  ✓ Training complete\n")

    # Evaluate (if metrics don't exist)
    metrics_path = Path("results/metrics") / f"{model_name}_test_metrics.json"
    if metrics_path.exists():
        print(f"  Metrics exist at {metrics_path}, skipping evaluation.")
    else:
        print(f"  Evaluating {model_name} on test set...")
        checkpoint_file = checkpoint_path if checkpoint_path.exists() else None
        if not checkpoint_file:
            print(f"  [Warning] No checkpoint found for {model_name}, cannot evaluate")
            continue
        cmd = [
            sys.executable,
            "training/evaluate.py",
            "--checkpoint", str(checkpoint_file),
            "--data_dir", DATA_DIR,
            "--model_name", model_name
        ]
        result = subprocess.run(cmd, check=True)
        if result.returncode != 0:
            print(f"[Error] Evaluation {model_name} failed!")
            continue
        print(f"  ✓ Evaluation complete\n")

print("[Step Final] Generating comparison...")
print("-"*80)
cmd = [sys.executable, "evaluation/compare_baselines.py", "--results_dir", RESULTS_DIR]
subprocess.run(cmd, check=True)
print("✓ Comparison plots saved to:", PLOTS_DIR)

print("\n" + "="*80)
print("PIPELINE COMPLETE")
print("="*80)
print(f"\nResults:")
print(f"  - Trained models: {RESULTS_DIR}")
print(f"  - Metrics: results/metrics/")
print(f"  - Plots: {PLOTS_DIR}")
print(f"  - Analysis notebook: notebooks/analysis.ipynb")
print("\nOpen notebooks/analysis.ipynb for final analysis and visualizations!")