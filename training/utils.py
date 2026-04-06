"""
Utility functions for training, evaluation, and data handling.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any

import torch
import numpy as np


def save_metrics(metrics_dict: Dict[str, Any], filename: str = None):
    """
    Save metrics dictionary to JSON file in results/metrics/.
    Filename format: {model_name}_{timestamp}.json
    """
    metrics_dir = "results/metrics"
    os.makedirs(metrics_dir, exist_ok=True)

    if filename is None:
        model_name = metrics_dict.get("model_name", "unknown")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{model_name}_{timestamp}.json"

    filepath = os.path.join(metrics_dir, filename)

    # Convert numpy types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        return obj

    serializable_dict = {}
    for key, value in metrics_dict.items():
        if isinstance(value, dict):
            serializable_dict[key] = {k: convert_to_serializable(v) for k, v in value.items()}
        else:
            serializable_dict[key] = convert_to_serializable(value)

    with open(filepath, 'w') as f:
        json.dump(serializable_dict, f, indent=2)

    print(f"[Metrics saved] {filepath}")
    return filepath


def load_metrics(filepath: str) -> Dict[str, Any]:
    """Load metrics from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def count_parameters(model: torch.nn.Module) -> int:
    """Count total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_latency(
    model: torch.nn.Module,
    input_shape: tuple,
    n_repeats: int = 100,
    warmup: int = 10,
    device: str = "cuda"
) -> float:
    """
    Measure average inference latency in milliseconds.
    Args:
        model: PyTorch model
        input_shape: Shape of input tensor (batch, seq_len, features)
        n_repeats: Number of timing iterations
        warmup: Number of warmup iterations (for GPU stabilization)
        device: 'cuda' or 'cpu'
    Returns:
        Average latency in milliseconds
    """
    model.eval()
    model.to(device)

    # Create dummy input
    dummy_input = torch.randn(input_shape, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
        torch.cuda.synchronize()

    # Timing
    times = []
    with torch.no_grad():
        for _ in range(n_repeats):
            if device == "cuda":
                torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = model(dummy_input)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

    return float(np.mean(times))


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Get model size in megabytes (including gradients if training)."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb