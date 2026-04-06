"""
Additional metric computations for MCS selection evaluation.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.metrics import confusion_matrix as sk_confusion_matrix


def compute_f1_score(
    confusion_matrix: np.ndarray,
    average: str = 'weighted'
) -> float:
    """
    Compute F1 score from confusion matrix.

    Args:
        confusion_matrix: [n_classes, n_classes]
        average: 'macro', 'micro', 'weighted', or None (return per-class)

    Returns:
        F1 score (float or array)
    """
    n_classes = confusion_matrix.shape[0]

    # True positives: diagonal
    tp = np.diag(confusion_matrix)
    # Predicted positives: sum over columns
    pred_pos = confusion_matrix.sum(axis=0)
    # Actual positives: sum over rows
    true_pos = confusion_matrix.sum(axis=1)

    # Precision and recall per class
    precision = np.divide(tp, pred_pos, out=np.zeros_like(tp, dtype=float), where=pred_pos>0)
    recall = np.divide(tp, true_pos, out=np.zeros_like(tp, dtype=float), where=true_pos>0)

    # F1 per class
    f1_per_class = 2 * (precision * recall) / (precision + recall + 1e-8)

    if average == 'macro':
        return float(np.nanmean(f1_per_class))
    elif average == 'micro':
        # Micro-averaged F1 = 2 * micro-precision * micro-recall / (micro-precision + micro-recall)
        micro_precision = tp.sum() / (pred_pos.sum() + 1e-8)
        micro_recall = tp.sum() / (true_pos.sum() + 1e-8)
        return 2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-8)
    elif average == 'weighted':
        weights = true_pos / true_pos.sum()
        return float((weights * f1_per_class).sum())
    else:
        return f1_per_class


def compute_cohen_kappa(confusion_matrix: np.ndarray) -> float:
    """
    Compute Cohen's kappa coefficient.
    Measures inter-rater agreement (here: model vs oracle).
    """
    n = confusion_matrix.sum()
    if n == 0:
        return 0.0

    observed = np.trace(confusion_matrix) / n

    expected = np.sum(
        (confusion_matrix.sum(axis=0) / n) *
        (confusion_matrix.sum(axis=1) / n)
    )

    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0

    kappa = (observed - expected) / (1 - expected)
    return float(kappa)


def compute_per_class_metrics(
    confusion_matrix: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute precision, recall, F1 for each class.

    Returns:
        {class_name: {"precision": ..., "recall": ..., "f1": ..., "support": ...}}
    """
    n_classes = confusion_matrix.shape[0]
    if class_names is None:
        class_names = [f"MCS_{i+1}" for i in range(n_classes)]

    metrics = {}

    for i in range(n_classes):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp
        fn = confusion_matrix[i, :].sum() - tp

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        support = confusion_matrix[i, :].sum()

        metrics[class_names[i]] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(support)
        }

    # Add macro averages
    macro_precision = np.mean([m["precision"] for m in metrics.values()])
    macro_recall = np.mean([m["recall"] for m in metrics.values()])
    macro_f1 = np.mean([m["f1"] for m in metrics.values()])

    metrics["macro_avg"] = {
        "precision": float(macro_precision),
        "recall": float(macro_recall),
        "f1": float(macro_f1),
        "support": int(confusion_matrix.sum())
    }

    # Weighted averages
    supports = np.array([m["support"] for m in metrics.values() if k != "macro_avg"])
    weights = supports / supports.sum()

    for key in ["precision", "recall", "f1"]:
        values = np.array([m[key] for m in metrics.values() if k != "macro_avg"])
        metrics[f"weighted_avg_{key}"] = float(np.sum(values * weights))

    return metrics


def load_metrics_file(filepath: str) -> Dict:
    """Load metrics from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def aggregate_metrics_across_models(
    metrics_dir: str,
    model_names: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Load metrics for multiple models and return as nested dict.

    Args:
        metrics_dir: Directory containing metrics JSON files
        model_names: List of model names to look for

    Returns:
        {model_name: {metric_name: value}}
    """
    all_metrics = {}

    for model_name in model_names:
        # Find the latest metrics file for this model
        metrics_path = Path(metrics_dir) / f"{model_name}_test_metrics.json"
        if not metrics_path.exists():
            print(f"[Warning] Metrics file not found for {model_name}: {metrics_path}")
            continue

        metrics = load_metrics_file(str(metrics_path))
        all_metrics[model_name] = metrics

    return all_metrics


def summarize_metrics_comparison(
    all_metrics: Dict[str, Dict[str, float]],
    key_metrics: List[str] = None
) -> Dict[str, List[float]]:

    if key_metrics is None:
        key_metrics = [
            "test_accuracy",
            "test_top3_accuracy",
            "throughput_ratio",
            "inference_latency_ms"
        ]

    comparison = {metric: [] for metric in key_metrics}

    for model_name, data in all_metrics.items():
        m = data.get("metrics", {})   # ✅ FIX: extract nested dict

        for metric in key_metrics:
            comparison[metric].append(m.get(metric, None))

    return comparison


def save_comparison_table(
    comparison: Dict[str, List[float]],
    model_names: List[str],
    save_path: str
):
    """Save comparison table as CSV."""
    import csv

    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
        writer.writerow(['Model'] + list(comparison.keys()))
        # Rows
        for i, model_name in enumerate(model_names):
            row = [model_name]
            for metric_values in comparison.values():
                val = metric_values[i] if i < len(metric_values) else ""
                if isinstance(val, float):
                    val = f"{val:.4f}"
                row.append(val)
            writer.writerow(row)

    print(f"Comparison table saved to: {save_path}")


# Export
__all__ = [
    'compute_f1_score',
    'compute_cohen_kappa',
    'compute_per_class_metrics',
    'load_metrics_file',
    'aggregate_metrics_across_models',
    'summarize_metrics_comparison',
    'save_comparison_table'
]