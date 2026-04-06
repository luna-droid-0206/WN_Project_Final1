"""
Visualization tools for MCS selection models.
Includes: confusion matrix, training curves, attention heatmaps, throughput plots.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib import cm


def set_plot_style():
    """Set consistent plot style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 11,
        'figure.figsize': (10, 6),
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10
    })


def plot_training_history(
    history: dict,
    save_path: Optional[str] = None,
    title: str = "Training History"
):
    """
    Plot training and validation loss/accuracy curves.

    Args:
        history: Dict with keys ['train_loss', 'train_acc', 'val_loss', 'val_acc']
        save_path: If provided, save figure to this path
        title: Plot title
    """
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train', linewidth=2, markersize=4)
    axes[0].plot(epochs, history['val_loss'], 'r-s', label='Validation', linewidth=2, markersize=4)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, [a*100 for a in history['train_acc']], 'b-o', label='Train', linewidth=2, markersize=4)
    axes[1].plot(epochs, [a*100 for a in history['val_acc']], 'r-s', label='Validation', linewidth=2, markersize=4)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 105)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training plot to: {save_path}")

    plt.show()


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    normalize: bool = True,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix"
):
    """
    Plot confusion matrix.

    Args:
        confusion_matrix: 2D array [n_classes, n_classes]
        normalize: If True, normalize by row (true class)
        class_names: List of class label names (e.g., MCS 1-15)
        save_path: If provided, save figure
        title: Plot title
    """
    set_plot_style()
    cm = confusion_matrix.copy()

    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
        fmt = '.2%' if normalize else 'd'
        vmax = 1.0 if normalize else None
    else:
        fmt = 'd'
        vmax = None

    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        vmax=vmax,
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    plt.xlabel('Predicted MCS')
    plt.ylabel('True MCS')
    plt.title(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to: {save_path}")

    plt.show()


def plot_attention_heatmaps(
    attn_weights_list: List[torch.Tensor],
    sample_idx: int = 0,
    n_heads_to_plot: int = 4,
    save_path: Optional[str] = None,
    title: str = "Attention Weights"
):
    """
    Visualize attention weights across layers and heads.
    Critical for demonstrating what the model "pays attention to" in the time window.

    Args:
        attn_weights_list: List of tensors [batch, n_heads, seq_len, seq_len] from each layer
        sample_idx: Which batch sample to visualize
        n_heads_to_plot: Number of attention heads to plot
        save_path: If provided, save figure
        title: Plot title
    """
    set_plot_style()

    n_layers = len(attn_weights_list)
    n_heads = attn_weights_list[0].size(1)

    # Select heads to plot (evenly distributed)
    head_indices = np.linspace(0, n_heads-1, min(n_heads_to_plot, n_heads), dtype=int)

    fig, axes = plt.subplots(
        n_layers, len(head_indices),
        figsize=(4*len(head_indices), 3*n_layers),
        squeeze=False
    )

    for layer_idx, layer_attn in enumerate(attn_weights_list):
        attn_sample = layer_attn[sample_idx].cpu().numpy()  # [n_heads, seq_len, seq_len]

        for head_plot_idx, head_idx in enumerate(head_indices):
            ax = axes[layer_idx, head_plot_idx]
            head_attn = attn_sample[head_idx]

            im = ax.imshow(head_attn, cmap='viridis', vmin=0, vmax=1, aspect='auto')
            ax.set_title(f"Layer {layer_idx+1}, Head {head_idx+1}")
            ax.set_xlabel("Key (past symbols)")
            ax.set_ylabel("Query (current symbol)")
            ax.set_xticks(range(head_attn.shape[1]))
            ax.set_yticks(range(head_attn.shape[0]))

    fig.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved attention heatmaps to: {save_path}")

    plt.show()


def plot_throughput_vs_snr(
    metrics_list: List[dict],
    model_names: List[str],
    save_path: Optional[str] = None
):
    """
    Bar chart comparing throughput ratios across models.

    Args:
        metrics_list: List of metrics dicts (from evaluate_model or loaded from JSON)
        model_names: Names for x-axis
        save_path: If provided, save figure
    """
    set_plot_style()

    throughput_ratios = [m.get('throughput_ratio', 0.0) for m in metrics_list]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, throughput_ratios, color=plt.cm.Set2.colors[:len(model_names)])

    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Oracle (100%)')
    plt.ylabel('Normalized Throughput Ratio')
    plt.xlabel('Model')
    plt.title('Throughput Comparison Across Models')
    plt.ylim(0, 1.1)
    plt.legend()

    # Add value labels on bars
    for bar, ratio in zip(bars, throughput_ratios):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{ratio:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved throughput plot to: {save_path}")

    plt.show()


def plot_class_accuracy_comparison(
    class_accuracies_dict: dict,
    save_path: Optional[str] = None
):
    """
    Bar chart comparing per-MCS class accuracy across models.

    Args:
        class_accuracies_dict: {model_name: [acc_class0, acc_class1, ..., acc_classN]}
        save_path: If provided, save figure
    """
    set_plot_style()

    n_classes = len(next(iter(class_accuracies_dict.values())))
    x = np.arange(1, n_classes+1)  # MCS indices 1-15

    width = 0.8 / len(class_accuracies_dict)
    offsets = np.linspace(-0.4 + width/2, 0.4 - width/2, len(class_accuracies_dict))

    plt.figure(figsize=(12, 6))

    for (model_name, accuracies), offset in zip(class_accuracies_dict.items(), offsets):
        plt.bar(x + offset, accuracies, width=width, label=model_name, alpha=0.8)

    plt.xlabel('MCS Class')
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy Comparison')
    plt.xticks(x, [str(i) for i in x])
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved class accuracy plot to: {save_path}")

    plt.show()


def plot_attention_temporal_patterns(
    attn_weights: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Visualize temporal attention patterns.
    Shows which past time steps are attended to when predicting current MCS.

    Args:
        attn_weights: [n_layers, n_heads, seq_len, seq_len] or
                      [batch, n_heads, seq_len, seq_len] (will average over batch/heads)
        save_path: If provided, save figure
    """
    set_plot_style()

    if attn_weights.ndim == 4:
        # Average over batch and heads
        attn_avg = attn_weights.mean(axis=(0, 1))  # [seq_len, seq_len]
    elif attn_weights.ndim == 3:
        # Average over heads: [n_layers, seq_len, seq_len] -> plot each layer
        attn_avg = attn_weights
    else:
        attn_avg = attn_weights

    if attn_avg.ndim == 3:
        n_layers = attn_avg.shape[0]
        fig, axes = plt.subplots(1, n_layers, figsize=(4*n_layers, 4), squeeze=False)
        for i in range(n_layers):
            ax = axes[0, i]
            im = ax.imshow(attn_avg[i], cmap='hot', aspect='auto', vmin=0, vmax=1)
            ax.set_title(f"Layer {i+1}")
            ax.set_xlabel("Key (past OFDM symbol)")
            ax.set_ylabel("Query (current symbol)")
            fig.colorbar(im, ax=ax)
    else:
        plt.figure(figsize=(6, 5))
        im = plt.imshow(attn_avg, cmap='hot', aspect='auto', vmin=0, vmax=1)
        plt.colorbar(im, label='Attention weight')
        plt.xlabel('Key (past OFDM symbol)')
        plt.ylabel('Query (current symbol)')
        plt.title('Average Attention Pattern')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved attention temporal plot to: {save_path}")

    plt.show()


# Export
__all__ = [
    'plot_training_history',
    'plot_confusion_matrix',
    'plot_attention_heatmaps',
    'plot_throughput_vs_snr',
    'plot_class_accuracy_comparison',
    'plot_attention_temporal_patterns',
    'set_plot_style'
]