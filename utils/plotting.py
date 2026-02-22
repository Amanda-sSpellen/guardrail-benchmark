# TODO: add timestamps
"""
plotting.py: Visualization utilities for benchmark results.

Provides functions to generate confusion matrices, latency comparisons,
and other performance visualizations for model evaluation reports.
"""

from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# Use non-interactive backend for file generation
matplotlib.use("Agg")


def plot_confusion_matrix(
    tp: int,
    tn: int,
    fp: int,
    fn: int,
    model_name: str = "Model",
    save_path: Optional[str | Path] = None
) -> Optional[str]:
    """
    Plot and save a confusion matrix visualization.
    
    Args:
        tp: True positives
        tn: True negatives
        fp: False positives
        fn: False negatives
        model_name: Name of the model for the title
        save_path: Optional path to save the figure (if None, returns figure object)
        
    Returns:
        Path where figure was saved, or None if save_path not provided
    """
    # Create confusion matrix
    cm = np.array([[tn, fp], [fn, tp]])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot heatmap
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Set ticks and labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Unsafe", "Predicted Safe"])
    ax.set_yticklabels(["Actually Unsafe", "Actually Safe"])
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight="bold")
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=16, fontweight="bold")
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return str(save_path)
    
    return None


def plot_latency_comparison(
    model_results: Dict[str, List[float]],
    save_path: Optional[str | Path] = None
) -> Optional[str]:
    """
    Plot and save a latency comparison between models.
    
    Creates a box plot showing latency distribution for each model,
    allowing comparison of performance (speed) across guardrail providers.
    
    Args:
        model_results: Dictionary mapping model names to lists of latency values (ms)
        save_path: Optional path to save the figure
        
    Returns:
        Path where figure was saved, or None if save_path not provided
    """
    if not model_results:
        raise ValueError("model_results cannot be empty")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data for box plot
    model_names = list(model_results.keys())
    latencies = [model_results[name] for name in model_names]
    
    # Create box plot
    bp = ax.boxplot(latencies, label=model_names, patch_artist=True)
    
    # Color the boxes
    colors = plt.get_cmap("Set3")(np.linspace(0, 1, len(model_names)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    
    # Customize plot
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_title("Latency Comparison Across Guardrail Models", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    
    # Rotate x labels if needed
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return str(save_path)
    
    return None


def plot_metrics_comparison(
    metrics_by_model: Dict[str, Dict[str, float]],
    metrics_to_plot: List[str] | None = None,
    save_path: Optional[str | Path] = None
) -> Optional[str]:
    """
    Plot and save a comparison of classification metrics across models.
    
    Creates a grouped bar chart comparing metrics like Accuracy, Precision,
    Recall, and F1 across different models.
    
    Args:
        metrics_by_model: Dictionary mapping model names to metric dictionaries
        metrics_to_plot: List of metric names to include (default: ['accuracy', 'precision', 'recall', 'f1'])
        save_path: Optional path to save the figure
        
    Returns:
        Path where figure was saved, or None if save_path not provided
    """
    if not metrics_by_model:
        raise ValueError("metrics_by_model cannot be empty")
    
    if metrics_to_plot is None:
        metrics_to_plot = ["accuracy", "precision", "recall", "f1"]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data
    model_names = list(metrics_by_model.keys())
    x = np.arange(len(model_names))
    width = 0.2
    
    # Plot each metric as a group of bars
    for idx, metric in enumerate(metrics_to_plot):
        values = [metrics_by_model[model].get(metric, 0) for model in model_names]
        ax.bar(x + idx * width, values, width, label=metric.capitalize())
    
    # Customize plot
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Classification Metrics Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.legend(loc="lower right")
    ax.set_ylim((0, 1.0))
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return str(save_path)
    
    return None
