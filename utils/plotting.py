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

from loguru import logger

# Use non-interactive backend for file generation
matplotlib.use("Agg")


def plot_confusion_matrix(
    tp: int,
    tn: int,
    fp: int,
    fn: int,
    model_name: str = "Model",
    save_path: Optional[str | Path] = None,
    normalize: bool = False,
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

    # Optionally normalize rows (true-class normalization)
    disp = cm.astype(float)
    if normalize:
        with np.errstate(all="ignore"):
            row_sums = disp.sum(axis=1, keepdims=True)
            disp = np.divide(disp, row_sums, where=row_sums != 0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot heatmap
    im = ax.imshow(disp, interpolation="nearest", cmap="Blues")
    
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
                    color="white" if cm[i, j] > cm.max() / 2.0 else "black",
                    fontsize=16, fontweight="bold")
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return str(save_path)
    
    return None


def plot_multiclass_confusion_matrix(
    cm: Dict[str, Dict[str, float]] | np.ndarray,
    class_names: List[str],
    model_name: str = "Model",
    normalize: bool = False,
    cmap: str = "Blues",
    save_path: Optional[str | Path] = None,
    category_distribution: Optional[Dict[str, float]] = None,
) -> Optional[str]:
    """
    Plot and save a multiclass confusion matrix.

    Accepts either a nested-dict confusion matrix (as produced by
    `calculate_multiclass_metrics().confusion_matrix`) or a numpy array
    together with `class_names`.
    """
    # Convert dict representation to numpy array if needed
    if isinstance(cm, dict):
        # # Preserve insertion order from the dict (metrics builds it that way)
        # if class_names is None:
        #     class_names = list(cm.keys())

        classes = class_names
        n = len(classes)
        arr = np.zeros((n, n))
        for i, true_cls in enumerate(classes):
            row = cm.get(true_cls, {})
            for j, pred_cls in enumerate(classes):
                arr[i, j] = row.get(pred_cls, 0)
    else:
        arr = np.array(cm)
        if class_names is None:
            # Fallback to numeric labels
            class_names = [str(i) for i in range(arr.shape[0])]

    # Optionally normalize rows (true-class normalization)
    disp = arr.astype(float)
    if normalize: # TODO: fix
        with np.errstate(all="ignore"):
            if category_distribution is None:
                row_sums = disp.sum(axis=1, keepdims=True)
            else:
                row_sums = np.array([[category_distribution[cat]] for cat in class_names])
            disp = np.divide(disp, row_sums, where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 0.6), max(6, len(class_names) * 0.6)))
    im = ax.imshow(disp, interpolation="nearest", cmap=cmap)
    plt.colorbar(im, ax=ax)

    # Ticks and labels
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight="bold")

    # Annotate cells with counts or percentages
    fmt = ".2f" # if normalize else "d"
    thresh = disp.max() / 3.0 if disp.size else 0
    for i in range(disp.shape[0]):
        for j in range(disp.shape[1]):
            val = disp[i, j]
            text = format(val if not normalize else val, fmt)
            ax.text(j, i, text, ha="center", va="center",
                    color="black" if disp[i, j] > thresh else "black",
                    fontsize=10)

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


def generate_confusion_matrices(
        metrics, 
        output_dir, 
        experiment_index, 
        model_name, 
        categories, 
        safe_categories: List[str],
        normalize=False,
    ) -> Dict[str, Path]:
    """
    Generate confusion matrices for binary and multiclass classification.
    
    Returns:
        Dictionary mapping visualization names to file paths
    """
    logger.info("Generating visualizations")
    
    if not metrics:
        raise ValueError("Must calculate metrics before generating visualizations")
    
    paths = {}
    index_prefix = f"{experiment_index:03d}_"
    
    # Confusion matrix for binary classification
    is_accumulative = metrics['accumulative']
    iterations = metrics['iterations']
    if is_accumulative:
        # Get averaged confusion matrix
        cm = metrics["binary_classification"]["average_confusion_matrix"]
    else: 
        cm = metrics["binary_classification"]
    
    labeled_cm = {}
    labeled_cm["safe"] = {
        "safe": cm["tp"],
        "unsafe": cm["fn"]
    }
    labeled_cm["unsafe"] = {
        "safe": cm["fp"],
        "unsafe": cm["tn"]
    }
        
    cm_path = output_dir / f"{experiment_index:03d}" / f"{index_prefix}binary{'_normalized' if normalize else ''}_confusion_matrix.png"
    # cat_dist: Dict[str, float] = {
    #     "safe": sum([metrics["category_distribution"][cat] for cat in safe_categories])/iterations,
    #     "unsafe": sum([
    #         dist
    #         for cat, dist in metrics["category_distribution"].items() 
    #         if cat not in safe_categories
    #     ])/iterations,
    # }
    plot_multiclass_confusion_matrix(
        cm=labeled_cm,
        class_names=["safe", "unsafe"],
        model_name=f"{model_name} (Safe vs Unsafe){f' (Averaged {iterations} iterations)' if is_accumulative else ''}{' (Normalized)' if normalize else ''}",
        save_path=cm_path,
        normalize=normalize,
        category_distribution=None, # cat_dist,
    )
    paths["binary_confusion_matrix"] = cm_path
    logger.info(f"Saved binary confusion matrix: {cm_path}")
    
    # Multiclass confusion matrix (if available)
    try:
        if is_accumulative:
            mc = metrics["multiclass_classification"]["average_confusion_matrix"]
        else:
            mc = metrics["multiclass_classification"]
    except Exception:
        mc = None

    if mc:
        mc_path = output_dir / f"{experiment_index:03d}" / f"{index_prefix}multiclass{'_normalized' if normalize else ''}_confusion_matrix.png"
        plot_multiclass_confusion_matrix(
            cm=mc,
            class_names=categories,
            model_name=f"{model_name} (Multiclass){f' (Averaged {iterations} iterations)' if is_accumulative else ''}{' (Normalized)' if normalize else ''}",
            save_path=mc_path,
            normalize=normalize,
            category_distribution=None, #{cat: dist/iterations for cat, dist in metrics["category_distribution"].items()},
        )
        paths["multiclass_confusion_matrix"] = mc_path
        logger.info(f"Saved multiclass confusion matrix: {mc_path}")

    return paths
