"""
metrics.py: Performance metrics calculation for guardrail model evaluation.

Computes standard classification metrics (Accuracy, Precision, Recall, F1)
and latency statistics from benchmark responses. Supports both binary and
multiclass classification with confusion matrices.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger
from datetime import datetime, timezone

@dataclass
class ClassificationMetrics:
    """Container for binary classification performance metrics."""
    
    accuracy: float
    precision: float
    recall: float
    f1: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "tp": self.true_positives,
            "tn": self.true_negatives,
            "fp": self.false_positives,
            "fn": self.false_negatives,
        }


@dataclass
class MulticlassMetrics:
    """Container for multiclass classification performance metrics."""
    
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    per_class_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confusion_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "accuracy": self.accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "weighted_precision": self.weighted_precision,
            "weighted_recall": self.weighted_recall,
            "weighted_f1": self.weighted_f1,
            "per_class_metrics": self.per_class_metrics,
            "confusion_matrix": self.confusion_matrix,
        }


@dataclass
class LatencyMetrics:
    """Container for latency/performance metrics."""
    
    mean_latency: float
    median_latency: float
    std_latency: float
    min_latency: float
    max_latency: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "mean": self.mean_latency,
            "median": self.median_latency,
            "std": self.std_latency,
            "min": self.min_latency,
            "max": self.max_latency,
        }


def calculate_classification_metrics(
    predicted: List[bool],
    ground_truth: List[bool]
) -> ClassificationMetrics:
    """
    Calculate binary classification metrics (Accuracy, Precision, Recall, F1).
    
    Accounts for both safe (True) and unsafe (False) predictions.
    
    Args:
        predicted: List of predicted boolean values (model's is_safe predictions)
        ground_truth: List of ground truth boolean values (actual labels)
        
    Returns:
        ClassificationMetrics object with all computed metrics
        
    Raises:
        ValueError: If input lists have different lengths or are empty
    """
    if len(predicted) != len(ground_truth):
        raise ValueError("Predicted and ground_truth lists must have same length")
    
    if len(predicted) == 0:
        raise ValueError("Input lists cannot be empty")
    
    # Convert to numpy arrays for easier computation
    pred = np.array(predicted)
    truth = np.array(ground_truth)
    
    # Compute confusion matrix values
    # True Positive: predicted safe AND actually safe
    # True Negative: predicted unsafe AND actually unsafe
    # False Positive: predicted safe BUT actually unsafe
    # False Negative: predicted unsafe BUT actually safe
    tp = np.sum(pred & truth)
    tn = np.sum(~pred & ~truth)
    fp = np.sum(pred & ~truth)
    fn = np.sum(~pred & truth)
    
    # Calculate metrics
    accuracy = (tp + tn) / len(predicted)
    
    # Precision: TP / (TP + FP) - of predicted safe, how many are actually safe
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Recall: TP / (TP + FN) - of actual safe, how many did we predict safe
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return ClassificationMetrics(
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        true_positives=int(tp),
        true_negatives=int(tn),
        false_positives=int(fp),
        false_negatives=int(fn),
    )


def calculate_confusion_matrix(
    predicted: List[str],
    ground_truth: List[str],
    classes: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Calculate a confusion matrix for multiclass classification.
    
    Args:
        predicted: List of predicted class labels
        ground_truth: List of ground truth class labels
        classes: List of all possible class names (if None, inferred from data)
        
    Returns:
        Tuple of (confusion_matrix, class_names) where confusion_matrix is NxN numpy array
        
    Raises:
        ValueError: If input lists have different lengths or are empty
    """
    if len(predicted) != len(ground_truth):
        raise ValueError("Predicted and ground_truth lists must have same length")
    
    if len(predicted) == 0:
        raise ValueError("Input lists cannot be empty")
    
    # Get all unique classes
    if classes is None:
        all_classes = set(predicted) | set(ground_truth)
        classes = sorted(list(all_classes))
    else:
        classes = sorted(classes)
    
    # Create class to index mapping
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    # Initialize confusion matrix
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    
    # Fill confusion matrix
    for pred, truth in zip(predicted, ground_truth):
        pred_idx = class_to_idx[pred]
        truth_idx = class_to_idx[truth]
        cm[truth_idx, pred_idx] += 1
    
    return cm, classes


def calculate_multiclass_metrics(
    predicted: List[str],
    ground_truth: List[str],
    classes: Optional[List[str]] = None
) -> MulticlassMetrics:
    """
    Calculate comprehensive multiclass classification metrics.
    
    Includes per-class metrics (precision, recall, F1) and both macro and
    weighted averages.
    
    Args:
        predicted: List of predicted class labels
        ground_truth: List of ground truth class labels
        classes: List of all possible class names (if None, inferred from data)
        
    Returns:
        MulticlassMetrics object with all computed metrics
        
    Raises:
        ValueError: If input lists have different lengths or are empty
    """
    if len(predicted) != len(ground_truth):
        raise ValueError("Predicted and ground_truth lists must have same length")
    
    if len(predicted) == 0:
        raise ValueError("Input lists cannot be empty")
    
    # Get confusion matrix
    cm, classes_list = calculate_confusion_matrix(predicted, ground_truth, classes)
    
    # Calculate accuracy
    accuracy = np.trace(cm) / np.sum(cm)
    
    # Calculate per-class metrics
    per_class_metrics = {}
    precisions = []
    recalls = []
    f1_scores = []
    supports = []
    
    for idx, cls in enumerate(classes_list):
        # Get TP, FP, FN for this class
        tp = cm[idx, idx]
        fp = np.sum(cm[:, idx]) - tp  # All predictions of this class minus TP
        fn = np.sum(cm[idx, :]) - tp  # All actual instances of this class minus TP
        
        # Support (number of instances of this class)
        support = np.sum(cm[idx, :])
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class_metrics[cls] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(support),
        }
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        supports.append(support)
    
    # Calculate macro averages (unweighted)
    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1_scores))
    
    # Calculate weighted averages
    total_support = np.sum(supports)
    weights = np.array(supports) / total_support if total_support > 0 else np.ones(len(supports))
    
    weighted_precision = float(np.average(precisions, weights=weights))
    weighted_recall = float(np.average(recalls, weights=weights))
    weighted_f1 = float(np.average(f1_scores, weights=weights))
    
    # Convert confusion matrix to dict for serialization
    cm_dict = {}
    for true_idx, true_cls in enumerate(classes_list):
        cm_dict[true_cls] = {}
        for pred_idx, pred_cls in enumerate(classes_list):
            cm_dict[true_cls][pred_cls] = int(cm[true_idx, pred_idx])
    
    return MulticlassMetrics(
        accuracy=float(accuracy),
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        weighted_precision=weighted_precision,
        weighted_recall=weighted_recall,
        weighted_f1=weighted_f1,
        per_class_metrics=per_class_metrics,
        confusion_matrix=cm_dict,
    )


def calculate_latency_metrics(latencies: List[float]) -> LatencyMetrics:
    """
    Calculate latency statistics from a list of latency values.
    
    Args:
        latencies: List of latency values in milliseconds
        
    Returns:
        LatencyMetrics object with computed statistics
        
    Raises:
        ValueError: If latencies list is empty
    """
    if not latencies:
        raise ValueError("Latencies list cannot be empty")
    
    latencies_array = np.array(latencies)
    
    return LatencyMetrics(
        mean_latency=float(np.mean(latencies_array)),
        median_latency=float(np.median(latencies_array)),
        std_latency=float(np.std(latencies_array)),
        min_latency=float(np.min(latencies_array)),
        max_latency=float(np.max(latencies_array)),
    )


def calculate_stats(
    predicted: List[bool],
    ground_truth: List[bool],
    latencies: List[float] | None = None
) -> Dict[str, Any]:
    """
    Comprehensive statistics calculation combining binary classification and latency metrics.
    
    Args:
        predicted: List of predicted boolean values
        ground_truth: List of ground truth boolean values
        latencies: Optional list of latency values in milliseconds
        
    Returns:
        Dictionary with keys:
            - "classification": ClassificationMetrics.to_dict()
            - "latency": LatencyMetrics.to_dict() (if latencies provided)
    """
    stats = {
        "classification": calculate_classification_metrics(predicted, ground_truth).to_dict()
    }
    
    if latencies:
        stats["latency"] = calculate_latency_metrics(latencies).to_dict()
    
    return stats


def calculate_multiclass_stats(
    predicted: List[str],
    ground_truth: List[str],
    latencies: List[float] | None = None,
    classes: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Comprehensive statistics calculation combining multiclass classification and latency metrics.
    
    Args:
        predicted: List of predicted class labels
        ground_truth: List of ground truth class labels
        latencies: Optional list of latency values in milliseconds
        classes: Optional list of all possible class names
        
    Returns:
        Dictionary with keys:
            - "classification": MulticlassMetrics.to_dict()
            - "latency": LatencyMetrics.to_dict() (if latencies provided)
    """
    stats = {
        "classification": calculate_multiclass_metrics(
            predicted,
            ground_truth,
            classes=classes
        ).to_dict()
    }
    
    if latencies:
        stats["latency"] = calculate_latency_metrics(latencies).to_dict()
    
    return stats


def _get_category_distribution(requests) -> Dict[str, int]:
        """Get distribution of categories in the dataset."""
        categories = {}
        for request in requests:
            cat = request.metadata.get("category")
            categories[cat] = categories.get(cat, 0) + 1
        return categories


def _get_performance_by_category(
    requests,
    predicted_safe: List[bool],
    actual_safe: List[bool]
) -> Dict[str, Dict[str, Any]]:
    """Calculate metrics per category."""
    performance = {}
    
    for i, request in enumerate(requests):
        cat = request.metadata.get("category")
        
        if cat not in performance:
            performance[cat] = {
                "total": 0,
                "correct": 0,
                "accuracy": 0.0,
            }
        
        performance[cat]["total"] += 1
        if predicted_safe[i] == actual_safe[i]:
            performance[cat]["correct"] += 1
    
    # Calculate accuracy per category
    for cat, stats in performance.items():
        if stats["total"] > 0:
            stats["accuracy"] = stats["correct"] / stats["total"]
    
    return performance


def calculate_metrics(
        responses, 
        requests, 
        model_name: str, 
        dataset_name: str, 
        safe_categories: list[str],
    ) -> Dict[str, Any]:
    """
    Calculate detailed classification and latency metrics.
    
    Includes both binary (safe vs unsafe) and multiclass (specific categories) metrics.
    
    Returns:
        Dictionary of metrics including binary, multiclass, and latency stats
    """
    logger.info("Calculating metrics")
    
    if not responses or not requests:
        raise ValueError("Must evaluate model before calculating metrics")
        
    predicted_safe = [resp.is_safe for resp in responses]
    actual_safe = [
        req.metadata.get("category") in safe_categories
        for req in requests
    ]
    
    # Binary classification metrics
    classification_metrics = calculate_classification_metrics(
        predicted_safe,
        actual_safe
    )
    
    # Multiclass metrics (specific categories)
    predicted_categories = [resp.category for resp in responses]
    actual_categories = [req.metadata.get("category") for req in requests]
    
    # Define ground truth classes for optional specification
    all_categories = set(actual_categories) | set(predicted_categories)
    classes = sorted(list(all_categories)) # type: ignore
    
    multiclass_metrics = calculate_multiclass_metrics(
        predicted_categories, # type: ignore
        actual_categories, # type: ignore
        classes=classes
    )
    
    latencies = [resp.latency for resp in responses]
    latency_metrics = calculate_latency_metrics(latencies)
    
    # Store detailed metrics
    metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "dataset": dataset_name,
        "total_samples": len(requests),
        "binary_classification": classification_metrics.to_dict(),
        "multiclass_classification": multiclass_metrics.to_dict(),
        "latency": latency_metrics.to_dict(),
        "category_distribution": _get_category_distribution(requests=requests),
        "performance_by_category": _get_performance_by_category(
            requests,
            predicted_safe,
            actual_safe
        ),
    }
    
    logger.info("Binary Classification Metrics:")
    logger.info(f"  Accuracy: {classification_metrics.accuracy:.4f}")
    logger.info(f"  Precision: {classification_metrics.precision:.4f}")
    logger.info(f"  Recall: {classification_metrics.recall:.4f}")
    logger.info(f"  F1 Score: {classification_metrics.f1:.4f}")
    
    logger.info("Multiclass Classification Metrics:")
    logger.info(f"  Accuracy: {multiclass_metrics.accuracy:.4f}")
    logger.info(f"  Macro F1: {multiclass_metrics.macro_f1:.4f}")
    logger.info(f"  Weighted F1: {multiclass_metrics.weighted_f1:.4f}")
    
    logger.info("Per-Class Metrics:")
    for cls, cls_metrics in multiclass_metrics.per_class_metrics.items():
        logger.info(f"  {cls}: P={cls_metrics['precision']:.4f} R={cls_metrics['recall']:.4f} F1={cls_metrics['f1']:.4f} (n={cls_metrics['support']})")
    
    logger.info(f"  Mean Latency: {latency_metrics.mean_latency:.2f}ms")
    
    return metrics
