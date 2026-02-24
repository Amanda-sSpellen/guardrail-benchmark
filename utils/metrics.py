"""
metrics.py: Performance metrics calculation for guardrail model evaluation.

Computes standard classification metrics (Accuracy, Precision, Recall, F1)
and latency statistics from benchmark responses. Supports both binary and
multiclass classification with confusion matrices.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
from loguru import logger
from datetime import datetime, timezone

from core.schema import GuardrailRequest, GuardrailResponse

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
    classes: List[str]
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

    # Create class to index mapping
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    # Initialize confusion matrix
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    
    # Fill confusion matrix
    for pred, truth in zip(predicted, ground_truth):
        if pred is not None:
            pred_idx = class_to_idx[pred if not isinstance(pred, list) else pred[0]] # TODO: handle None
            truth_idx = class_to_idx[truth]
            cm[truth_idx, pred_idx] += 1
    
    return cm, classes


def calculate_multiclass_metrics(
    predicted: List[str],
    ground_truth: List[str],
    classes: List[str]
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
    
    for idx, cls in enumerate(classes):
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
    for true_idx, true_cls in enumerate(classes):
        cm_dict[true_cls] = {}
        for pred_idx, pred_cls in enumerate(classes):
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


def calculate_accumulative_multiclass_metrics(
    responses,
    requests,
    classes: List[str],
) -> Dict[str, Any]:
    """
    Calculate aggregated multiclass metrics for accumulative executions.

    Returns a dict containing:
      - average_confusion_matrix: averaged confusion matrix (cells averaged over iterations)
      - per_class_mean_metrics: per-class mean precision/recall/f1 across iterations
      - macro_mean_metrics: macro-averaged precision/recall/f1 following the guide
      - micro_metrics: micro-averaged precision/recall/f1 computed from summed counts
      - classes: ordered list of classes
    """
    # Determine iterations and classes
    iterations_set = sorted({getattr(r, "iteration", 0) for r in responses if getattr(r, "iteration", None) is not None})
    if not iterations_set:
        iterations_set = [0]

    # initialize accumulators
    n = len(classes)
    cm_sum = np.zeros((n, n), dtype=float)
    per_iter_per_class_metrics: Dict[int, Dict[str, Dict[str, float]]] = {}

    for it in iterations_set:
        preds = []
        actuals = []
        for r in responses:
            if getattr(r, "iteration", 0) == it:
                preds.append(r.category)
                inst_idx = getattr(r, "instance_index", None)
                if inst_idx is None:
                    # fallback: try to match by order (may be unsafe)
                    inst_idx = len(actuals)
                actuals.append(requests[inst_idx].metadata.get("category"))

        if not preds:
            # empty iteration, skip
            continue

        cm_i, classes_i = calculate_confusion_matrix(preds, actuals, classes)
        # ensure same ordering
        cm_sum += cm_i.astype(float)

        # per-class metrics for this iteration
        iter_metrics = calculate_multiclass_metrics(preds, actuals, classes=classes).per_class_metrics
        per_iter_per_class_metrics[it] = iter_metrics

    # Average confusion matrix
    avg_cm = (cm_sum / max(1, len(iterations_set))).tolist()

    # Convert confusion matrix to dict for serialization
    cm_dict = {}
    for true_idx, true_cls in enumerate(classes):
        cm_dict[true_cls] = {}
        for pred_idx, pred_cls in enumerate(classes):
            cm_dict[true_cls][pred_cls] = int(avg_cm[true_idx][pred_idx])

    # Per-class mean metrics across iterations
    per_class_mean: Dict[str, Dict[str, float]] = {}
    for cls in classes:
        precisions = []
        recalls = []
        f1s = []
        for it, metrics in per_iter_per_class_metrics.items():
            cls_metrics = metrics.get(cls, {})
            precisions.append(cls_metrics.get("precision", 0.0))
            recalls.append(cls_metrics.get("recall", 0.0))
            f1s.append(cls_metrics.get("f1", 0.0))

        per_class_mean[cls] = {
            "precision": float(np.mean(precisions)) if precisions else 0.0,
            "recall": float(np.mean(recalls)) if recalls else 0.0,
            "f1": float(np.mean(f1s)) if f1s else 0.0,
        }

    # Macro-averaging following the guide: mean of per-class means
    macro_precision = float(np.mean([v["precision"] for v in per_class_mean.values()])) if per_class_mean else 0.0
    macro_recall = float(np.mean([v["recall"] for v in per_class_mean.values()])) if per_class_mean else 0.0
    macro_f1 = float(np.mean([v["f1"] for v in per_class_mean.values()])) if per_class_mean else 0.0

    # Micro-averaging: compute totals from summed confusion matrix (cm_sum)
    cm_total = cm_sum.astype(int)
    tp = int(np.trace(cm_total))
    fp = int((np.sum(cm_total, axis=0) - np.diag(cm_total)).sum())
    fn = int((np.sum(cm_total, axis=1) - np.diag(cm_total)).sum())

    micro_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    micro_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

    return {
        "classes": classes,
        "average_confusion_matrix": cm_dict,
        "per_class_mean_metrics": per_class_mean,
        "macro_mean_metrics": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1,
        },
        "micro_metrics": {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1,
        },
    }


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
    classes: List[str],
    latencies: List[float] | None = None,
) -> Dict[str, Any]:
    """
    Comprehensive statistics calculation combining multiclass classification and latency metrics.
    
    Args:
        predicted: List of predicted class labels
        ground_truth: List of ground truth class labels
        classes: List of all possible class names
        latencies: Optional list of latency values in milliseconds
        
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
        responses: List[GuardrailResponse], 
        requests: List[GuardrailRequest], 
        model_name: str, 
        dataset_name: str, 
        safe_categories: list[str],
        categories: Dict[str, str],
        iterations: int = 1,
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

    classification_metrics: ClassificationMetrics = None # type: ignore
    multiclass_metrics: MulticlassMetrics = None # type: ignore
    classes = list(categories.keys())

    # Detect accumulative runs: responses include `instance_index` and multiple responses per instance
    # is_accumulative = any(getattr(r, "instance_index", None) is not None for r in responses) and len(responses) > len(requests)

    if iterations > 1:
        # Binary accumulative metrics
        # per-iteration classification metrics
        iter_classification_metrics = []
        tp_total = tn_total = fp_total = fn_total = 0

        for it in range(iterations):
            preds = []
            actuals = []
            for r in responses:
                if r.iteration == it:
                    preds.append(r.is_safe)
                    inst_idx = r.instance_index
                    if inst_idx is None:
                        inst_idx = len(actuals)
                    actuals.append(requests[inst_idx].metadata.get("category") in safe_categories)

            if not preds:
                continue

            cm = calculate_classification_metrics(preds, actuals)
            iter_classification_metrics.append(cm)
            tp_total += cm.true_positives
            tn_total += cm.true_negatives
            fp_total += cm.false_positives
            fn_total += cm.false_negatives

        # Macro (mean over iterations) for binary
        if iter_classification_metrics:
            macro_accuracy = float(np.mean([m.accuracy for m in iter_classification_metrics]))
            macro_precision = float(np.mean([m.precision for m in iter_classification_metrics]))
            macro_recall = float(np.mean([m.recall for m in iter_classification_metrics]))
            macro_f1 = float(np.mean([m.f1 for m in iter_classification_metrics]))
        else:
            macro_accuracy = macro_precision = macro_recall = macro_f1 = 0.0

        # Micro from summed counts
        micro_precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        micro_recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

        # Latency: compute both raw-response latency metrics and per-instance mean latency metrics
        latencies_all = [r.latency for r in responses if getattr(r, "latency", None) is not None]
        latency_metrics = calculate_latency_metrics(latencies_all) if latencies_all else None

        per_instance_latencies = []
        for i, req in enumerate(requests):
            inst_lat = [r.latency for r in responses if getattr(r, "instance_index", None) == i and getattr(r, "latency", None) is not None]
            if inst_lat:
                per_instance_latencies.append(float(np.mean(inst_lat)))

        latency_per_instance_metrics = calculate_latency_metrics(per_instance_latencies) if per_instance_latencies else None

        # Performance by category: average instance accuracy across iterations
        performance = {}
        for i, req in enumerate(requests):
            cat = req.metadata.get("category")
            inst_resps = [r for r in responses if getattr(r, "instance_index", None) == i]
            if not inst_resps:
                inst_accuracy = 0.0
            else:
                correct = 0
                for r in inst_resps:
                    pred_safe = bool(getattr(r, "is_safe", False))
                    actual_safe = req.metadata.get("category") in safe_categories
                    if pred_safe == actual_safe:
                        correct += 1
                inst_accuracy = correct / len(inst_resps)

            if cat not in performance:
                performance[cat] = {"total": 0, "correct": 0.0, "accuracy": 0.0}
            performance[cat]["total"] += 1
            performance[cat]["correct"] += inst_accuracy

        for cat, stats in performance.items():
            if stats["total"] > 0:
                stats["accuracy"] = stats["correct"] / stats["total"]
        
        # Multiclass accumulative metrics
        acc_multi = calculate_accumulative_multiclass_metrics(responses, requests, classes=classes)

        metrics = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model_name,
            "dataset": dataset_name,
            "total_samples": len(requests),
            "accumulative": True,
            "iterations": iterations,
            "binary_classification": {
                "macro_mean": {
                    "accuracy": macro_accuracy,
                    "precision": macro_precision,
                    "recall": macro_recall,
                    "f1": macro_f1,
                },
                "micro": {
                    "precision": micro_precision,
                    "recall": micro_recall,
                    "f1": micro_f1,
                },
                "average_confusion_matrix": {
                    "tp": tp_total/iterations,
                    "tn": tn_total/iterations,
                    "fp": fp_total/iterations,
                    "fn": fn_total/iterations
                }
            },
            "multiclass_classification": {
                "average_confusion_matrix": acc_multi["average_confusion_matrix"],
                "per_class_mean_metrics": acc_multi["per_class_mean_metrics"],
                "macro_mean_metrics": acc_multi["macro_mean_metrics"],
                "micro_metrics": acc_multi["micro_metrics"],
            },
            "latency": latency_metrics.to_dict() if latency_metrics else None,
            "latency_per_instance": latency_per_instance_metrics.to_dict() if latency_per_instance_metrics else None,
            "category_distribution": _get_category_distribution(requests=requests),
            "performance_by_category": performance,
        }
    else:
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
            "accumulative": False,
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
    
    # logger.info("Binary Classification Metrics:")
    # logger.info(f"  Accuracy: {metrics['binary_classification']['accuracy']:.4f if not is_accumulative else metrics['binary_classification']['macro_mean']['accuracy']:.4f}")
    # logger.info(f"  Precision: {metrics['binary_classification']['precision']:.4f if not is_accumulative else metrics['binary_classification']['macro_mean']['precision']:.4f}")
    # logger.info(f"  Recall: {metrics['binary_classification']['recall']:.4f if not is_accumulative else metrics['binary_classification']['macro_mean']['recall']:.4f}")
    # logger.info(f"  F1 Score: {metrics['binary_classification']['f1']:.4f if not is_accumulative else metrics['binary_classification']['macro_mean']['f1']:.4f}")
    
    # logger.info("Multiclass Classification Metrics:")
    # logger.info(f"  Confusion Matrix: {metrics['multiclass_classification']['average_confusion'] if is_accumulative else metrics['multiclass_classification']['confusion_matrix']}")
    # # logger.info(f"  Accuracy: {metrics['multiclass_classification']['accuracy']:.4f}")
    # # logger.info(f"  Macro F1: {metrics['multiclass_classification']['macro_f1']:.4f}")
    # # logger.info(f"  Weighted F1: {metrics['multiclass_classification']['weighted_f1']:.4f}")
    
    # logger.info("Per-Class Metrics:")
    # for cls, cls_metrics in metrics['multiclass_classification']['per_class_metrics' if not is_accumulative else 'per_class_mean_metrics'].items():
    #     logger.info(f"  {cls}: P={cls_metrics['precision']:.4f} R={cls_metrics['recall']:.4f} F1={cls_metrics['f1']:.4f} (n={cls_metrics['support']})")

    # if isinstance(latency_metrics, LatencyMetrics): 
    #     logger.info(f"  Mean Latency: {latency_metrics.mean_latency:.2f}ms")
    
    return metrics
