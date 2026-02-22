"""
metrics.py: Performance metrics calculation for guardrail model evaluation.

Computes standard classification metrics (Accuracy, Precision, Recall, F1)
and latency statistics from benchmark responses.
"""

from typing import List, Dict, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class ClassificationMetrics:
    """Container for classification performance metrics."""
    
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
    Calculate classification metrics (Accuracy, Precision, Recall, F1).
    
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
    tp = np.sum(pred & truth)
    tn = np.sum(~pred & ~truth)
    fp = np.sum(pred & ~truth)
    fn = np.sum(~pred & truth)
    
    # TODO: Handle cases for negative samples
    # Calculate metrics
    accuracy = (tp + tn) / len(predicted)
    
    # Precision: TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Recall: TP / (TP + FN)
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
    Comprehensive statistics calculation combining classification and latency metrics.
    
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
