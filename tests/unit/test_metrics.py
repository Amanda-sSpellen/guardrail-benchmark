# Metric Math: Test utils/metrics.py by passing in "fake" lists of results and checking if the F1-score calculation is mathematically correct.

"""
test_metrics.py: Unit tests for metrics calculation functions.

This module tests the mathematical correctness of classification metrics
(Accuracy, Precision, Recall, F1) and latency statistics calculations.
Uses fake/synthetic data to verify metric computation accuracy.
"""

import pytest
from utils.metrics import (
    calculate_classification_metrics,
    calculate_latency_metrics,
    calculate_stats,
    # ClassificationMetrics,
    # LatencyMetrics,
)


@pytest.mark.unit
class TestCalculateClassificationMetrics:
    """Tests for classification metrics calculation."""
    
    def test_perfect_predictions(self):
        """Test metrics when all predictions are correct."""
        predicted = [True, True, False, False, True, False]
        ground_truth = [True, True, False, False, True, False]
        
        metrics = calculate_classification_metrics(predicted, ground_truth)
        
        assert metrics.accuracy == 1.0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1 == 1.0
        assert metrics.true_positives == 3
        assert metrics.true_negatives == 3
        assert metrics.false_positives == 0
        assert metrics.false_negatives == 0
    
    def test_all_false_predictions(self):
        """Test metrics when all predictions are False."""
        predicted = [False, False, False, False]
        ground_truth = [True, True, False, False]
        
        metrics = calculate_classification_metrics(predicted, ground_truth)
        
        # TP=0, TN=2, FP=0, FN=2
        assert metrics.true_positives == 0
        assert metrics.true_negatives == 2
        assert metrics.false_positives == 0
        assert metrics.false_negatives == 2
        assert metrics.accuracy == 0.5  # (0 + 2) / 4
        assert metrics.precision == 0.0  # 0 / (0 + 0) = undefined -> 0
        assert metrics.recall == 0.0  # 0 / (0 + 2) = 0
        assert metrics.f1 == 0.0
    
    def test_all_true_predictions(self):
        """Test metrics when all predictions are True."""
        predicted = [True, True, True, True]
        ground_truth = [True, True, False, False]
        
        metrics = calculate_classification_metrics(predicted, ground_truth)
        
        # TP=2, TN=0, FP=2, FN=0
        assert metrics.true_positives == 2
        assert metrics.true_negatives == 0
        assert metrics.false_positives == 2
        assert metrics.false_negatives == 0
        assert metrics.accuracy == 0.5  # (2 + 0) / 4
        assert metrics.precision == 0.5  # 2 / (2 + 2) = 0.5
        assert metrics.recall == 1.0  # 2 / (2 + 0) = 1.0
        assert metrics.f1 == pytest.approx(2 * (0.5 * 1.0) / (0.5 + 1.0))  # 2/3
    
    def test_f1_score_calculation(self):
        """Test F1-score calculation with known values.
        
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        """
        # Test case: TP=10, FP=10, FN=5 (perfect precision=0.67, recall=0.67)
        predicted = [True] * 15 + [False] * 5 + [True] * 5
        ground_truth = [True] * 10 + [False] * 5 + [True] * 5 + [False] * 5
        
        metrics = calculate_classification_metrics(predicted, ground_truth)
        
        # TP=10, FP=10, FN=5
        assert metrics.true_positives == 10
        assert metrics.false_positives == 10
        assert metrics.false_negatives == 5
        
        precision = 10 / (10 + 10)  # 0.5
        recall = 10 / (10 + 5)  # 0.667
        expected_f1 = 2 * (precision * recall) / (precision + recall)
        
        assert metrics.precision == pytest.approx(precision)
        assert metrics.recall == pytest.approx(recall)
        assert metrics.f1 == pytest.approx(expected_f1)
    
    def test_precision_recall_tradeoff(self):
        """Test F1-score with precision-recall tradeoff.
        
        Precision: TP / (TP + FP)
        Recall: TP / (TP + FN)
        """
        # High precision (few false positives), low recall (many false negatives)
        predicted = [True, False, False, False, False]
        ground_truth = [True, True, True, True, True]
        
        metrics = calculate_classification_metrics(predicted, ground_truth)
        
        # TP=1, FP=0, FN=4
        precision = 1 / (1 + 0)  # 1.0
        recall = 1 / (1 + 4)  # 0.2
        expected_f1 = 2 * (precision * recall) / (precision + recall)
        
        assert metrics.precision == pytest.approx(precision)
        assert metrics.recall == pytest.approx(recall)
        assert metrics.f1 == pytest.approx(expected_f1)
        assert metrics.f1 < metrics.precision  # F1 is harmonic mean
        assert not(metrics.f1 < metrics.recall)  # F1 is higher than recall here
    
    def test_balanced_metrics(self):
        """Test with balanced precision and recall."""
        # TP=50, FP=50, FN=50
        predicted = [True] * 100 + [False] * 50
        ground_truth = [True] * 50 + [False] * 50 + [True] * 50
        
        metrics = calculate_classification_metrics(predicted, ground_truth)
        
        # TP=50, FP=50, FN=50
        assert metrics.true_positives == 50
        assert metrics.false_positives == 50
        assert metrics.false_negatives == 50
        assert metrics.true_negatives == 0
        
        # precision = 50 / (50 + 50)  # 0.5
        # recall = 50 / (50 + 50)  # 0.5
        
        assert metrics.precision == pytest.approx(0.5)
        assert metrics.recall == pytest.approx(0.5)
        assert metrics.f1 == pytest.approx(0.5)  # F1 = precision = recall
    
    def test_zero_division_handling_precision(self):
        """Test F1 calculation when TP + FP = 0 (no positive predictions)."""
        predicted = [False, False, False]
        ground_truth = [False, False, False]
        
        metrics = calculate_classification_metrics(predicted, ground_truth)
        
        # TP=0, FP=0, so precision should be 0 (not undefined)
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0  # 0 / (0 + 0) = 1 (all negatives correctly predicted)
        assert metrics.f1 == 0.0  # 0 * 1 / (0 + 1) = 0
    
    def test_zero_division_handling_recall(self):
        """Test F1 calculation when TP + FN = 0 (no positive ground truth)."""
        predicted = [False, False, False]
        ground_truth = [False, False, False]
        
        metrics = calculate_classification_metrics(predicted, ground_truth)
        
        # This is a edge case where there are no positive samples
        assert metrics.recall == 0.0  # 0 / 0 is undefined, but we return 1
    
    def test_input_validation_length_mismatch(self):
        """Test that ValueError is raised for mismatched input lengths."""
        predicted = [True, False]
        ground_truth = [True, False, True]
        
        with pytest.raises(ValueError, match="same length"):
            calculate_classification_metrics(predicted, ground_truth)
    
    def test_input_validation_empty_lists(self):
        """Test that ValueError is raised for empty input lists."""
        with pytest.raises(ValueError, match="cannot be empty"):
            calculate_classification_metrics([], [])
    
    def test_single_sample(self):
        """Test metrics with a single sample."""
        # True positive
        metrics_tp = calculate_classification_metrics([True], [True])
        assert metrics_tp.accuracy == 1.0
        assert metrics_tp.precision == 1.0
        assert metrics_tp.recall == 1.0
        assert metrics_tp.f1 == 1.0
        
        # False positive
        metrics_fp = calculate_classification_metrics([True], [False])
        assert metrics_fp.accuracy == 0.0
        assert metrics_fp.precision == 0.0  # 0 / (0 + 1)
        assert metrics_fp.recall == pytest.approx(0.0)  # No ground truth positives
        assert metrics_fp.f1 == 0.0
    
    def test_metrics_to_dict(self):
        """Test conversion of metrics to dictionary."""
        metrics = calculate_classification_metrics([True, False], [True, False])
        
        metrics_dict = metrics.to_dict()
        
        assert isinstance(metrics_dict, dict)
        assert "accuracy" in metrics_dict
        assert "precision" in metrics_dict
        assert "recall" in metrics_dict
        assert "f1" in metrics_dict
        assert "tp" in metrics_dict
        assert "tn" in metrics_dict
        assert "fp" in metrics_dict
        assert "fn" in metrics_dict


@pytest.mark.unit
class TestCalculateLatencyMetrics:
    """Tests for latency metrics calculation."""
    
    def test_single_latency_value(self):
        """Test latency metrics with a single value."""
        latencies = [100.0]
        
        metrics = calculate_latency_metrics(latencies)
        
        assert metrics.mean_latency == 100.0
        assert metrics.median_latency == 100.0
        assert metrics.std_latency == 0.0
        assert metrics.min_latency == 100.0
        assert metrics.max_latency == 100.0
    
    def test_uniform_latency_values(self):
        """Test latency metrics with uniform values."""
        latencies = [50.0, 50.0, 50.0, 50.0]
        
        metrics = calculate_latency_metrics(latencies)
        
        assert metrics.mean_latency == 50.0
        assert metrics.median_latency == 50.0
        assert metrics.std_latency == 0.0
        assert metrics.min_latency == 50.0
        assert metrics.max_latency == 50.0
    
    def test_varied_latency_values(self):
        """Test latency metrics with varied values."""
        latencies = [10.0, 20.0, 30.0, 40.0, 50.0]
        
        metrics = calculate_latency_metrics(latencies)
        
        assert metrics.mean_latency == pytest.approx(30.0)
        assert metrics.median_latency == pytest.approx(30.0)
        assert metrics.min_latency == 10.0
        assert metrics.max_latency == 50.0
        assert metrics.std_latency > 0
    
    def test_standard_deviation_calculation(self):
        """Test that standard deviation is calculated correctly."""
        latencies = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        metrics = calculate_latency_metrics(latencies)
        
        # Expected mean = 3.0
        # Expected std = sqrt(((1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2) / 5)
        # = sqrt((4 + 1 + 0 + 1 + 4) / 5) = sqrt(2) ≈ 1.414
        expected_std = pytest.approx(1.414, abs=0.01)
        assert metrics.std_latency == expected_std
    
    def test_latency_to_dict(self):
        """Test conversion of latency metrics to dictionary."""
        latencies = [100.0, 200.0, 300.0]
        
        metrics = calculate_latency_metrics(latencies)
        latency_dict = metrics.to_dict()
        
        assert isinstance(latency_dict, dict)
        assert "mean" in latency_dict
        assert "median" in latency_dict
        assert "std" in latency_dict
        assert "min" in latency_dict
        assert "max" in latency_dict
    
    def test_empty_latencies_validation(self):
        """Test that ValueError is raised for empty latencies list."""
        with pytest.raises(ValueError, match="cannot be empty"):
            calculate_latency_metrics([])


@pytest.mark.unit
class TestCalculateStats:
    """Tests for comprehensive statistics calculation."""
    
    def test_stats_with_classification_only(self):
        """Test stats calculation with only classification data."""
        predicted = [True, True, False]
        ground_truth = [True, False, False]
        
        stats = calculate_stats(predicted, ground_truth)
        
        assert "classification" in stats
        assert "latency" not in stats
        assert isinstance(stats["classification"], dict)
        assert "f1" in stats["classification"]
    
    def test_stats_with_classification_and_latency(self):
        """Test stats calculation with both classification and latency data."""
        predicted = [True, True, False]
        ground_truth = [True, False, False]
        latencies = [100.0, 150.0, 120.0]
        
        stats = calculate_stats(predicted, ground_truth, latencies)
        
        assert "classification" in stats
        assert "latency" in stats
        assert isinstance(stats["classification"], dict)
        assert isinstance(stats["latency"], dict)
    
    def test_stats_with_empty_latencies(self):
        """Test that empty latencies list is handled correctly."""
        predicted = [True, False]
        ground_truth = [True, False]
        
        # Empty list or None should not add latency stats
        stats = calculate_stats(predicted, ground_truth, [])
        assert "latency" not in stats
        
        stats_none = calculate_stats(predicted, ground_truth, None)
        assert "latency" not in stats_none


@pytest.mark.unit
class TestF1ScoreEdgeCases:
    """Comprehensive tests for F1-score edge cases."""
    
    def test_f1_with_high_false_positives(self):
        """Test F1-score when false positives dominate."""
        predicted = [True] * 100
        ground_truth = [True] * 10 + [False] * 90
        
        metrics = calculate_classification_metrics(predicted, ground_truth)
        
        # TP=10, FP=90
        precision = 10 / 100  # 0.1
        recall = 10 / 10  # 1.0
        expected_f1 = 2 * (0.1 * 1.0) / (0.1 + 1.0)  # ≈ 0.182
        
        assert metrics.precision == pytest.approx(precision)
        assert metrics.recall == pytest.approx(recall)
        assert metrics.f1 == pytest.approx(expected_f1, rel=0.01)
    
    def test_f1_with_high_false_negatives(self):
        """Test F1-score when false negatives dominate."""
        predicted = [False] * 100
        ground_truth = [True] * 90 + [False] * 10
        
        metrics = calculate_classification_metrics(predicted, ground_truth)
        
        # # TP=0, FN=90, TN=10
        # precision = 0.0  # 0 / 0
        # recall = 0.0  # 0 / 90
        # expected_f1 = 0.0
        
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1 == 0.0
    
    def test_f1_comparison_different_scenarios(self):
        """Compare F1-scores across different confusion matrices."""
        # Scenario 1: Balanced
        pred1 = [True, True, False, False]
        truth1 = [True, False, True, False]
        metrics1 = calculate_classification_metrics(pred1, truth1)
        
        # Scenario 2: High recall, low precision
        pred2 = [True] * 10
        truth2 = [True] * 5 + [False] * 5
        metrics2 = calculate_classification_metrics(pred2, truth2)
        
        # High recall should give higher F1 than the balanced case with low precision
        assert metrics2.recall > metrics1.recall
        # But precision is lower
        assert metrics2.precision == metrics1.precision
