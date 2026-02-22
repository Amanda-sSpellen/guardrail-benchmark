"""
evaluator.py: High-level benchmark orchestration (the "Judge").

This module provides the Evaluator class which coordinates benchmark runs,
comparing multiple guardrail models against datasets and generating reports.
"""

from loguru import logger
from typing import List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone

from core.schema import GuardrailResponse
from core.base_model import GuardrailModel
from core.base_dataset import GuardrailDataset
from core.engine import AsyncRunner


@dataclass
class BenchmarkReport:
    """Container for benchmark results and metrics."""
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    models_evaluated: List[str] = field(default_factory=list)
    dataset_name: str = ""
    total_samples: int = 0
    results: Dict[str, List[GuardrailResponse]] = field(default_factory=dict)
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "models": self.models_evaluated,
            "dataset": self.dataset_name,
            "total_samples": self.total_samples,
            "metrics": self.metrics,
        }


class Evaluator:
    """
    High-level benchmark orchestrator for comparing guardrail models.
    
    The Evaluator coordinates the comparison of multiple guardrail models against
    standardized datasets, collecting responses, computing metrics, and generating reports.
    """
    
    def __init__(self, max_concurrency: int = 5):
        """
        Initialize the evaluator.
        
        Args:
            max_concurrency: Maximum concurrent requests per model
        """
        self.runner = AsyncRunner(max_concurrency=max_concurrency)
        self.report = None
    
    async def compare(
        self,
        models: List[GuardrailModel],
        dataset: GuardrailDataset,
        dataset_path: str
    ) -> BenchmarkReport:
        """
        Run a comprehensive benchmark comparing multiple models on a dataset.
        
        This method:
        1. Loads and standardizes the dataset
        2. Evaluates all models against all dataset samples
        3. Collects responses with timing information
        4. Generates a detailed benchmark report
        
        Args:
            models: List of GuardrailModel instances to compare
            dataset: GuardrailDataset instance for loading benchmark data
            dataset_path: Path or identifier for the dataset
            
        Returns:
            BenchmarkReport containing results, metrics, and evaluation details
            
        Raises:
            ValueError: If models list is empty or dataset path is invalid
            Exception: If any model evaluation fails
        """
        if not models:
            raise ValueError("At least one model must be provided for comparison")
        
        # Load and standardize dataset
        logger.info(f"Loading dataset from {dataset_path}")
        requests = dataset.load_and_standardize(dataset_path)
        
        if not requests:
            raise ValueError(f"Dataset at {dataset_path} contains no items")
        
        # Initialize report
        self.report = BenchmarkReport(
            models_evaluated=[model.model_name for model in models],
            dataset_name=dataset.name,
            total_samples=len(requests),
        )
        
        # Run each model against the dataset
        for model in models:
            logger.info(f"Evaluating model: {model.model_name}")
            responses = await self.runner.run_batch(model, requests)
            self.report.results[model.model_name] = responses
        
        return self.report
    
    def get_report(self) -> BenchmarkReport:
        """
        Get the most recent benchmark report.
        
        Returns:
            BenchmarkReport from the last compare() call
            
        Raises:
            RuntimeError: If no benchmark has been run yet
        """
        if self.report is None:
            raise RuntimeError("No benchmark has been run yet. Call compare() first.")
        return self.report
