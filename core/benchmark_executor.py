"""Generic benchmark executor.

Provides a lightweight, model/dataset-agnostic orchestration utility so
experiments only need to provide the main parameters (model class/kwargs,
dataset class/kwargs and the dataset path). It handles lazy instantiation,
dataset loading, evaluation using the existing `AsyncRunner` via `Evaluator`,
and returns the raw responses so callers can compute metrics and save results
using their existing helpers.

The executor is intentionally small and composable: callers may pass in
callables for post-processing (metrics/visualization) or handle those steps
themselves using the returned data.
"""

from typing import Any, Dict, Optional, Type, Callable
from dataclasses import dataclass
from loguru import logger

from core.evaluator import Evaluator
from core.base_model import GuardrailModel
from core.base_dataset import GuardrailDataset


@dataclass
class BenchmarkResult:
	requests: list
	responses: list
	model: GuardrailModel
	dataset: GuardrailDataset


class BenchmarkExecutor:
	"""Generic benchmark runner.

	Example usage:
		executor = BenchmarkExecutor(
			model_cls=OpenAIGeneralGuardrailModel,
			model_kwargs={...},
			dataset_cls=PTBRAcademicDataset,
			dataset_kwargs={...},
			max_concurrency=4,
		)

		result = await executor.run(dataset_path)

	The caller can then compute metrics using `result.requests` and
	`result.responses`.
	"""

	def __init__(
		self,
		model_cls: Optional[Type[GuardrailModel]] = None,
		model_instance: Optional[GuardrailModel] = None,
		model_kwargs: Optional[Dict[str, Any]] = None,
		dataset_cls: Optional[Type[GuardrailDataset]] = None,
		dataset_instance: Optional[GuardrailDataset] = None,
		dataset_kwargs: Optional[Dict[str, Any]] = None,
		max_concurrency: int = 5,
		batch_size: Optional[int] = None,
	) -> None:
		self.model_cls = model_cls
		self.model_instance = model_instance
		self.model_kwargs = model_kwargs or {}

		self.dataset_cls = dataset_cls
		self.dataset_instance = dataset_instance
		self.dataset_kwargs = dataset_kwargs or {}

		self.max_concurrency = max_concurrency
		self.batch_size = batch_size or 1

		# internal components
		self.evaluator = Evaluator(max_concurrency=max_concurrency)

	async def run(self, dataset_path: str, post_process: Optional[Callable] = None) -> BenchmarkResult:
		"""Run the benchmark: load dataset, evaluate model, return results.

		Args:
			dataset_path: Path to dataset file
			post_process: Optional callable(requests, responses) for additional processing

		Returns:
			BenchmarkResult with requests/responses and instantiated model/dataset
		"""
		# Instantiate dataset
		if self.dataset_instance is not None:
			dataset = self.dataset_instance
		elif self.dataset_cls is not None:
			dataset = self.dataset_cls(**self.dataset_kwargs)
		else:
			raise ValueError("Either dataset_cls or dataset_instance must be provided")

		logger.info(f"Loading dataset from {dataset_path}")
		dataset.load(dataset_path)
		requests = dataset.to_requests()

		# Log category distribution
		log_categories = {}
		for request in requests:
			cat = request.metadata.get("category")
			log_categories[cat] = log_categories.get(cat, 0) + 1

		logger.info("Category distribution:")
		for cat, count in sorted(log_categories.items()):
			logger.info(f"  {cat}: {count}")

		if not requests:
			raise ValueError(f"Dataset at {dataset_path} contains no items")

		# Instantiate model
		if self.model_instance is not None:
			model = self.model_instance
		elif self.model_cls is not None:
			model = self.model_cls(**self.model_kwargs)
		else:
			raise ValueError("Either model_cls or model_instance must be provided")

		logger.info(f"Evaluating model {model.model_name} on {len(requests)} samples")
		logger.info(f"Using max_concurrency={self.max_concurrency}, batch_size={self.batch_size}")
		logger.info(f"Model safe categories: {getattr(model, 'safe_categories', 'N/A')}")
		logger.info(f"System prompt: {getattr(model, 'system_prompt', 'N/A')}")
		logger.info(f"Model categories: {getattr(model, 'categories', 'N/A')}")

		responses = await self.evaluator.runner.run_batch(model, requests, batch_size=self.batch_size)

		# optional post-processing hook
		if post_process is not None:
			try:
				post_process(requests, responses)
			except Exception as e:
				logger.warning(f"post_process hook raised an exception: {e}")

		return BenchmarkResult(requests=requests, responses=responses, model=model, dataset=dataset)


