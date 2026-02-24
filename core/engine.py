"""
engine.py: The asynchronous execution engine (the "Conductor").

This module provides the AsyncRunner class which orchestrates concurrent evaluation
of guardrail models while respecting rate limits through semaphore-based concurrency control.
"""

import asyncio
from tqdm import tqdm
from typing import List, Optional, Union, Dict, Any
from core.schema import GuardrailRequest, GuardrailResponse
from core.base_model import GuardrailModel

class AsyncRunner:
    """
    Asynchronous batch execution engine for guardrail model evaluation.
    
    This class manages concurrent API calls using asyncio.Semaphore to prevent
    hitting provider rate limits. It handles batching, concurrency control, and
    tracks execution metrics.
    """
    
    def __init__(self, max_concurrency: int = 5):
        """
        Initialize the async runner.
        
        Args:
            max_concurrency: Maximum number of concurrent requests (default: 5)
        """
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency)
    
    async def _evaluate_with_semaphore(
        self,
        model: GuardrailModel,
        request: GuardrailRequest
    ) -> GuardrailResponse:
        """
        Wrapper to evaluate a single request with semaphore-based rate limiting.
        
        Args:
            model: The guardrail model to evaluate with
            request: The request to evaluate
            
        Returns:
            GuardrailResponse from the model
        """
        async with self.semaphore:
            return await model.evaluate(request)
    
    def expand_requests(self, requests: List[GuardrailRequest], iterations: int = 1) -> Dict[str, Any]:
        # Expand requests for accumulative executions (repeat each request `iterations` times)
        expanded_requests: List[GuardrailRequest] = []
        orig_indices: List[int] = []
        iteration_nums: List[int] = []
        for i, req in enumerate(requests):
            for it in range(max(1, iterations)):
                expanded_requests.append(req)
                orig_indices.append(i)
                iteration_nums.append(it)
        return {
            "requests": expanded_requests,
            "orig_indices": orig_indices,
            "iteration_nums": iteration_nums
        }
    

    async def run_batch(
        self,
        model: GuardrailModel,
        requests: List[GuardrailRequest],
        max_concurrency: Optional[int] = None,
        batch_size: Optional[int] = 1,
        iterations: int = 1,
        **kwargs,
    ) -> Dict[str, List[GuardrailRequest]|List[GuardrailResponse]]:
        """
        Run a batch of requests through a model with concurrency control.
        
        Uses asyncio.Semaphore to limit concurrent API calls, preventing rate limit
        violations while maximizing throughput for large-scale benchmarks.
        
        Args:
            model: The guardrail model to evaluate with
            requests: List of GuardrailRequest objects to evaluate
            max_concurrency: Optional override for concurrency limit.
                           If provided, temporarily updates the semaphore.
            batch_size: Optional override for batch size (default: 1).
        

        Returns:
            List of GuardrailResponse objects in the same order as input requests
            
        Raises:
            Exception: Re-raises any exceptions from individual model evaluations
        """

        if max_concurrency is not None:
            self.semaphore = asyncio.Semaphore(max_concurrency)

        expanded_requests_dict = self.expand_requests(requests=requests, iterations=iterations)
        expanded_requests = expanded_requests_dict["requests"]
        orig_indices = expanded_requests_dict["orig_indices"]
        iteration_nums = expanded_requests_dict["iteration_nums"]

        # Initialize response slots for expanded requests
        responses: List[Optional[Union[GuardrailResponse, Exception]]] = [None] * len(expanded_requests)

        # Wrapped evaluator that returns the expanded index
        async def _wrapped_eval(exp_index: int, req: GuardrailRequest):
            # slight stagger based on concurrency to avoid bursts
            await asyncio.sleep((exp_index % max(1, self.max_concurrency)) * 0.05)
            try:
                return exp_index, await self._evaluate_with_semaphore(model, req)
            except Exception as e:
                return exp_index, e

        # Execute using batching or individual calls over expanded requests
        if batch_size and batch_size > 1:
            for i in range(0, len(expanded_requests), batch_size):
                chunk = expanded_requests[i : i + batch_size]

                async with self.semaphore:
                    try:
                        if model.batch_evaluator is None:
                            raise ValueError("BatchEvaluator is not set for this model.")

                        batch_results = await model.batch_evaluator.evaluate_batch(
                            requests=chunk,
                            batch_size=batch_size,
                            **kwargs,
                        )

                        for j, result in enumerate(batch_results):
                            responses[i + j] = result

                    except Exception as e:
                        print(f"Batch {i//batch_size} failed: {e}")
        else:
            tasks = [_wrapped_eval(i, req) for i, req in enumerate(expanded_requests)]
            for coro in tqdm(asyncio.as_completed(tasks), total=len(expanded_requests), desc="Evaluating"):
                exp_index, result = await coro
                responses[exp_index] = result

        # Enrich GuardrailResponse objects with `instance_index` and `iteration` metadata
        for idx, res in enumerate(responses):
            if isinstance(res, GuardrailResponse):
                try:
                    # attach indices so callers can identify the original instance and iteration
                    res.instance_index = orig_indices[idx]
                    res.iteration = iteration_nums[idx]
                except Exception:
                    pass

        return {
            "requests": expanded_requests,
            "responses": [res for res in responses if isinstance(res, GuardrailResponse)]
        }
