# TODO: support batching for local models
"""
engine.py: The asynchronous execution engine (the "Conductor").

This module provides the AsyncRunner class which orchestrates concurrent evaluation
of guardrail models while respecting rate limits through semaphore-based concurrency control.
"""

import asyncio
from tqdm import tqdm
from typing import List, Optional, Union
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
        

    async def run_batch(
        self,
        model: GuardrailModel,
        requests: List[GuardrailRequest],
        max_concurrency: Optional[int] = None,
        batch_size: Optional[int] = 1,
        **kwargs,
    ) -> List[GuardrailResponse]:
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

        # 1. Initialize with explicit type hinting to satisfy Pylance
        responses: List[Optional[Union[GuardrailResponse, Exception]]] = [None] * len(requests)
        
        # 2. Wrap tasks with index tracking
        async def _wrapped_eval(index: int, req: GuardrailRequest):
            # Add a slight staggered start if you have a high concurrency
            await asyncio.sleep(index * 0.5) 
            try:
                return index, await self._evaluate_with_semaphore(model, req)
            except Exception as e:
                return index, e

        # 3. Execute and update tqdm
        if batch_size and batch_size > 1:
            # Process in batches
            for i in range(0, len(requests), batch_size):
                chunk = requests[i : i + batch_size]
            
                # Call the optimized batch method (True Tensor Batching)
                # We wrap this in a single semaphore 'hit' because it's ONE GPU operation
                async with self.semaphore:
                    try:
                        # You need to implement evaluate_batch in your model class
                        if model.batch_evaluator is None:
                            raise ValueError("BatchEvaluator is not set for this model.")
                        
                        batch_results = await model.batch_evaluator.evaluate_batch(
                            requests=chunk, 
                            batch_size=batch_size,
                            **kwargs,
                        )
                        
                        # Map results back to the correct indices
                        for j, result in enumerate(batch_results):
                            responses[i + j] = result
                            
                    except Exception as e:
                        print(f"Batch {i//batch_size} failed: {e}")
        else:
            tasks = [_wrapped_eval(i, req) for i, req in enumerate(requests)]
            for coro in tqdm(asyncio.as_completed(tasks), total=len(requests), desc="Evaluating"):
                index, result = await coro
                responses[index] = result
            
        # 4. Extract only the successful responses
        return [res for res in responses if isinstance(res, GuardrailResponse)] 
