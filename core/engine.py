"""
engine.py: The asynchronous execution engine (the "Conductor").

This module provides the AsyncRunner class which orchestrates concurrent evaluation
of guardrail models while respecting rate limits through semaphore-based concurrency control.
"""

import asyncio
from loguru import logger
from typing import List, Optional
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
        max_concurrency: Optional[int] = None
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
        
        Returns:
            List of GuardrailResponse objects in the same order as input requests
            
        Raises:
            Exception: Re-raises any exceptions from individual model evaluations
        """
        if max_concurrency is not None:
            self.semaphore = asyncio.Semaphore(max_concurrency)
            self.max_concurrency = max_concurrency
        
        # Create a task for each request
        tasks = [
            self._evaluate_with_semaphore(model, request)
            for request in requests
        ]

        logger.info(f"Running batch of {len(requests)} requests with max concurrency {self.max_concurrency}")
        
        # Execute all tasks concurrently with semaphore control
        responses = await asyncio.gather(*tasks)
        
        return responses
