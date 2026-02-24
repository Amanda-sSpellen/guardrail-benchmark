from abc import ABC, abstractmethod
from typing import Optional, List

from core.schema import GuardrailRequest, GuardrailResponse


class BatchEvaluator(ABC):
    """
    BatchEvaluator defines the interface for evaluating batches of 
    requests through a GuardrailModel.
    This allows for flexible implementations of batch processing and 
    concurrency control, which can be used by the AsyncRunner.
    """

    @abstractmethod
    async def evaluate_batch(
        self,
        requests: List[GuardrailRequest],
        batch_size: Optional[int] = 1,
        **kwargs,
    ) -> List[GuardrailResponse]:
        """
        Evaluate a batch of requests through the engine with concurrency and batching support.
        
        Args:
            model: The GuardrailModel to evaluate against.
            requests: List of GuardrailRequest objects to evaluate.
            max_concurrency: Optional override for concurrency limit.
                            If provided, temporarily updates the semaphore.
            batch_size: Optional override for batch size (default: 1).
        
        Returns:
            List of GuardrailResponse objects in the same order as input requests.
        """
        pass
