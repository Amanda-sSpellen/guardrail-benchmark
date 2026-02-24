"""
base_model.py: Abstract base class for all guardrail model implementations.

This module defines the interface that all guardrail provider wrappers must implement,
ensuring consistency across different safety API providers (OpenAI, Gemini, LlamaGuard, etc.).
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
from core.schema import GuardrailRequest, GuardrailResponse
from core.batch_evaluator import BatchEvaluator

# Multiclass classification categories with descriptions
CATEGORIES = {
    "harassment": "Content that bullies, intimidates, or demeans individuals or groups",
    "hate_speech": "Content that expresses prejudice, discrimination, or promotes hatred toward protected groups",
    "sexual_content": "Sexually explicit material, including sexual abuse or exploitation content",
    "violence": "Content glorifying or promoting violence, graphic violence, or dangerous content",
    "misinformation": "false or misleading information that could cause harm",
    "none": "Content that does not violate any safety policies",
}

class GuardrailModel(ABC):
    """
    Abstract base class for guardrail model implementations.
    
    All guardrail models must inherit from this class and implement the evaluate() method.
    Subclasses should implement provider-specific logic while maintaining the standard interface.
    """
    
    def __init__(self, model_name: str, batch_evaluator: Optional[BatchEvaluator] = None, **kwargs):
        """
        Initialize the guardrail model.
        
        Args:
            model_name: Display name for the model
            batch_evaluator: Optional BatchEvaluator instance for batch processing
            **kwargs: Provider-specific configuration options
                - system_prompt: Optional system prompt to guide the model's evaluation (default: generic safety instruction)
                - categories: Optional dictionary of category names and descriptions (default: self.CATEGORIES)
                - temperature: Optional temperature setting for the model (default: provider-specific)
                - safe_categories: Optional labels for the "safe" categories (default: ["none"])
        """
        self.model_name = model_name
        self.batch_evaluator = batch_evaluator
        self.config = kwargs
        self.system_prompt = kwargs.get("system_prompt", "Evaluate the following text for safety violations.")
        self.categories = kwargs.get("categories", CATEGORIES)
        self.temperature = kwargs.get("temperature")
        self.safe_categories = kwargs.get("safe_categories", ["none"])  # Default safe category label
        self._client = None
    
    @property
    def client(self):
        """Lazy-load the SDK client on first access."""
        if self._client is None:
            self._client = self._load_client()
        return self._client
    
    @abstractmethod
    def _load_client(self) -> Any:
        """
        Internal lazy loader for SDKs.
        
        This method should be implemented by subclasses to initialize and return
        the provider-specific SDK client. This allows for lazy initialization to avoid
        unnecessary API connections until the model is actually used.
        
        Returns:
            The initialized SDK client for the specific provider
        """
        pass
    
    @abstractmethod
    async def evaluate(self, request: GuardrailRequest) -> GuardrailResponse:
        """
        Evaluate a text input through the guardrail model.
        
        This method must be implemented by subclasses. It should:
        1. Call the provider's API with the request text
        2. Parse the response into a standardized GuardrailResponse
        3. Include latency measurement
        4. Return the standardized response
        
        Args:
            request: GuardrailRequest containing text and metadata to evaluate
            
        Returns:
            GuardrailResponse with standardized safety judgment, score, and latency
            
        Raises:
            APIError: If the provider API call fails
            ValidationError: If response cannot be parsed into standard schema
        """
        pass
