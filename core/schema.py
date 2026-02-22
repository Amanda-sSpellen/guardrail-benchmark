"""
schema.py: Pydantic models to standardize data flow across guardrail providers.

This module defines the core data models used for input/output standardization,
ensuring consistency regardless of the underlying guardrail provider.
"""

from pydantic import BaseModel, Field
from typing import Optional, Any
from datetime import datetime, timezone


class GuardrailRequest(BaseModel):
    """
    Standardized input request for guardrail evaluation.
    
    Attributes:
        text: The input text to evaluate for safety/policy violations
        metadata: Additional context (e.g., user_id, source, language)
    """
    text: str = Field(..., description="Text to evaluate")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional context")


class GuardrailResponse(BaseModel):
    """
    Standardized output response from a guardrail model.
    
    Attributes:
        is_safe: Boolean indicating whether the content passed safety checks
        score: Numerical safety score (typically 0-1 or 0-100 depending on model)
        category: Optional category of violation detected (e.g., "toxic", "spam")
        latency: Response time in milliseconds
        model_name: Name of the model that generated this response
        timestamp: When the response was generated
    """
    is_safe: bool = Field(..., description="Whether content passed safety checks")
    score: float = Field(..., description="Safety score from the model")
    category: Optional[str] = Field(default=None, description="Violation category if unsafe")
    latency: float = Field(..., description="Response latency in milliseconds")
    model_name: str = Field(..., description="Name of the guardrail model")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Response timestamp")
