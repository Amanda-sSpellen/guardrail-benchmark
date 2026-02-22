# TODO: update to dynamic category classification 
"""
openai_model.py: OpenAI moderations API wrapper for guardrail evaluation.

This module implements the GuardrailModel interface for OpenAI's moderation endpoint.
Uses openai.AsyncOpenAI for async API calls with comprehensive error handling.
"""

import asyncio
import time
from typing import Any
from datetime import datetime, timezone

try:
    from openai import AsyncOpenAI, APIError, APIConnectionError, RateLimitError
except ImportError as e:
    raise ImportError(
        "openai package is required for OpenAIGuardrailModel. "
        "Install it with: pip install openai"
    ) from e

from core.base_model import GuardrailModel
from core.schema import GuardrailRequest, GuardrailResponse


class OpenAIModerationGuardrailModel(GuardrailModel):
    """
    OpenAI Moderations API wrapper for guardrail evaluation.
    
    Uses OpenAI's moderation endpoint to evaluate text for policy violations.
    Supports async evaluation with automatic error handling and retry logic.
    """
    
    # OpenAI's moderation categories
    VIOLATION_CATEGORIES = {
        "harassment": "harassment",
        "harassment/threatening": "harassment",
        "hate": "hate_speech",
        "hate/defamatory": "hate_speech",
        "self-harm": "self_harm",
        "self-harm/intent": "self_harm",
        "self-harm/instructions": "self_harm",
        "sexual": "sexual_content",
        "sexual/minors": "sexual_content",
        "violence": "violence",
        "violence/graphic": "violence",
    }
    
    def _load_client(self) -> AsyncOpenAI:
        """
        Initialize and return AsyncOpenAI client.
        
        Uses api_key from config if provided, otherwise uses environment variable.
        
        Returns:
            AsyncOpenAI: Initialized async OpenAI client
            
        Raises:
            ValueError: If API key is not provided and not in environment
        """
        api_key = self.config.get("api_key")
        if not api_key:
            import os
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenAI API key not provided. Pass via config or set OPENAI_API_KEY environment variable."
                )
        
        return AsyncOpenAI(api_key=api_key)
    
    async def evaluate(self, request: GuardrailRequest) -> GuardrailResponse:
        """
        Evaluate text using OpenAI's moderation endpoint.
        
        Args:
            request: GuardrailRequest containing text and metadata
            
        Returns:
            GuardrailResponse with standardized safety judgment
            
        Raises:
            APIConnectionError: If connection to OpenAI fails
            RateLimitError: If rate limited by OpenAI
            APIError: For other API-related errors
            ValueError: If response format is invalid
        """
        start_time = time.time()
        
        try:
            # Call OpenAI moderation endpoint
            response = await self.client.moderations.create(
                model="text-moderation-latest",
                input=request.text
            )
            
            latency = (time.time() - start_time) * 1000  # Convert to ms
            
            # Parse the moderation response
            result = response.results[0]
            
            # Determine if content is safe
            is_safe = not result.flagged
            
            # Get the highest flagged category
            category = None
            if result.flagged:
                category_scores = result.category_scores.__dict__
                flagged_category = max(
                    category_scores.items(),
                    key=lambda x: x[1]
                )
                category = self.VIOLATION_CATEGORIES.get(
                    flagged_category[0],
                    flagged_category[0]
                )
            
            # Calculate safety score (0-1)
            # Use the maximum score from all categories as the safety indicator
            all_scores = result.category_scores.__dict__.values()
            score = max(all_scores) if all_scores else 0.0
            
            return GuardrailResponse(
                is_safe=is_safe,
                score=score,
                category=category,
                latency=latency,
                model_name=self.model_name,
                timestamp=datetime.now(timezone.utc)
            )
            
        except RateLimitError as e:
            latency = (time.time() - start_time) * 1000
            raise RateLimitError(
                f"OpenAI rate limit exceeded: {str(e)}",
                response=e.response,
                body=e.body
            ) from e
            
        except APIConnectionError as e:
            latency = (time.time() - start_time) * 1000
            raise APIConnectionError(
                f"Failed to connect to OpenAI API: {str(e)}"
            ) from e
            
        except APIError as e:
            latency = (time.time() - start_time) * 1000
            raise APIError(
                f"OpenAI API error: {str(e)}",
                response=e.response,
                body=e.body
            ) from e
            
        except (KeyError, IndexError, AttributeError) as e:
            latency = (time.time() - start_time) * 1000
            raise ValueError(
                f"Invalid response format from OpenAI: {str(e)}"
            ) from e
