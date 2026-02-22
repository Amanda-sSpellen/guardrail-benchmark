"""
openai_model.py: OpenAI API wrapper for guardrail evaluation.

This module implements the GuardrailModel interface for OpenAI's general API.
Uses openai.AsyncOpenAI for async API calls with structured output and comprehensive error handling.
"""

import time
from enum import Enum
from datetime import datetime, timezone
from pydantic import create_model

try:
    from openai import AsyncOpenAI
except ImportError as e:
    raise ImportError(
        "openai package is required for OpenAIGuardrailModel. "
        "Install it with: pip install openai" # TODO: update to optional dependency in pyproject.toml
    ) from e

from core.base_model import GuardrailModel
from core.schema import GuardrailRequest, GuardrailResponse


class OpenAIGeneralGuardrailModel(GuardrailModel):
    """
    OpenAI API wrapper for guardrail evaluation.
    
    Uses OpenAI's general GPT model to evaluate text for policy violations.
    Returns standardized multiclass classification results via structured output.
    """
    
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
        Evaluate text using OpenAI's general GPT model with structured output.
        
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
            # Build the categories description string
            categories_desc = "\n".join(
                [f"- {cat}: {desc}" for cat, desc in self.CATEGORIES.items()]
            )
            
            # Create system prompt with categories
            if self.system_prompt is not None:
                system_prompt = self.system_prompt + "\n\n" + f"Categories:\n{categories_desc}"
            else:
                system_prompt = f"""You are a content safety classifier. Evaluate the provided text and classify it into one of these categories:

{categories_desc}

Respond ONLY with valid JSON in the following format (no markdown, no code blocks, just raw JSON):
{{"category": "<category_name>", "explanation": "<brief explanation>"}}"""
            
        
            DynamicEnum = Enum("DynamicCategory", {k: k for k in self.categories.keys()})

            # Create a Pydantic model at runtime
            DynamicSchema = create_model(
                "DynamicSchema",
                category=(DynamicEnum, ...) # (Type, default_value)
            )

            # Call OpenAI general GPT endpoint with structured output
            response = await self.client.chat.completions.parse(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": request.text
                    }
                ],
                response_format=DynamicSchema,
                temperature=self.config.get("temperature", 1)
            )
            
            latency = (time.time() - start_time) * 1000  # Convert to ms
            
            category = response.choices[0].message.parsed.category.value
            is_safe = category in self.safe_categories
            
            return GuardrailResponse(
                is_safe=is_safe,
                score=-1,
                category=category,
                latency=latency,
                model_name=self.model_name,
                timestamp=datetime.now(timezone.utc),
                raw_response=response.choices[0].message.parsed.category  # Store raw response for debugging
            )
        
        except (AttributeError, TypeError, KeyError) as e:
            latency = (time.time() - start_time) * 1000
            raise ValueError(
                f"Invalid response format from OpenAI: {str(e)}"
            ) from e
        
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            raise Exception(
                f"Unexpected error during OpenAI evaluation: {str(e)}"
            ) from e
