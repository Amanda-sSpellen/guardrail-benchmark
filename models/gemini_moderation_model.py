# TODO: update to dynamic category classification 
"""
gemini_model.py: Google Gemini API wrapper for guardrail evaluation.

This module implements the GuardrailModel interface for Google's Gemini safety API.
Uses google.generativeai for content safety moderation with comprehensive error handling.
"""

import time
from typing import Any
from datetime import datetime, timezone

try:
    import google.generativeai as genai
    from google.api_core.exceptions import GoogleAPIError, Unauthenticated, TooManyRequests
except ImportError as e:
    raise ImportError(
        "google-generativeai package is required for GeminiGuardrailModel. "
        "Install it with: pip install google-generativeai"
    ) from e

from core.base_model import GuardrailModel
from core.schema import GuardrailRequest, GuardrailResponse


class GeminiModerationGuardrailModel(GuardrailModel):
    """
    Google Gemini API wrapper for content safety moderation.
    
    Uses Gemini's built-in safety filters to evaluate text for policy violations.
    Evaluates content using safety ratings across multiple categories.
    """
    
    # Mapping of Gemini safety categories to standardized categories
    SAFETY_CATEGORIES_MAP = {
        "HARM_CATEGORY_HARASSMENT": "harassment",
        "HARM_CATEGORY_HATE_SPEECH": "hate_speech",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "sexual_content",
        "HARM_CATEGORY_DANGEROUS_CONTENT": "violence",
        "HARM_CATEGORY_CIVIC_INTEGRITY": "misinformation",
    }
    
    def _load_client(self) -> genai.GenerativeModel:
        """
        Initialize and return Gemini GenerativeModel client.
        
        Uses api_key from config if provided, otherwise uses environment variable.
        
        Returns:
            genai.GenerativeModel: Initialized Gemini model client
            
        Raises:
            ValueError: If API key is not provided and not in environment
        """
        api_key = self.config.get("api_key")
        if not api_key:
            import os
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError(
                    "Google API key not provided. Pass via config or set GOOGLE_API_KEY environment variable."
                )
        
        genai.configure(api_key=api_key)
        
        # Initialize model - use gemini-pro for safety evaluation
        model_id = self.config.get("model_id", "gemini-pro")
        return genai.GenerativeModel(model_id)
    
    async def evaluate(self, request: GuardrailRequest) -> GuardrailResponse:
        """
        Evaluate text using Gemini's safety moderation features.
        
        Args:
            request: GuardrailRequest containing text and metadata
            
        Returns:
            GuardrailResponse with standardized safety judgment
            
        Raises:
            Unauthenticated: If API key is invalid
            TooManyRequests: If rate limited by Google API
            GoogleAPIError: For other API-related errors
            ValueError: If response format is invalid
        """
        start_time = time.time()
        
        try:
            # Configure safety settings to receive moderation data
            safety_settings = [
                {
                    "category": genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE,
                },
            ]
            
            # Generate content with safety metadata
            response = self.client.generate_content(
                request.text,
                safety_settings=safety_settings,
                stream=False
            )
            
            latency = (time.time() - start_time) * 1000  # Convert to ms
            
            # Parse safety ratings from response
            is_safe = True
            category = None
            max_score = 0.0
            
            if response.safety_ratings:
                for rating in response.safety_ratings:
                    category_name = self.SAFETY_CATEGORIES_MAP.get(
                        str(rating.category),
                        str(rating.category)
                    )
                    
                    # BLOCK_NONE = 1, LOW = 2, MEDIUM = 3, HIGH = 4
                    probability_score = float(rating.probability) / 4.0  # Normalize to 0-1
                    
                    if probability_score > max_score:
                        max_score = probability_score
                        if probability_score > 0.5:  # Threshold for flagging
                            is_safe = False
                            category = category_name
            
            return GuardrailResponse(
                is_safe=is_safe,
                score=max_score,
                category=category,
                latency=latency,
                model_name=self.model_name,
                timestamp=datetime.now(timezone.utc)
            )
            
        except Unauthenticated as e:
            latency = (time.time() - start_time) * 1000
            raise Unauthenticated(
                f"Authentication failed with Google API: {str(e)}"
            ) from e
            
        except TooManyRequests as e:
            latency = (time.time() - start_time) * 1000
            raise TooManyRequests(
                f"Rate limit exceeded by Google API: {str(e)}"
            ) from e
            
        except GoogleAPIError as e:
            latency = (time.time() - start_time) * 1000
            raise GoogleAPIError(
                f"Google API error: {str(e)}"
            ) from e
            
        except (AttributeError, TypeError, KeyError) as e:
            latency = (time.time() - start_time) * 1000
            raise ValueError(
                f"Invalid response format from Gemini: {str(e)}"
            ) from e
