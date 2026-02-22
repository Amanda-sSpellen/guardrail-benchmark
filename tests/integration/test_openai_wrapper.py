"""
test_openai_wrapper.py: Integration tests for OpenAIGeneralGuardrailModel.

This module tests the OpenAIGeneralGuardrailModel with mocked API responses,
verifying correct JSON parsing and interface consistency.
"""

import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from datetime import datetime

from models.openai_general_model import OpenAIGeneralGuardrailModel
from core.schema import GuardrailRequest, GuardrailResponse


@pytest.mark.integration
class TestOpenAIGeneralGuardrailModel:
    """Tests for OpenAIGeneralGuardrailModel with mocked API responses."""
    
    @pytest.fixture
    def openai_model(self):
        """Create a OpenAIGeneralGuardrailModel instance with mocked client."""
        model = OpenAIGeneralGuardrailModel(
            model_name="gpt-4o",
            api_key="mock_api_key"
        )
        # Mock the client
        model._client = AsyncMock()
        return model
    
    @pytest.mark.asyncio
    async def test_evaluate_safe_content(self, openai_model, safe_request):
        """Test evaluating safe content returns is_safe=True."""
        # 1. Setup the final data piece
        # Use a SimpleNamespace or a Mock with the specific attribute
        mock_parsed_data = SimpleNamespace(category="none")

        # 2. Build the nested mock structure
        mock_choice = AsyncMock()
        mock_choice.message.parsed = mock_parsed_data
        
        mock_response = AsyncMock()
        mock_response.choices = [mock_choice]
        
        # 3. Correct the path to the mocked method
        # Note: The function calls `self.client.chat.completions.parse`
        openai_model.client.chat.completions.parse = AsyncMock(return_value=mock_response)
        
        # Evaluate
        response = await openai_model.evaluate(safe_request)
        
        # Assertions
        assert isinstance(response, GuardrailResponse)
        assert response.is_safe is True
        assert response.category == "none"
        assert response.model_name == "gpt-4o"
    
    @pytest.mark.asyncio
    async def test_evaluate_violent_content(self, openai_model, violent_request):
        """Test evaluating violent content returns is_safe=False."""
        # Mock the API response
        mock_parsed_data = SimpleNamespace(category="violence")
        mock_choice = AsyncMock()
        mock_choice.message.parsed = mock_parsed_data
        mock_response = AsyncMock()
        mock_response.choices = [mock_choice]
        openai_model.client.chat.completions.parse = AsyncMock(return_value=mock_response)
        
        # Evaluate
        response = await openai_model.evaluate(violent_request)
        
        # Assertions
        assert response.is_safe is False
        assert response.category == "violence"
        assert response.score == -1
    
    @pytest.mark.asyncio
    async def test_evaluate_all_categories(self, openai_model):
        """Test that all category types are correctly parsed."""
        categories = ["none", "harassment", "hate_speech", "sexual_content", "violence", "misinformation"]
        
        for category in categories:
            # Create the mock response
            mock_parsed_data = SimpleNamespace(category=category)
            mock_choice = AsyncMock()
            mock_choice.message.parsed = mock_parsed_data
            mock_response = AsyncMock()
            mock_response.choices = [mock_choice]
            openai_model.client.chat.completions.parse = AsyncMock(return_value=mock_response)
            
            request = GuardrailRequest(text=f"Test {category}")
            response = await openai_model.evaluate(request)
            
            assert response.category == category
            assert response.is_safe == (category == "none")
    
    @pytest.mark.asyncio
    async def test_interface_consistency_with_gemini(self, openai_model):
        """Test that response structure matches OpenAI model."""        
        # Create mock responses for both models
        mock_parsed_data = SimpleNamespace(category="violence")
        mock_choice = AsyncMock()
        mock_choice.message.parsed = mock_parsed_data
        mock_response = AsyncMock()
        mock_response.choices = [mock_choice]
        openai_model.client.chat.completions.parse = AsyncMock(return_value=mock_response)
        
        request = GuardrailRequest(text="Kill the target")
        openai_response = await openai_model.evaluate(request)
        
        # Check that response has all required fields
        assert hasattr(openai_response, 'is_safe')
        assert hasattr(openai_response, 'score')
        assert hasattr(openai_response, 'category')
        assert hasattr(openai_response, 'latency')
        assert hasattr(openai_response, 'model_name')
        assert hasattr(openai_response, 'timestamp')
        
        # All fields have correct types
        assert isinstance(openai_response.is_safe, bool)
        assert isinstance(openai_response.score, (int, float))
        assert isinstance(openai_response.category, str)
        assert isinstance(openai_response.latency, (int, float))
        assert isinstance(openai_response.model_name, str)
        assert isinstance(openai_response.timestamp, datetime)
    
    def test_categories_dict_exists(self, openai_model):
        """Test that CATEGORIES dict is inherited from base class."""
        assert hasattr(openai_model, 'CATEGORIES')
        assert isinstance(openai_model.CATEGORIES, dict)
        assert 'none' in openai_model.CATEGORIES
        assert 'violence' in openai_model.CATEGORIES
        assert 'harassment' in openai_model.CATEGORIES
        assert 'hate_speech' in openai_model.CATEGORIES
        assert 'sexual_content' in openai_model.CATEGORIES
        assert 'misinformation' in openai_model.CATEGORIES
    
    @pytest.mark.asyncio
    async def test_latency_measurement(self, openai_model, safe_request):
        """Test that latency is properly measured.""" 
        # Create the mock response
        mock_parsed_data = SimpleNamespace(category="none")
        mock_choice = AsyncMock()
        mock_choice.message.parsed = mock_parsed_data
        mock_response = AsyncMock()
        mock_response.choices = [mock_choice]
        openai_model.client.chat.completions.parse = AsyncMock(return_value=mock_response)
        
        response = await openai_model.evaluate(safe_request)
        
        # Latency should be positive and reasonable
        assert response.latency > 0
        assert response.latency < 10000  # Less than 10 seconds
