# TODO: update mock response to dict
"""
test_gemini_wrapper.py: Integration tests for GeminiGeneralGuardrailModel.

This module tests the GeminiGeneralGuardrailModel with mocked API responses,
verifying correct JSON parsing and interface consistency.
"""

import pytest
from unittest.mock import MagicMock
from datetime import datetime

from models.gemini_general_model import GeminiGeneralGuardrailModel
from core.schema import GuardrailRequest, GuardrailResponse


@pytest.mark.integration
class TestGeminiGeneralGuardrailModel:
    """Tests for GeminiGeneralGuardrailModel with mocked API responses."""
    
    @pytest.fixture
    def gemini_model(self):
        """Create a GeminiGeneralGuardrailModel instance with mocked client."""
        model = GeminiGeneralGuardrailModel(
            model_name="gemini-2.0-flash",
            api_key="mock_api_key"
        )
        # Mock the client
        model._client = MagicMock()
        return model
    
    @pytest.mark.asyncio
    async def test_evaluate_safe_content(self, gemini_model, safe_request):
        """Test evaluating safe content returns is_safe=True."""
        # Create the mock response
        mock_response = MagicMock()
        mock_parsed_result = MagicMock()
        mock_parsed_result.category = "none"
        mock_response.parsed = mock_parsed_result
        
        gemini_model.client.generate_content.return_value = mock_response
        
        # Evaluate
        response = await gemini_model.evaluate(safe_request)
        
        # Assertions
        assert isinstance(response, GuardrailResponse)
        assert response.is_safe is True
        assert response.category == "none"
        assert response.score == -1
        assert response.model_name == "gemini-2.0-flash"
        assert response.latency > 0
    
    @pytest.mark.asyncio
    async def test_evaluate_violent_content(self, gemini_model, violent_request):
        """Test evaluating violent content returns is_safe=False."""
        # Mock the API response
        # Create the mock response
        mock_response = MagicMock()
        mock_parsed_result = MagicMock()
        mock_parsed_result.category = "violence"
        mock_response.parsed = mock_parsed_result
        
        gemini_model.client.generate_content.return_value = mock_response
        
        # Evaluate
        response = await gemini_model.evaluate(violent_request)
        
        # Assertions
        assert response.is_safe is False
        assert response.category == "violence"
        assert response.score == -1
    
    @pytest.mark.asyncio
    async def test_evaluate_all_categories(self, gemini_model):
        """Test that all category types are correctly parsed."""
        categories = ["none", "harassment", "hate_speech", "sexual_content", "violence", "misinformation"]
        
        for category in categories:
            # Create the mock response
            mock_response = MagicMock()
            mock_parsed_result = MagicMock()
            mock_parsed_result.category = category
            mock_response.parsed = mock_parsed_result
            
            gemini_model.client.generate_content.return_value = mock_response
            
            request = GuardrailRequest(text=f"Test {category}")
            response = await gemini_model.evaluate(request)
            
            assert response.category == category
            assert response.is_safe == (category == "none")
    
    @pytest.mark.asyncio
    async def test_interface_consistency_with_openai(self, gemini_model):
        """Test that response structure matches OpenAI model."""        
        # Create mock responses for both models
        # Create the mock response
        mock_response = MagicMock()
        mock_parsed_result = MagicMock()
        mock_parsed_result.category = "violence"
        mock_response.parsed = mock_parsed_result
        
        gemini_model.client.generate_content.return_value = mock_response
        
        request = GuardrailRequest(text="Kill the target")
        gemini_response = await gemini_model.evaluate(request)
        
        # Check that response has all required fields
        assert hasattr(gemini_response, 'is_safe')
        assert hasattr(gemini_response, 'score')
        assert hasattr(gemini_response, 'category')
        assert hasattr(gemini_response, 'latency')
        assert hasattr(gemini_response, 'model_name')
        assert hasattr(gemini_response, 'timestamp')
        
        # All fields have correct types
        assert isinstance(gemini_response.is_safe, bool)
        assert isinstance(gemini_response.score, (int, float))
        assert isinstance(gemini_response.category, str)
        assert isinstance(gemini_response.latency, (int, float))
        assert isinstance(gemini_response.model_name, str)
        assert isinstance(gemini_response.timestamp, datetime)
    
    def test_categories_dict_exists(self, gemini_model):
        """Test that CATEGORIES dict is inherited from base class."""
        assert hasattr(gemini_model, 'CATEGORIES')
        assert isinstance(gemini_model.CATEGORIES, dict)
        assert 'none' in gemini_model.CATEGORIES
        assert 'violence' in gemini_model.CATEGORIES
        assert 'harassment' in gemini_model.CATEGORIES
        assert 'hate_speech' in gemini_model.CATEGORIES
        assert 'sexual_content' in gemini_model.CATEGORIES
        assert 'misinformation' in gemini_model.CATEGORIES
    
    @pytest.mark.asyncio
    async def test_latency_measurement(self, gemini_model, safe_request):
        """Test that latency is properly measured.""" 
        # Create the mock response
        mock_response = MagicMock()
        mock_parsed_result = MagicMock()
        mock_parsed_result.category = "none"
        mock_response.parsed = mock_parsed_result
        
        gemini_model.client.generate_content.return_value = mock_response
        
        response = await gemini_model.evaluate(safe_request)
        
        # Latency should be positive and reasonable
        assert response.latency > 0
        assert response.latency < 10000  # Less than 10 seconds
