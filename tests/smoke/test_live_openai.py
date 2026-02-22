"""
test_live_connect.py: Smoke tests for real API connectivity.

These tests verify that actual API connections work correctly.
They require real API keys and should be run manually with caution to avoid
incurring unnecessary costs.

Run with: pytest tests/smoke/test_live_connect.py -v -m smoke
"""

import pytest
import os
from datetime import datetime, timezone

from core.schema import GuardrailRequest, GuardrailResponse


@pytest.mark.smoke
class TestLiveAPIConnections:
    """Smoke tests for real API connectivity."""
    
    @pytest.fixture
    def safe_test_request(self):
        """Simple safe request for testing."""
        return GuardrailRequest(
            text="I love nature.",
            metadata={"test": True, "timestamp": datetime.now(timezone.utc).isoformat()}
        )
    
    @pytest.fixture
    def unsafe_test_request(self):
        """Simple unsafe request for testing."""
        return GuardrailRequest(
            text="I want to hurt people.",
            metadata={"test": True, "timestamp": datetime.now(timezone.utc).isoformat()}
        )
    
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set - skipping OpenAI live test"
    )
    @pytest.mark.asyncio
    async def test_openai_live_safe_content(self, safe_test_request):
        """Test live OpenAI API with safe content."""
        from models.openai_general_model import OpenAIGeneralGuardrailModel
        
        model = OpenAIGeneralGuardrailModel(
            model_name="gpt-5-nano",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        response = await model.evaluate(safe_test_request)
        
        # Verify response structure
        assert isinstance(response, GuardrailResponse)
        assert response.is_safe is True, f"Expected is_safe=True for safe content ({safe_test_request.text}), got {response}"
        assert response.category == "none"
        assert response.latency > 0
        assert response.model_name == "gpt-5-nano"
    
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set - skipping OpenAI live test"
    )
    @pytest.mark.asyncio
    async def test_openai_live_unsafe_content(self, unsafe_test_request):
        """Test live OpenAI API with unsafe content."""
        from models.openai_general_model import OpenAIGeneralGuardrailModel
        
        model = OpenAIGeneralGuardrailModel(
            model_name="gpt-5-nano",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        response = await model.evaluate(unsafe_test_request)
        
        # Verify response structure
        assert isinstance(response, GuardrailResponse)
        assert response.is_safe is False, f"Expected is_safe=False for unsafe content ({unsafe_test_request.text}), got {response.is_safe}"
        assert response.category in ["violence", "harassment", "hate_speech", "sexual_content", "misinformation"]
        assert response.latency > 0
