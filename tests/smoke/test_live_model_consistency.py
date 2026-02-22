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

from core.schema import GuardrailRequest


@pytest.mark.smoke
class TestLiveAPIConnections:
    """Smoke tests for real API connectivity."""
    
    @pytest.fixture
    def safe_test_request(self):
        """Simple safe request for testing."""
        return GuardrailRequest(
            text="This is a safe and test message.",
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
        not os.getenv("OPENAI_API_KEY") or not os.getenv("GOOGLE_API_KEY"),
        reason="API keys not set - skipping comparison test"
    )
    @pytest.mark.asyncio
    async def test_model_consistency_across_providers(self, safe_test_request):
        """Test that both models agree on clearly safe content."""
        from models.gemini_general_model import GeminiGeneralGuardrailModel
        from models.openai_general_model import OpenAIGeneralGuardrailModel
        
        gemini_model = GeminiGeneralGuardrailModel(
            model_name="gemini-2.0-flash",
            api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        openai_model = OpenAIGeneralGuardrailModel(
            model_name="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        gemini_response = await gemini_model.evaluate(safe_test_request)
        openai_response = await openai_model.evaluate(safe_test_request)
        
        # Both should classify safe content as safe
        assert gemini_response.is_safe is True
        assert openai_response.is_safe is True
        assert gemini_response.category == "none"
        assert openai_response.category == "none"
