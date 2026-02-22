"""
conftest.py: Shared pytest fixtures and configuration for all tests.

This module provides common fixtures for models, requests, and test data
used across all test suites.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from typing import List

from core.schema import GuardrailRequest, GuardrailResponse
from core.base_model import GuardrailModel


class DummyGuardrailModel(GuardrailModel):
    """
    Dummy guardrail model for testing purposes.
    
    Returns deterministic responses based on text content to allow testing
    without actual API calls.
    """
    
    def _load_client(self):
        """Return None as dummy model doesn't need a real client."""
        return None
    
    async def evaluate(self, request: GuardrailRequest) -> GuardrailResponse:
        """
        Evaluate text using simple heuristics for testing.
        
        Args:
            request: GuardrailRequest to evaluate
            
        Returns:
            GuardrailResponse with dummy data
        """
        # Simple heuristic: check for keywords to determine category
        text_lower = request.text.lower()
        
        if any(word in text_lower for word in ["kill", "hurt", "attack"]):
            category = "violence"
        elif any(word in text_lower for word in ["hate", "despise", "detest"]):
            category = "hate_speech"
        elif any(word in text_lower for word in ["sex", "nude", "explicit"]):
            category = "sexual_content"
        elif any(word in text_lower for word in ["bully", "harass", "mock"]):
            category = "harassment"
        elif any(word in text_lower for word in ["fake", "false", "lie"]):
            category = "misinformation"
        else:
            category = "none"
        
        is_safe = category == "none"
        
        return GuardrailResponse(
            is_safe=is_safe,
            score=-1,
            category=category,
            latency=10.0,  # Dummy latency
            model_name=self.model_name,
            timestamp=datetime.now(timezone.utc)
        )


# Pytest configuration
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "functional: mark test as a functional test"
    )
    config.addinivalue_line(
        "markers", "smoke: mark test as a smoke test (requires real API)"
    )


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Request Fixtures
# ============================================================================

@pytest.fixture
def safe_request() -> GuardrailRequest:
    """Fixture for a safe GuardrailRequest."""
    return GuardrailRequest(
        text="This is a safe and appropriate message.",
        metadata={"source": "test", "user_id": "test_user"}
    )


@pytest.fixture
def violent_request() -> GuardrailRequest:
    """Fixture for a request with violent content."""
    return GuardrailRequest(
        text="I want to kill and hurt people.",
        metadata={"source": "test", "flag": "violent"}
    )


@pytest.fixture
def hate_request() -> GuardrailRequest:
    """Fixture for a request with hate speech."""
    return GuardrailRequest(
        text="I hate and despise certain groups of people.",
        metadata={"source": "test", "flag": "hate_speech"}
    )


@pytest.fixture
def sexual_request() -> GuardrailRequest:
    """Fixture for a request with sexual content."""
    return GuardrailRequest(
        text="Explicit sexual content here.",
        metadata={"source": "test", "flag": "sexual"}
    )


@pytest.fixture
def harassment_request() -> GuardrailRequest:
    """Fixture for a request with harassing content."""
    return GuardrailRequest(
        text="You're stupid and I will bully you.",
        metadata={"source": "test", "flag": "harassment"}
    )


@pytest.fixture
def misinformation_request() -> GuardrailRequest:
    """Fixture for a request with misinformation."""
    return GuardrailRequest(
        text="This is a fake and false claim.",
        metadata={"source": "test", "flag": "misinformation"}
    )


@pytest.fixture
def test_requests(
    safe_request,
    violent_request,
    hate_request,
    sexual_request,
    harassment_request,
    misinformation_request
) -> List[GuardrailRequest]:
    """Fixture providing a list of diverse test requests."""
    return [
        safe_request,
        violent_request,
        hate_request,
        sexual_request,
        harassment_request,
        misinformation_request,
    ]


# ============================================================================
# Model Fixtures
# ============================================================================

@pytest.fixture
def dummy_model() -> DummyGuardrailModel:
    """Fixture for a dummy guardrail model."""
    return DummyGuardrailModel(model_name="DummyModel")


@pytest.fixture
def dummy_model_batch() -> List[DummyGuardrailModel]:
    """Fixture providing multiple dummy models."""
    return [
        DummyGuardrailModel(model_name="DummyModel1"),
        DummyGuardrailModel(model_name="DummyModel2"),
        DummyGuardrailModel(model_name="DummyModel3"),
    ]


# ============================================================================
# Response Fixtures
# ============================================================================

@pytest.fixture
def safe_response() -> GuardrailResponse:
    """Fixture for a safe GuardrailResponse."""
    return GuardrailResponse(
        is_safe=True,
        score=-1,
        category="none",
        latency=15.0,
        model_name="TestModel",
        timestamp=datetime.now(timezone.utc)
    )


@pytest.fixture
def unsafe_response() -> GuardrailResponse:
    """Fixture for an unsafe GuardrailResponse."""
    return GuardrailResponse(
        is_safe=False,
        score=-1,
        category="violence",
        latency=20.0,
        model_name="TestModel",
        timestamp=datetime.now(timezone.utc)
    )
