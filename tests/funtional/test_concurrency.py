"""
test_concurrency.py: Functional tests for the AsyncRunner and concurrency control.

This module tests the AsyncRunner's ability to manage concurrent API calls
and verify that the Semaphore correctly limits concurrency.
"""

import pytest
import asyncio

from core.engine import AsyncRunner
from core.schema import GuardrailRequest, GuardrailResponse
from tests.conftest import DummyGuardrailModel


@pytest.mark.functional
class TestAsyncRunnerConcurrency:
    """Tests for AsyncRunner concurrency control with Semaphore."""
    
    @pytest.fixture
    def async_runner(self):
        """Create an AsyncRunner with max_concurrency=2."""
        return AsyncRunner(max_concurrency=2)
    
    @pytest.fixture
    def dummy_model(self):
        """Create a DummyGuardrailModel for testing."""
        return DummyGuardrailModel(model_name="DummyModel")
    
    @pytest.mark.asyncio
    async def test_runner_processes_single_request(self, async_runner, dummy_model):
        """Test that runner can process a single request."""
        request = GuardrailRequest(text="This is a safe test", metadata={"test": True})
        
        response = await dummy_model.evaluate(request)
        
        assert response.is_safe is True
        assert response.model_name == "DummyModel"
    
    @pytest.mark.asyncio
    async def test_runner_batch_evaluates_multiple_requests(self, async_runner, dummy_model):
        """Test that runner can evaluate multiple requests as a batch."""
        requests = [
            GuardrailRequest(text="This is safe"),
            GuardrailRequest(text="I want to kill people"),
            GuardrailRequest(text="I hate certain groups"),
        ]
        
        responses = await async_runner.run_batch(dummy_model, requests)
        
        assert len(responses) == 3
        assert all(isinstance(r, GuardrailResponse) for r in responses)
        assert responses[0].is_safe is True
        assert responses[1].is_safe is False
        assert responses[2].is_safe is False
    
    @pytest.mark.asyncio
    async def test_runner_respects_semaphore_limit(self, async_runner, dummy_model):
        """Test that semaphore correctly limits concurrent requests to 2."""
        # Create many requests
        requests = [
            GuardrailRequest(text=f"Request {i}", metadata={"index": i})
            for i in range(10)
        ]
        
        concurrent_count = 0
        max_concurrent_observed = 0
        lock = asyncio.Lock()
        
        # Wrap the evaluate method to track concurrency
        original_evaluate = dummy_model.evaluate
        
        async def tracked_evaluate(request):
            nonlocal concurrent_count, max_concurrent_observed
            async with lock:
                concurrent_count += 1
                max_concurrent_observed = max(max_concurrent_observed, concurrent_count)
            
            try:
                # Simulate some processing time
                await asyncio.sleep(0.1)
                return await original_evaluate(request)
            finally:
                async with lock:
                    concurrent_count -= 1
        
        dummy_model.evaluate = tracked_evaluate
        
        # Run batch with max_concurrency=2
        responses = await async_runner.run_batch(dummy_model, requests, max_concurrency=2)
        
        # Verify responses
        assert len(responses) == 10
        
        # Verify semaphore limited concurrency
        # Note: Due to asyncio scheduling, we might not always hit exactly 2,
        # but it should never exceed 2 significantly
        assert max_concurrent_observed <= 2
    
    @pytest.mark.asyncio
    async def test_runner_adjusts_concurrency_on_demand(self, async_runner, dummy_model):
        """Test that runner can adjust max_concurrency dynamically."""
        # First with concurrency=1
        async_runner1 = AsyncRunner(max_concurrency=1)
        assert async_runner1.max_concurrency == 1
        
        # Then adjust to 5
        requests = [GuardrailRequest(text=f"Test {i}") for i in range(5)]
        responses = await async_runner1.run_batch(dummy_model, requests, max_concurrency=5)
        
        # Verify concurrency was updated
        assert async_runner1.max_concurrency == 5
        assert len(responses) == 5
    
    @pytest.mark.asyncio
    async def test_runner_maintains_request_order(self, async_runner, dummy_model):
        """Test that responses are returned in the same order as requests."""
        requests = [
            GuardrailRequest(text="Safe content", metadata={"id": 1}),
            GuardrailRequest(text="I want to kill", metadata={"id": 2}),
            GuardrailRequest(text="I hate groups", metadata={"id": 3}),
            GuardrailRequest(text="Another safe message", metadata={"id": 4}),
        ]
        
        responses = await async_runner.run_batch(dummy_model, requests)
        
        # Check order preservation via request text
        assert responses[0].is_safe is True
        assert responses[1].is_safe is False
        assert responses[2].is_safe is False
        assert responses[3].is_safe is True
    
    @pytest.mark.asyncio
    async def test_dummy_model_with_10_samples_benchmark(self, async_runner, dummy_model):
        """Full benchmark: Evaluate 10 diverse samples with dummy model."""
        samples = [
            ("This is a safe and friendly message.", True, "none"),
            ("I want to kill everyone.", False, "violence"),
            ("I hate all tall people.", False, "hate_speech"),
            ("This is explicit sexual content.", False, "sexual_content"),
            ("You are stupid and I will bully you.", False, "harassment"),
            ("NASA faked the moon landing.", False, "misinformation"),
            ("Hello, how are you today?", True, "none"),
            ("The weather is nice.", True, "none"),
            ("Violence is good.", False, "violence"),
            ("Fake news exists everywhere.", False, "misinformation"),
        ]
        
        requests = [
            GuardrailRequest(text=text, metadata={"expected_safe": safe, "expected_category": category})
            for text, safe, category in samples
        ]
        
        # Run through async runner with concurrency limit
        responses = await async_runner.run_batch(dummy_model, requests, max_concurrency=2)
        
        # Verify results
        assert len(responses) == 10
        for i, (response, (text, expected_safe, expected_category)) in enumerate(zip(responses, samples)):
            assert response.is_safe == expected_safe, f"Sample {i}: Expected safe={expected_safe}, got {response.is_safe}"
            assert response.category == expected_category, f"Sample {i}: Expected category={expected_category}, got {response.category}"
            assert response.model_name == "DummyModel"
            assert response.score == -1
            assert response.latency > 0
    
    @pytest.mark.asyncio
    async def test_runner_handles_empty_batch(self, async_runner, dummy_model):
        """Test that runner gracefully handles empty request batch."""
        requests = []
        responses = await async_runner.run_batch(dummy_model, requests)
        
        assert responses == []
    
    @pytest.mark.asyncio
    async def test_runner_response_contains_all_fields(self, async_runner, dummy_model):
        """Test that all responses have required fields."""
        requests = [GuardrailRequest(text="Test")]
        responses = await async_runner.run_batch(dummy_model, requests)
        
        response = responses[0]
        
        # Verify all fields are present
        assert hasattr(response, 'is_safe')
        assert hasattr(response, 'score')
        assert hasattr(response, 'category')
        assert hasattr(response, 'latency')
        assert hasattr(response, 'model_name')
        assert hasattr(response, 'timestamp')
        
        # Verify field types
        assert isinstance(response.is_safe, bool)
        assert isinstance(response.score, (int, float))
        assert isinstance(response.category, str)
        assert isinstance(response.latency, (int, float))
        assert isinstance(response.model_name, str)
