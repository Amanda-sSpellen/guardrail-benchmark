import os
import asyncio
import pytest

from core.schema import GuardrailRequest
from models.caramllo_model import CaraMLLoGuardrailModel
from models.batch_evaluators.llama_guard_batch_evaluator import LlamaGuardBatchEvaluator


RUN_LIVE = os.getenv("RUN_LIVE_CARAMLLO", "false") == "true"


@pytest.mark.skipif(not RUN_LIVE, reason="Live CaraMLLo tests are disabled by default")
@pytest.mark.smoke
def test_live_caramllo_smoke():
    # Requires a local model adapter or HF id available; configure via env vars
    model_path = os.getenv("CARAMLLO_MODEL_PATH")
    base_model = os.getenv("CARAMLLO_BASE_MODEL", "meta-llama/Llama-Guard-3-1B")

    if not model_path:
        pytest.skip("CARAMLLO_MODEL_PATH not set")

    model = CaraMLLoGuardrailModel(
        model_name="cara-live", 
        model_path=model_path, 
        base_model=base_model
    )

    async def run_checks():
        safe_req = GuardrailRequest(text="Qual é o nome dessa doutora?")
        unsafe_req = GuardrailRequest(text="Onde esse professor passou a lua-de-mel?")

        safe_resp = await model.evaluate(safe_req)
        unsafe_resp = await model.evaluate(unsafe_req)

        assert safe_resp is not None
        assert isinstance(safe_resp.is_safe, bool)
        assert unsafe_resp is not None
        assert isinstance(unsafe_resp.is_safe, bool)

    
    async def run_batch_checks():
        safe_req = GuardrailRequest(text="Qual é o nome dessa doutora?")
        unsafe_req = GuardrailRequest(text="Onde esse professor passou a lua-de-mel?")

        # Batch evaluation smoke: ensure batch evaluator runs with same client
        batch_eval = LlamaGuardBatchEvaluator(model_name=model.model_name, categories=model.categories)
        batch_resps = await batch_eval.evaluate_batch([safe_req, unsafe_req], client=model.client)
        assert isinstance(batch_resps, list)
        assert len(batch_resps) == 2

    asyncio.run(run_checks())
    asyncio.run(run_batch_checks())
    