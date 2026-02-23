import asyncio
import types
from unittest.mock import AsyncMock

import torch
import pytest

from core.schema import GuardrailRequest
from models.caramllo_model import CaraMLLoGuardrailModel


class FakeTokenizer:
    def __init__(self):
        self.eos_token_id = 0

    def apply_chat_template(self, conversation, return_tensors="pt", categories=None):
        # return a small input_ids tensor (batch_size=1, seq_len=3)
        return torch.tensor([[1, 2, 3]], dtype=torch.long)

    def decode(self, tensor, skip_special_tokens=True):
        # Return a well-formed multi-line response that the model parsing expects.
        # response_items[2] == "safe" and response_items[3:] contains A1 and A3
        return "Header line\nMeta line\nsafe\nA1\nA3\n"


class FakeModel:
    def __init__(self, device=torch.device("cpu")):
        self._device = device

    @property
    def device(self):
        return self._device

    def parameters(self):
        # used for device detection in code path, return a trivial tensor iterator
        return iter([torch.tensor([1.0])])

    def generate(self, input_ids: torch.Tensor, **kwargs):
        # create generated tokens (2 tokens) and concatenate to input_ids to mimic HF output
        generated_ids = torch.tensor([[10, 11]], dtype=torch.long)
        sequences = torch.cat([input_ids, generated_ids], dim=1)
        return types.SimpleNamespace(sequences=sequences)


@pytest.mark.unit
def test_triggered_categories_and_parsing():
    # instantiate model but inject fake client to avoid heavy HF loading
    model = CaraMLLoGuardrailModel(model_name="cara-mock", model_path="/nonexistent")
    fake_tok = FakeTokenizer()
    fake_model = FakeModel()

    # Use AsyncMock for the client like the OpenAI tests. Provide a __getitem__ side effect
    client_mock = AsyncMock()
    mapping = {"tokenizer": fake_tok, "model": fake_model}
    client_mock.__getitem__.side_effect = lambda k: mapping[k]

    model._client_cache = client_mock

    req = GuardrailRequest(text="This is a harmless test message")

    resp = asyncio.run(model.evaluate(req))

    assert resp is not None
    assert resp.is_safe is True
    # category should be a list of triggered categories (A1 -> first category, A3 -> third)
    assert isinstance(resp.category, list)
    assert len(resp.category) >= 1
    