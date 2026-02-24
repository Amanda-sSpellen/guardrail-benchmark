# # TODO: fix
# import pytest
# from types import SimpleNamespace
# from unittest.mock import AsyncMock
# import torch

# from core.schema import GuardrailRequest, GuardrailResponse
# from models.caramllo_model import CaraMLLoGuardrailModel


# @pytest.mark.integration
# class TestCaraMLLoMock:
#     @pytest.fixture
#     def cara_model(self):
#         model = CaraMLLoGuardrailModel(model_name="cara-mock", model_path="/nonexistent")
#         # build fake tokenizer and model similar to unit mocks
#         class FakeTokenizer:
#             eos_token_id = 0

#             def apply_chat_template(self, conversation, return_tensors="pt", categories=None):
#                 return torch.tensor([[1, 2, 3]], dtype=torch.long)

#             def decode(self, tensor, skip_special_tokens=True):
#                 return "Header\nMeta\nsafe\nA1\n"

#         class FakeModel:
#             def __init__(self):
#                 self._device = torch.device("cpu")

#             @property
#             def device(self):
#                 return self._device

#             def parameters(self):
#                 return iter([torch.tensor([1.0])])

#             def generate(self, input_ids:torch.Tensor, **kwargs):
#                 sequences = torch.cat([input_ids, torch.tensor([[10, 11]])], dim=1)
#                 return SimpleNamespace(sequences=sequences)

#         client = AsyncMock()
#         mapping = {"tokenizer": FakeTokenizer(), "model": FakeModel()}
#         client.__getitem__.side_effect = lambda k: mapping[k]

#         model._client = client
#         return model

#     @pytest.mark.asyncio
#     async def test_evaluate_returns_response(self, cara_model):
#         req = GuardrailRequest(text="Olá, tudo bem?")
#         resp = await cara_model.evaluate(req)

#         assert isinstance(resp, GuardrailResponse)
#         assert resp.is_safe is True
#         assert resp.model_name == "cara-mock"
#         assert resp.latency > 0
        