import torch
import pytest

from core.schema import GuardrailRequest
from models.batch_evaluators.llama_guard_batch_evaluator import LlamaGuardBatchEvaluator


class FakeResult:
    def __init__(self, sequences):
        self.sequences = sequences
    
    def to(self, device):
        return self.sequences

class FakeTokenizer:
    def __init__(self):
        self.eos_token_id = 0
        self.pad_token_id = 0
        self.padding_side = "left"

    def apply_chat_template(self, conversation, tokenize=False, categories=None, add_generation_prompt=False):
        # Return an object that when passed to tokenizer() will be handled by fake tokenizer
        return "<PROMPT>"

    def __call__(self, prompts, return_tensors="pt", padding=True):
        # prompts is a list of strings; return simple batched dict
        batch_size = len(prompts)
        # fake input_ids: shape (batch_size, seq_len=3)
        return FakeResult(sequences={"input_ids": torch.tensor([[1, 2, 3]] * batch_size, dtype=torch.long)})

    def batch_decode(self, generated_tokens, skip_special_tokens=True):
        # generated_tokens is a tensor; produce simple decoded strings
        return ["Header\nMeta\nsafe\nA1\n", "Header\nMeta\nunsafe\nA2\n"]

    def decode(self, tensor, skip_special_tokens=True):
        return "Header\nMeta\nsafe\nA1\n"


class FakeModelOutput:
    def __init__(self, sequences):
        self.sequences = sequences

    def __getitem__(self, item):
        return self.sequences[item]


class FakeModel:
    def __init__(self, device=torch.device("cpu")):
        self._device = device

    @property
    def device(self):
        return self._device

    def generate(self, **kwargs):
        # inputs contains input_ids with shape (batch, seq_len)
        input_ids: torch.Tensor = kwargs.get("input_ids") # type: ignore
        batch = input_ids.shape[0]
        # create sequences: for each batch item append two tokens
        sequences = torch.cat([input_ids, torch.tensor([[10, 11]] * batch)], dim=1)
        return FakeModelOutput(sequences)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_batch_evaluator_parsing():
    categories = {"harassment": "..", "hate_speech": "..", "sexual_content": ".."}
    evaluator = LlamaGuardBatchEvaluator(model_name="lg-batch", categories=categories)

    tokenizer = FakeTokenizer()
    model = FakeModel()

    reqs = [GuardrailRequest(text="one"), GuardrailRequest(text="two")]

    responses = await evaluator.evaluate_batch(reqs, client={"tokenizer": tokenizer, "model": model})

    assert isinstance(responses, list)
    assert len(responses) == 2
    for resp in responses:
        assert hasattr(resp, "is_safe")
        assert hasattr(resp, "category")
        assert isinstance(resp.latency, float)
        