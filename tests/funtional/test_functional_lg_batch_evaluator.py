import pytest
import torch

from core.schema import GuardrailRequest
from models.batch_evaluators.llama_guard_batch_evaluator import LlamaGuardBatchEvaluator


class FakeResult:
    def __init__(self, sequences):
        self.sequences = sequences
    
    def to(self, device):
        return self.sequences

class BlankTokenizer:
    def __init__(self):
        self.eos_token_id = 0
        self.pad_token_id = 0
        self.padding_side = "left"

    def apply_chat_template(self, conversation, tokenize=False, categories=None, add_generation_prompt=False):
        return "<PROMPT>"

    def __call__(self, prompts, return_tensors="pt", padding=True):
        batch_size = len(prompts)
        return FakeResult(sequences={"input_ids": torch.tensor([[1, 2, 3]] * batch_size, dtype=torch.long)})

    def batch_decode(self, generated_tokens, skip_special_tokens=True):
        # Simulate edge cases: first blank output, second missing category code
        return ["\n\n\n", "Header\nMeta\nunsafe\n"]

    def decode(self, tensor, skip_special_tokens=True):
        return "\n\n\n"


class FakeModel:
    def __init__(self, device=torch.device("cpu")):
        self._device = device

    @property
    def device(self):
        return self._device

    def generate(self, **kwargs):
        input_ids: torch.Tensor = kwargs.get("input_ids") # type: ignore
        batch = input_ids.shape[0]
        sequences = torch.cat([input_ids, torch.tensor([[10, 11]] * batch)], dim=1)
        return type("O", (), {"sequences": sequences, "__getitem__": lambda self, i: sequences[i]})()


@pytest.mark.functional
@pytest.mark.asyncio
async def test_edge_case_parsing_blank_and_missing_codes():
    categories = {"harassment": "..", "hate_speech": "..", "sexual_content": ".."}
    evaluator = LlamaGuardBatchEvaluator(model_name="lg-batch", categories=categories)

    tokenizer = BlankTokenizer()
    model = FakeModel()

    reqs = [GuardrailRequest(text="one"), GuardrailRequest(text="two")]

    responses = await evaluator.evaluate_batch(reqs, client={"tokenizer": tokenizer, "model": model})

    assert len(responses) == 2
    # First response may be parsed as safe=False due to blank lines; ensure no crash and type correctness
    assert isinstance(responses[0].is_safe, bool)
    assert (responses[1].category is None) or isinstance(responses[1].category, list)
