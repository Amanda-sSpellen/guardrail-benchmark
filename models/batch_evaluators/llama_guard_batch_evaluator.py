from typing import Any, Dict, Optional, List
import time
import torch

from core.schema import GuardrailRequest, GuardrailResponse
from core.batch_evaluator import BatchEvaluator


class LlamaGuardBatchEvaluator(BatchEvaluator):
    """
    BatchEvaluator defines the interface for evaluating batches of 
    requests through a GuardrailModel.
    This allows for flexible implementations of batch processing and 
    concurrency control, which can be used by the AsyncRunner.
    """
    def __init__(self, model_name: str, categories: Dict[str, Any]):
        self.model_name = model_name
        self.categories = categories

    async def evaluate_batch(
        self,
        requests: List[GuardrailRequest],
        batch_size: Optional[int] = 1,
        **kwargs,
    ) -> List[GuardrailResponse]:
        """
        Evaluate a batch of requests through the engine with concurrency and batching support.
        
        Args:
            requests: List of GuardrailRequest objects to evaluate.
            batch_size: Optional override for batch size (default: 1).
            **kwargs: Additional keyword arguments, expected to include:
                - client: A dict containing the loaded `tokenizer` and `model` for inference
                - max_new_tokens: Optional override for max new tokens to generate (default: 20)
        
        Returns:
            List of GuardrailResponse objects in the same order as input requests.
        """

        client: Dict[str, Any] = kwargs.get("client", {})
        tokenizer = client.get("tokenizer")
        model = client.get("model")
        
        if tokenizer is None or model is None:
            raise ValueError("tokenizer and model must be provided in kwargs['client']")
        
        # 1. Prepare batch inputs
        formatted_prompts = [
            tokenizer.apply_chat_template(
                conversation=[{"role": "user", "content": [{"type": "text", "text": req.text}]}],
                tokenize=False,
                categories=self.categories,
                add_generation_prompt=True
            ) for req in requests
        ]
        
        tokenizer.padding_side = "left" # Crucial for batch generation
        tokenizer.pad_token = tokenizer.eos_token # Ensure pad token is defined
        inputs = tokenizer(
            formatted_prompts, 
            return_tensors="pt", 
            padding=True,
        ).to(model.device)

        gen_kwargs = {
            "max_new_tokens": int(kwargs.get("max_new_tokens", 20)),
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "use_cache": True,  # Speeds up generation by reusing hidden states
        }

        # 2. Inference Block
        start = time.time()
        with torch.no_grad(): # Prevents gradient memory overhead
            output = model.generate(
                **inputs,
                **gen_kwargs,
            )
        
        latency = (time.time() - start) * 1000.0 / len(requests)

        # 3. Parsing (Using batch_decode for speed)
        # Extract only the new tokens for each sequence in the batch
        input_len = inputs['input_ids'].shape[1]
        generated_tokens = output[:, input_len:]

        # 4. Parse and return list of GuardrailResponse objects
        # (Looping through the batch output and applying your parsing logic)
        responses = []
        for i, generated in enumerate(generated_tokens):
            response_text = tokenizer.decode(generated, skip_special_tokens=True)

            response_items = response_text.split("\n")

            is_safe = True if response_items[2].strip() == "safe" else False

            categories = {cat_name : False for cat_name in self.categories.keys()}

            for j, cat_name in enumerate(categories.keys()):
                if f"A{j+1}" in response_items[3:]:
                    categories[cat_name] = True
            triggered_categories = [cat for cat, triggered in categories.items() if triggered]

            responses.append(GuardrailResponse(
                model_name=self.model_name,
                is_safe=is_safe,
                score=-1.0,
                category=triggered_categories if triggered_categories else None,
                latency=latency,
            ))
        return responses
