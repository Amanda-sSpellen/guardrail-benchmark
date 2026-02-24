"""CaraMLLo local Guardrail model wrapper.

Implements a GuardrailModel that runs a locally fine-tuned Llama-Guard model
using HuggingFace transformers and PEFT. The class provides lazy loading of the
tokenizer/model and an async `evaluate()` that returns a `GuardrailResponse`.
"""

from __future__ import annotations

import time
from loguru import logger
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.base_model import GuardrailModel
from core.schema import GuardrailRequest, GuardrailResponse
from models.batch_evaluators.llama_guard_batch_evaluator import LlamaGuardBatchEvaluator


class CaraMLLoGuardrailModel(GuardrailModel):
    """Local CaraMLLo / Llama-Guard model wrapper.

    Expects `model_path`  which may be either a PEFT adapter path
    (folder or file) or a full model id. If a PEFT adapter is passed it will be
    loaded on top of the base model specified by `base_model` in the config or
    defaulting to `meta-llama/Llama-Guard-3-1B`.
    """
    def __init__(self, **config):
        super().__init__(**config)
        self._client_cache = None  # Initialize the cache
        self.batch_evaluator = LlamaGuardBatchEvaluator(model_name=self.model_name, categories=self.categories)

    def _load_client(self) -> Dict[str, Any]:
        """Lazy-load tokenizer and model.

        Returns a dict with keys `tokenizer` and `model`.
        """
        logger.info(f"Loading CaraMLLo model from {self.config.get('model_path')} with base {self.config.get('base_model')}")
        model_path = self.config.get("model_path")
        base_model = self.config.get("base_model", "meta-llama/Llama-Guard-3-1B")

        if not model_path:
            raise ValueError("`model_path` must be provided in config to load local model")

        # load tokenizer from base model
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

        # load base model
        device_map = self.config.get("device_map", "auto")
        dtype = self.config.get("torch_dtype")
        torch_dtype = getattr(torch, dtype) if dtype else None

        base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch_dtype, device_map=device_map)

        # If model_path points to a PEFT adapter, load it on top
        peft_adapter = Path(model_path)
        if peft_adapter.exists():
            model = PeftModel.from_pretrained(base, model_path)
        else:
            # fallback: assume model_path is a HF id for a full model
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype, device_map=device_map)

        # Keep in eval mode
        model.eval()

        # Ensure we are using the most efficient kernels
        if torch.cuda.is_available():
            model.to("cuda") # type: ignore
        
        return {"tokenizer": tokenizer, "model": model}

    @property
    def client(self) -> Dict[str, Any]:
        """Memoized client access to prevent re-loading weights."""
        if not self._client_cache:
            self._client_cache = self._load_client()
        logger.info("Using cached CaraMLLo client")
        return self._client_cache # type: ignore

    async def evaluate(self, request: GuardrailRequest) -> GuardrailResponse:
        """Evaluate the given text locally and return a GuardrailResponse.

        The method generates a short completion with the model and expects a
        JSON object like {"category": "...", "explanation": "..."} in the
        model output. If parsing fails, falls back to `category='none'`.
        """
        start = time.time()

        # Build category definition and input format
        conversation = [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": request.text
                },
            ],
        },]

        tokenizer = self.client["tokenizer"]
        model = self.client["model"]

        input_ids = tokenizer.apply_chat_template(
            conversation=conversation, 
            return_tensors="pt", 
            categories=self.categories, 
        ).to(model.device)['input_ids'] 
        
        prompt_len = input_ids.shape[1]

        # generation parameters
        gen_kwargs = {
            "max_new_tokens": int(self.config.get("max_new_tokens", 20)),
            "output_scores": True,
            "pad_token_id": tokenizer.eos_token_id or 0,
            "return_dict_in_generate": True,
            "use_cache": True,
        }

        try:
            start = time.time()
            with torch.no_grad(): # Prevents gradient memory overhead
                output = model.generate(
                    input_ids=input_ids,
                    **gen_kwargs,
                )
            
            generated = output.sequences[:, prompt_len:]

            response = tokenizer.decode(
                generated[0], skip_special_tokens=True
            )

            response_items = response.split("\n")

            is_safe = True if response_items[2].strip() == "safe" else False

            categories = {cat_name : False for cat_name in self.categories.keys()}

            # Determine which categories were triggered based on the model output. 
            # The fine-tuned model is expected to output category tokens like A1, A2, etc. 
            # corresponding to the categories in order.
            for i, cat_name in enumerate(categories.keys()):
                if f"A{i+1}" in response_items[3:]:
                    categories[cat_name] = True
            triggered_categories = [cat for cat, triggered in categories.items() if triggered]

            latency = (time.time() - start) * 1000.0

            return GuardrailResponse(
                is_safe=is_safe,
                score=-1.0,
                category=triggered_categories if triggered_categories else None,
                latency=latency,
                model_name=self.model_name,
                timestamp=datetime.now(timezone.utc),
                raw_response={"generated_text": response, "parsed": categories},
            )

        except Exception as e:
            latency = (time.time() - start) * 1000.0
            raise RuntimeError(f"Local model evaluation failed: {e}") from e


__all__ = ["CaraMLLoGuardrailModel"]
