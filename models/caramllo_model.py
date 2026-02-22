# TODO: update to dynamic category classification 
# Local execution of a fine-tuned Llama Guard model.

# Example implementation:
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer #, MllamaForConditionalGeneration, AutoProcessor, MllamaProcessor, GenerationConfig


def llama_guard_text_test(tokenizer, model, prompt, categories: dict[str, str]=None, excluded_category_keys: list[str]=[]):
    """
    This function uses the apply_chat_template helper function to tokenize and run inference on the provided inputs.
    The new templates support setting an arbitrary dictionary of categories or excluding the predefined categories
    by passing a list of the preexisting keys.
    """
    if categories is not None:
        input_ids = tokenizer.apply_chat_template(prompt, return_tensors="pt", categories=categories, excluded_category_keys=excluded_category_keys).to("cuda")
        # print("[personalized categories detected]")
    else:
        input_ids = tokenizer.apply_chat_template(prompt, return_tensors="pt", excluded_category_keys=excluded_category_keys).to("cuda")
    input_prompt = tokenizer.decode(input_ids[0], skip_special_tokens=False)


    prompt_len = input_ids.shape[1]
    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=20,
        output_scores=True,
        return_dict_in_generate=True,
        pad_token_id=0,
    )
    generated_tokens = output.sequences[:, prompt_len:]

    response = tokenizer.decode(
        generated_tokens[0], skip_special_tokens=True
    )
    return input_prompt, response


categories = None
# change categories when custom categories are needed
def test_prompt_with_llama_guard(prompt):
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
            ],
        },
    ]

    decoded_input_prompt, response = llama_guard_text_test(lg_small_text_tokenizer,
                                                        llama_guard_model,
                                                        conversation,
                                                        categories,
                                                        )
    # if "unsafe" in response:
    #     return True
    # else:
    #     return False
    response = response.split("\n")
    cats = {"safe": False, "unethical": False, "off-topic": False}
    if "A1" in response[3:]:
        cats["safe"] = True
    if "A2" in response[3:]:
        cats["unethical"] = True
    if "A3" in response[3:]:
        cats["off-topic"] = True
    
    return response[2], cats # str, {"safe": bool, "unethical": bool, "off-topic": bool}

import sys
from peft import PeftModel

lg_small_text_model_id = "meta-llama/Llama-Guard-3-1B"

# Loading the 1B text only model
lg_small_text_tokenizer = AutoTokenizer.from_pretrained(lg_small_text_model_id)
lg_small_text_model = AutoModelForCausalLM.from_pretrained(lg_small_text_model_id, torch_dtype=torch.bfloat16, device_map="auto")


filename = Path(sys.argv[1])
llama_guard_model = PeftModel.from_pretrained(lg_small_text_model, filename)

out_dir = Path(filename) / "tests"
out_dir.mkdir(exist_ok=True, parents=True)

print(f"Running fine-tuned model at {filename}")
run_10_tests(out_dir, filename.name, n=10)

print(f"Saving results at {out_dir}")
