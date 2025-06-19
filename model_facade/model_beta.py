import pathlib
from typing import Sequence

import torch
import tqdm
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

# File: model_beta.py
# Author: fancyfeast
# Modified by: nflamously
# Original License: Apache License 2.0 / unknown
# Changes:
# * Custom Llava Loader and inference script without generator


def load_llava(model_path: pathlib.Path):
    print("Loading LLAVA beta processor")
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
    processor.chat_template = """
{#- This block extracts the system message, so we can slot it into the right place. #}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content'] %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = "" %}
{%- endif %}

{{- "<|start_header_id|>system<|end_header_id|>" }}
{{- system_message }}
{{- "<|eot_id|>" }}


{%- set first_user_message = True %}
{%- for message in messages %}
    {%- if first_user_message and message['role'] == 'user' %}
		{%- set first_user_message = False %}
	    {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>

<|reserved_special_token_70|><|reserved_special_token_69|><|reserved_special_token_71|>'+ message['content'].replace('<|reserved_special_token_69|>', '').lstrip() + '<|eot_id|>' }}
	{%- else %}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>

'+ message['content'] + '<|eot_id|>' }}
	{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>

' }}
{%- endif %}
    """
    print("Loading LLAVA beta model")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="bfloat16", device_map=0
    )
    assert isinstance(
        model, LlavaForConditionalGeneration
    ), f"Expected LlavaForConditionalGeneration, got {type(model)}"
    model.eval()
    print("Model eval success")
    return [processor, model]


@torch.no_grad()
def inference(
    processor,
    model,
    images: list[Image.Image],
    original_prompt: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    show_prompt: bool,
    batch_size: int,
):
    if show_prompt:
        print(f"Custom prompt: {original_prompt}")

    if any(img is None for img in images):
        raise ValueError("One (or more) images was None")

    user_only_template: str = processor.apply_chat_template(
        [{"role": "user", "content": original_prompt.strip() + "\n"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    image_prompt_start = f"{original_prompt}\nassistant"
    prompts_data = []

    if any([img is None for img in images]):
        print(f"Warning invalid image provided.")
        return ""

    for i in tqdm.trange(0, len(images), batch_size, desc="Captioning Images"):
        chunk: Sequence[Image.Image] = images[i : i + batch_size]
        convos = [user_only_template] * len(chunk)
        inputs = processor(
            text=convos,
            images=chunk,
            return_tensors="pt",
        ).to("cuda")

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            generate_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                suppress_tokens=None,
                use_cache=True,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p if temperature > 0 else None,
            )

        # Strip input tokens from generated_ids since we do not need them.
        generated_tokens_only = generate_ids[:, inputs.input_ids.shape[1] :]

        decoded = processor.batch_decode(
            generated_tokens_only, skip_special_tokens=True
        )

        for img, desc in zip(chunk, decoded):
            prompts_data.append(
                {
                    "image": img,
                    "prompt": original_prompt,
                    "joycaption": desc.strip().replace("\n", ""),
                }
            )

    return prompts_data
