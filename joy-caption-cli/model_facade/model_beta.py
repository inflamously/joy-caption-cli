import pathlib

import torch
import tqdm
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoModel, LlamaForCausalLM, BitsAndBytesConfig
from captions.utils import break_list_into_chunks


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
    model = LlavaForConditionalGeneration.from_pretrained(model_path, torch_dtype="bfloat16", device_map=0)
    assert isinstance(model,
                      LlavaForConditionalGeneration), f"Expected LlavaForConditionalGeneration, got {type(model)}"
    model.eval()
    print("Model eval success")
    return [processor, model]


@torch.no_grad()
def inference(
        processor, model, images: list[Image.Image], original_prompt: str,
        temperature: float, top_p: float, max_new_tokens: int,
        show_prompt: bool, batch_size: int):
    torch.cuda.empty_cache()

    prompts_data = []

    if show_prompt: print(f"Custom Prompt: {original_prompt}")

    if any([img is None for img in images]):
        print(f"Warning invalid image provided.")
        return ""

    image_chunks = break_list_into_chunks(images, batch_size) if batch_size > 1 else images
    for chunk in tqdm.tqdm(image_chunks, desc="Processing images in batches"):
        convos = []
        for _ in tqdm.tqdm(chunk, desc="generating chat messages for images"):
            convo = [
                {
                    "role": "user",
                    "content": original_prompt.strip() + "\n",
                },
            ]

            convo_string = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
            assert isinstance(convo_string, str)
            convos.append(convo_string)

        # Process the inputs
        inputs = processor(text=convos, images=chunk, return_tensors="pt").to('cuda')
        inputs['pixel_values'] = inputs['pixel_values']

        generate_ids = model.generate(**inputs,
                                      max_new_tokens=max_new_tokens,
                                      do_sample=True if temperature > 0 else False,
                                      suppress_tokens=None,
                                      use_cache=True,
                                      temperature=temperature if temperature > 0 else None,
                                      top_k=None,
                                      top_p=top_p if temperature > 0 else None)

        batched_prompts = processor.batch_decode(generate_ids, skip_special_tokens=True)

        for idx in range(len(batched_prompts)):
            image_prompt: str = batched_prompts[idx]
            image_prompt_start = f"{original_prompt}\nassistant"
            image_prompt = image_prompt[
                           image_prompt.index(image_prompt_start) + len(image_prompt_start):].strip().replace(
                "\n", "")
            print("generated prompt: ", image_prompt.strip())
            prompts_data.append({
                "image": chunk[idx],
                "prompt": original_prompt,
                "joycaption": image_prompt,
            })

    return prompts_data
