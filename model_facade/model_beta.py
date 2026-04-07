import pathlib
import re
from typing import Sequence, Union

import torch
import tqdm
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch.nn.functional as F


# File: model_beta.py
# Author: fancyfeast
# Modified by: nflamously
# Original License: Apache License 2.0 / unknown
# Changes:
# * Custom Llava Loader and inference script without generator
# * Added confidence score extraction


def load_llava(model_path: Union[pathlib.Path, str]):
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


def calculate_token_confidence(logits, token_ids):
    """Calculate confidence scores for generated tokens"""
    probs = F.softmax(logits, dim=-1)
    token_probs = []

    for i, token_id in enumerate(token_ids):
        if i < len(logits):
            token_prob = probs[i, token_id].item()
            token_probs.append(token_prob)

    return token_probs


def calculate_entity_scores(tokens, token_probs, processor):
    """
    Calculate confidence scores for entities/phrases in the caption.
    Returns a list of (text, confidence) tuples.
    """

    current_entity = []
    current_probability = []
    entities = []
    handled_entities = []

    def add_token_to_current_entity(text, prob):
        if text not in entities:
            current_entity.append(text)
            current_probability.append(prob)

    def calculate_entity():
        entity_text = ''.join(current_entity).strip()

        if entity_text in handled_entities:
            return

        avg_confidence = sum(current_probability) / len(current_probability)
        entities.append({
            'text': entity_text,
            'confidence': avg_confidence
        })

        handled_entities.append(entity_text)

    for token, probability in zip(tokens, token_probs):
        token_text = processor.decode([token], skip_special_tokens=True)

        if not token_text or is_stop_word_or_symbol(token_text):
            continue

        if len(current_entity) <= 0:
            add_token_to_current_entity(token_text, probability)
            continue

        if token_text in handled_entities:
            continue

        if len(current_entity) > 0 and not token_text.startswith((' ', '.', ',', '!', '?')):
            add_token_to_current_entity(token_text, probability)
            continue
        else:
            calculate_entity()
            current_entity = []
            current_probability = []
            add_token_to_current_entity(token_text, probability)

    if len(current_entity) > 0:
        calculate_entity()

    return entities


def is_stop_word_or_symbol(text):
    # Remove if it's just symbols or whitespace
    if not re.search(r'[a-zA-Z0-9]', text):
        return True

    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
        'has', 'have', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'that', 'these', 'those',
        'it', 'its'
    }

    return text.lower().strip() in stop_words


def calculate_confidence_caption(result, token_ids, logits_stack, processor, confidence_threshold):
    # Calculate per-token confidence
    token_probs = calculate_token_confidence(logits_stack, token_ids)

    # Calculate entity-level scores
    entities = calculate_entity_scores(
        token_ids.tolist(),
        token_probs,
        processor
    )

    # Overall confidence (mean of all token probabilities)
    overall_confidence = sum(token_probs) / len(token_probs) if token_probs else 0.0

    result["overall_confidence"] = overall_confidence
    result["entities"] = entities

    # Filter caption by confidence threshold if specified
    if confidence_threshold > 0:
        filtered_entities = [
            ent for ent in entities
            if ent['confidence'] >= confidence_threshold
        ]
        # Override joycaption with filtered caption
        result["joycaption"] = ' '.join([ent['text'] for ent in filtered_entities])


@torch.no_grad()
def inference(
        processor, model, images: list[Image.Image], original_prompt: str, temperature: float,
        top_p: float, max_new_tokens: int, show_prompt: bool, batch_size: int, confidence_threshold: float = 0.0,
        return_confidence_scores: bool = False,
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
    prompts_data = []

    if any([img is None for img in images]):
        print(f"Warning invalid image provided.")
        return ""

    for i in tqdm.trange(0, len(images), batch_size, desc="Captioning Images"):
        chunk: Sequence[Image.Image] = images[i: i + batch_size]
        convos = [user_only_template] * len(chunk)
        inputs = processor(
            text=convos,
            images=chunk,
            return_tensors="pt",
        ).to("cuda")

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                suppress_tokens=None,
                use_cache=True,
                temperature=temperature,
                top_p=top_p if temperature > 0 else None,
                return_dict_in_generate=True,
                output_scores=True,  # Enable score output
            )

        generate_ids = outputs.sequences
        scores = outputs.scores  # Logits for each generation step

        # Strip input tokens from generated_ids
        generated_tokens_only = generate_ids[:, inputs.input_ids.shape[1]:]

        decoded = processor.batch_decode(
            generated_tokens_only, skip_special_tokens=True
        )

        for idx, (img, desc) in enumerate(zip(chunk, decoded)):
            result = {
                "image": img,
                "prompt": original_prompt,
                "joycaption": None if return_confidence_scores else desc.strip().replace("\n", ""),
            }

            if return_confidence_scores:
                calculate_confidence_caption(result,
                                             generated_tokens_only[idx],
                                             torch.stack([s[idx] for s in scores]),
                                             processor,
                                             confidence_threshold)

            prompts_data.append(result)

    return prompts_data
