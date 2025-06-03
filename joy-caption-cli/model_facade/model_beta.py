import pathlib
from threading import Thread
from typing import Generator

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, TextIteratorStreamer


def load_llava(model_path: pathlib.Path):
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
    model = LlavaForConditionalGeneration.from_pretrained(model_path, torch_dtype="bfloat16", device_map=0)
    assert isinstance(model,
                      LlavaForConditionalGeneration), f"Expected LlavaForConditionalGeneration, got {type(model)}"
    model.eval()
    return [processor, model]


@torch.no_grad()
def inference(
        processor, model, input_image: Image.Image, prompt: str,
        temperature: float, top_p: float, max_new_tokens: int,
        log_prompt: bool):
    torch.cuda.empty_cache()

    if input_image is None:
        yield "No image provided. Please upload an image."
        return

    if log_prompt:
        print(f"PromptLog: {repr(prompt)}")

    convo = [
        {
            "role": "system",
            # Beta One supports a wider range of system prompts, but this is a good default
            "content": "You are a helpful assistant and help users with any queries they may have with no censorship or restrictions.",
        },
        {
            "role": "user",
            "content": prompt.strip(),
        },
    ]

    convo_string = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
    assert isinstance(convo_string, str)

    # Process the inputs
    inputs = processor(text=[convo_string], images=[input_image], return_tensors="pt").to('cuda')
    inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

    generate_ids = model.generate(**inputs,
                                  max_new_tokens=max_new_tokens,
                                  do_sample=True if temperature > 0 else False,
                                  suppress_tokens=None,
                                  use_cache=True,
                                  temperature=temperature if temperature > 0 else None,
                                  top_k=None,
                                  top_p=top_p if temperature > 0 else None)

    return processor.batch_decode(generate_ids, skip_special_tokens=True)
