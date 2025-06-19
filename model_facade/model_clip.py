import pathlib

import torch
from PIL.Image import Image
from tqdm import trange
from transformers import BlipForConditionalGeneration, BlipProcessor


def load_clip(checkpoint: pathlib.Path):
    processor = BlipProcessor.from_pretrained(checkpoint, use_fast=True)
    model = BlipForConditionalGeneration.from_pretrained(checkpoint).to("cuda")
    return [processor, model]


@torch.no_grad()
def inference(
    processor,
    model,
    images: list[Image],
    prompt: str,
    batch_size: int,
    show_prompt: bool,
):
    prompts_data = []

    if show_prompt:
        print(f"prompt: {prompt}")

    for i in trange(0, len(images), batch_size, desc="Captioning Images"):
        batch = images[i : i + batch_size]
        inputs = processor(batch, prompt, return_tensors="pt").to("cuda")
        inputs = inputs["pixel_values"]
        tokens = model.generate(inputs)
        captions = processor.batch_decode(tokens, skip_special_tokens=True)
        for j in range(len(captions)):
            prompts_data.append(
                {
                    "image": batch[j],
                    "prompt": prompt,
                    "caption": captions[j],
                }
            )
    return prompts_data
