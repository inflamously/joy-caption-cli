from typing import List, Dict

import torch
import torchvision.transforms.functional
import PIL
import tqdm
from PIL.Image import Image
from torchvision import transforms
from transformers import PreTrainedTokenizerFast

from captions.utils import break_list_into_chunks


# File: prompt_image.py
# Author: fancyfeast
# Modified by: nflamously
# Original License: Apache License 2.0 / unknown
# Changes:
# * Rewritten some of the code for better performance and readability
# This code was originally authored by fancyfeast. All modifications are documented and follow the terms of the original license.

def select_prompt_type(caption_type: str, length: str | int, captions: Dict[str, List[str]]):
    # Build prompt
    # Based on "length" type and captions [0, 1, 2] we select a different prompt for input
    if length is None:
        map_idx = 0
    elif isinstance(length, int):
        map_idx = 1
    elif isinstance(length, str):
        map_idx = 2
    else:
        raise ValueError(f"Invalid joycaption length: {length}")

    return captions[caption_type][map_idx]


# VLM: Vision Language Model
def create_prompt_for_vlm(
        caption_type: str, caption_length: str | int, extra_options, name_input: str, custom_prompt: str,
        captions: Dict[str, List[str]]):
    torch.cuda.empty_cache()

    # 'any' means no length specified
    length = None if caption_length == "any" else caption_length

    if isinstance(length, str):
        try:
            length = int(length)
        except ValueError:
            pass

    prompt_str = select_prompt_type(caption_type, length, captions)

    # Add extra options
    if len(extra_options) > 0:
        prompt_str += " " + " ".join(extra_options)

    # Add name, length, word_count
    prompt_str = prompt_str.format(name=name_input, length=caption_length, word_count=caption_length)

    if custom_prompt.strip() != "":
        prompt_str = custom_prompt.strip()

    # For debugging
    # print(f"Prompt: {prompt_str}")

    return prompt_str


def process_images(images: List[PIL.Image.Image]) -> List[torch.Tensor]:
    def process_image(source_image: PIL.Image.Image) -> torch.Tensor:
        target_image = source_image.resize((384, 384),
                                           PIL.Image.LANCZOS) if source_image.width != 384 or source_image.height != 384 else source_image
        return transforms.functional.normalize(transforms.functional.pil_to_tensor(target_image).unsqueeze(
            0) / 255.0, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).to(dtype=torch.float16, device='cuda')

    return [process_image(image) for image in tqdm.tqdm(images, desc="Tensoring images")]


def create_conversation_token(tokenizer, prompt: str):
    # Build the conversation
    convo = [
        {
            "role": "system",
            "content": "You are a image captioner. Captions must be precise and are used in a diffusion model to learn intricate details of various training images.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    # Format the conversation
    convo_string = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
    convo_tokens = tokenizer.encode(convo_string, return_tensors="pt", add_special_tokens=False, truncation=False)
    convo_tokens = convo_tokens.squeeze(0)  # Squeeze just to make the following easier
    return convo_tokens


def create_prompt_tokens(tokenizer, prompt: str):
    prompt_tokens = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False, truncation=False)
    prompt_tokens = prompt_tokens.squeeze(0)
    return prompt_tokens


def calculate_preamble_length(tokenizer, convo_tokens, prompt_tokens):
    # Calculate where to inject the image
    eot_token = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    eot_id_indices = (convo_tokens == eot_token).nonzero(as_tuple=True)[0].tolist()
    assert len(eot_id_indices) == 2, f"Expected 2 <|eot_id|> tokens, got {len(eot_id_indices)}"
    return eot_id_indices[1] - prompt_tokens.shape[0]  # Number of tokens before the prompt


def generate_captions(
        text_model, clip_model, image_adapter, tokenizer: PreTrainedTokenizerFast,
        images: List[PIL.Image.Image], convo_tokens, preamble_len, batch_size: int = 1):
    if not images or len(images) <= 0: return

    with torch.no_grad():
        convo_embeds = text_model.model.embed_tokens(convo_tokens.unsqueeze(0).to('cuda'))

    resulting_captions: List[str] = []
    image_chunks = break_list_into_chunks(images, batch_size) if batch_size > 1 else images

    for chunk in tqdm.tqdm(image_chunks, desc="Processing images"):
        pixel_values = torch.stack(process_images(chunk if batch_size > 1 else [chunk])).squeeze(
            1)  # torch.stack(process_images(images)).squeeze() if len(images) > 1 else process_images(images)

        with torch.no_grad():
            # This results in Batch x Image Tokens x Features
            vision_outputs = clip_model(pixel_values=pixel_values, output_hidden_states=True)
            embedded_images = image_adapter(vision_outputs.hidden_states)
            del vision_outputs

        embeds = []
        tokens = []

        for n in range(0, embedded_images.shape[0]):
            embeds.append(torch.cat([
                convo_embeds[:, :preamble_len].to(dtype=convo_embeds.dtype),
                embedded_images[n:n + 1, :, :].to(dtype=convo_embeds.dtype),
                convo_embeds[:, preamble_len:],  # The prompt and anything after it
            ], dim=1).to('cuda'))

            tokens.append(torch.cat([
                convo_tokens[:preamble_len].unsqueeze(0),
                torch.zeros((1, embedded_images.shape[1]), dtype=torch.long),
                # Dummy tokens for the image (TODO: Should probably use a special token here so as not to confuse any generation algorithms that might be inspecting the input)
                convo_tokens[preamble_len:].unsqueeze(0),
            ], dim=1).to('cuda'))

        del embedded_images

        torch.cuda.empty_cache()

        # We take first because, embeds and tokens all shouuld be equal dimensions
        # TODO: Optimize later... attention_mask = torch.ones((1, convo_tokens.shape[1] + embedded_images.shape[1]), device='cuda') <- are sizes real, check?
        attention_mask = torch.ones_like(tokens[0])

        # Debugging
        # for token in tokens:
        #     print(f"Input to model: {repr(tokenizer.decode(token[0]))}")

        with torch.no_grad():
            generate_ids = text_model.generate(
                torch.stack(tokens).squeeze() if len(tokens) > 1 else torch.stack(tokens).squeeze(0),
                inputs_embeds=torch.stack(embeds).squeeze() if len(embeds) > 1 else torch.stack(embeds).squeeze(0),
                attention_mask=attention_mask,
                max_new_tokens=300,
                do_sample=True,
                suppress_tokens=None)
            # Uses the default which is temp=0.6, top_p=0.9

        # Skip prompting text using [:, tokens[0].shape[1]]
        trim_generate_ids = torch.stack(
            [generate_ids[n:n + 1, tokens[n].shape[1]:] for n in range(0, len(generate_ids))]).squeeze(1)
        batch_captions: List[str] = tokenizer.batch_decode(trim_generate_ids, skip_special_tokens=True,
                                                           clean_up_tokenization_spaces=False)
        resulting_captions.extend(batch_captions)

        del embeds, tokens
        torch.cuda.empty_cache()

    return resulting_captions


def caption_image(
        tokenizer, text_model, clip_model, image_adapter,
        image: PIL.Image.Image, caption_type: str, caption_length: str | int, extra_options: list[str],
        name_input: str, custom_prompt: str, captions: Dict[str, List[str]]):
    prompt_str = create_prompt_for_vlm(caption_type, caption_length, extra_options, name_input, custom_prompt,
                                       captions)
    # prompt_str is tokenized separately so we can do the calculations below
    convo_tokens = create_conversation_token(tokenizer, prompt_str)
    prompt_tokens = create_prompt_tokens(tokenizer, prompt_str)
    preamble_len = calculate_preamble_length(tokenizer, convo_tokens, prompt_tokens)
    captions = generate_captions(text_model, clip_model, image_adapter, tokenizer, [image], convo_tokens, preamble_len)

    return prompt_str, captions[0]


def caption_images(
        tokenizer, text_model, clip_model, image_adapter,
        images: List[Image],
        caption_type: str, caption_length: str | int, extra_options: list[str],
        name_input: str, custom_prompt: str, captions: Dict[str, List[str]],
        batch_size: int = 1):
    prompt_str = create_prompt_for_vlm(caption_type, caption_length, extra_options, name_input, custom_prompt, captions)
    convo_tokens = create_conversation_token(tokenizer, prompt_str)
    prompt_tokens = create_prompt_tokens(tokenizer, prompt_str)
    preamble_len = calculate_preamble_length(tokenizer, convo_tokens, prompt_tokens)

    captions = generate_captions(
        text_model, clip_model, image_adapter, tokenizer, images, convo_tokens, preamble_len, batch_size)

    return [{
        "image": images[img_idx],
        "prompt": prompt_str,
        "joycaption": captions[img_idx]
    } for img_idx in range(len(images))]
