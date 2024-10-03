from typing import List, Dict

import torch
import torchvision.transforms.functional as TVF
import PIL


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
        raise ValueError(f"Invalid caption length: {length}")

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
        target_image = source_image.resize(
            (384, 384), PIL.Image.LANCZOS) if source_image.width != 384 or source_image.height != 384 else source_image
        return TVF.normalize(TVF.pil_to_tensor(target_image).unsqueeze(
            0) / 255.0, [0.5], [0.5]).to('cuda')

    return [process_image(image) for image in images]


def create_conversation_token(tokenizer, prompt: str):
    # Build the conversation
    convo = [
        {
            "role": "system",
            "content": "You are a image captioner. You captions must be precise and are used by an diffusion model to learn intricate details of various training images.",
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


def generate_caption(
        text_model, clip_model, image_adapter, tokenizer, image: List[PIL.Image.Image], convo_tokens, preamble_len):
    convo_embeds = text_model.model.embed_tokens(convo_tokens.unsqueeze(0).to('cuda'))

    captions: List[str] = []

    pixel_values = process_images(image)

    for pixel_value in pixel_values:
        # This results in Batch x Image Tokens x Features
        with torch.amp.autocast_mode.autocast('cuda', enabled=True):
            vision_outputs = clip_model(pixel_values=pixel_value, output_hidden_states=True)
            embedded_images = image_adapter(vision_outputs.hidden_states)
            embedded_images = embedded_images.to('cuda')

        # Construct the input
        input_embeds = torch.cat([
            convo_embeds[:, :preamble_len],  # Part before the prompt
            embedded_images.to(dtype=convo_embeds.dtype),  # Image
            convo_embeds[:, preamble_len:],  # The prompt and anything after it
        ], dim=1).to('cuda')

        input_ids = torch.cat([
            convo_tokens[:preamble_len].unsqueeze(0),
            torch.zeros((1, embedded_images.shape[1]), dtype=torch.long),
            # Dummy tokens for the image (TODO: Should probably use a special token here so as not to confuse any generation algorithms that might be inspecting the input)
            convo_tokens[preamble_len:].unsqueeze(0),
        ], dim=1).to('cuda')
        attention_mask = torch.ones_like(input_ids)

        # Debugging
        print(f"Input to model: {repr(tokenizer.decode(input_ids[0]))}")

        generate_ids = text_model.generate(
            input_ids,
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=300, do_sample=True,
            suppress_tokens=None)  # Uses the default which is temp=0.6, top_p=0.9

        # Trim off the prompt
        generate_ids = generate_ids[:, input_ids.shape[1]:]
        if generate_ids[0][-1] == tokenizer.eos_token_id or generate_ids[0][-1] == tokenizer.convert_tokens_to_ids(
                "<|eot_id|>"):
            generate_ids = generate_ids[:, :-1]

        captions.append(
            tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0])

    return captions


def caption_image(
        tokenizer, text_model, clip_model, image_adapter,
        images: List[PIL.Image.Image], caption_type: str, caption_length: str | int, extra_options: list[str],
        name_input: str, custom_prompt: str, captions: Dict[str, List[str]]):
    prompt_str = create_prompt_for_vlm(caption_type, caption_length, extra_options, name_input, custom_prompt,
                                       captions)
    # prompt_str is tokenized separately so we can do the calculations below
    convo_tokens = create_conversation_token(tokenizer, prompt_str)
    prompt_tokens = create_prompt_tokens(tokenizer, prompt_str)
    preamble_len = calculate_preamble_length(tokenizer, convo_tokens, prompt_tokens)
    captions = generate_caption(text_model, clip_model, image_adapter, tokenizer, images, convo_tokens, preamble_len)

    return prompt_str, captions[0]


def caption_images(
        tokenizer, text_model, clip_model, image_adapter,
        images: List[PIL.Image.Image], caption_type: str, caption_length: str | int, extra_options: list[str],
        name_input: str, custom_prompt: str, captions: Dict[str, List[str]]):
    predicted_captions: Dict[PIL.Image.Image, {"prompt": str, "caption": str}] = {}
    for image in images:
        prompt_str, caption = caption_image(
            tokenizer, text_model, clip_model, image_adapter,
            images, caption_type, caption_length, extra_options, name_input,
            custom_prompt, captions)
        predicted_captions[image] = {"prompt": prompt_str, "caption": caption}
    return predicted_captions
