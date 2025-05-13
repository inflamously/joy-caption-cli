import os.path
import pathlib
import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast, LlamaForCausalLM, \
    BitsAndBytesConfig
from image_adapter import ImageAdapter


# File: model.py
# Author: fancyfeast
# Modified by: nflamously
# Original License: Apache License 2.0 / unknown
# Changes:
# * split import and loaders into own functions to be used. Extends app with a state class.
# * Use proper loading of LORA and using PEFT and propery MODEL to load weights in and getting ready to action
# This code was originally authored by fancyfeast. All modifications are documented and follow the terms of the original license.


def load_clip_model(clip_path: str):
    print("Loading CLIP")
    clip_model = AutoModel.from_pretrained(clip_path, torch_dtype=torch.float16).to('cuda')
    clip_model = clip_model.vision_model
    return clip_model


def load_vision_model(checkpoint_path: pathlib.Path, clip_model):
    print("Loading custom vision model")
    clip_model_path = checkpoint_path / "clip_model.pt"
    if not os.path.exists(clip_model_path):
        raise Exception("Failed to load CLIP model, path ${} does not exist.".format(clip_model_path))

    checkpoint = torch.load(checkpoint_path / "clip_model.pt", map_location='cuda', weights_only=True)
    checkpoint = {k.replace("_orig_mod.module.", ""): v for k, v in checkpoint.items()}
    clip_model.load_state_dict(checkpoint)
    del checkpoint
    clip_model.eval()
    clip_model.requires_grad_(False)


def load_tokenizer(checkpoint_path: pathlib.Path):
    # Tokenizer
    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path / "text_model", use_fast=True)
    assert (isinstance(tokenizer, PreTrainedTokenizer) or
            isinstance(tokenizer, PreTrainedTokenizerFast)), f"Tokenizer is of type {type(tokenizer)}"
    return tokenizer


def load_llm(checkpoint_path: pathlib.Path):
    print("Loading custom text model")
    print(f"Loading LORA for base model: {checkpoint_path}")
    # Load the base model (adjust device_map, load_in_8bit, torch_dtype as needed for your hardware)
    base_model = LlamaForCausalLM.from_pretrained(
        "unsloth/Meta-Llama-3.1-8B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(load_in_8bit=True)
    )

    print(f"Loading PEFT adapter from")
    # Load the adapter onto the base model
    model = PeftModel.from_pretrained(
        base_model,
        checkpoint_path / "text_model",
        # torch_dtype=torch.float16,  # Usually inherits from base model
        device_map="auto"
    )

    # Optional: Merge adapters into the base model if you don't need to switch adapters later
    return model.merge_and_unload()


def load_image_adapter(checkpoint_path: pathlib.Path, clip_model, text_model):
    print("Loading image adapter")
    image_adapter = ImageAdapter(clip_model.config.hidden_size, text_model.config.hidden_size, False, False, 38,
                                 False).to('cuda')
    image_adapter.load_state_dict(
        torch.load(checkpoint_path / "image_adapter.pt", map_location="cpu", weights_only=True))
    image_adapter.eval()
    return image_adapter
