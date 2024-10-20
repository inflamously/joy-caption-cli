import os.path
import pathlib
import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast, \
    AutoModelForCausalLM
from image_adapter import ImageAdapter
from state import APP_STATE


# File: model.py
# Author: fancyfeast
# Modified by: nflamously
# Original License: Apache License 2.0 / unknown
# Changes:
# * split import and loaders into own functions to be used. Extends app with a state class.
# This code was originally authored by fancyfeast. All modifications are documented and follow the terms of the original license.

# Setup Models
def load_models(clip_model_name: str, checkpoint_path: pathlib.Path):
    APP_STATE["clip_model"] = load_clip_model(clip_model_name)
    load_vision_model(checkpoint_path, APP_STATE["clip_model"])
    APP_STATE["tokenizer"] = load_tokenizer(checkpoint_path)
    APP_STATE["text_model"] = load_llm(checkpoint_path)
    APP_STATE["image_adapter"] = load_image_adapter(checkpoint_path, APP_STATE["clip_model"], APP_STATE["text_model"])


def load_clip_model(clip_path: str):
    print("Loading CLIP")
    clip_model = AutoModel.from_pretrained(clip_path)
    clip_model = clip_model.vision_model
    return clip_model


def load_vision_model(checkpoint_path: pathlib.Path, clip_model):
    print("Loading custom vision model")
    clip_model_path = checkpoint_path / "clip_model.pt"
    if not os.path.exists(clip_model_path):
        raise Exception("Failed to load CLIP model, path ${} does not exist.".format(clip_model_path))

    checkpoint = torch.load(checkpoint_path / "clip_model.pt", map_location='cpu', weights_only=True)
    checkpoint = {k.replace("_orig_mod.module.", ""): v for k, v in checkpoint.items()}
    clip_model.load_state_dict(checkpoint)
    del checkpoint

    clip_model.eval()
    clip_model.requires_grad_(False)
    clip_model.to("cuda")


def load_tokenizer(checkpoint_path: pathlib.Path):
    # Tokenizer
    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path / "text_model", use_fast=True)
    assert (isinstance(tokenizer, PreTrainedTokenizer) or
            isinstance(tokenizer, PreTrainedTokenizerFast)), f"Tokenizer is of type {type(tokenizer)}"
    return tokenizer


def load_llm(checkpoint_path: pathlib.Path):
    print("Loading custom text model")
    text_model = AutoModelForCausalLM.from_pretrained(checkpoint_path / "text_model", device_map=0,
                                                      torch_dtype=torch.float16)
    text_model.eval()
    return text_model


def load_image_adapter(checkpoint_path: pathlib.Path, clip_model, text_model):
    print("Loading image adapter")
    image_adapter = ImageAdapter(clip_model.config.hidden_size, text_model.config.hidden_size, False, False, 38, False)
    image_adapter.load_state_dict(
        torch.load(checkpoint_path / "image_adapter.pt", map_location="cpu", weights_only=True))
    image_adapter.eval()
    image_adapter.to("cuda")
    return image_adapter
