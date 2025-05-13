# Setup Models
import pathlib
from state import APP_STATE
from model_facade.model_alpha import load_vision_model, load_clip_model, load_llm, load_tokenizer, load_image_adapter
import torch

# File: model_selection.py
# Author: nflamously
# Original License: Apache License 2.0

def load_models(clip_model_name: str, checkpoint_path: pathlib.Path, model_type: str):
    torch.clear_autocast_cache()

    if APP_STATE["clip_model"]:
        del APP_STATE["clip_model"]
        APP_STATE["clip_model"] = None
    if APP_STATE["tokenizer"]:
        del APP_STATE["tokenizer"]
        APP_STATE["tokenizer"] = None
    if APP_STATE["text_model"]:
        del APP_STATE["text_model"]
        APP_STATE["text_model"] = None
    if APP_STATE["image_adapter"]:
        del APP_STATE["image_adapter"]
        APP_STATE["image_adapter"] = None

    if model_type == "alpha":
        APP_STATE["clip_model"] = load_clip_model(clip_model_name)
        load_vision_model(checkpoint_path, APP_STATE["clip_model"])
        APP_STATE["tokenizer"] = load_tokenizer(checkpoint_path)
        APP_STATE["text_model"] = load_llm(checkpoint_path)
        APP_STATE["image_adapter"] = load_image_adapter(checkpoint_path, APP_STATE["clip_model"],
                                                        APP_STATE["text_model"])
    elif model_type == "beta":
        APP_STATE["text_model"] = load_llm("fancyfeast/llama-joycaption-beta-one-hf-llava")