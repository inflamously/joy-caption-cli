# Setup Models
import torch

from model_facade import model_alpha, model_beta, model_clip
from state import APP_STATE

# File: model_selection.py
# Author: nflamously
# Original License: Apache License 2.0


def load_model():
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

    model_type = APP_STATE["model_type"]

    if model_type == "alpha":
        checkpoint_path = APP_STATE["checkpoint_path"]
        APP_STATE["clip_model"] = model_alpha.load_clip_model(
            APP_STATE["clip_model_name"]
        )
        model_alpha.load_vision_model(checkpoint_path, APP_STATE["clip_model"])
        APP_STATE["tokenizer"] = model_alpha.load_tokenizer(checkpoint_path)
        APP_STATE["text_model"] = model_alpha.load_llm(checkpoint_path)
        APP_STATE["image_adapter"] = model_alpha.load_image_adapter(
            checkpoint_path, APP_STATE["clip_model"], APP_STATE["text_model"]
        )
    elif model_type == "beta":
        [processor, model] = model_beta.load_llava(APP_STATE["checkpoint_path"])
        APP_STATE["text_model"] = model
        APP_STATE["processor"] = processor

    elif model_type == "clip":
        [processor, model] = model_clip.load_clip(APP_STATE["checkpoint_path"])
        APP_STATE["text_model"] = model
        APP_STATE["processor"] = processor


def supported_joycaption_models():
    return ["alpha", "beta"]


def supported_clip_models():
    return ["clip"]
