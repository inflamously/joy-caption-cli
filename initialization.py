import os
import pathlib

from config.load import load_config
from model_selection import supported_clip_models, supported_joycaption_models
from state import APP_STATE

# File: initialization.py
# Author: nflamously
# Original License: Apache License 2.0


def setup_config(model_type: str = "alpha"):
    APP_STATE["model_type"] = model_type

    app_config = load_config("config/config.json")["model"][model_type]
    APP_STATE["checkpoint_path"] = (
        pathlib.Path(app_config["checkpoint_path"])
        if os.path.exists(app_config["checkpoint_path"])
        else app_config["checkpoint_path"]
    )

    if model_type == "alpha":
        APP_STATE["clip_model_name"] = app_config["clip_model"]
        APP_STATE["caption_map"] = app_config["caption_types"]
    elif model_type == "beta":
        APP_STATE["text_model"] = app_config["checkpoint_path"]
        APP_STATE["caption_map"] = app_config["caption_types"]
    elif model_type == "clip":
        pass
    else:
        raise Exception(
            f"model_type {model_type} is not supported, available: {[*supported_joycaption_models(), *supported_clip_models()]}"
        )
