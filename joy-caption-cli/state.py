import pathlib
from typing import Any, TypedDict


# File: state.py
# Author: nflamously
# Original License: Apache License 2.0

# TODO: Could be Type near future
class AppState(TypedDict):
    clip_model: Any
    clip_model_name: str | None
    tokenizer: Any
    text_model: Any
    image_adapter: Any
    caption_map: Any
    checkpoint_path: pathlib.Path | None
    processor: Any
    model_type: str


def init_app_state() -> AppState:
    return {
        "clip_model": None,
        "clip_model_name": None,
        "tokenizer": None,
        "text_model": None,
        "image_adapter": None,
        "caption_map": None,
        "checkpoint_path": None,
        "processor": None,
        "model_type": "",
    }


APP_STATE: AppState = init_app_state()
