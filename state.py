import pathlib
from typing import Any, TypedDict


# TODO: Could be Type near future
class AppState(TypedDict):
    clip_model: Any
    clip_model_name: str | None
    tokenizer: Any
    text_model: Any
    image_adapter: Any
    caption_map: Any
    checkpoint_path: pathlib.Path | None


APP_STATE: AppState = {
    "clip_model": None,
    "clip_model_name": None,
    "tokenizer": None,
    "text_model": None,
    "image_adapter": None,
    "caption_map": None,
    "checkpoint_path": None
}
