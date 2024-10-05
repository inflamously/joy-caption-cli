from typing import Any, TypedDict

# TODO: Could be Type near future
class AppState(TypedDict):
    clip_model: Any
    tokenizer: Any
    text_model: Any
    image_adapter: Any
    caption_map: Any

APP_STATE: AppState = {
    "clip_model": None,
    "tokenizer": None,
    "text_model": None,
    "image_adapter": None,
    "caption_map": None
}