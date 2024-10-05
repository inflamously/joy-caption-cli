import json
import os
from typing import List

import PIL

from prompt_image import caption_images
from state import APP_STATE


def process_caption_files(
        files: List[str], output: str,
        caption_type: str = "Descriptive", caption_length: str = "long",
        name: str = "", extra_options=None, custom_prompt: str = "", batch_size: int = 1):
    if extra_options is None:
        extra_options = []

    if any(map(lambda f: not os.path.exists(f), files)):
        raise FileNotFoundError

    if not APP_STATE["caption_map"]:
        raise Exception("config.json -> captions.map cannot be undefined!")

    images = [PIL.Image.open(file).convert('RGB').resize((384, 384), PIL.Image.LANCZOS) for file in files]
    options = extra_options if extra_options else []
    image_caption_list = caption_images(
        APP_STATE["tokenizer"], APP_STATE["text_model"], APP_STATE["clip_model"], APP_STATE["image_adapter"],
        images, caption_type, caption_length, options, name, custom_prompt, APP_STATE["caption_map"], batch_size)

    if not output or output == "json":
        for image_caption in image_caption_list:
            print(json.dumps({
                "prompt": image_caption["prompt"],
                "caption": image_caption["caption"]
            }))
    else:
        if output == 'text':
            for file_idx in range(len(files)):
                caption = image_caption_list[file_idx]["caption"]
                with open(files[file_idx][:-4] + '.txt', 'w') as f:
                    f.write(caption)