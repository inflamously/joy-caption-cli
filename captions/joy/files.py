import json
import os
import codecs
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import List

import PIL
from tqdm import tqdm

from model_facade import model_beta
from prompt_image import caption_images
from state import APP_STATE


# File: files.py
# Author: nflamously
# Original License: Apache License 2.0

def transform_image(file: str):
    return PIL.Image.open(file).convert('RGB').resize((384, 384), PIL.Image.LANCZOS)


def transform_images(files: List[str]):
    print("processing images")
    with ThreadPoolExecutor() as executor:
        images = list(tqdm(executor.map(transform_image, files), total=len(files)))
    return images


def process_caption_files(
        files: List[str], output: str,
        caption_type: str = "Descriptive", caption_length: str = "long",
        name: str = "", extra_options=None, custom_prompt: str = "", batch_size: int = 1, prompt_prefix: str = ""):
    if extra_options is None:
        extra_options = []

    if any(map(lambda x: not os.path.exists(x), files)):
        raise FileNotFoundError

    if not APP_STATE["caption_map"]:
        raise Exception("config.json -> captions.map cannot be undefined!")

    images = transform_images(files)
    options = extra_options if extra_options else []

    image_caption_list = process_captions(images, caption_type, caption_length, options, name, custom_prompt,
                                          batch_size)

    if not output or output == "json":
        for image_caption in image_caption_list:
            print(json.dumps({
                "prompt": image_caption["prompt"],
                "joycaption": image_caption["joycaption"]
            }))
    elif output == 'text':
        for file_idx in range(len(files)):
            caption_prefix: str = f"{prompt_prefix}, " if prompt_prefix else ""
            caption = caption_prefix + image_caption_list[file_idx]["joycaption"]
            filepath = Path(files[file_idx])
            filename = filepath.stem + ".txt"
            directory_path = filepath.parent
            with codecs.open(os.path.join(directory_path, filename), 'w', 'utf-8') as f:
                f.write(caption)


def process_captions(images, caption_type="", caption_length="", extra_options=None,
                     name="", custom_prompt="", batch_size: int = 1, max_new_tokens: int = 256,
                     temperature: float = 0.6, top_p: float = 0.9, debug_prompt: bool = False):
    if extra_options is None:
        extra_options = {}
    model_type = APP_STATE["model_type"]

    if model_type == "alpha":
        return caption_images(
            APP_STATE["tokenizer"], APP_STATE["text_model"], APP_STATE["clip_model"], APP_STATE["image_adapter"],
            images, caption_type, caption_length, extra_options, name,
            custom_prompt, APP_STATE["caption_map"], batch_size)
    elif model_type == "beta":
        try:
            # TODO: Define prompt length
            next_prompt = APP_STATE["caption_map"][caption_type][0] + custom_prompt
        except:
            next_prompt = custom_prompt or ""

        return model_beta.inference(
            APP_STATE["processor"],
            APP_STATE["text_model"],
            images,
            next_prompt,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            show_prompt=debug_prompt,
            batch_size=batch_size
        )
    else:
        raise Exception(f"unknown model type {model_type}")
