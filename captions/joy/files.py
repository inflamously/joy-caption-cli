import codecs
import json
import os
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import List

import PIL
from pandas.core.window.doc import kwargs_scipy
from tqdm import tqdm

from model_facade import model_beta
from prompt_image import caption_images
from state import APP_STATE


# File: files.py
# Author: nflamously
# Original License: Apache License 2.0


def transform_image(file: str):
    return PIL.Image.open(file).convert("RGB").resize((384, 384), PIL.Image.LANCZOS)


def transform_images(files: List[str]):
    print("processing images")
    with ThreadPoolExecutor() as executor:
        images = list(tqdm(executor.map(transform_image, files), total=len(files)))
    return images


def process_caption_files(**kwargs):
    files = kwargs.get("images")
    output = kwargs.get("output") or ""
    prompt_prefix = kwargs.get("prompt_prefix") or ""
    prompt_suffix = kwargs.get("prompt_suffix") or ""
    confidence_score = kwargs.get("confidence_score") or 0.0

    if any(map(lambda file: not os.path.exists(file), files)):
        raise FileNotFoundError

    if not APP_STATE["caption_map"]:
        raise Exception("config.json -> captions.map cannot be undefined!")

    images = transform_images(files)

    image_caption_list = process_captions(
        images=images,
        caption_type=kwargs.get("caption_type") or "Descriptive",
        caption_length=kwargs.get("caption_length") or "long",
        extra_options=kwargs.get("extra_options") or [],
        name=kwargs.get("name") or "",
        custom_prompt=kwargs.get("custom_prompt") or "",
        batch_size=kwargs.get("batch_size") or 1,
        temperature=kwargs.get("temperature") or 0.6,
        confidence_threshold=confidence_score,
        return_confidence_scores=confidence_score > 0.0
    )

    if not output or output == "json":
        for image_caption in image_caption_list:
            print(
                json.dumps(
                    {
                        "prompt": image_caption["prompt"],
                        "joycaption": image_caption["joycaption"],
                    }
                )
            )
    elif output == "text":
        for file_idx in range(len(files)):
            caption_prefix: str = f"{prompt_prefix}, " if prompt_prefix else ""
            caption_suffix: str = f", {prompt_suffix}" if prompt_suffix else ""
            caption = caption_prefix + image_caption_list[file_idx]["joycaption"] + caption_suffix
            caption = caption.replace(".", "") + "." if caption_suffix else caption
            filepath = Path(files[file_idx])
            filename = filepath.stem + ".txt"
            directory_path = filepath.parent
            with codecs.open(os.path.join(directory_path, filename), "w", "utf-8") as f:
                f.write(caption)


def process_captions(**kwargs):
    print(f"processing captions with arguments:\n({kwargs})")

    images = kwargs.get("images")
    caption_type = kwargs.get("caption_type") or "Descriptive"
    caption_length = kwargs.get("caption_length") or "long"
    extra_options = kwargs.get("extra_options") or []
    name = kwargs.get("name") or ""
    custom_prompt = kwargs.get("custom_prompt") or ""
    batch_size = kwargs.get("batch_size") or 1
    max_new_tokens = kwargs.get("max_new_tokens") or 256
    temperature = kwargs.get("temperature") or 0.6
    top_p = kwargs.get("top_p") or 0.9
    debug_prompt = kwargs.get("debug_prompt") or False
    confidence_threshold = kwargs.get("confidence_score") or 0.0
    return_confidence_scores = confidence_threshold > 0.0

    if extra_options is None:
        extra_options = {}
    model_type = APP_STATE["model_type"]

    if model_type == "alpha":
        return caption_images(
            APP_STATE["tokenizer"],
            APP_STATE["text_model"],
            APP_STATE["clip_model"],
            APP_STATE["image_adapter"],
            images,
            caption_type,
            caption_length,
            extra_options,
            name,
            custom_prompt,
            APP_STATE["caption_map"],
            batch_size,
        )
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
            batch_size=batch_size,
            confidence_threshold=confidence_threshold,
            return_confidence_scores=return_confidence_scores
        )
    else:
        raise Exception(f"unknown model type {model_type}")
