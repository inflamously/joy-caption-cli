import codecs
import json
import os.path
from pathlib import Path

import PIL
import click
import torch

from tqdm import trange, tqdm
from captions.images_query import query_images
from initialization import setup_config
from model_facade.model_clip import inference
from model_selection import load_model
from state import APP_STATE


# File: folder.py
# Author: nflamously
# Original License: Apache License 2.0

@click.group("folder")
def folder():
    pass


@click.command('caption')
@click.argument('path')
@click.option('--output', type=str, default="text")
@click.option('--batch_size', default=1)
@click.option('--custom_prompt', default='')
@click.option('--prompt_prefix', default='')
def caption_folder(path: str, output: str, batch_size: int, custom_prompt: str, prompt_prefix: str):
    setup_config("clip")
    load_model()
    _caption_folder(path, output, custom_prompt, batch_size, prompt_prefix=prompt_prefix)


def _caption_folder(path: str, output: str, custom_prompt: str, batch_size: int, prompt_prefix: str = ""):
    files = query_images(path)
    images = [PIL.Image.open(image_path).convert('RGB') for image_path in tqdm(files, desc="Processing images")]

    image_caption_list = inference(APP_STATE["processor"],
              APP_STATE["text_model"], images, prompt=custom_prompt if custom_prompt else "a photograph of",
              batch_size=batch_size, show_prompt=True)

    if not output or output == "json":
        for image_caption in image_caption_list:
            print(json.dumps({
                "prompt": image_caption["prompt"],
                "caption": image_caption["caption"]
            }))
    elif output == 'text':
        for file_idx in range(len(files)):
            caption_prefix: str =  f"{prompt_prefix}, " if prompt_prefix else ""
            caption = caption_prefix + image_caption_list[file_idx]["caption"]
            filepath = Path(files[file_idx])
            filename = filepath.stem + ".txt"
            directory_path = filepath.parent
            with codecs.open(os.path.join(directory_path, filename), 'w', 'utf-8') as f:
                f.write(caption)


folder.add_command(caption_folder)
