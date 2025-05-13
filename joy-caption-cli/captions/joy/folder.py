import os
import click

from captions.images_query import query_images
from captions.joy.files import process_caption_files
from model_selection import load_models
from state import APP_STATE


# File: folder.py
# Author: nflamously
# Original License: Apache License 2.0

@click.command('folder')
@click.argument('path')
@click.option('--output', type=str, default="text")
@click.option('--name', default='')
@click.option('--caption_type', default='Descriptive')
@click.option('--caption_length', default='long')
@click.option('--extra_options', '-ex', multiple=True)
@click.option('--custom_prompt', default='')
@click.option('--batch_size', default=1)
def caption_folder(
        path: str, output: str, name: str, caption_type: str, caption_length: str,
        extra_options: list[str], custom_prompt: str, batch_size: int):
    # Load Models on captioning
    load_models(APP_STATE['clip_model_name'], APP_STATE['checkpoint_path'], APP_STATE['model_type'])
    process_caption_folder(path, output, name, caption_type, caption_length, extra_options, custom_prompt, batch_size)


def process_caption_folder(
        path: str, output: str, name: str, caption_type: str, caption_length: str,
        extra_options: list[str], custom_prompt: str, batch_size: int = 1):
    if not os.path.exists(path):
        raise Exception("Path does not exist")

    images = query_images(path)

    # Process Images
    process_caption_files(images, output, caption_type, caption_length, name, extra_options, custom_prompt, batch_size)
