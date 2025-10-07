import os

import click

from captions.images_query import query_images
from captions.joy.files import process_caption_files
from initialization import setup_config
from model_selection import load_model, supported_joycaption_models


# File: folder.py
# Author: nflamously
# Original License: Apache License 2.0


@click.group("folder")
def folder():
    pass


@click.command("caption")
@click.argument("path")
@click.argument("model_type", type=click.Choice(supported_joycaption_models()))
@click.option("--output", type=str, default="text")
@click.option("--name", default="")
@click.option("--caption_type", default="Descriptive")
@click.option("--caption_length", default="long")
@click.option("--extra_options", "-ex", multiple=True)
@click.option("--custom_prompt", default="")
@click.option("--batch_size", default=1)
@click.option("--prompt_prefix", default="")
@click.option("--prompt_suffix", default="")
@click.option("--temperature", default=0.6)
@click.option("--confidence_score", default=0.0)
def caption_folder(**kwargs):
    setup_config(kwargs["model_type"])
    load_model()
    process_caption_folder(**kwargs)


def process_caption_folder(**kwargs):
    if not os.path.exists(kwargs["path"]):
        raise Exception("Path does not exist")

    process_caption_files(images=query_images(kwargs["path"]), **kwargs)


folder.add_command(caption_folder)
