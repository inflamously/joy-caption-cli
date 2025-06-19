from typing import List

import click

from captions.joy.files import process_caption_files
from initialization import setup_config
from model_selection import load_model, supported_joycaption_models
from state import APP_STATE


# File: file.py
# Author: nflamously
# Original License: Apache License 2.0
@click.group("file")
def file():
    pass


@click.command("caption")
@click.argument("file")
@click.argument("model_type", type=click.Choice(supported_joycaption_models()))
@click.option("--output", type=str)
@click.option("--name", default="")
@click.option("--caption_type", default="Descriptive")
@click.option("--caption_length", default="long")
@click.option("--extra_options", "-ex", multiple=True)
@click.option("--custom_prompt", default="")
@click.option("--prompt_prefix", default="")
def caption_file(
    model_type: str,
    file: str,
    output: str,
    caption_type: str,
    caption_length: str,
    name: str,
    extra_options: List[str],
    custom_prompt: str,
    prompt_prefix: str,
):
    setup_config(model_type)
    load_model()
    _process_caption_file(
        file,
        output,
        caption_type,
        caption_length,
        name,
        extra_options,
        custom_prompt,
        prompt_prefix,
    )


def _process_caption_file(
    f: str,
    output: str,
    caption_type: str = "Descriptive",
    caption_length: str = "long",
    name: str = "",
    extra_options=None,
    custom_prompt: str = "",
    prompt_prefix: str = "",
):
    process_caption_files(
        [f],
        output,
        caption_type,
        caption_length,
        name,
        extra_options,
        custom_prompt,
        1,
        prompt_prefix,
    )


file.add_command(caption_file)
