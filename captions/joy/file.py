import click

from captions.joy.files import process_caption_files
from initialization import setup_config
from model_selection import load_model, supported_joycaption_models


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
@click.option("--temperature", default=0.6)
def caption_file(**kwargs):
    setup_config(kwargs["model_type"])
    load_model()
    process_caption_file(**kwargs)


def process_caption_file(**kwargs):
    process_caption_files(images=[kwargs["file"]], **kwargs)


file.add_command(caption_file)
