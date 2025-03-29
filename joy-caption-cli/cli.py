import pathlib

import click

from captions.joy.file import caption_file as joy_caption_file
from captions.joy.folder import caption_folder as joy_caption_folder
from captions.joy.profiling import caption_profile_image as joy_caption_profile_image

from captions.clip.folder import caption_folder as clip_caption_folder

from config.load import load_config
from state import APP_STATE
from initialization import setup_config


# File: cli.py
# Author: nflamously
# Original License: Apache License 2.0

@click.group()
def cli():
    pass


@click.command()
def version():
    return "1.0"


@click.group('joycaption')
def caption():
    pass


@click.group('clip_caption')
def clip_caption():
    pass


if __name__ == '__main__':
    setup_config()

    # Joy Caption
    caption.add_command(joy_caption_file)
    caption.add_command(joy_caption_folder)
    caption.add_command(joy_caption_profile_image)

    # Stable Diffusion 1.5 -> CLIP
    clip_caption.add_command(clip_caption_folder)

    cli.add_command(clip_caption)
    cli.add_command(version)
    cli.add_command(caption)

    cli()
