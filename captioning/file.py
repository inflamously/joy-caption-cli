from typing import List

import click

from captioning.files import process_caption_files


@click.command('file')
@click.argument('file')
@click.option('--output', type=str)
@click.option('--name', default='')
@click.option('--caption_type', default='Descriptive')
@click.option('--caption_length', default='long')
@click.option('--extra_options', '-ex', multiple=True)
@click.option('--custom_prompt', default='')
def caption_file(
        file: str, output: str, caption_type: str, caption_length: str,
        name: str, extra_options: List[str], custom_prompt: str):
    _process_caption_file(file, output, caption_type, caption_length, name, extra_options, custom_prompt)

def _process_caption_file(
        file: str, output: str, caption_type: str = "Descriptive", caption_length: str = "long",
        name: str = "", extra_options=None, custom_prompt: str = ""):
    process_caption_files([file], output, caption_type, caption_length, name, extra_options, custom_prompt)
