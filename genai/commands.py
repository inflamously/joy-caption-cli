import click

from genai.caption_check import caption_check


@click.group("genai")
def genai():
    pass


genai.add_command(caption_check)
