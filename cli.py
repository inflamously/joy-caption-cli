import pathlib
import click

from captioning.file import caption_file
from captioning.folder import caption_folder
from captioning.profiling import caption_profile_image
from config.load import load_config
from state import APP_STATE


@click.group()
def cli():
    pass


@click.group('caption')
def caption():
    pass


if __name__ == '__main__':
    config = load_config('config/config.json')

    # Read Config
    APP_STATE["checkpoint_path"] = pathlib.Path(config['checkpoint_path'])
    APP_STATE["caption_map"] = config['captions']["map"]
    APP_STATE["clip_model_name"] = config['clip_model']

    # Application
    caption.add_command(caption_file)
    caption.add_command(caption_folder)
    caption.add_command(caption_profile_image)
    cli.add_command(caption)

    cli()
