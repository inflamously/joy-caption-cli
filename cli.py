import pathlib
import click
from tqdm import tqdm


from captioning.file import caption_file
from captioning.folder import caption_folder
from captioning.profiling import caption_profile_image
from config.load import load_config
from model import load_clip_model, load_vision_model, load_tokenizer, load_llm, load_image_adapter
from state import APP_STATE


@click.group()
def cli():
    pass


@click.group('caption')
def caption():
    pass


# Setup Models
def load_models(clip_model_name: str):
    APP_STATE["clip_model"] = load_clip_model(clip_model_name)
    load_vision_model(checkpoint_path, APP_STATE["clip_model"])
    APP_STATE["tokenizer"] = load_tokenizer(checkpoint_path)
    APP_STATE["text_model"] = load_llm(checkpoint_path)
    APP_STATE["image_adapter"] = load_image_adapter(checkpoint_path, APP_STATE["clip_model"], APP_STATE["text_model"])


if __name__ == '__main__':
    config = load_config()

    # Read Config
    checkpoint_path = pathlib.Path(config['checkpoint_path'])
    APP_STATE["caption_map"] = config['captions']["map"]

    # Load Models
    load_models(config['clip_model'])

    # Application
    caption.add_command(caption_file)
    caption.add_command(caption_folder)
    caption.add_command(caption_profile_image)

    cli.add_command(caption)
    cli()
