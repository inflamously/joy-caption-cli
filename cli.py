import json
import os.path
import pathlib
from typing import Dict, Any, List

import PIL.Image
import click
from pydantic.v1 import PathNotExistsError
from tqdm import tqdm

from model import load_clip_model, load_vision_model, load_tokenizer, load_llm, load_image_adapter
from prompt_image import caption_image

# Global State
clip_model = None
tokenizer = None
text_model = None
image_adapter = None
captions_map = None


@click.group()
def cli():
    pass


@click.group('caption')
def caption():
    pass


def _process_caption_file(
        file: str, output: str, caption_type: str = "Descriptive", caption_length: str = "long",
        name: str = "", extra_options=None, custom_prompt: str = ""):
    if extra_options is None:
        extra_options = []
    if not os.path.exists(file):
        raise FileNotFoundError

    if not captions_map:
        raise Exception("config.json -> captions.map cannot be undefined!")

    img = PIL.Image.open(file).convert('RGB').resize((384, 384), PIL.Image.LANCZOS)

    prompt, caption = caption_image(
        tokenizer, text_model, clip_model, image_adapter, img, caption_type, caption_length,
        extra_options if extra_options else [], name, custom_prompt, captions_map)

    if not output:
        print("Prompt used:", prompt)
        print("Caption retrieved:", caption)
    else:
        if output == 'text':
            with open(file[:-4] + '.txt', 'w') as f:
                f.write(caption)


@click.command('file')
@click.argument('file')
@click.option('--output', type=str)
@click.option('--name', default='')
@click.option('--caption_type', default='Descriptive')
@click.option('--caption_length', default='long')
@click.option('--extra_options', '-ex', multiple=True)
@click.option('--custom_prompt', default='')
def caption_file(file: str, output: str, caption_type: str, caption_length: str,
                 name: str, extra_options: List[str], custom_prompt: str):
    _process_caption_file(file, output, caption_type, caption_length, name, extra_options, custom_prompt)


def _process_caption_folder(
        path: str, output: str, name: str, caption_type: str,
        caption_length: str, extra_options: List[str], custom_prompt: str):
    if not os.path.exists(path):
        raise PathNotExistsError

    images = []
    image_extensions = ['.jpg', '.jpeg', '.png']

    # Recurse all images and their paths.
    for root, dirs, files in os.walk(path):
        for file in files:
            for ext in image_extensions:
                if file.endswith(ext):
                    images.append(os.path.join(root, file))

    # Process Images
    idx = 1
    for img in tqdm(images, desc="captioning images"):
        _process_caption_file(img, output, caption_type, caption_length, name, extra_options, custom_prompt)
        idx += 1


@click.command('folder')
@click.argument('path')
@click.option('--output', type=str, default="text")
@click.option('--name', default='')
@click.option('--caption_type', default='Descriptive')
@click.option('--caption_length', default='long')
@click.option('--extra_options', '-ex', multiple=True)
@click.option('--custom_prompt', default='')
def caption_folder(path: str, output: str, name: str, caption_type: str, caption_length: str, extra_options: List[str],
                   custom_prompt: str):
    _process_caption_folder(path, output, name, caption_type, caption_length, extra_options, custom_prompt)


@click.command('profile_image')
def caption_profile_image():
    import cProfile, pstats
    profiler = cProfile.Profile(builtins=False)
    profiler.enable()
    _process_caption_file("./assets/example.png", "text")
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats()


def load_config() -> Dict[str, Any]:
    result = None
    with open('config.json', 'r') as jsonfile:
        result = json.load(jsonfile)
    return result


# Setup Models
def load_models():
    global clip_model, tokenizer, text_model, image_adapter

    clip_model = load_clip_model(clip_model_name)
    load_vision_model(checkpoint_path, clip_model)
    tokenizer = load_tokenizer(checkpoint_path)
    text_model = load_llm(checkpoint_path)
    image_adapter = load_image_adapter(checkpoint_path, clip_model, text_model)


if __name__ == '__main__':
    config = load_config()

    # Read Config
    checkpoint_path = pathlib.Path(config['checkpoint_path'])
    clip_model_name = config['clip_model']
    captions_map = config['captions']["map"]

    # Load Models
    load_models()

    # Application
    caption.add_command(caption_file)
    caption.add_command(caption_folder)
    caption.add_command(caption_profile_image)

    cli.add_command(caption)
    cli()
