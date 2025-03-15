import os.path

import PIL
import click

from tqdm import tqdm
from captions.images_query import query_images
import clip_interrogator

from captions.utils import break_list_into_chunks


# File: folder.py
# Author: nflamously
# Original License: Apache License 2.0

@click.command('folder')
@click.argument('path')
@click.option("--prefix", type=str, default="")
@click.option('--output', type=str, default="text")
@click.option('--batch_size', default=1)
def caption_folder(path: str, prefix:str, output: str, batch_size: int):
    _caption_folder(path, prefix, output, batch_size)


def _caption_folder(path: str, prefix: str, output: str, batch_size: int):
    images = [(image_path, PIL.Image.open(image_path).convert('RGB')) for image_path in tqdm(query_images(path))]
    config = clip_interrogator.Config(clip_model_name="ViT-L-14/openai", quiet=True)
    interrogator = clip_interrogator.Interrogator(config)
    captions = []
    for image_batch in tqdm(break_list_into_chunks(images, batch_size)):
        for path, image in image_batch:
            raw_caption = interrogator.generate_caption(image)
            caption_dict = {}
            caption_tags = raw_caption.split(" ")
            new_caption = prefix + ", " if len(prefix) > 0 else ""
            for tag in caption_tags:
                if tag not in caption_dict:
                    caption_dict[tag] = tag
            for _, v in caption_dict.items():
                new_caption += v + " "
            captions.append((path, new_caption))
    for target_path, caption in captions:
        filepath, fileextension = os.path.splitext(target_path)
        target_text_file = target_path[:-len(fileextension)] + ".txt"
        with open(target_text_file, "w") as f:
            f.write(caption)
