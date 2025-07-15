from typing import Any, Generator

import PIL
import tqdm

from captions.query_files import query_files, query_root_files


# File: images_query.py
# Author: nflamously
# Original License: Apache License 2.0

def load_image(path):
    return PIL.Image.open(path)


def query_images(path, walk_tree=True) -> list[str]:
    extensions = [".jpg", ".jpeg", ".png", ".webp"]
    if walk_tree:
        return query_files(path, extensions)
    else:
        return query_root_files(path, extensions)


def stream_image_files(paths: list[str], batch_size=128) -> Generator[tuple[list[Any], list[str]], Any, None]:
    for idx in tqdm.trange(0, len(paths), batch_size, desc="Processing batched images"):
        batch = paths[idx: idx + batch_size]
        images_to_be_processed = []
        for batch_item in batch:
            try:
                images_to_be_processed.append(PIL.Image.open(batch_item))
            except Exception as e:
                print(f"Exception \"{e}\" occured while reading image:", batch_item)
        yield images_to_be_processed, batch
