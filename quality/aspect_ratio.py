import os
import shutil
from pathlib import Path

import PIL
import click

from captions.images_query import query_images, stream_image_files
from quality.image_utils import store_images
from quality.label_utils import store_label_map, create_label_folder, increment_label_in_map


@click.command("aspectratio")
@click.argument("folder")
@click.option("--walk_tree", is_flag=True)
@click.option("--stream_batch_size", type=int, default=1000)
def aspectratio(folder: str, walk_tree, stream_batch_size) -> None:
    try:
        PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True

        source_path = Path(folder)
        image_paths = query_images(folder, walk_tree)
        aspect_ratio_map = {
            "error": 0,
        }

        image_index = 0
        for batched_images, batched_paths in stream_image_files(image_paths, batch_size=stream_batch_size):
            for idx in range(len(batched_images)):
                image_index += 1
                try:
                    print(f"Determining aspect ratio for image {batched_paths[idx]}")
                    img = batched_images[idx]
                    w, h = img.width, img.height
                    if w > h:
                        h, w = w, h
                    ratio = str(round(w / float(h), 2))

                    label_path = os.path.join(source_path, ratio)
                    source_image_path = batched_paths[idx]
                    store_images(
                        label_path,
                        batched_paths[idx],
                        os.path.join(label_path, os.path.basename(source_image_path)),
                    )
                    aspect_ratio_map = increment_label_in_map(aspect_ratio_map, ratio)
                except Exception as e:
                    print(f"Error processing image {idx}: {e}")
                    aspect_ratio_map["error"] += 1
        store_label_map(aspect_ratio_map, source_path)
    except Exception as e:
        print("Exception occured, cannot use brisque to validate image quality due to:", e)
