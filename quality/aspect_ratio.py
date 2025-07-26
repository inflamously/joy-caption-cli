import os
from pathlib import Path

import PIL
import click

from captions.images_query import query_images, stream_image_files
from quality.image_utils import store_image
from quality.label_utils import store_label_map, increment_label_in_map


@click.command("aspectratio")
@click.argument("folder")
@click.option("--walk_tree", is_flag=True)
@click.option("--root_folder_only", is_flag=True)
@click.option("--stream_batch_size", type=int, default=1000)
def aspectratio(folder: str, walk_tree, root_folder_only, stream_batch_size) -> None:
    try:
        source_path = Path(folder)
        image_paths = query_images(folder, walk_tree)

        if len(image_paths) == 0:
            raise Exception(f"No images found in {folder}")

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
                    source_image_label_path_filename = os.path.join(label_path, f"{image_index}_{os.path.basename(source_image_path)}")
                    store_image(
                        label_path,
                        batched_paths[idx],
                        source_image_label_path_filename,
                        copy_to_subfolder=not root_folder_only
                    )
                    aspect_ratio_map = increment_label_in_map(aspect_ratio_map, ratio)
                except Exception as e:
                    print(f"Error processing image {idx}: {e}")
                    aspect_ratio_map["error"] += 1
        store_label_map(aspect_ratio_map, source_path)
    except Exception as e:
        print("Exception occured due to:", e)
