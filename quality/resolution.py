import math
import os
from pathlib import Path

import PIL
import click
import numpy as np
from PIL import ImageFile

from captions.images_query import query_images, stream_image_files
from quality.data_category_score import score_to_quality_label_pyiqa
from quality.image_utils import store_image
from quality.label_utils import create_label_folder, increment_label_in_map, store_label_map


def process_brisque_score(score: float) -> float:
    norm_score = score / 100.0  # Make it 0-1
    return 1 - norm_score  # Invert score from 0.3 to -> 0.7 (Higher better)


# Good weights: [0.0, 1.0, 0.5]
def get_label_from_scores(brisque, maniqa, dbcnn):
    brisque_norm_score = process_brisque_score(brisque)
    final_score = np.average([brisque_norm_score, maniqa, dbcnn], weights=[0.0, 1.0, 0.1]) * 100  # Rescale to 0-100
    return score_to_quality_label_pyiqa(math.trunc(final_score)), final_score  # Truncate for proper score.


@click.command("resolution")
@click.argument("folder")
@click.option("--walk_tree", is_flag=True)
@click.option("--root_folder_only", is_flag=True)
@click.option("--stream_batch_size", type=int, default=1000)
def resolution(folder: str, walk_tree, root_folder_only, stream_batch_size) -> None:
    try:
        PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True

        source_path = Path(folder)
        image_paths = query_images(folder, walk_tree)

        if len(image_paths) == 0:
            raise Exception(f"No images found in {folder}")

        resolution_map = {
            "error": 0,
        }

        image_index = 0
        for batched_images, batched_paths in stream_image_files(image_paths, batch_size=stream_batch_size):
            for idx in range(len(batched_images)):
                image_index += 1
                try:
                    img = batched_images[idx]
                    w, h = img.width, img.height
                    res_list = [512, 768, 1024, 1280, 1536, 2048, 4096, 8192]
                    resolution = None
                    for res_item in res_list:
                        if w < res_item:
                            resolution = f"{res_item}x"
                            break
                        elif h < res_item:
                            resolution = f"x{res_item}"
                            break

                    if not resolution:
                        resolution = "NxN"

                    print(f"Determined resolution bucket {resolution} for image {batched_paths[idx]}")

                    label_path = os.path.join(source_path, resolution)
                    source_image_path = batched_paths[idx]
                    source_image_label_path_filename = os.path.join(label_path,
                                                                    f"{image_index}_{os.path.basename(source_image_path)}")
                    store_image(
                        label_path,
                        batched_paths[idx],
                        source_image_label_path_filename,
                        copy_to_subfolder=not root_folder_only
                    )
                    resolution_map = increment_label_in_map(resolution_map, resolution)
                except Exception as e:
                    print(f"Error processing image {idx}: {e}")
                    resolution_map["error"] += 1
        store_label_map(resolution_map, source_path)
    except Exception as e:
        print("Exception occured due to:", e)
