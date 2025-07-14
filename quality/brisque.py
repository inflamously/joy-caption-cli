import math
import os
import shutil

import click
import numpy as np
from PIL import ImageFile
from brisque.brisque_implementation import BrisqueImplementation

from captions.images_query import query_images, stream_image_files
from quality.label_utils import create_label_folder, increment_label_in_map, store_label_map
from quality.scoring import brisque_score_to_quality_label


def get_label_from_score(score):
    label_brisque = brisque_score_to_quality_label(score)
    return f"{label_brisque}".lower()


@click.command("brisque")
@click.argument("folder")
@click.option("--output")
@click.option("--implementation", default=BrisqueImplementation.Pytorch)
@click.option("--walk_tree", is_flag=True)
@click.option("--stream_batch_size", type=int, default=1000)
def brisque_check(folder, output, implementation, stream_batch_size, walk_tree):
    try:
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        target_path = output if output and len(output) > 0 else folder
        image_paths = query_images(folder, walk_tree)

        from brisque.brisque import BRISQUE

        bri = BRISQUE(implementation)
        quality_label_map = {
            "unclassified": 0,
        }

        image_index = 0
        for batched_images, batched_paths in stream_image_files(image_paths, batch_size=stream_batch_size):
            features = [np.asarray(image) for image in batched_images]
            for idx in range(len(batched_images)):
                image_index += 1
                try:
                    score = bri.score(features[idx])
                    label = get_label_from_score(score)

                    print(f"Rating image at [{batched_paths[idx]}] with a score of [{math.trunc(score)}]")

                    quality_path = os.path.join(target_path, label)
                    create_label_folder(quality_path)

                    source_image_path = batched_paths[idx]
                    source_image_name = str(image_index) + "_" + os.path.basename(source_image_path)
                    target_image_path = os.path.join(quality_path, source_image_name)

                    shutil.copyfile(source_image_path, target_image_path)
                    quality_label_map = increment_label_in_map(quality_label_map, label)
                except Exception as e:
                    print(f"Error processing image {idx}: {e}")
                    quality_label_map["unclassified"] += 1
        store_label_map(quality_label_map, target_path)
    except Exception as e:
        print("Exception occured, cannot use brisque to validate image quality due to:", e)
