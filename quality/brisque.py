import math
import os
import shutil

import PIL
import click
import numpy as np

from captions.images_query import query_images
from quality.label_utils import create_label_folder, increment_label_in_map, store_label_map
from quality.scoring import brisque_score_to_quality_label


@click.command("brisque")
@click.argument("folder")
@click.option("--output")
def brisque_check(folder, output):
    try:
        target_path = output if output and len(output) > 0 else folder

        image_paths = query_images(folder)
        images = [PIL.Image.open(image_path) for image_path in image_paths]
        images_features = [np.asarray(image) for image in images]
        from brisque.brisque import BRISQUE
        import tqdm
        bri = BRISQUE(url=False)
        quality_label_map = {}

        for idx in tqdm.trange(0, len(image_paths)):
            try:
                score = bri.multi_score(images_features[idx])
                label_brisque = brisque_score_to_quality_label(score)
                label = f"{label_brisque}".lower()

                print(f"Rating image at [{image_paths[idx]}] with a score of [{math.trunc(score)}]")

                quality_path = os.path.join(target_path, label)

                create_label_folder(quality_path)

                source_image_path = image_paths[idx]
                source_image_name = os.path.basename(source_image_path)
                target_image_path = os.path.join(quality_path, source_image_name)

                shutil.copyfile(source_image_path, target_image_path)
                quality_label_map = increment_label_in_map(quality_label_map, label)
            except Exception as e:
                print(f"Error processing image {idx}: {e}")
                quality_label_map["unclassified"] += 1
            store_label_map(quality_label_map, target_path)
    except Exception as e:
        print("Exception occured, cannot use brisque to validate image quality due to:", e)
