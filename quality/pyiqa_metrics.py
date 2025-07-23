import math
import shutil

import click
import numpy as np
import tqdm
from PIL import ImageFile

from captions.images_query import query_images, stream_image_files
from quality.data_category_score import score_to_quality_label_pyiqa
from quality.label_utils import create_label_folder, increment_label_in_map, store_label_map


def process_brisque_score(score: float) -> float:
    norm_score = score / 100.0  # Make it 0-1
    return 1 - norm_score  # Invert score from 0.3 to -> 0.7 (Higher better)


# Good weights: [0.0, 1.0, 0.5]
def get_label_from_scores(brisque, maniqa, dbcnn):
    brisque_norm_score = process_brisque_score(brisque)
    final_score = np.average([brisque_norm_score, maniqa, dbcnn], weights=[0.0, 1.0, 0.1]) * 100  # Rescale to 0-100
    return score_to_quality_label_pyiqa(math.trunc(final_score)), final_score  # Truncate for proper score.


@click.command("pyiqa_metrics")
@click.argument("folder")
@click.option("--output")
@click.option("--walk_tree", is_flag=True)
@click.option("--stream_batch_size", type=int, default=1000)
def pyiqa_metrics(folder, output, stream_batch_size, walk_tree):
    try:
        # !/usr/bin/env python3
        import cv2, os, pandas as pd, pyiqa, torch
        from pathlib import Path

        # ---------- config ----------
        # METRICS = ["brisque", "maniqa", "dbcnn"]  # any PyIQA NR metric
        METRICS = ["maniqa", "dbcnn"]  # any PyIQA NR metric
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # ----------------------------

        # create metric objects
        iqa = {m: pyiqa.create_metric(m, device=DEVICE) for m in METRICS}

        ImageFile.LOAD_TRUNCATED_IMAGES = True

        target_path = output if output and len(output) > 0 else folder
        image_paths = query_images(folder, walk_tree)

        quality_label_map = {
            "unclassified": 0,
        }

        image_index = 0
        for batched_images, batched_paths in stream_image_files(image_paths, batch_size=stream_batch_size):
            for idx in tqdm.trange(len(batched_images)):
                image_index += 1
                try:
                    scores = np.array([0.0])

                    for m in METRICS:
                        scores = np.append(scores, float(iqa[m](batched_images[idx]).item()))

                    label, score = get_label_from_scores(scores[0], scores[1], scores[2])

                    print(f"Rating image at [{batched_paths[idx]}] with a scores of [{score}]")

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
