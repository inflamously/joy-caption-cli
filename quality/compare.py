import os
import shutil
from pathlib import Path

import PIL
import click
import cv2
import numpy as np
import torch

from captions.images_query import query_images, stream_image_files
from quality.label_utils import store_label_map, increment_label_in_map, create_label_folder


@torch.no_grad()
def phash_tensor(image):
    # 1. Load 32×32 grayscale [32,32]
    image = image.convert('L').resize((32, 32), PIL.Image.Resampling.LANCZOS)
    x = torch.from_numpy(np.array(image, dtype=np.float32)) / 255.0

    # 2. 2-D DCT via FFT
    def dct2d(img_array):
        # rows
        X1 = torch.fft.fft(img_array, dim=-1)
        X1.real[..., 0] *= 0.5 ** 0.5
        X1.real[..., 1:] *= 2 ** 0.5
        X1.imag.zero_()
        # cols
        X2 = torch.fft.fft(X1.real, dim=-2)
        X2.real[..., 0, :] *= 0.5 ** 0.5
        X2.real[..., 1:, :] *= 2 ** 0.5
        return X2.real

    dct = dct2d(x)[:8, :8]  # keep 8×8 low-freq
    med = dct.median()
    bits = (dct > med).flatten()  # 64 bools
    return bits


def phash_compare(img_a: PIL.Image.Image, img_b: PIL.Image.Image):
    h1 = phash_tensor(img_a)
    h2 = phash_tensor(img_b)
    hash_scores = [(h1 != h2).sum().item()]
    return np.array(hash_scores).mean()


@click.command("compare")
@click.argument("folder")
@click.option("--walk_tree", is_flag=True)
@click.option("--stream_batch_size", type=int, default=1000)
def compare(folder: str, walk_tree, stream_batch_size) -> None:
    try:
        source_path = Path(folder)
        image_paths = query_images(folder, walk_tree)
        similiarity_map = {
            "similiar": 0,
            "error": 0,
        }

        image_index = 0
        for batched_images, batched_paths in stream_image_files(image_paths, batch_size=stream_batch_size):
            similiar_images = []
            for idx_x in range(len(batched_images)):
                for idx_y in range(len(batched_images)):
                    if idx_x == idx_y:
                        continue  # Ignore same image.
                    try:
                        distance = phash_compare(batched_images[idx_x], batched_images[idx_y])
                        print(
                            f"Comparing images {batched_paths[idx_x]} and {batched_paths[idx_y]} with score \"{distance}\"")
                        if distance <= 15:
                            label = "similiar"
                            label_path = os.path.join(source_path, label)
                            source_image_name = f" {Path(batched_paths[idx_x]).name}"
                            similiar_images.append({
                                "source": batched_paths[idx_x],
                                "target": os.path.join(label_path, source_image_name)
                            })
                        else:
                            label = "unique"
                        increment_label_in_map(similiarity_map, label)
                    except Exception as e:
                        print(f"Error processing image {idx_x}: {e}")
                        similiarity_map["error"] += 1
                image_index += 1
            if similiarity_map["similiar"] > 0:
                create_label_folder(os.path.join(source_path, "similiar"))
            for image_path in similiar_images:
                shutil.copyfile(image_path["source"], image_path["target"])
        store_label_map(similiarity_map, folder)
    except Exception as e:
        print("Exception occured due to:", e)
