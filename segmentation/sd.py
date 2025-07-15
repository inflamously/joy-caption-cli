import os.path
from pathlib import Path
from typing import List

import PIL
import click
import tqdm

from captions.images_query import load_image, query_images, stream_image_files
from segmentation.gen2seg_wrapper import segment_image, get_model
from segmentation.operations import kmeans_color_distance, unique_color_bboxes, draw_bboxes, nms_color_boxes, \
    bboxes_to_points, draw_points, aspect_ratio_bboxes_from_points, multi_crop_images_from_bboxes


def matrix_crop_image(img: PIL.Image.Image, pipe, threshold=20, kmeans_iter=20, min_clusters=2) -> List[
    PIL.Image.Image]:
    seq_image = segment_image(img, pipe)
    seq_image, centers, labels = kmeans_color_distance(seq_image, threshold, kmeans_iter, min_clusters)
    bboxes = unique_color_bboxes(seq_image)
    points = bboxes_to_points(nms_color_boxes(bboxes))
    bboxes = aspect_ratio_bboxes_from_points(points, aspect_ratios=1 / 1, size=1024)
    return multi_crop_images_from_bboxes(img, bboxes)


def save_crops_to_folder(crops, image_path):
    try:
        image_path_item = Path(image_path)
        target_directory = os.path.join(image_path_item.parent, "segmentation")
        if not os.path.exists(target_directory):
            os.mkdir(target_directory)
        idx = 1
        for crop in crops:
            target_image_path = os.path.join(target_directory, f"{idx}_{image_path_item.name}")
            crop.save(target_image_path)
            idx += 1
    except Exception as e:
        print(f"An error occurred: {e}")


@click.group("sd")
def sd():
    pass


@click.command("single")
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--model_name', type=str, default='reachomk/gen2seg-sd')
@click.option('--use_cuda', type=bool, default=True)
@click.option('--output-format', type=click.Choice(['image', 'json']), default='image',
              help='The format of the output.')
def single_segment_image(image_path, model_name, use_cuda, output_format):
    try:
        orig_image = load_image(image_path)
        pipe = get_model(use_cuda, model_name)
        crops = matrix_crop_image(img=orig_image, pipe=pipe)
        if output_format == 'image':
            save_crops_to_folder(crops, image_path)
        else:
            pass
    except Exception as e:
        print(f"An error occurred: {e}")


@click.command("multi")
@click.argument('path', type=click.Path(exists=True))
@click.option('--model_name', type=str, default='reachomk/gen2seg-sd')
@click.option('--use_cuda', type=bool, default=True)
@click.option('--output-format', type=click.Choice(['image', 'json']), default='image', help='The format of the output.')
def multi_segment_image(path, model_name, use_cuda, output_format):
    images = query_images(path)
    pipe = get_model(use_cuda, model_name)
    try:
        for batched_images, batched_paths in stream_image_files(images):
            for idx in tqdm.trange(0, len(batched_paths), 1, desc="Processing batched images"):
                crops = matrix_crop_image(img=batched_images[idx], pipe=pipe)
                if output_format == 'image':
                    save_crops_to_folder(crops, batched_paths[idx])
                else:
                    pass
    except Exception as e:
        print(f"An error occurred: {e}")


sd.add_command(single_segment_image)
sd.add_command(multi_segment_image)
