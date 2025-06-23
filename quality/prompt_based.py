import os
import shutil

import click

from captions.images_query import query_images
from captions.joy.files import transform_images, process_captions
from initialization import setup_config
from model_selection import load_model, supported_joycaption_models
from quality.label_utils import create_label_folder, increment_label_in_map, store_label_map
from quality.scoring import prompt_score_to_quality_label


def custom_prompt():
    return """
Analyze this image's technical quality only. Ignore the subject matter completely and focus exclusively on image fidelity. Classify into exactly ONE of these categories:
'low_quality': Visibly degraded image with any of these issues: pixelation, heavy compression artifacts, excessive noise/grain, significant blur, or very low resolution.
'standard_quality': Acceptable image with moderate technical issues: some noise, minor compression artifacts, or slight blur. Not pristine but serviceable.
'high_quality': Clear, sharp image with good resolution and minimal defects. Almost no noise or compression artifacts.
'premium_quality': Exceptional technical fidelity. Perfect sharpness, no visible noise, no compression artifacts, high resolution.
Focus solely on technical image quality defects. Output only one category name without explanation.
"""


@click.command("prompt_based")
@click.argument("folder")
@click.argument("model_type", type=click.Choice(supported_joycaption_models()))
@click.option("--output")
@click.option("--batch_size", default=1)
def prompt_based_check(folder: str, model_type: str, batch_size: int, output: str = ""):
    setup_config(model_type)
    load_model()
    image_paths = query_images(folder)
    images = transform_images(image_paths)

    # Using the vision model to assess quality instead of categorizing content
    quality_assessments = process_captions(
        images,
        custom_prompt=custom_prompt(),
        batch_size=batch_size,
        max_new_tokens=10,
        temperature=0,
        top_p=0.9,
    )

    if not quality_assessments or len(quality_assessments) == 0:
        return

    target_path = output if output and len(output) > 0 else folder

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    quality_destination_map = {}

    for idx in range(len(quality_assessments)):
        try:
            assessment = quality_assessments[idx]
            quality_label = prompt_score_to_quality_label(assessment["joycaption"].lower().strip())
            # Handle possible variations in model output

            quality_path = os.path.join(target_path, quality_label)
            create_label_folder(quality_path)

            source_image_path = image_paths[idx]
            source_image_name = os.path.basename(source_image_path)
            target_image_path = os.path.join(quality_path, source_image_name)

            shutil.copyfile(source_image_path, target_image_path)
            quality_destination_map = increment_label_in_map(quality_destination_map, quality_label)
        except Exception as e:
            print(f"Error processing image {idx}: {e}")
            quality_destination_map = increment_label_in_map(quality_destination_map, "unclassified")

    store_label_map(quality_destination_map, target_path)

    # Print summary
    print("\nQuality Classification Summary:")
    for quality, count in quality_destination_map.items():
        if count > 0:
            print(f"  {quality}: {count} images")
