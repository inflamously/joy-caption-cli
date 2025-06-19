import json
import math
import os
import shutil

import PIL
import click
import numpy as np

from captions.images_query import query_images
from captions.joy.files import process_captions, transform_images
from initialization import setup_config
from model_selection import load_model, supported_joycaption_models


def score_to_quality_label(score: int) -> str:
    if score > 100:
        return "Rejected"
    elif score <= 0:
        return "Flawless"
    elif score <= 20:
        return "Excellent"
    elif score <= 40:
        return "Fair"
    elif score <= 60:
        return "Mediocre"
    elif score <= 80:
        return "Defective"
    else:
        return "Rejected"


@click.command("brisque_quality")
@click.argument("folder")
@click.option("--output")
def brisque_quality_check(folder, output):
    try:
        target_path = output if output and len(output) > 0 else folder

        image_paths = query_images(folder)
        images = [PIL.Image.open(image_path) for image_path in image_paths]
        from brisque.brisque import BRISQUE
        import tqdm
        bri = BRISQUE(url=False)
        quality_label_map = {}

        for idx in tqdm.trange(0, len(image_paths)):
            try:
                ndarr = np.asarray(images[idx])
                score = bri.score(ndarr)
                label = score_to_quality_label(score)

                print(f"Rating image at [{image_paths[idx]}] with a score of [{math.trunc(score)}]")

                quality_path = os.path.join(target_path, label)

                if not os.path.exists(quality_path):
                    print(f"Creating quality folder '{label}'")
                    os.makedirs(quality_path)

                source_image_path = image_paths[idx]
                source_image_name = os.path.basename(source_image_path)
                target_image_path = os.path.join(quality_path, source_image_name)

                shutil.copyfile(source_image_path, target_image_path)
                if label in quality_label_map:
                    quality_label_map[label] += 1
                else:
                    quality_label_map[label] = 1
            except Exception as e:
                print(f"Error processing image {idx}: {e}")
                quality_label_map["unclassified"] += 1
            # Save quality distribution results
            with open(os.path.join(target_path, "quality_results.json"), "w") as f:
                json.dump(quality_label_map, f)
    except Exception as e:
        print("Exception occured, cannot use brisque to validate image quality due to:", e)


@click.command("quality")
@click.argument("folder")
@click.argument("model_type", type=click.Choice(supported_joycaption_models()))
@click.option("--output")
@click.option("--batch_size", default=1)
def quality_check(folder: str, model_type: str, batch_size: int, output: str = ""):
    setup_config(model_type)
    load_model()
    image_paths = query_images(folder)
    images = transform_images(image_paths)

    # Using the vision model to assess quality instead of categorizing content
    quality_assessments = process_captions(
        images,
        custom_prompt="""
Analyze this image's technical quality only. Ignore the subject matter completely and focus exclusively on image fidelity. Classify into exactly ONE of these categories:
'low_quality': Visibly degraded image with any of these issues: pixelation, heavy compression artifacts, excessive noise/grain, significant blur, or very low resolution.
'standard_quality': Acceptable image with moderate technical issues: some noise, minor compression artifacts, or slight blur. Not pristine but serviceable.
'high_quality': Clear, sharp image with good resolution and minimal defects. Almost no noise or compression artifacts.
'premium_quality': Exceptional technical fidelity. Perfect sharpness, no visible noise, no compression artifacts, high resolution.
Focus solely on technical image quality defects. Output only one category name without explanation.
""",
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

    quality_destination_map = {
        "low_quality": 0,
        "standard_quality": 0,
        "high_quality": 0,
        "premium_quality": 0,
        "unclassified": 0,
    }

    for idx in range(len(quality_assessments)):
        try:
            assessment = quality_assessments[idx]
            quality_label = assessment["joycaption"].lower().strip()

            # Handle possible variations in model output
            if "low" in quality_label:
                quality_label = "low_quality"
            elif "standard" in quality_label:
                quality_label = "standard_quality"
            elif "high" in quality_label:
                quality_label = "high_quality"
            elif "premium" in quality_label:
                quality_label = "premium_quality"
            else:
                quality_label = "unclassified"

            quality_path = os.path.join(target_path, quality_label)

            if not os.path.exists(quality_path):
                print(f"Creating quality folder '{quality_label}'")
                os.makedirs(quality_path)

            source_image_path = image_paths[idx]
            source_image_name = os.path.basename(source_image_path)
            target_image_path = os.path.join(quality_path, source_image_name)

            shutil.copyfile(source_image_path, target_image_path)
            quality_destination_map[quality_label] += 1

        except Exception as e:
            print(f"Error processing image {idx}: {e}")
            quality_destination_map["unclassified"] += 1

    # Save quality distribution results
    with open(os.path.join(target_path, "quality_results.json"), "w") as f:
        json.dump(quality_destination_map, f)

    # Print summary
    print("\nQuality Classification Summary:")
    for quality, count in quality_destination_map.items():
        if count > 0:
            print(f"  {quality}: {count} images")
