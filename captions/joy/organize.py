import json
import os.path
import shutil

import click
import pathvalidate

from captions.images_query import query_images
from captions.joy.files import process_captions, transform_images
from initialization import setup_config
from model_selection import supported_models, load_model


@click.command("organize")
@click.argument('folder')
@click.argument('model_type', type=click.Choice(supported_models()))
@click.option('--output')
@click.option('--batch_size', default=1)
def organize_folder(folder: str, model_type: str, batch_size: int, output: str = ""):
    setup_config(model_type)
    load_model()
    image_paths = query_images(folder)
    images = transform_images(image_paths)
    captions = process_captions(images,
                                custom_prompt="You are an expert classifier. Analyze the given image and describe it using only one word that reflects its high-level category or theme. Your output must be a single word with no additional commentary or explanation. Examples: woman, clothing, outdoors, comics, photography, costume, man, swimwear, animal, armor, transportation, architecture, city, cartoon, car, food, astronomy, modern art, cat, robot, landscape, dog, latex clothing, dragon, fantasy, sports car, post apocalyptic, photorealistic, game character, sci-fi..",
                                batch_size=batch_size,
                                max_new_tokens=5,
                                temperature=0.1,
                                top_p=0.9)

    if not captions or len(captions) == 0:
        return

    target_path = output if output and len(output) > 0 else folder

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    caption_destination_map = {}

    for idx in range(len(captions)):
        try:
            caption_map = captions[idx]
            caption = pathvalidate.sanitize_filename(caption_map["joycaption"].lower())
            caption_path = os.path.join(target_path, caption)
            if not os.path.exists(caption_path):
                # TODO: Add image count?
                print(f"creating folder '{caption}'")
                os.makedirs(caption_path)
                caption_destination_map[caption] = 0
            source_image_path = image_paths[idx]
            source_image_name = os.path.basename(source_image_path)
            target_image_path = os.path.join(caption_path, source_image_name)
            shutil.copyfile(source_image_path, target_image_path)
            caption_destination_map[caption] += 1
        except Exception as e:
            print(f"Error: {e}")

    with open(os.path.join(target_path, "organize_results.json"), "w") as f:
        json.dump(caption_destination_map, f)
