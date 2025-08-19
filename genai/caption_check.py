import os.path
from pathlib import Path

import click
import tqdm

from captions.images_query import query_text
from genai.models.model_loader import load_model


@click.command("caption_check")
@click.argument("folder")
@click.argument("model", type=click.Choice(["flux.dev", "flux.schnell"]))
@click.option("--walk_tree", is_flag=True)
@click.option("--interactive", is_flag=False)
def caption_check(folder: str, model: str, walk_tree, interactive) -> None:
    try:
        print(f"Loading model {model} for caption checking.")
        wrapper = load_model(model)
        if not wrapper:
            raise Exception(f"No model loaded for {model}")

        text_files = query_text(folder, walk_tree)

        if len(text_files) == 0:
            raise Exception(f"No text files found in {folder}")

        target_image_folder = os.path.join(folder, f"caption_examples_{model}")

        for text_file in tqdm.tqdm(text_files, desc="Prompting captions:"):
            with open(text_file, "r", encoding="utf-8") as f:
                text_file_path = Path(text_file)
                prompt = f.readline()

                # Skip in case image exists
                image_path = os.path.join(target_image_folder, f"{text_file_path.stem}.png")
                if os.path.exists(image_path):
                    print(f"Skipping {image_path} as it already exists.")
                    continue

                images = wrapper.generate(
                    prompt=prompt,
                    **wrapper.config()
                )

                if not os.path.exists(target_image_folder):
                    os.makedirs(target_image_folder)

                if isinstance(images, list):
                    for idx in range(len(images)):
                        images[idx].save(os.path.join(target_image_folder, f"{text_file_path.stem}_{idx}.png"))
                else:
                    images.save(image_path)

                if not prompt or len(prompt) == 0:
                    raise Exception(f"No text file or proper prompt found in {text_file}")
    except Exception as e:
        print("Exception occured at due to:", e)
