import torch
from PIL import Image
import numpy as np
import time
import click
from transformers import AutoModel

from segmentation.gen2seg.gen2seg_sd_pipeline import gen2segSDPipeline


@click.command("sd")
@click.argument('image_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--model_name', type=str, default='reachomk/gen2seg-sd')
@click.option('--use_cuda', type=bool, default=True)
def segment_image_sd(image_path, output_path, model_name, use_cuda):
    """
    Generates an instance segmentation map for a given image using a pretrained gen2seg model.
    """
    try:
        if use_cuda and not torch.cuda.is_available():
            print("Warning: CUDA is not available. Falling back to CPU.")
            use_cuda = False

        device = "cuda" if use_cuda else "cpu"

        print(f"Loading model '{model_name}'...")
        pipe = gen2segSDPipeline.from_pretrained(
            model_name,
            use_safetensors=True,
        ).to(device)
        print("Model loaded successfully.")

        print(f"Loading image from '{image_path}'...")
        image = Image.open(image_path).convert("RGB")
        orig_res = image.size

        print("Running inference...")
        with torch.no_grad():
            start_time = time.time()
            try:
                seg = pipe(image).prediction.squeeze()
            except Exception as e:
                print(f"An error occurred during inference: {e}")
                return
            end_time = time.time()
        print(f"Inference completed in {end_time - start_time:.2f} seconds.")
        print(f"Saving output to '{output_path}'...")
        seg_array = np.array(seg).astype(np.uint8)
        Image.fromarray(seg_array).resize(orig_res, Image.LANCZOS).show()#.save(output_path)
        print("Output saved successfully.")
    except Exception as e:
        print(f"An error occurred during inference within gen2seq: {e}")


if __name__ == '__main__':
    segment_image_sd()
