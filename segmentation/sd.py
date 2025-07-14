import torch
from PIL import Image
import numpy as np
import time
import click
import cv2
import json

from segmentation.gen2seg.gen2seg_sd_pipeline import gen2segSDPipeline


@click.command("sd")
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--output_path', type=click.Path())
@click.option('--model_name', type=str, default='reachomk/gen2seg-sd')
@click.option('--use_cuda', type=bool, default=True)
@click.option('--output-format', type=click.Choice(['image', 'json']), default='json', help='The format of the output.')
def segment_image_sd(image_path, output_path, model_name, use_cuda, output_format):
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

        # Resize segmentation to original image resolution
        seg_image = Image.fromarray(np.array(seg).astype(np.uint8)).resize(orig_res, Image.LANCZOS)
        seg_array = np.array(seg_image)
        seg_array = cv2.cvtColor(seg_array, cv2.COLOR_BGR2GRAY)

        if output_format == 'image':
            print(f"Saving output image to '{output_path}'...")
            # TODO
            # seg_image.save(output_path)
            print("Output saved successfully.")
        
        elif output_format == 'json':
            print("Converting segmentation to bounding boxes...")
            # Find unique colors, ignoring black (background)
            unique_colors = np.unique(seg_array)
            unique_colors = unique_colors[unique_colors != 0]

            bounding_boxes = []
            for color in unique_colors:
                # Create a mask for the current color
                mask = np.uint8(seg_array == color) * 255
                
                # Find contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Combine all contours for the current color into a single one
                    all_points = np.concatenate(contours)
                    x, y, w, h = cv2.boundingRect(all_points)
                    bounding_boxes.append({
                        'label': int(color),
                        'box': [x, y, x + w, y + h] # top-left and bottom-right corners
                    })

            print(f"Saving bounding boxes to '{output_path}'...")
            with open(output_path, 'w') as f:
                json.dump(bounding_boxes, f, indent=4)
            print("Bounding boxes saved successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    segment_image_sd()
