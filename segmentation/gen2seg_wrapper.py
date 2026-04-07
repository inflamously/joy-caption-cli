import time

import PIL.Image
import numpy as np
import torch

from captions.images_query import load_image
from segmentation.gen2seg.gen2seg_sd_pipeline import gen2segSDPipeline


def get_device(use_cuda=True):
    if use_cuda and not torch.cuda.is_available():
        print("Warning: CUDA is not available. Falling back to CPU.")
        use_cuda = False

    return "cuda" if use_cuda else "cpu"


def predict(pipe, image):
    print("Running inference...")
    with torch.no_grad():
        start_time = time.time()
        try:
            seg = pipe(image).prediction.squeeze()
            end_time = time.time()
            print(f"Inference completed in {end_time - start_time:.2f} seconds.")
            return seg
        except Exception as e:
            print(f"An error occurred during inference: {e}")
            return None


def get_model(use_cuda, model_name):
    device = get_device(use_cuda)

    print(f"Loading model '{model_name}'...")
    pipe = gen2segSDPipeline.from_pretrained(
        model_name,
        use_safetensors=True,
    ).to(device)
    print("Model loaded successfully.")

    return pipe


def segment_image(image: PIL.Image.Image, pipe):
    image_res = image.size
    seg = predict(pipe, image)
    return PIL.Image.fromarray(np.array(seg).astype(np.uint8)).resize(image_res, PIL.Image.LANCZOS)
