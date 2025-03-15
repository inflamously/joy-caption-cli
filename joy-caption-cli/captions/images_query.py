import os

# File: images_query.py
# Author: nflamously
# Original License: Apache License 2.0

def query_images(path) -> list[str | bytes]:
    images = []
    image_extensions = ['.jpg', '.jpeg', '.png']

    # Recurse all images and their paths.
    for root, dirs, files in os.walk(path):
        for file in files:
            for ext in image_extensions:
                if file.endswith(ext):
                    images.append(os.path.join(root, file))

    return images
