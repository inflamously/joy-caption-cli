from captions.query_files import query_files


# File: images_query.py
# Author: nflamously
# Original License: Apache License 2.0


def query_images(path) -> list[str | bytes]:
    return query_files(path, [".jpg", ".jpeg", ".png", ".webp"])
