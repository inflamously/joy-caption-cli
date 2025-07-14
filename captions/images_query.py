from captions.query_files import query_files, query_root_files


# File: images_query.py
# Author: nflamously
# Original License: Apache License 2.0


def query_images(path, walk_tree=True) -> list[str | bytes]:
    extensions = [".jpg", ".jpeg", ".png", ".webp"]
    if walk_tree:
        return query_files(path, extensions)
    else:
        return query_root_files(path, extensions)
