import os
import shutil
from pathlib import Path


def should_copy_subfolder_to_path(label_path: str, source_image_path: str):
    parent_a = Path(label_path).parent
    parent_b = Path(source_image_path).parent

    if parent_a.name == parent_b.name:
        return False
    else:
        return True


def store_images(label_path: str, source_image_path: str, target_image_path: str):
    if should_copy_subfolder_to_path(label_path, source_image_path):
        target_path_for_image = os.path.join(label_path, Path(source_image_path).parent.name,
                                             Path(target_image_path).name)
    else:
        target_path_for_image = os.path.join(label_path, target_image_path)
    os.makedirs(os.path.dirname(target_path_for_image), exist_ok=True)
    shutil.copyfile(source_image_path, target_path_for_image)
