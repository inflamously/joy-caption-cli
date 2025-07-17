import os
import shutil

from quality.label_utils import create_label_folder


def store_images(label_path: str, source_image_path: str, target_image_path: str):
    create_label_folder(label_path)
    target_image_path = os.path.join(label_path, target_image_path)
    shutil.copyfile(source_image_path, target_image_path)
