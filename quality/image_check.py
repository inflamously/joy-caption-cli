from pathlib import Path

import click

from captions.images_query import query_images, stream_image_files


@click.command("image_check")
@click.argument("folder")
@click.option("--walk_tree", is_flag=True)
@click.option("--root_folder_only", is_flag=True)
@click.option("--stream_batch_size", type=int, default=1000)
def image_check(folder: str, walk_tree, root_folder_only, stream_batch_size) -> None:
    try:
        image_paths = query_images(folder, walk_tree)

        if len(image_paths) == 0:
            raise Exception(f"No images found in {folder}")

        images_left = len(image_paths)

        for batched_images, batched_paths in stream_image_files(image_paths, batch_size=stream_batch_size):
            for idx in range(len(batched_paths)):
                # print(f"Image processed {batched_paths[idx]}")
                images_left -= 1

        if images_left != 0:
            raise Exception(
                f"Error occurred processing images found in {folder} with flag --walk_tree={walk_tree} and --root_folder_only={root_folder_only}")
    except Exception as e:
        print("Exception occured at due to:", e)
