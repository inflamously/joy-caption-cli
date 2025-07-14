import os

import click

__REPO_PATH = os.path.join(os.path.dirname(__file__), 'gen2seg')


def has_gen2seg_installed():
    return os.path.exists(__REPO_PATH)


from segmentation.sd import segment_image_sd


@click.group("segmentation")
def segmentation():
    pass


segmentation.add_command(segment_image_sd)
