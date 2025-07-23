import click

from quality.aspect_ratio import aspectratio
from quality.compare import compare
from quality.label_check import label_check
from quality.brisque import brisque_check
from quality.pyiqa_metrics import pyiqa_metrics
from quality.organize import organize_folder
from quality.prompt_based import prompt_based_check
from quality.resolution import resolution


@click.group("quality")
def quality_check():
    pass


quality_check.add_command(organize_folder)
quality_check.add_command(prompt_based_check)
quality_check.add_command(brisque_check)
quality_check.add_command(pyiqa_metrics)
quality_check.add_command(label_check)
quality_check.add_command(aspectratio)
quality_check.add_command(compare)
quality_check.add_command(resolution)
