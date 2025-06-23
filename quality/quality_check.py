import click

from quality.label_check import label_check
from quality.brisque import brisque_check
from quality.organize import organize_folder
from quality.prompt_based import prompt_based_check


@click.group("quality")
def quality_check():
    pass


quality_check.add_command(organize_folder)
quality_check.add_command(prompt_based_check)
quality_check.add_command(brisque_check)
quality_check.add_command(label_check)
