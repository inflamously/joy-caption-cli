import click

@click.command("labels")
@click.argument("folder")
# @click.argument("model_type", type=click.Choice(supported_joycaption_models()))
@click.option("--output")
def label_check(folder, output):
    pass
