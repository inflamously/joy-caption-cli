import click

from captions.joy.file import _process_caption_file
from initialization import setup_config
from model_selection import load_model, supported_joycaption_models

# File: profiling.py
# Author: nflamously
# Original License: Apache License 2.0


@click.command("profile_image")
@click.argument("model_type", type=click.Choice(supported_joycaption_models()))
def caption_profile_image(model_type: str):  #
    setup_config(model_type)
    load_model()
    import cProfile
    import pstats

    profiler = cProfile.Profile(builtins=False)
    profiler.enable()
    _process_caption_file("../assets/example.png", "text")
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("tottime")
    stats.print_stats()
