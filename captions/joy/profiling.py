import click

from captions.joy.file import _process_caption_file
from model import load_models
from state import APP_STATE

# File: profiling.py
# Author: nflamously
# Original License: Apache License 2.0

@click.command('profile_image')
def caption_profile_image():
    load_models(APP_STATE['clip_model_name'], APP_STATE['checkpoint_path'])
    import cProfile, pstats
    profiler = cProfile.Profile(builtins=False)
    profiler.enable()
    _process_caption_file("../assets/example.png", "text")
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats()
