import click

from captioning.file import _process_caption_file


@click.command('profile_image')
def caption_profile_image():
    import cProfile, pstats
    profiler = cProfile.Profile(builtins=False)
    profiler.enable()
    _process_caption_file("../assets/example.png", "text")
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats()