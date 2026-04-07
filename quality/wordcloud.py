import json
import os
import click
import tqdm
import wordcloud

from captions.images_query import query_text


@click.command("wordcloud")
@click.argument("folder")
@click.option("--walk_tree", is_flag=True)
def wordcloud_builder(folder, walk_tree):
    try:
        text_paths = query_text(folder, walk_tree)
        story = ""
        for text_path in tqdm.tqdm(text_paths):
            with open(text_path, "r", encoding="utf-8") as f:
                for line in f:
                    story = story + " " + line
        cloud = wordcloud.WordCloud(width=2048, height=1024, max_font_size=100, min_font_size=10,
                                    background_color='white').generate(story)
        cloud.to_image().save(os.path.join(folder, "wordcloud.png"))
        words = {k: v * 100 for k, v in cloud.words_.items()}
        with open(os.path.join(folder, "wordcloud.json"), "w", encoding="utf-8") as f:
            f.write(json.dumps(words, indent=4))
    except Exception as e:
        print("Exception occured:", e)
