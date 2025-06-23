import click
import tqdm

from nltk.tokenize import word_tokenize
from nltk import download
from nltk.corpus import stopwords
from captions.query_files import query_files
from quality.label_utils import increment_label_in_map, store_label_map


@click.command("labels")
@click.argument("folder")
@click.option("--output")
def label_check(folder, output):
    target_path = output if output and len(output) > 0 else folder

    download('punkt_tab')
    download('stopwords')

    text_files = query_files(folder, ['.txt'])
    if len(text_files) == 0:
        print("No caption files have been found.")
        return

    captions = []
    for text_file in text_files:
        with open(text_file, 'r', encoding='utf-8') as f:
            captions.append(f.readline().strip())

    if len(captions) == 0:
        print("No captions have been found.")
        return

    label_map = {}
    stop_words = stopwords.words('english')
    for caption in tqdm.tqdm(captions):
        caption_split = word_tokenize(caption)
        for label in caption_split:
            if label.isalpha() and label.lower() not in stop_words:
                increment_label_in_map(label_map, label)

    label_map_numbered = dict(sorted(label_map.items(), key=lambda item: item[1], reverse=True))

    store_label_map(label_map_numbered, target_path)
