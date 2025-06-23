import json
import os


def store_label_map(label_map: dict, target_path: str):
    # Save quality distribution results
    with open(os.path.join(target_path, "quality_results.json"), "w") as f:
        json.dump(label_map, f)


def create_label_folder(path):
    if not os.path.exists(path):
        print(f"Creating folder at '{path}'")
        os.makedirs(path)


def increment_label_in_map(labelmap, label):
    if label in labelmap:
        labelmap[label] += 1
    else:
        labelmap[label] = 1
    return labelmap
