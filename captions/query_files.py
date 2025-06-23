import os


def query_files(path: str, extensions: list) -> list:
    results = []
    for root, dirs, files in os.walk(path):
        for file in files:
            for ext in extensions:
                if file.endswith(ext):
                    results.append(os.path.join(root, file))
    return results