import os


def query_files(path: str, extensions: list) -> list:
    results = []
    for root, dirs, files in os.walk(path):
        for file in files:
            for ext in extensions:
                if file.endswith(ext):
                    results.append(os.path.join(root, file))
    return results


def query_root_files(path: str, extensions: list) -> list[str]:
    results = []
    for file in os.listdir(path):
        for ext in extensions:
            if file.endswith(ext):
                results.append(os.path.join(path, file))
    return results
