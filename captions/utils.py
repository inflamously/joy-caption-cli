
# File: utils.py
# Author: nflamously
# Original License: Apache License 2.0

def break_list_into_chunks(inputs: list, chunk_size: int) -> list:
    return [inputs[i * chunk_size:(i + 1) * chunk_size] for i in range((len(inputs) + chunk_size - 1) // chunk_size)]