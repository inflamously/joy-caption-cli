import json
from typing import Dict, Any

# File: load.py
# Author: nflamously
# Original License: Apache License 2.0

def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r',  encoding="utf-8") as jsonfile:
        return json.load(jsonfile)