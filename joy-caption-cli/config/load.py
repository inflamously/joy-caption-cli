import json
from typing import Dict, Any

# File: load.py
# Author: nflamously
# Original License: Apache License 2.0

def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as jsonfile:
        return json.load(jsonfile)