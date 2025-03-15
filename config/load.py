import json
from typing import Dict, Any


def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as jsonfile:
        return json.load(jsonfile)