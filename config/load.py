import json
from typing import Dict, Any


def load_config() -> Dict[str, Any]:
    with open('config.json', 'r') as jsonfile:
        return json.load(jsonfile)