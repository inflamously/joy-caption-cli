import pathlib

from config.load import load_config
from state import APP_STATE

# File: initialization.py
# Author: nflamously
# Original License: Apache License 2.0

def setup_config():
    config = load_config('config/config.json')

    # Read Config
    APP_STATE["checkpoint_path"] = pathlib.Path(config['checkpoint_path'])
    APP_STATE["caption_map"] = config['captions']["map"]
    APP_STATE["clip_model_name"] = config['clip_model']
