import os
import yaml


def load_config(config_path: str = None) -> dict:

    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'parameters.yml')

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    return cfg


