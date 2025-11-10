"""Module providing input/output handling functions."""

import yaml

def _read_config(file_path):
    """Function reading the configuration parameters from a YAML file."""
    with open(file_path, encoding="utf-8") as f:
        return yaml.safe_load(f)