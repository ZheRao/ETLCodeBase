"""
Docstring for ETLCodeBase.utils.filesystem

Purpose:
    Standardized methods for read/write methods

Exposed API:
    - `read_configs`
"""
from importlib.resources import files
import json

# config file reads
def read_configs(config_type:str, name:str) -> dict:
    """
    Given
        - config file type: io, contract, ...
        - config file name

    Reads and return configurations stored in json_configs/config_type/name, e.g., json_configs/io/path.json
    """
    p = files("ETLCodeBase.json_configs").joinpath(f"{config_type}/{name}")
    return json.loads(p.read_text())