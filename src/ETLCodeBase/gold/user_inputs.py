"""
Docstring for ETLCodeBase.gold.user_inputs

Purpose:
    Handles dynamic user input in `.csv` files

Exposed API:
    - `process_units`
"""

import pandas as pd
from pathlib import Path

from ETLCodeBase.utils.filesystem import read_configs
from ETLCodeBase.gold._helpers import classify_pillar

def process_units(write_out:bool=True) -> pd.DataFrame:
    """ 
    Input:
        - write_out: whether to write out the final csv file

    Output:
        - df: processed units data frame

    """
    path_config = read_configs(config_type="io", name="path.json")
    units_path = Path(path_config["root"]) / Path(path_config["gold"]["payroll"])
    df = pd.read_csv(units_path/"Unit.csv", dtype={"Location":str, "Unit":float})
    df["Location"] = df["Location"].str.strip()
    mapping_config = read_configs(config_type="contracts", name="mapping.json")
    doc_rename = mapping_config["location"]["units"]
    df["Location"] = df["Location"].replace(doc_rename)
    df = classify_pillar(df=df)
    if write_out:
        df.to_csv(units_path.parent/ "OtherTables" /"Unit_PowerBI.csv",index=False)
    return df