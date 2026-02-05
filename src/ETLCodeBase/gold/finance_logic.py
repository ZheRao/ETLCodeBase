"""
Docstring for ETLCodeBase.gold.finance_logic

Purpose:
    This script converts silver QBO PL into gold QBO table to serve many downstream tasks

Exposed API:
    - `process_finance()`
"""

import pandas as pd
from pathlib import Path 

from ETLCodeBase.utils.filesystem import read_configs












def process_finance(write_out:bool=True) -> pd.DataFrame:
    """
    Input:
        - write_out: controls whether final resulting df got written to disk
    
    Output:
        - df: transformed QBO df
    
    Note:
        - it reclassify accounts
        - it standardizes locations, classify pillars
        - it revises signs
    """
    print("\nStarting Finance Operational Project Transformation\n")
    