"""
Docstring for ETLCodeBase.gold._helpers

Purpose:
    Common methods for gold layer business logic application across all QBO & QBO Time data

Exposed API:
    - `run_cattle_split()` - add additional split transactions and offset transactions
"""



import pandas as pd
from pathlib import Path
import datetime as dt
import numpy as np

from ETLCodeBase.gold.finance_logic import process_finance
from ETLCodeBase.utils.filesystem import read_configs


def _generate_year_month(month_anchor: int, year_anchor:int) -> list[str]:
    """
    Purpose:
        - given year & month anchor, generate year-month_name backward, one year from the anchor (12 pairs)
        - return the ordered year-month_name list
    """
    months = np.array([
        "Jan","Feb","Mar","Apr","May","Jun",
        "Jul","Aug","Sep","Oct","Nov","Dec"
    ], dtype=str)
    original_index = [(x+month_anchor-1) for x in range(11,-1,-1)]      # original index before mod 12, e.g., [13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]
    index_list = [x%12 for x in original_index]                         # actual index for month_list, e.g., [1, 0, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]
    month_list = list(months[index_list])                               # e.g., ['Feb', 'Jan', 'Dec', 'Nov', 'Oct', 'Sep', 'Aug', 'Jul', 'Jun', 'May', 'Apr', 'Mar']
    year_list = [year_anchor if original_index[x] >= 12 else year_anchor-1 for x in range(12)]  # e.g., [2026, 2026, 2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025]
    column_list = [str(year_list[x]) +"-"+ month_list[x] for x in range(12)]    # e.g., ['2026-Feb', '2026-Jan', '2025-Dec', '2025-Nov', '2025-Oct', '2025-Sep', ... , '2025-Mar']
    return column_list


def _extract_recent_inventory(df: pd.DataFrame) -> str:
    """
    Purpose:
        - read user cattle inventory input
        - isolate the most recent inventory
        - returns the column name, e.g., '2026-Jan'
    Note: 
        - currently it only searches the past year, no fallback if inventory doesn't exist yet - search for more years or raise error
        - and it ensures the inventory is at least 2/3 filled
    """
    today = dt.date.today()
    df_len = len(df)
    column_candidate = _generate_year_month(month_anchor=today.month, year_anchor=today.year)
    for i in range(len(column_candidate)):
        if (column_candidate[i] in df.columns) and (df[column_candidate[i]].isna().sum() < df_len * 2/3):
            return column_candidate[i]
        

def _rename_location(df:pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
        - rename locations so it is consistent with QBO
    """
    map_config = read_configs(config_type="contracts", name="mapping.json")
    cattle_location = map_config["location"]["cattle_inputs"]
    df["Location"] = df["Location"].replace(cattle_location)
    return df


def _extract_perc(df: pd.DataFrame, inventory_column:str, path_config:dict) -> pd.DataFrame:
    """
    Purpose:
        - takes the full cattle inventory df, compute Airdrie, Eddystone, Walkdeck H and HD split, percentage based
    Note:
        - future: add tests for sum to 1 per location + 0 <= perc <= 1 for all percentages
    """
    split_location = ["Airdrie (H)", "Airdrie (HD)", "Eddystone (H)", "Eddystone (HD)", "Waldeck (H)", "Waldeck (HD)"]
    df_split = df[df["Location"].isin(split_location)].copy(deep=True).reset_index(drop=True).rename(columns={inventory_column:"inventory"})
    df_split.loc[df_split["Location"].str.contains("(H)", regex=False), "inventory"] *= 365 # convert H to HD to compute percentage
    # compute and append total inventory in HD for each location
    for l in ["Airdrie", "Eddystone", "Waldeck"]:
        mask = df_split["Location"].str.contains(l, case=False)
        total = df_split[mask].inventory.sum()
        df_split.loc[mask, "total_inventory"] = total
    # compute percentage split
    df_split["inventory_perc"] = df_split["inventory"] / df_split["total_inventory"]
    out_path = Path(path_config["root"]) / path_config["reference"]
    df_split.to_csv(out_path/"cattle_inventory_perc.csv",index=False)
    return df_split


def _create_entries(mode:str, info:dict, original_df:pd.DataFrame, split_map:pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
        - Create splitting or offset transactions for the cattle split logic
    """
    amount_columns = ["Amount", "AmountAdj", "AmountCAD", "AmountDisplay"]
    mode = mode.lower()
    if mode not in ["split","offset"]: raise ValueError(f"'mode' must be entered as one of 'split' or 'offset', entered {mode}")
    # create copied dataframe and label as synthetic 
    df = original_df.copy(deep=True).reset_index(drop=True); df["record_type"] = info["record_type"]; df["classification_source"] = info["classification_source"]
    # for splitting: change to new location names, use mapping (location -> %) to append % values - fully vectorized
    if mode == "split": 
        df["Pillar"] = info["Pillar"]
        df["Location"] += info["location_str_addition"]
        df["split_perc"] = df["Location"].map(split_map).fillna(-1)
        if -1 in df["split_perc"]: raise ValueError("negative percentage detected for splitting operation, location-percentage mapping failed")
        df["Memo"] = "Split " + (df["split_perc"] * 100).round(2).astype(str) + "%-" + df["Memo"]
    # for offset: multiply amount columns by -1
    else:
        df["split_perc"] = -1.0
        df["Memo"] = "Offset-" + df["Memo"]
    # multiply % with actual amounts
    for col in amount_columns:
        df[col] *= df["split_perc"]
    return df

def run_cattle_split(qbo:pd.DataFrame) -> pd.DataFrame:
    """
    Input:
        - qbo: original Gold QBO table
    Output:
        - qbo with added split entries and offset records
    """
    cattle = qbo[qbo["Location"].isin(["Airdrie", "Eddystone (cattle)", "Waldeck"])].copy()
    cattle["Location"] = cattle["Location"].replace({"Eddystone (cattle)":"Eddystone"})

    # compute % allocation mapping table
    path_config = read_configs(config_type="io", name="path.json")
    ## read inventory CSV, select most recent inventory column
    path = Path(path_config["root"]) / path_config["user_inputs"]["cattle_inventory"]
    df = pd.read_csv(path/"Units.csv")
    inventory_column = _extract_recent_inventory(df=df)
    df = df.loc[:,["Location", inventory_column]].copy()
    df = _rename_location(df=df)
    ## compute percentage
    df_split = _extract_perc(df=df, inventory_column=inventory_column, path_config=path_config)
    split_map = df_split.set_index("Location")["inventory_perc"]

    # create additioanl entries
    cattle_h_info = {
        "record_type": "SYNTHETIC_SPLIT",
        "classification_source": "CATTLE_INVENTORY",
        "Pillar": "Cattle-CowCalf",
        "location_str_addition": " (H)"
    }
    cattle_hd_info = {
        "record_type": "SYNTHETIC_SPLIT",
        "classification_source": "CATTLE_INVENTORY",
        "Pillar": "Cattle-Feedlot",
        "location_str_addition": " (HD)"
    }
    cattle_offset_info = {
        "record_type": "SYNTHETIC_OFFSET",
        "classification_source": "CATTLE_INVENTORY",
    }
    cattle_offset = _create_entries(original_df=cattle,mode="offset",info=cattle_offset_info, split_map=split_map)
    cattle_h = _create_entries(original_df=cattle,mode="split",info=cattle_h_info, split_map=split_map)
    cattle_hd = _create_entries(original_df=cattle,mode="split",info=cattle_hd_info, split_map=split_map)
    qbo = pd.concat([cattle_h,cattle_hd, cattle_offset,cattle], ignore_index=True)
    qbo["Location"] = qbo["Location"].replace({"Eddystone": "Eddystone (cattle)"})
    return qbo

