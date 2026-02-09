"""
Docstring for ETLCodeBase.gold.finance_budget

Purpose:
    Read formatted budget input csv, consolidate with QBO PL actuals, output Power BI ready Excel file

Exposed API:
    - `compose_budget_actual()`
"""

import pandas as pd
from pathlib import Path 

from ETLCodeBase.utils.filesystem import read_configs
from ETLCodeBase.gold._helpers import classify_pillar, accid_reroute

def _extract_actuals(root:Path) -> pd.DataFrame:
    """
    Purpose:
        - read and return the df for actuals with `Location`, `AccNum`, `FiscalYear`, `Month`, `DataType`, `AmountCAD` (not to confused with `AmountCAD` from QBO PL, this corresponds to `AmountDisplay`)
    """
    path = root / "Actuals" / "actuals.csv"
    return pd.read_csv(path)

def _extract_budget_25(root:Path) -> pd.DataFrame:
    """
    Purpose:
        - read computed 2025 budget file without reprocessing everything from raw Excel file, budget generation engine is in a different script
        - columns
            - `Location`, `AccNum`, `FiscalYear`, `Month`, `DataType`, `AmountCAD`
    """
    path = root / "Budgets" / "2025" / "budget.csv"
    df = pd.read_csv(path, usecols=["Location", "AccNum", "FiscalYear", "Month", "DataType", "AmountCAD"])
    df = df[df["Location"]!="Calderbank (grain)"]
    budget_location_rename = {"Airdrie (grain)": "Airdrie", "Airdrie (cattle)": "Airdrie", "Calderbank (cattle)": "Calderbank",
                              "Airdrie (corporate)": "Airdrie", "Seeds USA":"Arizona (produce)"}
    df["Location"] = df["Location"].replace(budget_location_rename)
    return df


def _extract_budget(root:Path, year:list[int]=[2026]) -> pd.DataFrame:
    """
    Purpose:
        - read all targeted budget file, standardize column and add `FiscalYear` column if not available

    Note:
        - Place holder for actual 2026 budget code
    """
    return 1

def _budget_accnum_reroute(budget:pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
        - reroute the accnum for budget based on contract that finance provided
    """
    if "AccNum" not in budget.columns: raise KeyError("'AccNum' missing in budget columns for AccNum reroute")
    acc_contract = read_configs(config_type="contracts", name="acc.contract.json")
    accnum_reroute = acc_contract["budget_reroutes"]["accnum_reroute"]
    budget["AccNum"] = budget["AccNum"].replace(accnum_reroute)
    return budget

def _accid_map(df:pd.DataFrame, path_config:str) -> pd.DataFrame:
    """
    Purpose: 
        - take the budget + actual df, map `AccID` into df based on `AccNum`
    """
    if "AccNum" not in df.columns: raise KeyError("'AccNum' missing in consolidated Budget and Actual for mapping AccID")
    acc_path = Path(path_config["root"]) / Path(path_config["gold"]["finance_operational"]) / "Account_table.csv"
    acc = pd.read_csv(acc_path)
    acc = acc[acc["AccNum"].notna()]    # AccNum must be non-missing
    acc = acc[acc["Active"]]            # avoid non-active accounts what share same AccNum with active accounts
    acc_mapping = (
        acc[["AccNum", "AccID"]]
        .dropna()
        .drop_duplicates(subset=["AccNum"])
        .set_index("AccNum")["AccID"]
        .to_dict()
    )
    df["AccID"] = df["AccNum"].map(acc_mapping).fillna("Unsuccessful")
    alert = df[(df["AccID"]=="Unsuccessful")&(df["AmountCAD"]>0)]
    print("\nAccID mapping alerts: ")
    print(alert)
    return df

def _add_fx(df:pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
        - read last fx from data pull from config and append to df
    """
    fx_config = read_configs(config_type="state",name="fx.json")
    fx = fx_config["fx"]
    df["FXRate"] = fx
    return df


def compose_budget_actual(write_out:bool=True) -> pd.DataFrame:
    """
    Purpose:
        - consolidate actual and budget monthly stat
        - additional columns: `AccID`, `FXRate`
    Input: 
        - write_out: whether to write Excel to disk
    Output:
        - df: consolidated data frame with
            - `Location`, `AccNum`, `FiscalYear`, `Month`, `DataType`, `AccID`, `FXRate`, `AmountCAD` (not to confused with `AmountCAD` from QBO PL, this corresponds to `AmountDisplay`)
    """
    print("\nStarting Budget to Actual Transformation\n")
    path_config = read_configs(config_type="io", name="path.json")
    root = Path(path_config["root"]) / Path(path_config["gold"]["budget"])
    budget = _extract_budget_25(root=root)
    budget = _budget_accnum_reroute(budget=budget)
    actual = _extract_actuals(root=root)
    print(f"Location missing in actual: {(set(budget.Location.unique()) - set(actual.Location.unique()))}")
    print(f"Location missing in budget: {(set(actual.Location.unique()) - set(budget.Location.unique()))}")
    df = pd.concat([budget, actual], ignore_index=True)

    df = _accid_map(df=df, path_config=path_config)

    df = accid_reroute(df=df)

    df = _add_fx(df=df)

    df= classify_pillar(df=df)

    if write_out:
        print("\nSaving ...")
        df.to_csv(root / "PowerBI" / "BudgetActual.csv", index=False)
        df.to_excel(root / "PowerBI" / "BudgetActual.xlsx", sheet_name="Budget", index=False)
        pillar_root = Path(path_config["root"]) / Path(path_config["gold"]["pillar_dashboard"])
        for pillar in ["Grain", "Cattle", "Seed", "Produce"]:
            df[df["Pillar"] == pillar].to_excel(pillar_root/pillar/"BudgetActual.xlsx", sheet_name="Budget", index=False)

    return df

