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
    df["Location"] = df["Location"].replace({"Outlook": "SK Produce"})
    budget_location_rename = {"Airdrie (grain)": "Airdrie", "Airdrie (cattle)": "Airdrie", "Calderbank (cattle)": "Calderbank",
                              "Airdrie (corporate)": "Airdrie", "Seeds USA":"Arizona (produce)"}
    df["Location"] = df["Location"].replace(budget_location_rename)
    return df

def _format_budget_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
        - unpivot the amount per month, turn horizontal records into vertical records
    """
    df = df.melt(
        id_vars=["Pillar", "Currency", "Location", "Category", "Account", "FiscalYear"],
        var_name="Month",
        value_name="Amount"
    )
    df = df.dropna(subset=["Amount"])
    df["Amount"] = df["Amount"].astype(float)
    df = df[df["Amount"]!=0].reset_index(drop=True)
    return df

def _convert_to_cad(df:pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
        - based on `Currency` column, apply fx rate for `USD`
        - create `AmountCAD` column
    """
    fx_config = read_configs(config_type="state", name="fx.json")
    fx = fx_config["fx"]
    m = df["Currency"].eq("USD")
    df["AmountCAD"] = df["Amount"]* (1 + m * (fx-1))
    return df

def _accnum_extract(df:pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
        - extract `AccNum` from `Account`
        - e.g., extract "MSL566500" from "MSL 566500 Supplies"
    """
    if "Account" not in df.columns: raise KeyError("Missing 'Account' from Budget df, required to extract AccNum")
    parts = df["Account"].str.split(" ", n=2, expand=True)
    df["AccNum"] = parts[0] + parts[1]
    return df


def _apply_location_mapping(df:pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
        - apply the contract to standardize and combine locations
    """
    map_contract = read_configs(config_type="contracts", name="mapping.json")
    budget_location_map = map_contract["location"]["budgets"]
    df["Location"] = df["Location"].replace(budget_location_map)
    return df



def _extract_budget(root:Path, year:list[int]=[2026]) -> pd.DataFrame:
    """
    Purpose:
        - read all targeted budget file, standardize column and add `FiscalYear` column if not available
        - input columns
            - `Pillar`, `Currency`, `Location`, `Category`, `Account`, `November`, ...
        - output columns
            - `Location`, `AccNum`, `FiscalYear`, `Month`, `DataType`, `AmountCAD`
    """
    path = root / "Budgets" / "excel_formatted"
    df_list = []
    for y in year:
        df = pd.read_csv(path / f"budgets_{y}.csv")
        df["FiscalYear"] = y
        df_list.append(df)
    df = pd.concat(df_list, ignore_index=True)
    df = df.loc[df["Location"]!="Delaware",:]
    df = _format_budget_input(df=df)
    df = _convert_to_cad(df=df)
    df = _accnum_extract(df=df)
    df["DataType"] = "Budget"
    df = df.loc[:,["Location", "AccNum", "FiscalYear", "Month", "DataType", "AmountCAD"]]
    df = _apply_location_mapping(df=df)
    
    return df

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
    print(alert.groupby(by=["Location","FiscalYear","AccNum","AccID"]).agg({"AmountCAD":"sum"}))
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
    budget_25 = _extract_budget_25(root=root)
    year = [2026]
    budget_rest = _extract_budget(root=root,year=year)
    budget = pd.concat([budget_25, budget_rest], ignore_index=True)
    budget = _budget_accnum_reroute(budget=budget)
    budget = _accid_map(df=budget, path_config=path_config)
    actual = _extract_actuals(root=root)
    print(f"Location missing in actual: {(set(budget.Location.unique()) - set(actual.Location.unique()))}")
    print(f"Location missing in budget_{max(year)}: {(set(actual.Location.unique()) - set(budget[budget["FiscalYear"]==max(year)].Location.unique()))}")
    df = pd.concat([budget, actual], ignore_index=True)

    # df = accid_reroute(df=df)

    df = _add_fx(df=df)

    df= classify_pillar(df=df)

    if write_out:
        print("\nSaving ...")
        df.to_csv(root / "PowerBI" / "BudgetActual.csv", index=False)
        df.to_excel(root / "PowerBI" / "BudgetActual.xlsx", sheet_name="Budget", index=False)
        pillar_root = Path(path_config["root"]) / Path(path_config["gold"]["pillar_dashboard"])
        for pillar in ["Grain", "Cattle", "Seed", "Produce"]:
            df[df["Pillar"].str.contains(pillar)].to_excel(pillar_root/pillar/"BudgetActual.xlsx", sheet_name="Budget", index=False)

    return df

