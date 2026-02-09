"""
Docstring for ETLCodeBase.gold.finance_logic

Purpose:
    This script converts silver QBO PL into gold QBO table to serve many downstream tasks

Exposed API:
    - `process_finance()`
"""

import pandas as pd
from pathlib import Path 
import datetime as dt
import json
from importlib.resources import files

from ETLCodeBase.utils.filesystem import read_configs
from ETLCodeBase.gold._helpers import classify_pillar, standardize_product, accid_reroute




def _process_dates(df:pd.DataFrame) -> pd.DataFrame:
    """
    Purposes:
        - Convert `TransactionDate` to datetime format
        - Compute fiscal year
        - Crate `Month` for month name in string
    """
    if "TransactionDate" not in df.columns: raise KeyError("'TransactionDate' column missing for processing silver QBO PL report")
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])
    df["FiscalYear"] = df.TransactionDate.apply(lambda x: x.year + 1 if x.month >= 11 else x.year)
    df["Month"] = df["TransactionDate"].dt.month_name()
    return df

def _add_seed_location(df:pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
        - add `Location` for Seed pillar; `Location` is missing from raw QBO PL reports
    """
    if "Corp" not in df.columns: raise KeyError("'Corp' column is missing for assigning locations for Seed Pillar for processing silver QBO PL report")
    df.loc[df["Corp"]=="MSL", "Location"] = "Seeds"
    df.loc[df["Corp"]=="NexGen", "Location"] = "NexGen"
    df.loc[df["Corp"]=="MSUSA", "Location"] = "Arizona (produce)"
    return df

def _process_location(df:pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
        - Add Loaction for Seed Pillar
        - rename `Location` to `LocationRaw`
        - fillna with "Missing"
        - read location contract and standardize location names
        - printout unaccounted locations
    """
    df = _add_seed_location(df=df)
    df = df.rename(columns={"Location":"LocationRaw"})
    df["Location"] = df["LocationRaw"]
    df = df.fillna(value={"Location": "Missing"})
    mapping_config = read_configs(config_type="contracts",name="mapping.json")
    clean_location = mapping_config["location"]["qbo"]
    df["Location"] = df["Location"].replace(clean_location)
    fact_config = read_configs(config_type="contracts",name="facts.json")
    fact_location = fact_config["locations"]["Produce"] + fact_config["locations"]["Cattle"] + fact_config["locations"]["Grain"] + fact_config["locations"]["Seed"] + fact_config["locations"]["Others"]
    unaccounted_location = list(set(df["Location"].unique()) - set(fact_location))
    if len(unaccounted_location) > 0:
        print(f"In contract, locations unaccounted for - {unaccounted_location}")
    return df

def _adjust_records_location_corp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
        - "MPUSA" + `Location` missing = "Arizona (produce)" + "Produce" Pillar
        - `Location` == "Arizona (produce)" -> "MPUSA" - not relevant now, we never look at `Corp`
        - FY >= 2024, `Location` contains "Arizona" -> "Produce" Pillar + "Arizona (produce)" Location
        - "BritishColumbia (produce)" -> "MPL"
        - "Outlook" -> "MPL"
    """
    df.loc[((df["Corp"] == "MPUSA")&(df["Location"].isna())), "Location"] = "Arizona (produce)"
    df.loc[((df["Corp"] == "MPUSA")&(df["Location"]=="Missing")), "Location"] = "Arizona (produce)"
    df.loc[((df["Corp"] == "MPUSA")&(df["Location"] == "Arizona (produce)")), "Pillar"] = "Produce"
    df.loc[df["Location"] == "Arizona (produce)", "Corp"] = "MPUSA"
    df.loc[((df["FiscalYear"] >= 2024) & (df["Location"].str.contains("Arizona",case=False))),"Pillar"] = "Produce"
    df.loc[((df["FiscalYear"] >= 2024) & (df["Location"].str.contains("Arizona",case=False))),"Location"] = "Arizona (produce)"
    df.loc[df["Location"] == "BritishColumbia (produce)", "Corp"] = "MPL"
    df.loc[df["Location"]=="Outlook", "Corp"] = "MPL"
    return df

def _process_accounts(write_out:bool, acc_path_silver:Path, acc_out_root:Path) -> pd.DataFrame:
    """
    Purpose:
        - read silver account table
        - identify and standardize product
        - write out as gold account table if `write_out` == True
    """ 
    accounts = pd.read_csv(acc_path_silver)
    accounts = standardize_product(df=accounts,for_budget=False)
    if write_out:
        accounts.to_csv(acc_out_root/"Account_table.csv", index=False)
        accounts.to_excel(acc_out_root/"Account_table.xlsx", sheet_name = "Account", index=False)
    return accounts

def _revise_signs(df:pd.DataFrame, accounts:pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
        - reverse the signs for expense accounts entries in QBO df
    """
    if "ProfitType" not in accounts.columns: raise KeyError("'ProfitType' column missing, unable to revise signs for silver QBO PL table")
    if "AmountCAD" not in df.columns: raise KeyError("'AmountCAD' column missing, unable to revise signs for silver QBO PL table")
    expense_accounts = accounts[accounts["ProfitType"].isin(["Cost of Goods Sold", "Direct Operating Expenses", "Operating Overheads", "Other Expense"])]
    df["AmountDisplay"] = df.apply(lambda x: -x["AmountCAD"] if x["AccID"] in expense_accounts.AccID.unique() else x["AmountCAD"], axis=1)
    return df

def _prepare_actuals_for_budget(df: pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
        - create summarized actuals by `Location`, `AccNum`, `FiscalYear`, `Month`, with aggregated `AmountDisplay`
        - save to budget location for consolidation with budgets
    """
    df = df[df["FiscalYear"] >= 2024].copy()
    df["AccName"] = df["AccName"].str.strip()
    actuals = df.groupby(["Location","AccNum", "FiscalYear", "Month", "AccID"]).agg({"AmountDisplay":"sum"}).reset_index(drop=False)
    actuals["DataType"] = "Actual"
    actuals = actuals.rename(columns={"AmountDisplay": "AmountCAD"})
    return actuals


def _write_fx(fx:float) -> None:
    """
    Purpose:
        - write fx out as a system state
    """
    path = files("ETLCodeBase.json_configs").joinpath("state/fx.json")
    meta = {
        "fx": fx,
        "timestamp": dt.datetime.now().isoformat()
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)



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
    path_config = read_configs(config_type="io",name="path.json")
    df = pd.read_csv(Path(path_config["root"]) / Path(path_config["silver"]["PL"]) / "ProfitAndLoss.csv", dtype={"Class":str, "ClassID":str})
    if len(df.FXRate.unique()) != 1: raise ValueError(f"Expected silver QBO PL FX rate to be singular across all records, got {df.FXRate.unique()}")
    fx = df.loc[0,"FXRate"]
    _write_fx(fx=fx)
    
    df = _process_dates(df=df)
    df = _process_location(df=df)
    df = classify_pillar(df=df)
    df = _adjust_records_location_corp(df=df)
    df = accid_reroute(df=df)

    acc_path_silver = Path(path_config["root"]) / Path(path_config["silver"]["Dimension"]) / "Account.csv"
    out_root = Path(path_config["root"]) / Path(path_config["gold"]["finance_operational"])
    accounts = _process_accounts(write_out=write_out, acc_path_silver=acc_path_silver, acc_out_root=out_root)

    print("Revising Signs ...")
    df = _revise_signs(df=df,accounts=accounts)

    print("Saving ...")
    if write_out:
        df.to_csv(out_root/"PL.csv", index=False)
        df.to_excel(out_root/"PL.xlsx", sheet_name="Transactions", index=False)

        budget_out_path = Path(path_config["root"]) / Path(path_config["gold"]["budget"]) / "Actuals"
        actuals = _prepare_actuals_for_budget(df=df)
        actuals.to_csv(budget_out_path / "actuals.csv", index=False)

        pillar_out_root = Path(path_config["root"]) / Path(path_config["gold"]["pillar_dashboard"])
        for pillar in ["Grain", "Cattle", "Seed", "Produce"]:
            df[df["Pillar"]==pillar].to_excel(pillar_out_root/pillar/"PL.xlsx", sheet_name="Transactions", index=False)





    return df
    