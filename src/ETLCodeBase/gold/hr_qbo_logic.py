"""
Docstring for ETLCodeBase.gold.hr_qbo_logic

Purpose:
    Branch payroll data from QBO gold table, then apply payperiods + transformation that serves downstream Power BI HR Dashboard

Exposed API
    - `hr_qbo`
"""

import pandas as pd
from pathlib import Path
import datetime as dt

from ETLCodeBase.utils.filesystem import read_configs
from ETLCodeBase.gold._helpers import process_pp

def _locate_accounts(path_config:dict) -> pd.DataFrame:
    """
    Purpose:
        - read and load silver account table
        - filter for payroll related accounts
    """
    path = Path(path_config["root"]) / Path(path_config["silver"]["Dimension"])
    # load and filter accounts for wages and contract labor
    acc = pd.read_csv(path / "Account.csv")
    acc = acc[(acc["Category"].isin(["Wages and benefits - direct","Wages and benefits - overhead"])) | (acc["AccNum"].isin(["MFAZ595001","MFBC536030"]))]
    return acc

def _load_transactions(path_config:dict, account:pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
        - load gold QBO PL table and perform basic transformations
    """
    path = Path(path_config["root"]) / Path(path_config["gold"]["finance_operational"])
    df = pd.read_csv(path/"PL.csv")
    df = df[df["AccID"].isin(account.AccID.unique())]
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])
    df = df[df["TransactionDate"]>=dt.datetime(2021,12,20)].reset_index(drop=True)
    df = df[~df["Memo"].str.contains("Accrual",case=False,na=False)]
    return df

def _retrieve_bc_ranches() -> list[str]:
    """
    Purpose:
        - retrieve all the locations for BC for summarized stats in BC for payroll project
    """
    fact_config = read_configs(config_type="contracts", name="facts.json")
    bc_ranches = fact_config["bc_ranches"]
    return bc_ranches + ["BritishColumbia (corporate)"]

def _clean_location(df:pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
        - clean up location in payroll data to align with HR Dashboard logic
    """
    df.loc[df["Location"]=="Eddystone (corporate)", "Pillar"] = "Unclassified"
    df.loc[df["Location"]=="Eddystone (corporate)", "Location"] = "Unassigned"
    df.loc[df["Location"]=="Legacy", "Location"] = "Unassigned"
    df.loc[(df["Location"].str.contains("corporate",case=False,na=False)&(df["Location"]!="BritishColumbia (corporate)")),"Location"] = "Corporate"
    ## move BC ranches into BC Cattle
    bc_ranches = _retrieve_bc_ranches()
    df.loc[(df["Location"].isin(bc_ranches)), "Location"] = "BritishColumbia (cattle)"
    df.loc[df["Location"] == "BritishColumbia (cattle)", "Pillar"] = "Cattle-CowCalf"
    return df

def _compute_total_units_bc(units:pd.DataFrame) -> float:
    """
    Purpose:
        - calculate total units for BC locations (including all the sub-location for BC)
    """
    total_bc = 0
    bc_ranches = _retrieve_bc_ranches()
    for l in bc_ranches+["BritishColumbia (corporate)"]:
        total_bc += units.loc[units["Location"]==l, "Unit"].item()
    return total_bc


def _per_unit_calculation(summarized:pd.DataFrame, path_config:dict) -> pd.DataFrame:
    """
    Purpose:
        - load units table, perform per-unit calculation on summarized table (per location per PP)
    """
    path = Path(path_config["root"]) / Path(path_config["gold"]["payroll"])
    path = path.parent/ "OtherTables" / "Unit_PowerBI.csv"
    units = pd.read_csv(path, dtype={"Location":str, "Unit":float},usecols=["Location", "Unit"])
    # accomodate consolidated BC stats
    total_bc = _compute_total_units_bc(units=units)
    units.loc[units["Location"]=="BritishColumbia (cattle)", "Unit"] = total_bc
    # final clean up
    units["Unit"] = units["Unit"].replace({0: 1})
    print(f"Location not in Summarized table: {set(units.Location.unique()) - set(summarized.Location.unique()) - set(_retrieve_bc_ranches())}")
    print(f"Location not in Units table: {set(summarized.Location.unique()) - set(units.Location.unique())}")
    # compute per-unit
    summarized = pd.merge(summarized, units, on="Location", how="left")
    summarized["CostPerUnit"] = summarized["AmountDisplay"] / summarized["Unit"] * 26
    summarized["Count"] = 1
    return summarized


def _summarizing(df:pd.DataFrame, path_config:dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Purpose:
        - create 3 levels of summaries
            1. by location per PP + per-unit calculation
            2. by location (for all PP)
            3. by pillar (for all PP)
    """
    summarized = pd.DataFrame(df.groupby(["Location","PPName","Pillar","FiscalYear","Cycle","PPNum"]).agg({"AmountDisplay":"sum"}).reset_index(drop=False))
    quick_check = df.groupby(["Location","PPName"]).agg({"AmountDisplay":"sum"}).reset_index(drop=False)
    if len(summarized) != len(quick_check): 
        raise ValueError(f"Duplicated value detected for Level 1 Summary (per Location per PP) - length of summary {len(summarized)} - length of groupby 'Location' & 'PPName' only {len(quick_check)}")
    # per unit calculation
    summarized = _per_unit_calculation(summarized=summarized,path_config=path_config)
    # summarized by location (for all PP)
    summarized2 = summarized.groupby(by=["Location","FiscalYear","Pillar"]).agg({"CostPerUnit":"mean", "Count":"sum"}).reset_index(drop=False)
    summarized2 = summarized2.rename(columns={"CostPerUnit":"Avg CostPerUnit"})
    quick_check = summarized.groupby(by=["Location","FiscalYear"]).agg({"CostPerUnit":"mean"})
    if len(summarized2) != len(quick_check):
        raise ValueError(f"Duplicated value detected for Level 2 Summary (per Location) - length of summary {len(summarized2)} - length of groupby 'Location' & 'FiscalYear' only {len(quick_check)}")
    # summarized by Pillar (for all PP)
    summarized3 = summarized2.groupby(by=["FiscalYear","Pillar"]).agg({"Avg CostPerUnit":"mean", "Count":"sum"}).reset_index(drop=False)
    quick_check = summarized.groupby(by=["Pillar","FiscalYear"]).agg({"CostPerUnit":"mean"})
    if len(summarized3) != len(quick_check):
        raise ValueError(f"Duplicated value detected for Level 3 Summary (per Pillar) - length of summary {len(summarized3)} - length of groupby 'Pillar' & 'FiscalYear' only {len(quick_check)}")
    return summarized, summarized2, summarized3




def hr_qbo(write_out:bool=True) -> pd.DataFrame:
    """
    Input:
        - write_out: write out to disk

    Output:
        - transformed data frame for QBO Payroll data
    """
    print("\nStarting Payroll Project Transformation\n")
    path_config = read_configs(config_type="io", name="path.json")
    # get accounts
    acc = _locate_accounts(path_config=path_config)
    # get transactions
    df = _load_transactions(path_config=path_config,account=acc)
    # classify payperiods
    df = process_pp(df=df,date_col="TransactionDate")
    # clean up locations
    df = _clean_location(df=df)
    # summarizing
    print("Summarizing ...")
    summarized, summarized2, summarized3 = _summarizing(df=df, path_config=path_config)
    # saving
    if write_out:
        path_df = Path(path_config["root"]) / Path(path_config["gold"]["payroll"])
        path_summary = Path(path_config["root"]) / Path(path_config["gold"]["hr_combined"]) / "CSV"
        df.to_excel(path_df/"Payroll.xlsx", sheet_name="Payroll", index=False)
        summarized.to_csv(path_summary/ "payroll_summarized1.csv", index=False)
        summarized2.to_csv(path_summary/ "payroll_summarized2.csv", index=False)
        summarized3.to_csv(path_summary/ "payroll_summarized3.csv", index=False)
    
    return df




