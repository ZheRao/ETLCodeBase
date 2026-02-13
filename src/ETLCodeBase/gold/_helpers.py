"""
Docstring for ETLCodeBase.gold._helpers

Purpose:
    Common methods for gold layer business logic application across all QBO & QBO Time data

Exposed API:
    - `classify_pillar` - classify and create a new column `Pillar` based on `Location` column
    - `standardize_product` - from account names, identify and standardize `commodity`
    - `accid_reroute` - reroute `AccID` based on Finance contract
    - `process_pp` - apply the payperiod number classification based on transactions date, process payperiod columns, and return the new dataframe
    - `determine_fy` - identify fiscal year based on a date column
"""

import pandas as pd
from pathlib import Path
import datetime as dt

from ETLCodeBase.utils.filesystem import read_configs

def _pillar_classification(entry:pd.Series) -> str:
    """ 
    Input:
        - entry: a row from df
    
    Output:
        - classification in string format
    """
    location = entry["Location"]
    if not isinstance(location, str) or (location == "Missing"):
        return "Missing"
    location = location.lower()
    if "produce" in location:
        return "Produce"
    elif "grain" in location:
        return "Grain"
    elif "corporate" in location:
        return "Unclassified"
    match location:
        case "hafford"|"kamsack"|"prince albert"|"raymore"|"regina"|"swift current"|"the pas"|"camp 4"|"fly creek"|"havre"|"yorkton"|"colorado"|"billings"|"delaware":
            return "Grain"
        case "outlook"|"seeds usa":
            return "Produce"
        case "eddystone (cattle)"|"waldeck"|"airdrie":
            return "Cattle-Feedlot"
        case "ashcroft"|"diamond s"|"fraser river ranch"|"home ranch"|"moon ranch"|"wolf ranch"|"calderbank"|"bc cattle mfl"|"britishcolumbia (cattle)":
            return "Cattle-CowCalf"
        case "seeds"|"nexgen":
            return "Seed"
        case _:
            return "Unclassified"

def classify_pillar(df:pd.DataFrame) -> pd.DataFrame:
    """
    Input:
        - df: dataframe with `Location` column

    Output:
        - df: with additional `Pillar` column

    Note:
        - for efficiency, this function creates a one-to-one map of location -> pillar, than map back to original df, to avoid unnecessary duplicated assignment
    """
    if "Location" not in df.columns: raise KeyError("Missing 'Location' column for pillar classification")
    df["Location"] = df["Location"].str.strip()
    location = pd.DataFrame(
        data = df.Location.unique(),
        columns = ["Location"]
    )
    location["Pillar"] = location.apply(lambda x: _pillar_classification(x), axis=1)
    location_mapping = location.set_index("Location")["Pillar"].to_dict()
    df["Pillar"] = df["Location"].map(location_mapping)
    return df

def _identify_product(entry: pd.Series, for_budget:bool=False) -> str:
    """ 
    Input:
        - entry: one row from pandas data frame
        - for_budget: whether this is for budget, different column of reference, small adjustments for naming (to match budget from finance)

    Output:
        - commodity name in string

    Note:
        - no seed product allocation
    """
    if not for_budget:
        if entry["Corp"] in ["MSL", "MSUSA", "NexGen"]:
            return "SeedProduct"
    accname = entry["AccName"].lower() if not for_budget else entry["AccFull"].lower()
    if "float" in accname:
        return "Others"
    facts_config = read_configs(config_type="contracts", name="facts.json")
    products = facts_config["commodities"]["Produce"] + facts_config["commodities"]["Grain"] + facts_config["commodities"]["Cattle"]
    for x in products:
        if x.lower() in accname:
            if for_budget:
                match x:
                    case "Market Garden"|"CSA":
                        return "Market Garden / CSA"
                    case "Corn Maze":
                        return "Prairie Pathways"
                return x 
            return x
    if "straw" in accname or "forage" in accname or "hay bale" in accname:
        if for_budget: 
            return "Hay/Silage" 
        else: 
            return "Forage"
    return "Others"

def standardize_product(df:pd.DataFrame, for_budget:bool=False) -> pd.DataFrame:
    """
    Input:
        - df: original dataframe
        - for_budget: whether this product standardization is for budget
    
    Output:
        - df: with `Commodity` column
    
    Note:
        - should be called at Account's level
        - not using for_budget yet, budget default to copying Excel budget sheet
    """
    if for_budget:
        if "AccFull" not in df.columns: raise KeyError("Column used to standardize product is not found, for classifying budget accounts, missing 'AccFull'")
    else:
        if "AccName" not in df.columns: raise KeyError("Column used to standardize product is not found, missing 'AccName'")
    df["Commodity"] = df.apply(lambda x: _identify_product(x), axis=1)
    return df

def accid_reroute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input:
        - df: data frame with `AccID`

    Output:
        - df: data frame with `AccID` rerouted
    """
    if "AccID" not in df.columns: raise KeyError("'AccID' column missing for rerouting acc IDs for silver QBO PL report")
    acc_contract = read_configs(config_type="contracts",name="acc.contract.json")
    accid_reroute = acc_contract["actuals_reroutes"]["accid_reroute"]
    df["AccID"] = df["AccID"].replace(accid_reroute)
    return df

def process_pp(df:pd.DataFrame, date_col:str, write_out:bool=False) -> pd.DataFrame:
    """ 
    Input:
        - df: data frame where pay period details needs to be mapped onto, based on date column
        - date_col: column name for the date column
        - write_out: write the processed payperiod mapping table to disk?
            - caution: should be done with care, manual modification to the file needed afterwards (do it only when the payperiod Excel input file is modified)
            - duplicated problem - e.g., 23-PP22 has Fiscal year 2023 and 2024 - rank by PPName and FiscalYear, than drop the earlier FiscalYear record
    
    Output:
        - df: combined with payperiods info

    Purpose:
        - read `Payperiods.csv`, assigned payperiods based on which adjust intervel a transaction date falls into
    """
    # load payperiods
    path_config = read_configs(config_type="io",name="path.json")
    path = Path(path_config["root"]) / Path(path_config["gold"]["payroll"])
    payperiods = pd.read_csv(path/"Payperiods.csv")
    payperiods["START"] = pd.to_datetime(payperiods["START"])
    payperiods["END"] = pd.to_datetime(payperiods["END"])
    payperiods = payperiods.loc[:,["PP","START","END","Cycle","FiscalYear"]]
    payperiods = payperiods.rename(columns={"PP":"PPNum"})
    payperiods["PPName"] = payperiods["Cycle"].astype(str).str.slice(2) + "-" + "PP" + payperiods["PPNum"].astype(str).str.zfill(2)
    # shift transaction dates - AZ: left 12 days, others: left 5 days
    offset_days = {"Arizona (produce)": 12, "Outlook": 12}
    df["days_offset"] = df["Location"].map(offset_days).fillna(5)
    df["date_shifted"] = df[date_col] - pd.to_timedelta(df["days_offset"],unit="days")
    df = df[df["date_shifted"]>=dt.datetime(2021,12,20)].copy(deep=True)
    # construct interval index object for all periods
    idx = pd.IntervalIndex.from_arrays(
        left = payperiods["START"],
        right = payperiods["END"],
        closed = "both"
    )
    # determine which payperiod a transaction date belongs to by identifying the positional index inside the interval index object
    pos = idx.get_indexer(df["date_shifted"])
    # extract payperiod info based on positional indices
    ppnum = payperiods["PPNum"].to_numpy()
    ppname = payperiods["PPName"].to_numpy()
    cycle = payperiods["Cycle"].to_numpy()
    df["PPNum"] = ppnum[pos]
    df["PPName"] = ppname[pos]
    df["Cycle"] = cycle[pos]
    # create mapping for max fiscal year per payperiod to determine which fiscal year a payperiod should bleong to
    mapping_table = df.groupby(["days_offset","PPName"]).agg({"FiscalYear":"max"}).reset_index(drop=False)
    df = df.drop(columns=["FiscalYear"])
    df = pd.merge(df, mapping_table, on=["days_offset", "PPName"], how="left")
    # drop intermediate columns
    df = df.drop(columns=["days_offset", "date_shifted"])
    if write_out:
        df.loc[:,["PPName", "PPNum", "Cycle", "FiscalYear"]].drop_duplicates().to_csv(path.parent/ "OtherTables" / "PayPeriods.csv", index=False)
    return df

def determine_fy(df:pd.DataFrame, date_col:str) -> pd.DataFrame:
    """
    Purpose:
        - identify fiscal year based on date column
    
    Input: 
        - df
        - date_col
    
    Output: 
        - df with `FiscalYear` column
    """
    df[date_col] = pd.to_datetime(df[date_col])
    if "Month" not in df.columns: df["Month"] = df[date_col].dt.month_name()
    df["FiscalYear"] = df[date_col].dt.year 
    mask = df["Month"].isin(["November", "December"])
    df.loc[mask, "FiscalYear"] = df.loc[mask, "FiscalYear"] + 1
    return df

