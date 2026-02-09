"""
Docstring for ETLCodeBase.gold._helpers

Purpose:
    Common methods for gold layer business logic application across all QBO & QBO Time data

Exposed API:
    - `classify_pillar` - classify and create a new column `Pillar` based on `Location` column
    - `standardize_product` - from account names, identify and standardize `commodity`
    - `accid_reroute` - reroute `AccID` based on Finance contract
"""

import pandas as pd

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


