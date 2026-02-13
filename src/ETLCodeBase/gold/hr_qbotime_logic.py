"""
Docstring for ETLCodeBase.gold.hr_qbotime_logic

Purpose:
    Transform QBO Time data (labour hours), part of HR Dashboard

Exposed API
    - `hr_time`
"""

import pandas as pd
from pathlib import Path 
import datetime as dt


from ETLCodeBase.gold._helpers import process_pp, classify_pillar, determine_fy
from ETLCodeBase.utils.filesystem import read_configs

# should be moved to general _helpers
from ETLCodeBase.gold.hr_qbo_logic import _retrieve_bc_ranches


def _combine_bc(group:pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
        - merge BC locations into one, align with payroll QBO table
    """
    group = group.rename(columns={"Location": "Location (detail)"})
    group["Location"] = group["Location (detail)"]
    bc_ranches = _retrieve_bc_ranches()
    group.loc[(group["Location (detail)"].isin(bc_ranches)), "Location"] = "BritishColumbia (cattle)"
    return group

def _clean_location(group:pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
        - clean up location in Group table
    """
    ## Arizona - all produce
    group.loc[((group["corp_short"]=="A")&(group["location_name"]=="Monette Farms AZ")), "Location"] = "Arizona (produce)"
    group.loc[((group["corp_short"]=="A")&(group["location_name"]=="Monette Produce USA")), "Location"] = "Arizona (produce)"
    group.loc[((group["corp_short"]=="A")&(group["location_name"]=="Monette Seeds USA")), "Location"] = "Arizona (produce)"
    ## BC
    group.loc[((group["corp_short"]=="BC")&(group["location_name"]=="Ashcroft Ranch")), "Location"] = "Ashcroft"
    group.loc[((group["corp_short"]=="BC")&(group["location_name"]=="Cache/Fischer/Loon")), "Location"] = "BritishColumbia (cattle)"
    group.loc[((group["corp_short"]=="BC")&(group["location_name"].str.contains("silage", case=False))), "Location"] = "BritishColumbia (cattle)"
    group.loc[((group["corp_short"]=="BC")&(group["location_name"]=="Diamond S Ranch")), "Location"] = "Diamond S"
    group.loc[((group["corp_short"]=="BC")&(group["location_name"]=="Fraser River Ranch")), "Location"] = "Fraser River Ranch"
    group.loc[((group["corp_short"]=="BC")&(group["location_name"]=="Home Ranch (70 Mile, LF/W, BR)")), "Location"] = "Home Ranch"
    group.loc[((group["corp_short"]=="BC")&(group["location_name"]=="Moon Ranch")), "Location"] = "Moon Ranch"
    group.loc[((group["corp_short"]=="BC")&(group["location_name"]=="Produce")), "Location"] = "BritishColumbia (produce)"
    group.loc[((group["corp_short"]=="BC")&(group["location_name"]=="Wolf Ranch")), "Location"] = "Wolf Ranch"
    group.loc[((group["corp_short"]=="BC")&(group["location_name"]=="SAWP")), "Location"] = "BritishColumbia (produce)"
    group.loc[((group["corp_short"]=="BC")&(group["location_name"]=="SAWP Produce")), "Location"] = "BritishColumbia (produce)"
    ## Outlook
    group.loc[((group["corp_short"]=="O")), "Location"] = "Outlook"
    ## others
    group.loc[((group["corp_short"]=="CM")&(group["location_name"]=="Yorkton")), "Location"] = "Yorkton"
    group.loc[((group["corp_short"]=="CM")&(group["location_name"]=="Airdrie")), "Location"] = "Airdrie"
    group.loc[((group["corp_short"]=="CM")&(group["location_name"]=="BC")), "Location"] = "Unassigned"
    group.loc[((group["corp_short"]=="CM")&(group["location_name"]=="Calderbank")), "Location"] = "Calderbank"
    group.loc[((group["corp_short"]=="CM")&(group["location_name"]=="Eddystone")), "Location"] = "Eddystone (unspecified)"
    group.loc[((group["corp_short"]=="CM")&(group["location_name"]=="Hafford")), "Location"] = "Hafford"
    group.loc[((group["corp_short"]=="CM")&(group["location_name"]=="Kamsack")), "Location"] = "Kamsack"
    group.loc[((group["corp_short"]=="CM")&(group["location_name"]=="MFUSA Billings")), "Location"] = "Billings"
    group.loc[((group["corp_short"]=="CM")&(group["location_name"]=="MFUSA Box Elder")), "Location"] = "Havre"
    group.loc[((group["corp_short"]=="CM")&(group["location_name"]=="Nexgen Seeds")), "Location"] = "NexGen"
    group.loc[((group["corp_short"]=="CM")&(group["location_name"]=="Prince Albert")), "Location"] = "Prince Albert"
    group.loc[((group["corp_short"]=="CM")&(group["location_name"]=="Raymore")), "Location"] = "Raymore"
    group.loc[((group["corp_short"]=="CM")&(group["location_name"]=="Regina")), "Location"] = "Regina"
    group.loc[((group["corp_short"]=="CM")&(group["location_name"]=="Russel Approvals")), "Location"] = "Unassigned"
    group.loc[((group["corp_short"]=="CM")&(group["location_name"]=="Seeds")), "Location"] = "Seeds"
    group.loc[((group["corp_short"]=="CM")&(group["location_name"]=="Swift Current")), "Location"] = "Swift Current"
    group.loc[((group["corp_short"]=="CM")&(group["location_name"]=="The Pas")), "Location"] = "The Pas"
    group.loc[((group["corp_short"]=="CM")&(group["location_name"]=="Waldeck")), "Location"] = "Waldeck"
    unclassified = group[group["Location"].isna()].location_name.unique()
    if len(unclassified) > 0: print(f"\nUnclassified location - {unclassified}\n")
    return group

def _merge_tables(group:pd.DataFrame, users:pd.DataFrame, timesheets:pd.DataFrame, jobcode:pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
        - merge dimensions into timesheets table for consolidated view
    """
    timesheets_len, users_len = len(timesheets), len(users)
    # merge location into users
    users = pd.merge(users, group.loc[:,["group_id", "location_name", "Location", "Location (detail)"]].drop_duplicates(), on="group_id", how="left")
    # merge users into timesheets
    timesheets = pd.merge(timesheets,users.loc[:,["user_id", "group_id", "username", "full_name", "location_name","Location","Location (detail)", "first_name", "last_name"]], on="user_id", how="left")
    # merge job into timesheets
    timesheets = pd.merge(timesheets, jobcode.loc[:,["jobcode_id","job_name","type"]].rename(columns={"type":"job_type"}), on="jobcode_id", how="left")
    # sanity check
    if (len(users) != users_len) or (len(timesheets) != timesheets_len):
        raise ValueError(f"duplicated records found, timesheets - {timesheets_len} vs {len(timesheets)}; users - {users_len} vs {len(users)}")
    return timesheets


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
    print(f"Location not in Summarized Time table: {set(units.Location.unique()) - set(summarized.Location.unique()) - set(_retrieve_bc_ranches())}")
    print(f"Location not in Units table: {set(summarized.Location.unique()) - set(units.Location.unique())}")
    # compute per-unit
    summarized = pd.merge(summarized, units, on="Location", how="left")
    summarized["HoursPerUnit"] = summarized["duration"] / summarized["Unit"] * 26
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
    summarized = pd.DataFrame(df.groupby(["Location","PPName","Pillar","FiscalYear","Cycle","PPNum"]).agg({"duration":"sum"}).reset_index(drop=False))
    quick_check = df.groupby(["Location","PPName"]).agg({"duration":"sum"}).reset_index(drop=False)
    if len(summarized) != len(quick_check): 
        raise ValueError(f"Duplicated value detected for Level 1 Time Summary (per Location per PP) - length of summary {len(summarized)} - length of groupby 'Location' & 'PPName' only {len(quick_check)}")
    # per unit calculation
    summarized = _per_unit_calculation(summarized=summarized,path_config=path_config)
    # summarized by location (for all PP)
    summarized2 = summarized.groupby(by=["Location","FiscalYear","Pillar"]).agg({"HoursPerUnit":"mean", "Count":"sum"}).reset_index(drop=False)
    summarized2 = summarized2.rename(columns={"HoursPerUnit":"Avg HoursPerUnit"})
    quick_check = summarized.groupby(by=["Location","FiscalYear"]).agg({"duration":"mean"})
    if len(summarized2) != len(quick_check):
        raise ValueError(f"Duplicated value detected for Level 2 Time Summary (per Location) - length of summary {len(summarized2)} - length of groupby 'Location' & 'FiscalYear' only {len(quick_check)}")
    # summarized by Pillar (for all PP)
    summarized3 = summarized2.groupby(by=["FiscalYear","Pillar"]).agg({"Avg HoursPerUnit":"mean", "Count":"sum"}).reset_index(drop=False)
    quick_check = summarized[summarized["Pillar"]!="Missing"].groupby(by=["Pillar","FiscalYear"]).agg({"duration":"sum"})
    if len(summarized3) != len(quick_check):
        raise ValueError(f"Duplicated value detected for Level 3 Summary (per Pillar) - length of summary {len(summarized3)} - length of groupby 'Pillar' & 'FiscalYear' only {len(quick_check)}")
    return summarized, summarized2, summarized3

def hr_time(write_out:bool=True) -> pd.DataFrame:
    """
    Input: 
        - write_out: write out final data frame to disk

    Output:
        - transformed gold table
    """
    print("\nStarting QBO Time Project Transformation\n")
    path_config = read_configs(config_type="io", name="path.json")
    silver_root = Path(path_config["root"]) / Path(path_config["silver"]["Time"])

    # load and prepare silver tables
    timesheets = pd.read_csv(silver_root/"timesheets.csv")
    jobcode = pd.read_csv(silver_root/"jobcodes.csv")
    users = pd.read_csv(silver_root/"users.csv")
    group = pd.read_csv(silver_root/"group.csv")
    print(f"Read {len(timesheets)} timesheet records, {len(jobcode)} jobcodes, {len(users)} users, {len(group)} groups")

    # clean up location in group table
    group = _clean_location(group=group)

    # consolidate BC
    group = _combine_bc(group=group)

    # merge tables
    timesheets = _merge_tables(group=group, users=users, timesheets=timesheets, jobcode=jobcode)

    # determine Fiscal Year
    timesheets = determine_fy(df=timesheets, date_col="date")

    # classify PP
    timesheets = process_pp(df=timesheets, date_col="date", write_out=False)

    # modify location for BC0
    timesheets.loc[timesheets["user_id"] == "BC6107856", "Location"] = "Unassigned"

    # classify Pillars
    timesheets = classify_pillar(df=timesheets)
    timesheets.loc[timesheets["Pillar"] == "Missing", "Pillar"] = "Unclassified"

    # summarizing
    print("Summarizing ...")
    summarized, summarized2, summarized3 = _summarizing(df=timesheets, path_config=path_config)

    # saving
    print("Saving ...")
    if write_out:
        path_df = Path(path_config["root"]) / Path(path_config["gold"]["QBOTime"])
        path_summary = Path(path_config["root"]) / Path(path_config["gold"]["hr_combined"]) / "CSV"
        timesheets.to_excel(path_df/"QBOTime.xlsx", sheet_name = "QBOTime", index=False)
        summarized.to_csv(path_summary/ "time_summarized1.csv", index=False)
        summarized2.to_csv(path_summary/ "time_summarized2.csv", index=False)
        summarized3.to_csv(path_summary/ "time_summarized3.csv", index=False)
    
    return timesheets

