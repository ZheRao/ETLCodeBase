import pandas as pd
import numpy as np 
import datetime
from pathlib import Path
import os
from typing import List
import argparse

rawpath = Path("../../Database/Bronze/Traction")
rawpath_new = Path("../../Database/Bronze/Traction/New")
outpath = Path("../../Database/Silver/Traction")
os.listdir(rawpath)

def get_ticket_pattern(df, num_digits):
    df["ticket_start"] = df["Ticket #"].str.slice(0,num_digits)
    df["ticket_length"] = df["Ticket #"].str.len().astype(str)
    df["ticket_pattern"] = df["ticket_start"] + "-" + df["ticket_length"]
    return df
def get_delivery_method(entry, train_locations):
    if entry in train_locations:
        return "Train"
    else:
        return "Truck"
def get_product(entry):
    entry = entry.lower()
    if "durum" in entry:
        return "Durum"
    elif "wheat" in entry:
        return "Wheat"
    elif "canola" in entry:
        return "Canola"
    elif ("chickpea" in entry) or ("garbanzo bean" in entry):
        return "Chickpeas"
    elif ("peas" in entry) or ("field pea" in entry):
        return "Peas"
    elif "barley" in entry:
        return "Barley"
    elif "green lentil" in entry:
        return "Green Lentils"
    elif "red lentil" in entry:
        return "Red Lentils"
    elif "oats" in entry:
        return "Oats"
    elif "corn" in entry:
        return "Corn"
    else:
        return "Others"
def get_location(entry):
    entry = entry.lower()
    if "moose jaw" in entry:
        return "Moose Jaw"
    elif "battleford" in entry:
        return "North Battleford"
    elif "north vancouver" in entry:
        return "North Vancouver"
    elif "thunder bay" in entry:
        return "Thunder Bay Ontario"
    elif "clavet" in entry:
        return "Clavet"
    elif "prince rupert" in entry:
        return "Prince Rupert"
    elif "yorkton" in entry:
        return "Yorkton"
    elif "corinne" in entry:
        return "Corinne"
    elif "saskatoon" in entry:
        return "Saskatoon"
    elif "hamlin" in entry:
        return "Hamlin"
    else:
        return "Others"
def determine_terminal(value):
    if "Cargill" in value:
        return "Cargill"
    elif "viterra" in value.lower():
        return "Viterra"
    elif "Louis Dreyfus" in value:
        return "Louis Dreyfus"
    elif "G3" in value:
        return "G3"
    elif ("P & H" in value) or ("P&H" in value):
        return "Parrish & Heimbecker"
    elif "Richardson" in value:
        return "Richardson Pioneer"
    else:
        return "Others"
def determine_farm_canada(x, target="From"):
    if target == "To":
        if x["Terminal"] != "Others":
            return "Invalid"
    entry = x[target]
    if ("Meath Park Yard" in entry) or ("PA:" in entry) or ("W.Star" in entry):
        return "Prince Albert"
    elif "PAS:" in entry:
        return "The Pas"
    elif ("KAM:" in entry) or ("Yorkton" in entry) or ("Kamsack" in entry):
        return "Kamsack"
    elif ("Hafford" in entry) or ("N.Battleford" in entry) or ("Maymont" in entry):
        return "Hafford"
    elif "RAY:" in entry or ("Raymore" in entry):
        return "Raymore"
    elif ("REG:" in entry) or ("Regina" in entry) or ("MJ " in entry) or ("Moose Jaw" in entry):
        return "Regina"
    elif "Outlook" in entry:
        return "Outlook (JV)"
    elif ("Nexgen" in entry):
        return "Nexgen"
    elif ("Swift" in entry) or ("GMD " in entry) or ("SF:" in entry) or ("Wymark" in entry) or ("NF:" in entry) or ("SC:" in entry):
        return "Swift Current"
    else:
        return "Invalid"
def determine_customer_canada(entry):
    terminal = entry["Terminal"]
    farm = entry["Farm Destination"]
    if terminal != "Others":
        return terminal 
    customer = entry["To"].lower()
    if "adm:" in customer:
        return "ADM"
    elif "bunge" in customer:
        return "Bunge Canada"
    elif "dg global" in customer:
        return "DG Global"
    elif "etg commodities" in customer:
        return "ETG Commodities"
    elif "grains connect" in customer:
        return "Grains Connect"
    elif ("monette seeds" in customer) or ("simpsons seeds" in customer):
        return "Monette Seeds"
    elif "paterson grain" in customer:
        return "Paterson Grain"
    elif "belle pulses" in customer:
        return "Belle Pulses"
    elif "prairie malt" in customer:
        return "Prairie Malt ULC Biggar"
    elif "purely canada" in customer:
        return "Purely Canada"
    elif "sunnydale foods" in customer:
        return "Sunnydale Foods"
    elif "adroit " in customer:
        return "Adroit"
    elif ("agt " in customer) or ("agt-" in customer):
        return "AGT"
    elif "ldm foods" in customer:
        return "LDM Foods"
    elif "victoria pulse" in customer:
        return "Victoria Pulse"
    elif "bridge agri" in customer:
        return "Bridge Agri"
    elif "gfi vigro" in customer:
        return "GFI Vigro"
    elif "jgl " in customer:
        return "JGL Commodities"
    elif "remple seeds" in customer:
        return "Remple Seeds"
    elif farm != "Invalid":
        return "Farm"
    else:
        return "Invalid"
def get_farm_usa(entry, target="From"):
    location = entry[target].lower()
    havre_locations = ["bitz yard", "wilson east", "wilson west", "havre", "sally yard", "hadford", "toluca"]
    if ("camp 4" in location) or ("fort smith" in location) or ("c4" in location) or ("c4." in location):
        return "Camp 4"
    elif ("camp 1" in location) or ("fly creek" in location) or ("fc " in location):
        return "Fly Creek"
    contained = False 
    for item in havre_locations:
        if item in location:
            contained = True 
            break 
    if contained:
        return "Havre"
    else:
        return "Invalid"
    
def get_customers_usa(entry):
    farm = entry["Farm Destination"]
    customer = entry["To"].lower()

    if "united grain" in customer:
        return "United Grain"
    elif "columbia grain" in customer:
        return "Columbia Grain"
    elif ("viterra" in customer) or ("gavilon" in customer) or ("huntley" in customer) or ("pile" in customer):
        return "Viterra USA"
    elif "lyft commodity" in customer:
        return "Lyft Commodity Trading"
    elif "monette seeds" in customer:
        return "Monette Seeds USA"
    elif "redwood" in customer:
        return "Redwood Group"
    elif "sinamco" in customer:
        return "Sinamco Trading"
    elif "gsl" in customer:
        return "GSL"
    
    if farm != "Invalid":
        return "Farm"
    else:
        return "Invalid"

def process_traction_data(paths:List[Path|str] = [rawpath,rawpath_new], country:str = "CA", light_load:str = "True", date:str|None = None, outpath:Path = outpath):
    if not (outpath/"Excel").exists():
        os.makedirs(outpath/"Excel")
    
    # reading
    print("Reading Raw Files ...")
    if light_load.lower() == "false":
        if country == "CA":
            path = paths[0] / "Traction_202001_202503"
            traction1 = pd.read_csv(path/"1-Traction_2020Jan_2020Oct.csv", dtype={"To Ticket #": str, "Contract":str})
            traction2 = pd.read_csv(path/"2-Traction_2020Nov_2021Nov.csv", dtype={"To Ticket #": str, "Contract":str})
            traction3 = pd.read_csv(path/"3-Traction_2021Dec_2022Nov.csv", dtype={"To Ticket #": str, "Contract":str})
            traction4 = pd.read_csv(path/"4-Traction_2022Dec_2023Aug.csv", dtype={"To Ticket #": str, "Contract":str})
            traction5 = pd.read_csv(path/"5-Traction_2023Sep_2024Jun.csv", dtype={"To Ticket #": str, "Contract":str})
            traction6 = pd.read_csv(path/"6-Traction_2024Jul_2024Nov.csv", dtype={"To Ticket #": str, "Contract":str})
            traction7 = pd.read_csv(path/"7-Traction_2024Dec_2025Mar.csv", dtype={"To Ticket #": str, "Contract":str})
            traction = pd.concat([traction1, traction2, traction3, traction4, traction5, traction6, traction7],ignore_index=True)
            del traction1, traction2, traction3, traction4, traction5, traction6, traction7
            print(f"Loaded {len(traction)} number of records, proceeding to processing")
        else:
            path = paths[0] / "Traction_us_202210_202503"
            traction_us1 = pd.read_csv(path/"US_Traction_Oct2022_May2024.csv")
            traction_us2 = pd.read_csv(path/"US_Traction_June2024_Mar2025.csv")
            traction_us = pd.concat([traction_us1, traction_us2], ignore_index=True)
            del traction_us1, traction_us2 
            print(f"Loaded {len(traction_us)} number of records, proceeding to processing")
    else:
        path = paths[1]
        if country == "CA":
            assert len(date) == 4, "date must be in the form of MMDD"
            traction = pd.read_csv(path/(f"traction_{date}.csv"), dtype={"To Ticket #": str})
        else:
            traction_us = pd.read_csv(path/(f"traction_us_{date}.csv"), dtype={"To Ticket #": str})
    


    # processing
    print("Processing ...")
    if country == "CA":
        traction = traction.rename(columns={"Timestamp":"Date", "Crop":"Product"})
        traction["Date"] = pd.to_datetime(traction["Date"])
        traction["Last Updated"] = pd.to_datetime(traction["Last Updated"])
        traction["Date"] = traction["Date"].dt.date
        traction["Last Updated"] = traction["Last Updated"].dt.date
        traction = traction.drop(columns=["Files","Transfer type","BOL #"])
        traction["Terminal"] = traction["To"].apply(lambda x: determine_terminal(x))
        traction["Direct Delivery"] = traction.apply(lambda x: True if (("direct" in x.To.lower()) | ("direct" in x.From.lower()) | ("field" in x.From.lower()) | ("field" in x.To.lower())) 
                                                                else False,axis=1)
        traction["Farm Origin"] = traction.apply(lambda x: determine_farm_canada(x), axis=1)
        traction["Farm Destination"] = traction.apply(lambda x: determine_farm_canada(x,target="To"),axis=1)
        traction = traction.rename(columns = {"Product": "Product Detail"})
        traction["Product"] = traction["Product Detail"].apply(lambda x: get_product(x))
        traction["Customer"] = traction.apply(lambda x: determine_customer_canada(x), axis=1)
        traction = traction.drop(columns=["Terminal"])
        traction["Internal Transfer"] = traction.apply(lambda x: False if ((x["Direct Delivery"]) or (x["Customer"] != "Farm")) else True, axis=1)
        traction["Settled"] = traction["Settled"].apply(lambda x: "Yes" if x == True else "No")
        print(f"{len(traction)} records processed")
    else:
        traction_us = traction_us.rename(columns={"Timestamp":"Date", "Crop":"Product"})
        traction_us["Date"] = pd.to_datetime(traction_us["Date"])
        traction_us["Date"] = traction_us["Date"].dt.date
        traction_us = traction_us.drop(columns=["Files","Transfer type","BOL #", "Lot #"])
        traction_us["To Ticket #"] = traction_us["To Ticket #"].astype(str)
        traction_us = traction_us.rename(columns={"Product":"Product Detail"})
        traction_us["Product"] = traction_us["Product Detail"].apply(lambda x: get_product(x))
        traction_us["Direct Delivery"] = traction_us.apply(lambda x: True if (("direct" in x.To.lower()) | ("direct" in x.From.lower())) else False,axis=1)
        traction_us["Pile Delivery"] = traction_us.apply(lambda x: True if (("field" in x.To.lower()) | ("field" in x.From.lower())) else False, axis=1)
        traction_us["Farm Origin"] = traction_us.apply(lambda x: get_farm_usa(x, target="From"), axis=1)
        traction_us["Farm Destination"] = traction_us.apply(lambda x: get_farm_usa(x, target="To"), axis=1)
        traction_us["Customer"] = traction_us.apply(lambda x: get_customers_usa(x), axis=1)
        traction_us["Internal Transfer"] = traction_us.apply(lambda x: False if ((x["Direct Delivery"]) or (x["Customer"] != "Farm")) else True, axis=1)
        traction_us = traction_us.rename(columns={"To Final Qty.":"To Final Qty-bu"})
        canola = 55
        others = 60
        traction_us["To Final Qty."] = traction_us.apply(lambda x: x["To Final Qty-bu"] * canola / 2204 if x["Product"] == "Canola" else x["To Final Qty-bu"] * others / 2204, axis=1)
        traction_us["To Final Qty-ST"] = traction_us.apply(lambda x: x["To Final Qty-bu"] * canola / 2000 if x["Product"] == "Canola" else x["To Final Qty-bu"] * others / 2000, axis=1)
        traction_us["Settled"] = traction_us["Settled"].apply(lambda x: "Yes" if x == True else "No")
        print(f"{len(traction_us)} records processed")
    

    ## saving
    print("Saving files ...")
    if light_load.lower() == "true":
        print("Loading and Appending Old files ...")
        if country == "CA":
            traction_old = pd.read_csv(outpath/"traction.csv")
            traction_old["Date"] = pd.to_datetime(traction_old["Date"])
            traction_old["Date"] = traction_old["Date"].dt.date
            if traction_old.Date.max() >= traction.Date.min():
                traction_old = traction_old[traction_old["Date"] < traction.Date.min()]
            print(f"Old records - {len(traction_old)}")
            traction = pd.concat([traction, traction_old], ignore_index=True)
            traction = traction.drop_duplicates()
            print(f"Total records - {len(traction)}")
        else:
            traction_us_old = pd.read_csv(outpath/"traction_us.csv")
            traction_us_old["Date"] = pd.to_datetime(traction_us_old["Date"])
            traction_us_old["Date"] = traction_us_old["Date"].dt.date
            if traction_us_old.Date.max() >= traction_us.Date.min():
                traction_us_old = traction_us_old[traction_us_old["Date"] < traction_us.Date.min()]
            print(f"Old records - {len(traction_us_old)}")
            traction_us = pd.concat([traction_us, traction_us_old], ignore_index=True)
            traction_us = traction_us.drop_duplicates()
            print(f"Total records - {len(traction_us)}")
    
    if country == "CA":
        traction.to_csv(outpath/"traction.csv", index=False)
        traction.to_excel(outpath/"Excel"/"traction.xlsx", index=False, sheet_name="traction")
    else:
        traction_us.to_csv(outpath/"traction_us.csv",index=False)
        traction_us.to_excel(outpath/"Excel"/"traction_us.xlsx",index=False,sheet_name="traction_us")



def main(args):
    if (args.country.lower() == "ca") or (args.country.lower() == "both"):
        print("Processing Canada ...")
        date = args.date if args.date else None
        process_traction_data(country="CA",light_load=args.light_load, date=date)
    if (args.country.lower() == "usa") or (args.country.lower() == "both"):
        print("Processing USA ...")
        date = args.date if args.date else None
        process_traction_data(country="USA", light_load=args.light_load, date=date)
    print("All Done!")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Transforming Traction Data"
    )
    parser.add_argument(
        "--light_load",
        required=True,
        help="True = only process the new file and append to the old file"
    )
    parser.add_argument(
        "--country",
        required=True,
        help="load CA or USA or both"
    )
    parser.add_argument(
        "--date",
        required=False,
        help="if chose light_load, enter the date of the latest file to extract"
    )
    args = parser.parse_args()
    main(args)