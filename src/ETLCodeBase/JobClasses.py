import pandas as pd
import numpy as np
from pathlib import Path 
import os
import datetime as dt
import requests
import json
from intuitlib.client import AuthClient
import urllib.parse
from urllib.parse import urlparse, unquote, urljoin
from time import perf_counter
import re
import yaml
from io import StringIO
from bs4 import BeautifulSoup
import time
from ETLCodeBase.utils.filesystem import read_configs


class Job:
    def __init__(self):
        base_dir = Path("c:/Users/ZheRao/OneDrive - Monette Farms/Monette Farms Team Site - Innovation Projects/Production/Database")
        self.json_download_path = Path("c:/Users/ZheRao/OneDrive - Monette Farms/Desktop/Work Files/Projects/5 - HP Data")
        self.base_dir = base_dir
        self.today = dt.date.today()
        month_format = "".join(["0",str(self.today.month)]) if self.today.month < 10 else str(self.today.month)
        self.us_companies = ["MFUSA", "MFAZ", "MSUSA", "MPUSA"]
        self.company_names = self.us_companies + ["MSL", "NexGen", "MFBC", "MPL", "MFL"]
        self.raw_path = {
            "QBO": {
                "Raw": base_dir/"Bronze"/"QBO"/"Raw"/f"{self.today.year}_{self.today.month}",
                "GL": base_dir/"Bronze"/"QBO"/"GeneralLedger",
                "PL": base_dir/"Bronze"/"QBO"/"ProfitAndLoss",
                "Time":base_dir/"Bronze"/"QBOTime",
                "APAR": base_dir/"Bronze"/"QBO"/"APAR"
            },
            "Delivery": {"Traction":base_dir/"Bronze"/"Traction", "HP":base_dir/"Bronze"/"HarvestProfit"},
            "Auth": {"QBO":base_dir/"Bronze"/"Authentication"/"QBO", "QBOTime": base_dir/"Bronze"/"Authentication"/"QBOTime",
                     "Harvest Profit": base_dir/"Bronze"/"Authentication"/"Harvest Profit"},
            "Log": base_dir/"Load_History"/f"{self.today.year}"/month_format
        }
        self.silver_path = {
            "QBO": {
                "Dimension_time": base_dir/"Silver"/"QBO"/"Dimension"/f"{self.today.year}_{self.today.month}",
                "Dimension": base_dir/"Silver"/"QBO"/"Dimension",
                "Raw": base_dir/"Silver"/"QBO"/"Fact"/"Raw",
                "PL": base_dir/"Silver"/"QBO"/"Fact"/"ProfitAndLoss",
                "GL": base_dir/"Silver"/"QBO"/"Fact"/"GeneralLedger",
                "Time": base_dir/"Silver"/"QBOTime",
                "APAR": base_dir/"Silver"/"QBO"/"Fact"/"APAR"
            },
            "Delivery": {"Traction":base_dir/"Silver"/"Traction", "HP":base_dir/"Silver"/"HarvestProfit"}
        }
        
    
    def get_fx(self):
        key  = os.getenv("ALPHAVANTAGE_KEY")
        url  = ("https://www.alphavantage.co/query?"
                "function=CURRENCY_EXCHANGE_RATE"
                "&from_currency=USD&to_currency=CAD"
                f"&apikey={key}")
        rate = float(requests.get(url, timeout=10).json()
                    ["Realtime Currency Exchange Rate"]["5. Exchange Rate"])
        self.fx = rate
    
    def create_log(self, path: Path) -> None:
        self.check_file(path)
        day_format = "".join(["0",str(self.today.day)]) if self.today.day < 10 else str(self.today.day)
        self.log = open(path/(day_format+"_Log.txt"), "a")

    
    def close_log(self):
        self.log.close()

    def check_file(self, path: Path) -> None:
        if not Path.exists(path):
            os.makedirs(path)
    
    def formulate_date(self, df:pd.DataFrame, date_cols:list[str], drop_time:bool=True) -> pd.DataFrame:
        """ 
            format the date columns into datetime format
        """
        assert len(set(date_cols) - set(df.columns)) == 0, "Not all columns passed are inside dataframe passed"
        for col in date_cols:
            df[col] = pd.to_datetime(df[col], utc=True)
            if drop_time: df[col] = df[col].dt.date
        return df

class QBOETL(Job):
    """
        optionally run QBO API raw, GL, PL extraction and transformation
    """

    def __init__(self, use_live_fx: bool=True): 
        super().__init__()
        self.names = ["Deposit","CreditMemo", "VendorCredit", "Invoice", "SalesReceipt", "Purchase", "Bill", "JournalEntry"]
        self.other_names = ["Account", "Customer", "Department", "Item", "Term", "Vendor", "Class", "Employee"]
        self.PL_raw_cols = ["TransactionDate", "TransactionType", "DocNumber", "Name", "Location", "Class", "Memo", "SplitAcc", "Amount", "Balance"]
        self.GL_raw_cols = ["TransactionDate", "TransactionType", "DocNumber", "IsAdjust", "Name", "Memo", "SplitAcc", "Amount", "Balance"]
        self.acctype_QBO_expense = ["Expense","Cost of Goods Sold", "Other Expense"]
        self.profittype_cube_expense = ["Cost of Goods Sold", "Direct Operating Expenses", "Operating Overheads", "Other Expense"]
        fx_config = read_configs(config_type="state",name="fx.json")
        default_fx = fx_config["fx"]
        if use_live_fx:
            try:
                self.get_fx()
                print(f"FX - {self.fx}")
            except:
                print(f"cannot extract live FX rate, default to {default_fx}")
                self.fx = default_fx
        else:
            print(f"not pulling live FX - rate - {default_fx}")
            self.fx = default_fx
        
    
    # QBO Extraction methods
    def _refresh_auth_client(self, company: str, secret: dict[str, str]) -> AuthClient:
        """ 
            create auth_client object for company called with, return auth_client for data extraction
        """
        mode = "production"
        # create auth_client object
        if company in ["MFUSA","MPUSA","MFAZ","MSUSA"]:
            auth_client = AuthClient(client_id = secret["USA"]["client_id"],
                            client_secret = secret["USA"]["client_secret"],
                            redirect_uri = "https://developer.intuit.com/v2/OAuth2Playground/RedirectUrl",
                            environment = mode)
        else:
            auth_client = AuthClient(client_id = secret["CA"]["client_id"],
                                    client_secret = secret["CA"]["client_secret"],
                                    redirect_uri = "https://developer.intuit.com/v2/OAuth2Playground/RedirectUrl",
                                    environment = mode)
        # assign tokens
        with open(self.raw_path["Auth"]["QBO"]/"tokens.json", "r") as f:
            tokens = json.load(f)
        auth_client.access_token = tokens[company]["access_token"]
        auth_client.refresh_token = tokens[company]["refresh_token"]
        auth_client.realm_id = tokens[company]["realm_id"]
        # refresh
        auth_client.refresh()
        # save refreshed tokens
        tokens[company]["access_token"] = auth_client.access_token 
        tokens[company]["refresh_token"] = auth_client.refresh_token 
        tokens[company]["realm_id"] = auth_client.realm_id 
        with open(self.raw_path["Auth"]["QBO"]/"tokens.json", "w") as f:
            json.dump(tokens, f, indent=4)
        return auth_client 

    def _pull_raw(self, table_names: str, auth_client: AuthClient, company: str) -> None:
        """ 
            pull all the raw tables from QBO and store them as json files in /Bronze/QBO/Raw/year_month
        """
        base_url = 'https://quickbooks.api.intuit.com'
        for name in table_names:
            start, max_results = 1, 100
            data = []
            total_records = 0 
            path = self.raw_path["QBO"]["Raw"] / company 
            self.check_file(path)
            while True:
                if name in ["Account", "Customer", "Department", "Item", "Term", "Vendor", "Class", "Employee"]:
                    query = f"SELECT * FROM {name} WHERE Active IN (true,false) STARTPOSITION {start} MAXRESULTS {max_results}"
                else:
                    query = f"SELECT * FROM {name} STARTPOSITION {start} MAXRESULTS {max_results}"
                encoded_query = urllib.parse.quote(query)
                url = f"{base_url}/v3/company/{auth_client.realm_id}/query?query={encoded_query}"
                headers = {
                    'Authorization': f'Bearer {auth_client.access_token}',
                    'Accept': 'application/json'
                }
                response = requests.get(url, headers=headers)
                if (response.json().get("QueryResponse", -1) == -1) or (len(response.json()["QueryResponse"]) < 1):
                    break
                else:
                    info = response.json()["QueryResponse"][name]
                    data.extend(info)
                    total_records += len(info)
                    if len(info) < max_results:
                        break
                    start += max_results
            self.log.write(f"{name} - total records - {total_records}\n")
            if total_records != 0:
                file_path = f"{path}/{name}.json"
                with open(file_path, "w", encoding="utf-8", newline="\n") as f:
                    json.dump(data, f, indent=4)

    def _pull_reports(self, auth_client: AuthClient, company: str, light_load: bool=True, report_type: str="PL",
                      minor_version: int=75) -> None:
        """ 
            pull PL and GL (optional light_load for the latest fiscal year) and store them as json files at 
                PL: Bronze/QBO/ProfitAndLoss
                GL: Bronze/QBO/GeneralLedger
            
        """
        # ensure parameters are entered as required
        assert report_type == "PL" or report_type == "GL", "report_type must be entered as one of {PL, GL}"
        
        # determine year and month start for light and full load
        month = 10
        if light_load:
            year = 2024
        else:
            year = 2018
        start = dt.date(year, month, 1) 

        # hyperparameters
        REALM_ID, TOKEN, BASE_URL = auth_client.realm_id, auth_client.access_token, "https://quickbooks.api.intuit.com"
        headers = {
            "Authorization": f"Bearer {TOKEN}",
            "Accept": "application/json"
        }
        report_name = "ProfitAndLossDetail" if report_type == "PL" else "GeneralLedger"

        # define path for storage
        path_out = self.raw_path["QBO"][report_type] / company 
        self.check_file(path_out)

        # extraction
        while start <= self.today:
            # determine slice end (updates EndPeriod automatically), quarter end for Q1, Q4 ends in 31st, and for Q2, Q3 ends in 30
            match start.month:
                case 10:
                    end = dt.date(start.year, 12, 31)
                    year = end.year + 1 
                case 1:
                    end = dt.date(start.year, start.month + 2, 31)
                case 4|7:
                    end = dt.date(start.year, start.month+2, 30) 
            # build URL for report extraction with filters and columns
            params = {
                "minorversion": minor_version,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "columns": "all"
            }
            resp = requests.get(f"{BASE_URL}/v3/company/{REALM_ID}/reports/{report_name}", headers=headers, params=params)
            resp.raise_for_status()
            slice_json = resp.json()
            # save report content
            rows = slice_json.get("Rows", {}).get("Row", []) # extract content (i.e., entries) of report
            if rows:
                with open(path_out/(str(start.year)+"_"+str(start.month)+".json"), "w") as f: # for light_load, it will only overwrite 
                    json.dump(slice_json, f, indent=4)
            
            # advance start for next slice
            start = dt.date(end.year, end.month+1, 1) if year == end.year else dt.date(end.year+1, 1, 1)
    
    def _pull_APAR(self, auth_client: AuthClient, company:str, minor_version: int=75, report_type: str="AP") -> None:
        """ 
            This function pulls APAgingDetail report (for now), for all outstanding transactions
        """
        # ensure report type is entered correctly
        report_type = report_type.upper()
        assert report_type in ["AP", "AR"], f"please use 'AP' or 'AR' for report type, entered {report_type}"
        # hypterparameters
        params = {
            "minorversion": minor_version,
            "report_date": "-".join([str(self.today.year+1),str(self.today.month),str(self.today.day)])
        }
        REALM_ID, TOKEN, BASE_URL = auth_client.realm_id, auth_client.access_token, "https://quickbooks.api.intuit.com"
        headers = {
            "Authorization": f"Bearer {TOKEN}",
            "Accept": "application/json"
        }
        report_name = "AgedPayableDetail" if report_type == "AP" else "AgedReceivableDetail"
        # define path for storage
        path_out = self.raw_path["QBO"]["APAR"] / report_name / str(self.today.year) / str(self.today.month) / company 
        self.check_file(path_out)
        # extract
        resp = requests.get(f"{BASE_URL}/v3/company/{REALM_ID}/reports/{report_name}", headers=headers, params=params)
        resp.raise_for_status()
        results = resp.json()
        with open(path_out/(str(self.today.day) + ".json"), "w") as f:
            json.dump(results, f, indent=4)


    # QBO Raw Processing methods
    def _raw_get_lineitem(self, line: pd.Series):
        """ 
            extraction lines within raw tables that has non-summarized inforamtion
        """
        lineitems = []
        skip_list = ['DescriptionOnly','SubTotalLineDetail']
        for i in range(len(line)):
            item = line[i].get("DetailType", -1)
            # extract line item if it has an id, and the line doesn't contain summarized information
            if (line[i].get("Id", -1) != -1) and (item not in skip_list):
                lineitems.append(line[i])
        return lineitems
    
    def _raw_sep_entity(self, df: pd.DataFrame) -> pd.DataFrame:
        """ 
            create VendorID, CustomerID, EmployeeID column based on entity type
        """
        assert "EntityType" in df.columns and "EntityID" in df.columns, "Entity column missing from dataframe"
        for i in range(len(df)):
            if isinstance(df.loc[i,"EntityType"],str):
                match df.loc[i,"EntityType"].title():
                    case "Vendor":
                        df.loc[i,"VendorID"] = df.loc[i,"EntityID"]
                    case "Customer":
                        df.loc[i,"CustomerID"] = df.loc[i,"EntityID"]
                    case "Employee":
                        df.loc[i,"EmployeeID"] = df.loc[i,"EntityID"]
        return df

    def _pre_process_cubefile(self) -> None:
        """ 
            This function converts the new excel cube acc classification file into the old csv file format for the program to process
        """
        # location excel file
        cube_file = [name for name in os.listdir(self.raw_path["QBO"]["Raw"].parent) if ".xlsx" in name]
        assert len(cube_file) == 1, "multiple cube files detected"
        cube_file = cube_file[0]
        # read excel file
        maps = pd.read_excel(self.raw_path["QBO"]["Raw"].parent/cube_file,sheet_name="Account",usecols=["Current Name", "Parent Dimension Member"])
        maps = maps.rename(columns={"Current Name":"child", "Parent Dimension Member":"parent"})
        # recreate account:income statement:... structure for each account
        records = []
        structure = ["Account", "Income Statement"]
        for i in range(1, len(maps)):
            # print()
            # print(i)
            # print(f"structure {structure}")
            # print(f"records {records}")
            child, parent = maps.loc[i, :]
            idx = structure.index(parent)
            structure = structure[:idx+1]
            structure.append(child)
            if self._raw_split_name(child)[0]:
                records.append(":".join(structure))
        # save file
        pd.DataFrame({"Account":records}).to_csv(self.raw_path["QBO"]["Raw"].parent/"Cube Dimensions - Monette Farms.csv",index=False)

    def _raw_split_name(self, accname: str):
        """ 
            helper function for formatting cube file, split a full account name into corp + accnum + accname if the name is a legit accname
                the return list is formatted as [is_acc, corp, accnum, accname], -1 is used when it is not an actual account
                expected actual accname to have this format: 'MFL 000000 Sales'
        """
        name_list = accname.split(" ")
        if len(name_list) < 3:
            return False, -1, -1, -1
        if (name_list[0] in self.company_names) and (len(name_list[1]) == 6) and (name_list[1].isnumeric()):
            return True, name_list[0], name_list[1], accname[len((" ".join(name_list[:2])))+1:]
        return False, -1, -1, -1
    
    def _raw_formulate_accounts(self, df:pd.DataFrame) -> pd.DataFrame:
        """ 
            convert cube account file into pandas dataframe, file contains one column 'Account' with accounts hierarchy embeded in one string
        """
        df = df.reset_index(drop=True)
        for i in range(len(df)):
            # extract account hierarchy list
            full_name = df.loc[i,"Account"]
            # print(full_name)
            name_list = full_name.split(":")
            # for irregular accounts classification from the finance team, skip, e.g., Account: MFBC 626080 Miscellaneous Insurance!!
            if len(name_list) < 4:
                continue
            # determine the last item in the hierarchy list is actually an account and it is part of the corps_list, otherwise ignore this entry
            fullname = name_list[-1]
            is_account, corp, accnum, accname = self._raw_split_name(fullname)
            if not is_account:
                continue 
            # extract account info for the child account
            df.loc[i,"AccNum"] = corp + accnum 
            df.loc[i,"AccName"] = accname 
            df.loc[i,"Corp"] = corp 
            root_corp = corp 
            # extract   AccountingType (e.g., Income Statement/Balance Sheet, ...),
            #           Profititem (e.g., Sales Revenue/Direct Operating Expenses, ...),
            #           Category (e.g., Seed Processing/Grain Revenue, ...),
            #           Subcategory - optional (e.g., Grain - cash settlements/COS - Freight - Insurance & Other, ...)
            df.loc[i,"AccountingType"] = name_list[0]
            df.loc[i,"ProfitType"] = name_list[1]
            df.loc[i,"Category"] = name_list[2]
            ## determine if the next field is parent account or subcategory
            is_account, corp, accnum, accname = self._raw_split_name(name_list[3])
            if (not is_account) and (root_corp not in name_list[3]):
                df.loc[i,"Subcategory"] = name_list[3]
        return df[df["AccNum"].notnull()]

    def _raw_construct_LinkedTxn_table(self, df:pd.DataFrame, df_type:str, name:str="LinkedTxn_Mapping") -> None:
        """ 
            construct LinkedTxn tables from bill, invoie, salesreceipts for identifying what activities are transactions from bank accounts associated with
        """
        path = self.silver_path["QBO"]["Raw"] / "LinkedTxn"
        self.check_file(path)
        df = df.loc[:,["TransactionID","LinkedTxn","AccID", "Corp"]]
        df["TransactionID"] = df["TransactionID"].apply(lambda x: x.split("-")[1])
        df = pd.json_normalize(df.to_dict(orient="records"), meta=["TransactionID","AccID","Corp"], record_path=["LinkedTxn"], sep="_")
        df = df[df["TxnId"].notna()]
        df["TxnId"] = df["Corp"] + df["TxnId"]
        df.to_csv(Path(path) / (name+"_"+df_type+".csv"), index=False)
    
    def _raw_read_jsons(self, df_type:str, concat_cols:list[str], is_fact: bool = False,
               first_cols:list[str] = []) -> pd.DataFrame:
        """ 
            read all raw tables for df_type (e.g., bill), combine them, flatten and convert from json format to csv
        """
        dfs: list[pd.DataFrame] = []
        missing = []
        match df_type.lower():
            case "salesreceipt":
                df_type = "SalesReceipt"
            case "journalentry":
                df_type = "JournalEntry"
            case "creditmemo":
                df_type = "CreditMemo"
            case "vendorcredit":
                df_type = "VendorCredit"
            case _:
                df_type = df_type.title()
        print(f"Processing {df_type}")
        self.log.write(f"\nProcessing {df_type}\n")
        for corp in self.company_names:
            self.log.write(f"Processing {df_type} {corp}\n")
            path = self.raw_path["QBO"]["Raw"] / corp / f"{df_type}.json"
            if not path.exists():
                missing.append(corp)
                continue
            # read file
            df = pd.read_json(path, dtype={col:str for col in concat_cols})
            # replace string version of NULL to actual NULL values - to avoid id like MFL-nan
            if "AcctNum" in df.columns:
                df["AcctNum"] = df["AcctNum"].replace({"None":np.nan, "nan":np.nan})
            if "DocNumber" in df.columns:
                df["DocNumber"] = df["DocNumber"].replace({"None":np.nan, "nan":np.nan})
            if len(df) < 1:
                continue
            # flatten json format
            df = pd.json_normalize(df.to_dict(orient="records"),sep="_")
            # process line items for further flatten
            if is_fact:
                df["Line"] = df["Line"].apply(lambda x: self._raw_get_lineitem(x))
                first_cols_copy = [col for col in first_cols if col in df.columns]
                df = df.loc[:, first_cols_copy]
                df = pd.json_normalize(df.to_dict(orient="records"), record_path=["Line"], meta=first_cols, record_prefix="Line_", sep="_",errors="ignore")
            if 'LinkedTxn' in concat_cols:  # should concat LinkedTxn_Id, not LinkedTxn column
                concat_cols.remove('LinkedTxn')
            # concat corp with ID
            for col in concat_cols:
                middle = "" 
                if col == "DocNumber":
                    middle = "-"
                if col in df.columns and not df[col].isna().all():
                    mask = df[col].notna()
                    df.loc[mask,col] = corp + middle + df[col].astype(str)
            df["Corp"] = corp
            dfs.append(df)
        if len(missing) > 0:
            self.log.write(f"\nCorps missing {df_type}: {missing}\n\n")
        # concat all dfs
        if not dfs:     # if dfs is empty
            return pd.DataFrame()
        df = pd.concat(dfs, ignore_index=True)
        return df
    
    def _raw_processing_combined(self, df: pd.DataFrame, df_col: list[str], df_rename: dict[str,str], df_type: str, 
                                    process_line:bool = False) -> None:
        """ 
            for a df_type, after combining all corps, this function transforms them
        """
        # only take relevant columns
        df = df.loc[:, df_col]
        match df_type.lower():
            case "salesreceipt":
                df_type = "SalesReceipt"
            case "journalentry":
                df_type = "JournalEntry"
            case "vendorcredit":
                df_type = "VendorCredit"
            case "creditmemo":
                df_type = "CreditMemo"
            case _:
                df_type = df_type.title()
        # if table contains line items - facts
        if process_line:
            df["Line_Id"] = df["Line_Id"].astype(str)
            df["TransactionID"] = df_type[0] + "-" + df["Id"] + "-" + df["Line_Id"] # create a unique ID for each line item
            df["TransactionType"] = df_type
            df = df.rename(columns={"Id": "TransactionID_partial"})
            path = self.silver_path["QBO"]["Raw"]
            mode = "Fact"
        # dimensions
        else:
            path = self.silver_path["QBO"]["Dimension_time"]
            path2 = self.silver_path["QBO"]["Dimension"] / "CSV"
            self.check_file(path2)
            mode = "Dimension"
        self.check_file(path)
        # drop Line column - it's already been expanded
        if "Line" in df.columns:
            df = df.drop(columns=["Line"])
        # rename columns and format dates
        df = df.rename(columns=df_rename)
        df = self.formulate_date(df,drop_time=True,date_cols=["CreateDate","LastUpdateDate"])
        # append Acc to Invoice and SalesReceipt because they are only tied to Item table - must process dimensions first
        if df_type in ["Invoice","SalesReceipt"]:
            item = pd.read_csv(self.silver_path["QBO"]["Dimension_time"]/"Item.csv")
            # should only link incomeacc because Invoice and SR are income
            df = pd.merge(df,item.loc[:,["ItemID","IncomeAccountID"]].rename(columns={"IncomeAccountID":"AccID"}), on="ItemID", how="left")
        # append Acc to bill where bill_type is Item_based
        if df_type == "Bill":
            item = pd.read_csv(self.silver_path["QBO"]["Dimension_time"]/"Item.csv")
            df_itembased = df[df["ItemID"].notna()].copy(deep=True)
            df_itembased = df_itembased.drop(columns=["AccID"])
            df = df[df["ItemID"].isna()]
            df_itembased = pd.merge(df_itembased,item.loc[:,["ItemID","ExpenseAccountID"]].rename(columns={"ExpenseAccountID":"AccID"}), on="ItemID", how="left")
            df = pd.concat([df,df_itembased],ignore_index=True)
        # construct LinkedTxn table if applicable
        if "LinkedTxn" in df.columns:
            self._raw_construct_LinkedTxn_table(df.copy(deep=True),df_type=df_type)
            df = df.drop(columns=['LinkedTxn'])
        # process accounts - merge cube classification
        if df_type.lower() == "account":
            account_cube = pd.read_csv(self.silver_path["QBO"]["Dimension"]/"accounts_cube_classified.csv")
            account1 = df[df["AccNum"].isna()].copy(deep=True)
            account = df[df["AccNum"].notna()]
            account = pd.merge(account,account_cube.loc[:,["AccNum", "AccountingType", "ProfitType" ,"Category","Subcategory"]],on="AccNum",how="left")
            df = pd.concat([account,account1],ignore_index=True)
            df["DisplayName"] = df["AccNum"] + " " + df["AccName"]
        # split Entity columns
        if "EntityID" in df.columns:
            df = self._raw_sep_entity(df)
            df = df.drop(columns=["EntityType", "EntityID"])
        # save processed files
        df.to_csv(path/ f"{df_type}.csv",index=False) 
        if mode == "Dimension":
            df.to_csv(path2/ f"{df_type}.csv",index=False) 
    
    def _raw_cube_processing(self) -> None:
        """ 
            process and save cube accounts from bronze to silver
        """
        self._pre_process_cubefile()
        account_cube = pd.read_csv(self.raw_path["QBO"]["Raw"].parent/"Cube Dimensions - Monette Farms.csv")
        account_cube = account_cube[account_cube["Account"].str.startswith("Account:")].copy(deep=True)
        account_cube["Account"] = account_cube["Account"].str.replace("Account:","")
        account_cube = account_cube[~account_cube["Account"].str.contains("DNU")].copy(deep=True)
        account_cube = self._raw_formulate_accounts(account_cube)
        account_cube = account_cube.sort_values(by=["AccNum"]).drop_duplicates(subset=["AccNum"],keep="last",ignore_index=True)
        account_cube.to_csv(self.silver_path["QBO"]["Dimension"]/"accounts_cube_classified.csv",index=False)
        #return account_cube
    
    def _raw_transform(self) -> None:
        """ 
            Raw Processing
        """
        self.log.write("\n\nPerforming Raw Transformation\n\n")
        print("\nRaw Transformation ...")
        self._raw_cube_processing()
        # Dimensions
        ## account
        concat_cols = ["Id","AcctNum","ParentRef_value"]
        account_all = self._raw_read_jsons(df_type="account", concat_cols=concat_cols)
        account_col = ["Id", "AcctNum", "Name", "FullyQualifiedName", "Classification", "AccountType", "SubAccount", "Active", "CurrencyRef_value", "ParentRef_value",
                    "MetaData_CreateTime", "MetaData_LastUpdatedTime", "Corp", "AccountSubType"]
        account_rename = {"Id": "AccID", "Name": "AccName", "CurrencyRef_value": "CurrencyID", "ParentRef_value": "ParentAccID",
                        "MetaData_CreateTime": "CreateDate", "MetaData_LastUpdatedTime": "LastUpdateDate", "AcctNum":"AccNum",
                        "AccountSubType": "DetailType"}
        self._raw_processing_combined(account_all,df_col=account_col,df_rename=account_rename,df_type="account")
        ## item
        items = self._raw_read_jsons(df_type="Item", concat_cols=["Id","IncomeAccountRef_value","ExpenseAccountRef_value","ParentRef_value"])
        item_col = ["Id", "Name", "Description","FullyQualifiedName", "Active", "Taxable", "Type", "IncomeAccountRef_value", "ExpenseAccountRef_value",
                    "MetaData_CreateTime", "MetaData_LastUpdatedTime", "SubItem", "ParentRef_value", "Level", "Corp"]
        item_rename = {"Id": "ItemID", "Name": "ItemName","IncomeAccountRef_value": "IncomeAccountID", "ExpenseAccountRef_value": "ExpenseAccountID",
                    "MetaData_CreateTime": "CreateDate", "MetaData_LastUpdatedTime": "LastUpdateDate", "ParentRef_value": "ParentID"}
        self._raw_processing_combined(items,df_col=item_col,df_rename=item_rename,df_type="Item")
        ## class
        Class = self._raw_read_jsons(df_type="class",concat_cols=["Id","ParentRef_value"])
        Class_col = ["Id", "Name", "FullyQualifiedName", "Active", "SubClass", "ParentRef_value", "MetaData_CreateTime", "MetaData_LastUpdatedTime", "Corp"]
        Class_rename = {"Id": "ClassID", "Name": "ClassName", "ParentRef_value": "ParentID", 
                        "MetaData_CreateTime": "CreateDate", "MetaData_LastUpdatedTime": "LastUpdateDate"}
        self._raw_processing_combined(Class,df_col=Class_col,df_rename=Class_rename,df_type="class")
        ## farm
        farm = self._raw_read_jsons(df_type="department",concat_cols=["Id"])
        farm_col = ["Id", "Name", "FullyQualifiedName", "SubDepartment", "Active", "MetaData_CreateTime", "MetaData_LastUpdatedTime","Corp"]
        farm_rename = {"Id": "FarmID", "Name": "FarmName", "MetaData_CreateTime": "CreateDate", "MetaData_LastUpdatedTime": "LastUpdateDate"}
        self._raw_processing_combined(farm,farm_col,farm_rename,"farm")
        ## customer 
        customer = self._raw_read_jsons(df_type="Customer",concat_cols=["Id","SalesTermRef_value"])
        customer_col = ["Id", "FullyQualifiedName", "Active", "Balance", "CurrencyRef_value", "BillAddr_Id", "BillAddr_Line1", "BillAddr_City", "BillAddr_CountrySubDivisionCode",
                        "BillAddr_PostalCode", "PrimaryEmailAddr_Address", "PrimaryPhone_FreeFormNumber", "Mobile_FreeFormNumber",
                        "MetaData_CreateTime", "MetaData_LastUpdatedTime", "SalesTermRef_value", "Corp"]
        customer_rename = {"Id": "CustomerID", "FullyQualifiedName": "CustomerName", "CurrencyRef_value": "CurrencyID", "BillAddr_Id": "BillAddr_ID", 
                        "BillAddr_CountrySubDivisionCode": "BillAddr_Province", "PrimaryEmailAddr_Address": "Email", 
                        "PrimaryPhone_FreeFormNumber": "Phone", "Mobile_FreeFormNumber": "MobilePhone",
                        "MetaData_CreateTime": "CreateDate", "MetaData_LastUpdatedTime": "LastUpdateDate",
                        "SalesTermRef_value":"TermID"}
        self._raw_processing_combined(customer,customer_col,customer_rename,"customer")
        ## vendor
        vendor = self._raw_read_jsons(df_type="vendor",concat_cols=["Id"])
        vendor_col = ["Id", "DisplayName", "Active", "Vendor1099", "Balance", "PrimaryPhone_FreeFormNumber","PrimaryEmailAddr_Address",
                    "CurrencyRef_value", "BillAddr_Id", "BillAddr_Line1", "BillAddr_City", "BillAddr_CountrySubDivisionCode","BillAddr_PostalCode",
                    "MetaData_CreateTime", "MetaData_LastUpdatedTime", "Corp"]
        vendor_rename = {"Id": "VendorID", "DisplayName": "VendorName", "PrimaryPhone_FreeFormNumber": "Phone",
                        "PrimaryEmailAddr_Address": "Email", "CurrencyRef_value": "CurrencyID", "BillAddr_Id": "BillAddr_ID", 
                        "BillAddr_CountrySubDivisionCode": "BillAddr_Province",
                        "MetaData_CreateTime": "CreateDate", "MetaData_LastUpdatedTime": "LastUpdateDate"}
        self._raw_processing_combined(vendor,vendor_col,vendor_rename,"vendor")
        ## term
        term = self._raw_read_jsons(df_type="term",concat_cols=["Id"])
        term_rename = {"Id": "TermID", "Name": "TermName", "MetaData_CreateTime": "CreateDate", "MetaData_LastUpdatedTime": "LastUpdateDate"}
        self._raw_processing_combined(term,term.columns,term_rename,"term")
        # facts
        ## invoice
        invoice_cols = ["Id", "TxnDate", "DepartmentRef_value", "CurrencyRef_value", "CustomerRef_value", "ClassRef_value",
                    "TotalAmt", "Line","MetaData_CreateTime", "MetaData_LastUpdatedTime","DocNumber", "LinkedTxn"]
        concat_cols = ["Id","DepartmentRef_value","CustomerRef_value","DocNumber","LinkedTxn_TxnId","ClassRef_value",
                    "Line_SalesItemLineDetail_ClassRef_value", "Line_SalesItemLineDetail_ItemRef_value"]
        invoice = self._raw_read_jsons(df_type="Invoice",is_fact=True,first_cols=invoice_cols,concat_cols=concat_cols)
        invoice_cols_other = ["Line_Id","Line_Description", "Line_Amount", "Line_SalesItemLineDetail_UnitPrice","Line_SalesItemLineDetail_Qty",
                                "Line_SalesItemLineDetail_ClassRef_value", "Line_SalesItemLineDetail_ItemRef_value","Corp"]
        invoice_cols.extend(invoice_cols_other)
        invoice_rename = {"TxnDate":"TransactionDate", "DepartmentRef_value":"FarmID", 
                            "CurrencyRef_value":"CurrencyID","CustomerRef_value":"CustomerID", "MetaData_CreateTime": "CreateDate", 
                            "MetaData_LastUpdatedTime": "LastUpdateDate","ClassRef_value":"ClassID"}
        invoice_rename_other = {"Line_Description":"TransactionEntered", "Line_SalesItemLineDetail_UnitPrice":"UnitPrice",
                            "Line_SalesItemLineDetail_Qty":"Qty", "Line_SalesItemLineDetail_ClassRef_value":"ClassID2", "Line_Amount":"Amount",
                            "Line_SalesItemLineDetail_ItemRef_value":"ItemID",}
        invoice_rename.update(invoice_rename_other)
        self._raw_processing_combined(invoice,invoice_cols,invoice_rename,df_type="invoice",process_line=True)
        ## SalesReceipt
        sales_col = ["Id", "TxnDate", "DepartmentRef_value", "CurrencyRef_value", "CustomerRef_value", "TotalAmt","Line","MetaData_CreateTime", "MetaData_LastUpdatedTime","DocNumber",
                    "LinkedTxn", "ClassRef_value"]
        concat_cols = ["Id", "DepartmentRef_value", "CustomerRef_value", "DocNumber","LinkedTxn_TxnId","ClassRef_value",
                        "Line_SalesItemLineDetail_ItemRef_value", "Line_SalesItemLineDetail_ClassRef_value"]
        sales = self._raw_read_jsons(df_type="salesreceipt",concat_cols=concat_cols,is_fact=True,first_cols=sales_col)
        sales_col_other = ["Line_Id", "Line_Description", "Line_Amount", "Line_SalesItemLineDetail_ItemRef_value",
                        "Line_SalesItemLineDetail_ClassRef_value", "Line_SalesItemLineDetail_UnitPrice",
                        "Line_SalesItemLineDetail_Qty","Corp"]
        sales_col.extend(sales_col_other)
        sales_rename = {"TxnDate": "TransactionDate", "DepartmentRef_value": "FarmID", "CurrencyRef_value": "CurrencyID", "CustomerRef_value": "CustomerID",
                        "MetaData_CreateTime": "CreateDate", "MetaData_LastUpdatedTime": "LastUpdateDate","ClassRef_value":"ClassID"}
        sales_rename_other = {"Line_Description": "TransactionEntered", 
                        "Line_Amount": "Amount", "Line_SalesItemLineDetail_ItemRef_value": "ItemID",
                        "Line_SalesItemLineDetail_ClassRef_value": "ClassID2", "Line_SalesItemLineDetail_UnitPrice": "UnitPrice",
                        "Line_SalesItemLineDetail_Qty": "Qty"}
        sales_rename.update(sales_rename_other)
        self._raw_processing_combined(sales,sales_col,sales_rename,df_type="salesreceipt",process_line=True)
        ## bill
        bill_col = ["Id", "Balance", "TxnDate", "Line", "TotalAmt", "PrivateNote","SalesTermRef_value","DepartmentRef_value","CurrencyRef_value","VendorRef_value",
                "APAccountRef_value","MetaData_CreateTime", "MetaData_LastUpdatedTime","DocNumber","LinkedTxn","DueDate"]
        concat_cols = ["Id", "SalesTermRef_value", "DepartmentRef_value", "VendorRef_value", "APAccountRef_value", "DocNumber", "LinkedTxn_TxnId",
                    "Line_AccountBasedExpenseLineDetail_AccountRef_value","Line_AccountBasedExpenseLineDetail_ClassRef_value", "Line_ItemBasedExpenseLineDetail_ItemRef_value"]
        bill = self._raw_read_jsons(df_type="bill",concat_cols=concat_cols,is_fact=True,first_cols=bill_col)
        bill_col_other = ["Line_Id", "Line_Description", "Line_Amount", "Corp",
                        "Line_AccountBasedExpenseLineDetail_AccountRef_value",
                        "Line_AccountBasedExpenseLineDetail_ClassRef_value",
                        "Line_ItemBasedExpenseLineDetail_ItemRef_value",
                        "Line_ItemBasedExpenseLineDetail_UnitPrice",
                        "Line_ItemBasedExpenseLineDetail_Qty",]
        bill_col.extend(bill_col_other)
        bill_rename={"TxnDate":"TransactionDate", "SalesTermRef_value":"TermID","DepartmentRef_value":"FarmID","CurrencyRef_value":"CurrencyID","VendorRef_value":"VendorID",
                    "APAccountRef_value":"APAccID","MetaData_CreateTime": "CreateDate", "MetaData_LastUpdatedTime": "LastUpdateDate",
                    }
        bill_rename_other = {"Line_Description":"TransactionEntered", "Line_Amount":"Amount",
                    "Line_AccountBasedExpenseLineDetail_AccountRef_value":"AccID",
                    "Line_AccountBasedExpenseLineDetail_ClassRef_value":"ClassID",
                    "Line_ItemBasedExpenseLineDetail_ItemRef_value":"ItemID",
                    "Line_ItemBasedExpenseLineDetail_UnitPrice":"UnitPrice",
                    "Line_ItemBasedExpenseLineDetail_Qty":"Qty"}
        bill_rename.update(bill_rename_other)
        self._raw_processing_combined(bill,bill_col,bill_rename,"bill",process_line=True)
        ## purchase
        purchase_col = ["Id", "PaymentType", "Credit", "TotalAmt", "DocNumber", "TxnDate", "PrivateNote", "Line", "AccountRef_value", "EntityRef_value", "EntityRef_type", "DepartmentRef_value", 
                    "CurrencyRef_value","MetaData_CreateTime", "MetaData_LastUpdatedTime"]
        concat_cols = ["Id","AccountRef_value","DocNumber","EntityRef_value","DepartmentRef_value","LinkedTxn_TxnId",
                    "Line_AccountBasedExpenseLineDetail_AccountRef_value","Line_AccountBasedExpenseLineDetail_ClassRef_value"]

        purchase = self._raw_read_jsons(df_type="purchase",concat_cols=concat_cols,is_fact=True,first_cols=purchase_col)
        purchase_col_other = ["Line_Id", "Line_Description", "Line_Amount","Line_AccountBasedExpenseLineDetail_AccountRef_value","Line_AccountBasedExpenseLineDetail_ClassRef_value", 
                            "Corp"]
        purchase_col.extend(purchase_col_other)
        purchase_rename={"Credit":"IsCreditCardRefund", "TxnDate":"TransactionDate", "AccountRef_value":"BankAccID", "EntityRef_value":"EntityID",
                        "EntityRef_type":"EntityType", "DepartmentRef_value":"FarmID", "CurrencyRef_value":"CurrencyID",
                        "MetaData_CreateTime": "CreateDate", "MetaData_LastUpdatedTime": "LastUpdateDate"}
        purchase_rename_other = {"Line_Description":"TransactionEntered", "Line_Amount":"Amount",
                        "Line_AccountBasedExpenseLineDetail_AccountRef_value":"AccID",
                        "Line_AccountBasedExpenseLineDetail_ClassRef_value":"ClassID"}
        purchase_rename.update(purchase_rename_other)
        self._raw_processing_combined(purchase,purchase_col,purchase_rename,df_type="purchase",process_line=True)
        ## journal entry
        concat_cols_other = ["Line_JournalEntryLineDetail_AccountRef_value","Line_JournalEntryLineDetail_Entity_EntityRef_value",
                        "Line_JournalEntryLineDetail_DepartmentRef_value","Line_JournalEntryLineDetail_ClassRef_value"]
        journal_col = ["Id", "DocNumber", "TxnDate", "Adjustment", "Line", "PrivateNote", "CurrencyRef_value","MetaData_CreateTime", "MetaData_LastUpdatedTime"]
        concat_cols = ["Id", "DocNumber","LinkedTxn_TxnId"]
        concat_cols.extend(concat_cols_other)
        journal = self._raw_read_jsons(df_type="journalentry",concat_cols=concat_cols,is_fact=True,first_cols=journal_col)
        journal_col.extend(["Line_Id", "Line_Description", "Line_Amount", "Line_JournalEntryLineDetail_PostingType", "Line_JournalEntryLineDetail_AccountRef_value",
                            "Line_JournalEntryLineDetail_Entity_Type", "Line_JournalEntryLineDetail_Entity_EntityRef_value",
                            "Line_JournalEntryLineDetail_DepartmentRef_value","Line_JournalEntryLineDetail_ClassRef_value", "Corp"])
        journal_rename={"Adjustment":"IsAdjustment", "TxnDate":"TransactionDate", "CurrencyRef_value":"CurrencyID",
                        "MetaData_CreateTime": "CreateDate", "MetaData_LastUpdatedTime": "LastUpdateDate",
                        "Line_Description":"TransactionEntered", "Line_Amount":"Amount", "Line_JournalEntryLineDetail_PostingType":"JEType",
                        "Line_JournalEntryLineDetail_AccountRef_value":"AccID", "Line_JournalEntryLineDetail_Entity_Type":"EntityType",
                        "Line_JournalEntryLineDetail_Entity_EntityRef_value":"EntityID", "Line_JournalEntryLineDetail_DepartmentRef_value":"FarmID",
                        "Line_JournalEntryLineDetail_ClassRef_value":"ClassID"}
        self._raw_processing_combined(journal,journal_col,journal_rename,df_type="journalentry",process_line=True)
        ## vendor credit
        vc_col = ["Id", "DocNumber", "TxnDate", "Line", "TotalAmt", "PrivateNote", "MetaData_CreateTime", "MetaData_LastUpdatedTime", "DepartmentRef_value", "CurrencyRef_value", "VendorRef_value",
                "APAccountRef_value"]
        concat_cols = ["Id", "DocNumber","APAccountRef_value","DepartmentRef_value", "VendorRef_value", 
                    "Line_AccountBasedExpenseLineDetail_AccountRef_value","Line_AccountBasedExpenseLineDetail_ClassRef_value",
                    "Line_AccountBasedExpenseLineDetail_CustomerRef_value","Line_ItemBasedExpenseLineDetail_ItemRef_value"]
        vc = self._raw_read_jsons(df_type="vendorcredit",concat_cols=concat_cols,is_fact=True,first_cols=vc_col)
        vc_col.extend(["Line_Id", "Line_Description", "Line_Amount", "Line_AccountBasedExpenseLineDetail_AccountRef_value","Line_AccountBasedExpenseLineDetail_ClassRef_value",
                "Corp", "Line_AccountBasedExpenseLineDetail_CustomerRef_value","Line_ItemBasedExpenseLineDetail_ItemRef_value"])
        vc_rename = {"TxnDate": "TransactionDate", "DepartmentRef_value":"FarmID", "CurrencyRef_value": "CurrencyID", "VendorRef_value":"VendorID", "APAccountRef_value":"APAccID",
                    "Line_Description":"TransactionEntered", "Line_Amount":"Amount","Line_AccountBasedExpenseLineDetail_AccountRef_value":"AccID",
                    "Line_AccountBasedExpenseLineDetail_ClassRef_value":"ClassID","Line_AccountBasedExpenseLineDetail_CustomerRef_value":"CustomerID",
                    "Line_ItemBasedExpenseLineDetail_ItemRef_value":"ItemID","MetaData_CreateTime": "CreateDate", "MetaData_LastUpdatedTime": "LastUpdateDate"}
        self._raw_processing_combined(vc,vc_col,vc_rename,df_type="vendorcredit",process_line=True)
        ## deposit
        dp_col = ["Id", "DocNumber", "TxnDate", "PrivateNote", "Line", "DepartmentRef_value", "DepositToAccountRef_value",'MetaData_CreateTime', 'MetaData_LastUpdatedTime',"CurrencyRef_value"]
        concat_cols = ["Id", "DocNumber", "DepartmentRef_value", "DepositToAccountRef_value", "Line_DepositLineDetail_AccountRef_value","Line_DepositLineDetail_Entity_value",
                    "Line_DepositLineDetail_PaymentMethodRef_value"]
        dp = self._raw_read_jsons(df_type="deposit",concat_cols=concat_cols,is_fact=True,first_cols=dp_col)
        dp_col.extend(["Line_Id", "Line_Amount", "Line_DepositLineDetail_AccountRef_value", "Line_Description","Line_DepositLineDetail_Entity_value","Line_DepositLineDetail_Entity_type",
                    "Line_DepositLineDetail_PaymentMethodRef_value","Corp"])
        dp_rename = {"TxnDate":"TransactionDate","DepartmentRef_value":"FarmID","DepositToAccountRef_value":"BankAccID","MetaData_CreateTime": "CreateDate", "MetaData_LastUpdatedTime": "LastUpdateDate",
                    "CurrencyRef_value":"CurrencyID","Line_DepositLineDetail_PaymentMethodRef_value":"PaymentID",
                    "Line_Amount":"Amount", "Line_DepositLineDetail_AccountRef_value":"AccID", "Line_Description":"TransactionEntered",
                    "Line_DepositLineDetail_Entity_value":"EntityID", "Line_DepositLineDetail_Entity_type":"EntityType"}
        self._raw_processing_combined(dp,dp_col,dp_rename,df_type="deposit",process_line=True)
        ## credit memo
        concat_cols = ["Id","DocNumber","DepartmentRef_value","CustomerRef_value","Line_SalesItemLineDetail_ItemRef_value","Line_SalesItemLineDetail_ClassRef_value"]
        cm_col = ["Id","DocNumber","TxnDate","Line","TotalAmt","Balance","CustomerMemo_value","PrivateNote", 'MetaData_CreateTime', 'MetaData_LastUpdatedTime',"DepartmentRef_value","CurrencyRef_value",
                "CustomerRef_value"]
        cm = self._raw_read_jsons(df_type="creditmemo",concat_cols=concat_cols,is_fact=True,first_cols=cm_col)
        cm_col.extend(["Line_Id","Line_Description","Line_Amount","Line_SalesItemLineDetail_ItemRef_value","Line_SalesItemLineDetail_UnitPrice","Line_SalesItemLineDetail_Qty",
                    "Line_SalesItemLineDetail_ClassRef_value", "Corp"])
        cm_rename = {"TxnDate":"TransactionDate","MetaData_CreateTime": "CreateDate", "MetaData_LastUpdatedTime": "LastUpdateDate","DepartmentRef_value":"FarmID","CurrencyRef_value":"CurrencyID",
                    "CustomerRef_value":"CustomerID","CustomerMemo_value":"CustomerMemo", "Line_SalesItemLineDetail_ClassRef_value":"ClassID",
                    "Line_Description":"TransactionEntered","Line_Amount":"Amount","Line_SalesItemLineDetail_ItemRef_value":"ItemID",
                    "Line_SalesItemLineDetail_UnitPrice":"UnitPrice","Line_SalesItemLineDetail_Qty":"Qty"}
        self._raw_processing_combined(cm,cm_col,cm_rename,df_type="creditmemo",process_line=True)               

    def _report_adjust_sign(self, entry: pd.Series, target_col:str="AmountAdj") -> float:
        """ 
            adjust the sign of transactions according to the column that the amount is being used
            3 scenarios:  1. adjust signs = QBO PL report --> entry point might be string                         -> column "AmountAdj"
                          2. adjust USD to CAD --> entry point is float                                           -> column "AmountCAD"
        """
        assert target_col in ["AmountAdj", "AmountCAD"], "please pass target_col as one of (AmountAdj, AmountCAD)"
        if target_col == "AmountAdj":
            if isinstance(entry["Amount"], str) and (len(entry["Amount"]) == 0):   # if the amount is empty string, return 0 
                return 0
            if entry["AccID"] in self.account_QBO_expense:  # if it's expense account, reverse the sign
                return -float(entry["Amount"])
            return float(entry["Amount"])
        else:
            return entry["AmountAdj"] * self.fx if entry["Corp"] in self.us_companies else entry["AmountAdj"]

    def _report_transform(self, report_type:str = "PL", light_load:bool = True) -> pd.DataFrame:
        """ 
            transform PL or GL from json nested format to flat pandas dataframe and save as csv
        """
        assert report_type in ["PL", "GL"], "please pass report_type as one of (PL, GL)"
        mode = "Light Load" if light_load else "Full Load"
        # prepare for extraction 
        records_all = []
        print(f"\nBegin {report_type} Transformation - {mode}...")
        self.log.write(f"\n\nBegin {report_type} Transformation - {mode}\n\n")
        # extraction
        for name in self.company_names:
            print(f"Processing {name}...")
            self.log.write(f"\nProcessing {name}\n")
            # determine the columns in raw report for the current company - PL only ******* also APAR
            if report_type == "PL":
                if name in ["MSUSA", "MSL"]: # these companies don't have Class column
                    report_columns = [x for x in self.PL_raw_cols if x != "Location"]        
                elif name in ["NexGen"]:
                    report_columns = [x for x in self.PL_raw_cols if ((x != "Location") & (x != "Class"))]  # this company doesn't have Location and Class column
                else:
                    report_columns = self.PL_raw_cols
            else:
                report_columns = self.GL_raw_cols
            # iterate through all json files (saved by year_month) for the current company
            path = self.raw_path["QBO"][report_type] / name
            files = os.listdir(path)
            # if light_load, only load and transform data in this FY
            if light_load:
                files = [file1 for file1 in files if str(self.today.year) in file1 or file1 == f"{self.today.year-1}_10.json"]
            for file_name in files:
                with open(path/file_name, "r") as f:
                    df = json.load(f)
                records = []    # records for the company
                # not separating this function because access to records list
                def _extract(current_level: dict[str,any], parent1_id: str="", parent2_id: str="", parent2_name: str="") -> None:
                    """ 
                        recursive function that focus on current_leve, if reached the leaf node, append, if not, call itself with next_level, while recording current id as parent_id
                    """
                    if current_level["type"] == "Data":  # reached a leaf node, append and stop
                        sub_record = {}
                        for i in range(len(report_columns)): # iterate through each column + value inside the json format
                            if (i == 2) and (current_level["ColData"][i]["value"]): 
                                # for DocNumber: use format: MFl-XXXXX, don't need to extract id
                                sub_record["DocNumber"] = name + "-" + current_level["ColData"][i]["value"]
                                continue 
                            sub_record[report_columns[i]] = current_level["ColData"][i]["value"]
                            # add id and append company in front of the id for SplitAccID, TransactionID, NameID (vendorID,...)
                            if (report_columns[i] == "SplitAcc") and (current_level["ColData"][i].get("id", -1) != -1):
                                sub_record["SplitAccID"] = name + current_level["ColData"][i]["id"]
                            elif (report_columns[i] == "TransactionType") and (current_level["ColData"][i].get("id", -1) != -1):
                                sub_record["TransactionID_partial"] = name + current_level["ColData"][i]["id"]
                            elif (report_columns[i] == "Name") and (current_level["ColData"][i].get("id", -1) != -1):
                                sub_record["NameID"] = name + current_level["ColData"][i]["id"]
                            elif (report_columns[i] == "Class") and (current_level["ColData"][i].get("id", "") != ""):
                                sub_record["ClassID"] = name + current_level["ColData"][i]["id"]
                            elif (report_columns[i] == "Location") and (current_level["ColData"][i].get("id", "") != ""):
                                sub_record["FarmID"] = name + current_level["ColData"][i]["id"]
                        # add account information, which is at the parent level of the records
                        sub_record["AccName"] = parent2_name[6:] 
                        sub_record["AccID"] = name + parent2_id if len(parent2_id)>=1 else name + parent1_id 
                        sub_record["Corp"] = name
                        sub_record["AccNum"] = name + parent2_name[:6]
                        # append to records for this company
                        records.append(sub_record)
                    else:       # if not leaf node
                        if current_level.get("Header", -1) != -1:   # reached a node that contains a legit path, record account information 
                            idx = current_level["Header"]["ColData"][0].get("id", -1)
                            parent2_name = current_level["Header"]["ColData"][0]["value"]
                            if idx != -1:   # if we are looking at an account
                                parent1_id = parent2_id     # account id 2 levels up to any transactions 
                                parent2_id = idx    # account id that is the parent account of transactions 
                        # recursively call _extract on every sub-records
                        if current_level.get("Rows", -1) != -1:
                            if current_level["Rows"].get("Row",-1) != -1:
                                for j in range(len(current_level["Rows"]["Row"])):
                                    _extract(current_level["Rows"]["Row"][j], parent1_id = parent1_id, parent2_id = parent2_id, parent2_name = parent2_name)
                # start the recursive process
                for k in range(len(df["Rows"]["Row"])):
                    _extract(df["Rows"]["Row"][k])
                self.log.write(f"\nFinished Processing {file_name}, total records added - {len(records)}\n")
                records_all.extend(records)
        self.log.write(f"\n\nFinished Processing All Files, Total Records {len(records_all)}\n\n")
        # processing all records
        records_all = pd.DataFrame(records_all)
        records_all = records_all[records_all["TransactionDate"]!="Beginning Balance"] # ignore beginning blanace summary records, because its format is different than others
        if report_type == "GL":
            records_all["TransactionType"] = records_all["TransactionType"].str.replace("Bill Payment (Cheque)", "BillPaymentCheck")
            records_all["TransactionType"] = records_all["TransactionType"].str.replace("Bill Payment (Check)", "BillPaymentCheck")
        # self.records_dev = records_all
        records_all["TransactionDate"] = pd.to_datetime(records_all["TransactionDate"])
        records_all["AmountAdj"] = records_all.apply(lambda x: self._report_adjust_sign(x, target_col="AmountAdj"), axis=1) # can be improved, see _report_APAR_transform()
        records_all["AmountCAD"] = records_all.apply(lambda x: self._report_adjust_sign(x, target_col="AmountCAD"), axis=1) # can be improved, see _report_APAR_transform()
        records_all["FXRate"] = self.fx
        records_all["Country"] = records_all["Corp"].apply(lambda x: "USA" if x in self.us_companies else "Canada")  # can be improved, see _report_APAR_transform()
        self.log.write(f"\nAfter omitting Beginning Balance entries, Total length is {len(records_all)}\n")
        self.log.write(f"\nSpot USD/CAD ={self.fx}\n")
        self.log.write(f"\nFinished Processing all Files for {report_type}, Total Records {len(records_all)}\n\n")
        print(f"\nFinished Processing all Files for {report_type}, Total Records {len(records_all)}\n")
        return records_all

    def _report_merge(self, mode: str, df_new: pd.DataFrame, path_old: Path) -> pd.DataFrame:
        """ 
            if light_load, load df_old and perform merge
        """
        assert mode in ["PL", "GL"], "mode must be one of [PL, GL]" 
        if mode == "PL":
            df_old = pd.read_csv(path_old,dtype={"Class":str, "ClassID":str})
        else:
            df_old = pd.read_csv(path_old)
        
        assert ((list(set(df_old.columns)-set(df_new.columns))==[]) and (list(set(df_new.columns)-set(df_old.columns))==[])), \
                f"{mode} new and old df columns don't match - old {df_old.columns} , new {df_new.columns}"
        df_old["TransactionDate"], df_new["TransactionDate"] = pd.to_datetime(df_old["TransactionDate"]), pd.to_datetime(df_new["TransactionDate"])
        # truncate the old df to transactions where dates are prior to the earlist transaction date in df_new
        df_old = df_old[df_old["TransactionDate"]<df_new["TransactionDate"].min()]
        df_new = pd.concat([df_new, df_old])
        if mode == "PL": df_new["FXRate"] = self.fx
        print(f"\nLoaded and merged old records for {mode}, total records - {len(df_new)}")
        self.log.write(f"\n\nLoaded and merged old records for {mode}, total records - {len(df_new)}\n\n")
        return df_new

    def _report_APAR_transform(self,report_type:str = "AP") -> None:
        """ 
            This function transforms Outstanding APAR reports from json raw format extracted from QBO API into dataframes
        """
        columns_full = ["Date", "TransactionType", "DocNumber", "Vendor", "Farm", "DueDate", "PastDue", "Amount", "OpenBalance"]
        additional_columns = ["VendorID", "TransactionTypeID", "FarmID", "AmountCAD"]
        all_columns = columns_full + additional_columns
        dtypes = {"Date":str, "TransactionType":str, "DocNumber":str, "Vendor":str, "Farm":str, "DueDate":str, "PastDue":float, "Amount":float, "OpenBalance":float,
                  "VendorID":str, "TransactionTypeID":str, "FarmID":str, "AmountCAD":float}
        # ensure report type is entered correctly
        report_type = report_type.upper()
        assert report_type in ["AP", "AR"], f"please use 'AP' or 'AR' for report type, entered {report_type}"
        report_name = "AgedPayableDetail" if report_type == "AP" else "AgedReceivableDetail"
        df_final = pd.DataFrame()
        folder = self.raw_path["QBO"]["APAR"] / report_name / str(self.today.year) / str(self.today.month)

        print(f"\nBegin {report_name} Transformation ...")
        self.log.write(f"\n\nBegin {report_name} Transformation \n\n")
        
        for company in self.company_names:
            country = "USA" if company in self.us_companies else "Canada"
            print(f"Processing {company}...")
            self.log.write(f"\nProcessing {company}\n")
            # load json file
            path = folder / company
            with open(path/f"{self.today.day}.json", "r") as f:
                data = json.load(f)
            # # record report date
            # header = data["Header"]
            # if header["Option"][0]["Name"] == "report_date":
            #     report_day = header["Option"][0]["Value"]
            # else:
            #     report_day = header["Time"].split("T")[0]
            if company in ["MSUSA", "MSL", "NexGen"]:
                columns = [x for x in columns_full if x!= "Farm"]
            else:
                columns = columns_full
            company_df = pd.DataFrame(columns=all_columns).astype(dtypes)
            # inner layer 1: different APAR Categories, e.g., 1-30 days over due
            row = data["Rows"]["Row"]
            row = row[:-1]
            for k in range(len(row)):
                APCategory_df = pd.DataFrame(columns=all_columns).astype(dtypes)
                ## record APAR Catetory
                APCategory = row[k]["Header"]["ColData"][0]["value"]
                ## inner layer 2: different entries for each category
                rows = row[k]["Rows"]["Row"]
                if len(rows) < 1:
                    continue
                for j in range(len(rows)):
                    if rows[j]["type"] == "Data":
                        entry = rows[j]["ColData"]
                    else:
                        continue 
                    row_df = {}
                    for i in range(len(entry)):
                        # if column value contains ID, append company code and add ID
                        if columns[i] == "Vendor":
                            row_df.update({"VendorID": company + entry[i]["id"]})
                        elif columns[i] == "TransactionType":
                            row_df.update({"TransactionTypeID":company + entry[i]["id"]})
                        elif columns[i] == "Farm":
                            if company not in ["MSUSA", "MSL", "NexGen"]:
                                row_df.update({"FarmID":company + entry[i]["id"]})
                        # if it is DocNumber column, append campany code else, just append column value
                        if columns[i] == "DocNumber":
                            row_df.update({columns[i]:company + "-" + entry[i]["value"]})
                        elif (columns[i] == "Farm") and (company not in ["MSUSA", "MSL", "NexGen"]):
                            row_df.update({columns[i]: entry[i]["value"]})
                        # convert numbers from string                            
                        elif (columns[i] == "OpenBalance") or (columns[i] == "PastDue") or (columns[i] == "Amount"):
                            amount = entry[i]["value"]
                            amount = float(amount) if amount != "" else 0
                            row_df.update({columns[i]: amount})
                            if columns[i] == "Amount":
                                amountCAD = amount * self.fx if country == "USA" else amount
                                row_df.update({"AmountCAD": amountCAD})
                        else:
                            row_df.update({columns[i]: entry[i]["value"]})
                    #print(row_df)
                    APCategory_df.loc[len(APCategory_df)] = row_df
                # append category if the dataframe is not empty
                if len(APCategory_df) > 0:
                    APCategory_df["APCategory"] = APCategory
                    company_df = pd.concat([company_df,APCategory_df],ignore_index=True) 
                else:
                    print(f"Empty APCategory {APCategory} for {company}")
            company_df["Corp"] = company 
            company_df["Country"] = country
            df_final = pd.concat([df_final, company_df], ignore_index=True)
        
        df_final["FXRate"] = self.fx
        df_final["ReportDate"] = self.today
        self.log.write(f"\nFinished Processing all Files for {report_name}, Total Records {len(df_final)}\n\n")
        print(f"\nFinished Processing all Files for {report_name}, Total Records {len(df_final)}\n")
        print("Saving ...")
        path_out = self.silver_path["QBO"]["APAR"]/report_name/str(self.today.year)/str(self.today.month)
        self.check_file(path_out)
        df_final.to_csv(path_out/(str(self.today.day)+".csv"), index=False)

    # flow
    def extract(self, load_raw: bool=True, load_pl: bool=True, load_gl: bool=True, light_load: bool=True, extract_only:list[str,None]=[]) -> None:
        
        # extract QBO client secret
        with open(self.raw_path["Auth"]["QBO"]/"client_secrets.json", "r") as f:
            secret = json.load(f)
        mode = "Light Load" if light_load else "Full Load"
        print(f"\nStart QBO Extraction - {mode} ...")
        self.log.write(f"\nStart QBO Extraction - {mode}\n")
        companies = self.company_names if len(extract_only) == 0 else extract_only
        for company in companies:
            print(f"Extracting {company}")
            self.log.write(f"\nExtracting {company}\n" + "Raw Summary\n")
            # refresh tokens
            auth_client = self._refresh_auth_client(company, secret)
            # extract outstanding APAR reports
            self._pull_APAR(auth_client=auth_client,company=company,report_type="AP")
            # extract GL - always for Weekly Banking Project
            if load_gl: self._pull_reports(auth_client=auth_client,company=company,light_load=light_load,report_type="GL")
            # extract PL
            if load_pl: self._pull_reports(auth_client=auth_client,company=company,light_load=light_load,report_type="PL")
            # extract raw
            if load_raw: self._pull_raw(table_names=self.names+self.other_names, auth_client=auth_client, company=company)
        print("Finished QBO Extraction ...")
        self.log.write("\nFinished QBO Extraction\n\n")
        
    def transform(self, light_load:bool=True, process_raw:bool=True, process_gl:bool=True) -> None:
        print("\nStarting QBO Transformation ...")
        self.log.write("\nStart QBO Transformation\n")
        # QBO_Raw_Processing.py
        if process_raw: self._raw_transform()
        # read account table for PL, GL transformation
        self.account = pd.read_csv(self.silver_path["QBO"]["Dimension_time"]/"Account.csv")
        self.account_QBO_expense = self.account[self.account["AccountType"].isin(self.acctype_QBO_expense)].AccID.unique()
        # transform GL & PL 
        for mode in ["PL", "GL"]:
            if not process_gl and (mode == "GL"):
                continue
            path_old = (self.silver_path["QBO"]["PL"]/"ProfitAndLoss.csv") if mode == "PL" else (self.silver_path["QBO"]["GL"]/"GeneralLedger.csv")
            df_new = self._report_transform(report_type=mode,light_load=light_load)
            # if light load mode, load and merge with old records
            if light_load:
                df_new = self._report_merge(mode=mode, df_new=df_new, path_old=path_old)
            df_new.to_csv(path_old, index=False)
        # APAR transformation
        self._report_APAR_transform(report_type="AP")
        print("Finished QBO Transformation ...")
        self.log.write("\nFinished QBO Transformation\n\n")
            
    def run(self, QBO_light:bool=True, extract:bool=True, extract_only:list[str,None]=[], AP_only:bool = False, PL_only: bool=False) -> None:
        # measure time 
        start = perf_counter()
        load_raw = load_gl = process_raw = process_gl = load_pl = True
        if PL_only or AP_only:
            load_raw = load_gl = False 
            if AP_only:
                load_pl = False
            process_raw = process_gl = False

        # start logging
        self.create_log(path=self.raw_path["Log"])
        self.log.write("\n"*4 + "*"*100 + "\n\nStart of QBO Pipeline\n\n")
        print("\nStart QBO Pipeline ...")
        
        # QBO Extracting
        if extract: self.extract(light_load=QBO_light, extract_only=extract_only, load_raw=load_raw, load_gl=load_gl, load_pl=load_pl)

        # QBO Transformation
        self.transform(light_load=QBO_light, process_raw=process_raw, process_gl=process_gl)

        end = perf_counter()
        self.log.write(f"\n\nQBO Pipeline Took {(end-start)/60:.3f} minutes\n\n" + "*****"*20 + "\n"*4)
        print(f"QBO Pipeline Finished with {(end-start)/60:.3f} minutes\n")

        # close logging
        self.close_log()

class QBOTimeETL(Job):
    """ 
        run QBO Time API extraction and transformation
    """
    def __init__(self):
        super().__init__()
        self.url = "https://reset.tsheets.com/api/v1/"
        self.corps_QBOTime = ["CanadaMontana", "Outlook", "Arizona", "BritishColumbia"]
        self.tokens = pd.read_json(self.raw_path["Auth"]["QBOTime"]/"tokens.json")
        self.last_sunday = (self.today - dt.timedelta(days=(self.today.weekday() + 1)%7))
        self.period_begin = (self.last_sunday - dt.timedelta(days=20))  # record three weeks' records
        self.folder = self.raw_path["QBO"]["Time"]/self.last_sunday.isoformat()
        self.check_file(self.folder)
    
    def _QBOTime_extraction(self, category:str, querystring:dict[str,str|int], headers:dict[str,str], corp:str) -> int:
        """ 
            extract timesheets or group tables from QBOTime API,
                timesheets table will process timesheets, users, and jobcodes
                groups contain locations
                optionally directly pull jobcodes - not normally used
        """
        assert category in ["timesheets", "groups", "jobcodes"], "category must be one of [timesheets, groups, jobcodes]"
        querystring["page"] = 1 # reset page to 1
        payload = ""
        temp, temp2, temp3 = {}, {}, {} # placeholder for appending table information, timesheets = 3 tables, group = 1 table 
        end = False # indicator for reaching end of table data from API
        count_call = 0
        # extraction
        while not end:
            count_call += 1
            response = requests.request("GET", self.url+category, data=payload, headers=headers, params=querystring)
            response.raise_for_status()
            data = response.json()
            # turn on the end indicator if end of table is reached from API
            if not data["more"]:
                end=True 
            temp.update(data["results"][category])
            if category == "timesheets":
                temp2.update(data["supplemental_data"]["users"])
                temp3.update(data["supplemental_data"]["jobcodes"])
            querystring["page"] += 1 # increase page for extracting more data
        # print info and save
        if category == "timesheets":
            self.log.write(f"Finished extracting {category}, there are {len(temp)} entries with {len(temp2)} users and {len(temp3)} jobcodes\n")
            with open(self.folder/f"timesheets_{corp}.json", "w") as f:
                json.dump(temp,f,indent=4)
            with open(self.folder/f"users_{corp}.json", "w") as f:
                json.dump(temp2,f,indent=4)
            with open(self.folder/f"jobcodes_{corp}.json", "w") as f:
                json.dump(temp3,f,indent=4)
        else:
            self.log.write(f"Finished extracting {category}, there are {len(temp)} entries\n")
            if category == "jobcodes":
                with open(self.folder/f"jobcodes_source_{corp}.json", "w") as f:
                    json.dump(temp,f,indent=4)
            else:
                with open(self.folder/f"{category}_{corp}.json", "w") as f:
                    json.dump(temp,f,indent=4)
        return count_call
    
    def _load_new_file(self, category:str) -> pd.DataFrame:
        """ 
            reads raw json format files and convert them into pandas dataframe for further processing
        """
        assert category in ["timesheets", "groups", "jobcodes", "users"], "category must be one of [timesheets, groups, jobcodes, users]"
        temp_new = pd.DataFrame()
        for corp in self.corps_QBOTime:
            # read and combine
            with open(self.raw_path["QBO"]["Time"]/str(self.last_sunday)/("_".join([category,corp])+".json")) as f:
                temp = json.load(f)
            temp = list(temp.values())
            temp = pd.json_normalize(temp)
            # add corp
            temp["corp"] = corp
            corp_short = "".join(re.findall(r'[A-Z]',corp))
            temp["corp_short"] = corp_short
            temp_new = pd.concat([temp_new, temp])
        return temp_new
    
    def transform(self) -> None:
        """ 
            transform raw json QBOTime data from json format to tabular csv format
                transformations: 
                    1. add corp and corp_short
                    2. format star, end, date
                    3. convert duration from seconds to hours
                    4. create weekof column
        """
        self.check_file(self.silver_path["QBO"]["Time"])
        
        # timesheet
        ## load old file
        temp_old = pd.read_csv(self.silver_path["QBO"]["Time"]/"timesheets.csv")
        temp_old["date"] = pd.to_datetime(temp_old["date"])
        ## read new file
        temp_new = self._load_new_file(category="timesheets")
        ### truncate columns
        columns = ["id", "state", "user_id", "jobcode_id", "start", "end", "duration", "date", "tz", "type", "on_the_clock", "notes", "last_modified", "created_by_user_id", "corp", "corp_short"]
        temp_new = temp_new.loc[:, columns]
        ### rename columns
        renames = {"duration":"duration_seconds","id":"timesheet_id"}
        temp_new = temp_new.rename(columns=renames)
        ### append corp into id columns
        for col in [column for column in temp_new.columns if "id" in column]:
            temp_new[col] = temp_new[col].astype(str)
            temp_new[col] = temp_new["corp_short"] + temp_new[col]
        ### format start, end, date
        temp_new["start"] = temp_new["start"].str.slice(0,19)
        temp_new["end"] = temp_new["end"].str.slice(0,19)
        temp_new["start"] = pd.to_datetime(temp_new["start"])
        temp_new["end"] = pd.to_datetime(temp_new["end"])
        temp_new["date"] = pd.to_datetime(temp_new["date"])
        ### convert duration into hours
        temp_new["duration"] = temp_new["duration_seconds"] / 3600
        ### create weekof column
        temp_new["weekof"] = temp_new["date"] - pd.to_timedelta(temp_new["date"].dt.weekday, unit="D")
        temp_new["weekof"] = temp_new["weekof"].astype(str) 
        ## dedup and append
        overlap = list(set(temp_new.weekof.unique()).intersection(set(temp_old.weekof.unique())))
        temp_old = temp_old[~temp_old["weekof"].isin(overlap)]
        temp_new = pd.concat([temp_new, temp_old])
        ## save
        temp_new.to_csv(self.silver_path["QBO"]["Time"]/"timesheets.csv", index=False)
        self.log.write(f"Total timesheets processed - {len(temp_new)}\n")
        print(f"Total timesheets processed - {len(temp_new)}")
        
        # users
        ## load old file
        temp_old = pd.read_csv(self.silver_path["QBO"]["Time"]/"users.csv")
        ## read new file
        temp_new = self._load_new_file(category="users")
        ## append corp into id
        for col in [column for column in temp_new.columns if "id" in column]:
            temp_new[col] = temp_new[col].astype(str)
            temp_new[col] = temp_new["corp_short"] + temp_new[col]
        ## rename
        columns = ["id", "first_name", "last_name", "group_id", "active", "employee_number", "username", "created","last_active","last_modified", "corp_short", "corp"]
        renames = {"id": "user_id"}
        temp_new = temp_new.loc[:,columns]
        temp_new = temp_new.rename(columns=renames)
        temp_new["full_name"] = temp_new["first_name"] + " " + temp_new["last_name"]
        ## dedup and append
        temp_new = pd.concat([temp_new, temp_old])
        temp_new = temp_new.drop_duplicates(subset="user_id",keep='first')
        ## save
        temp_new.to_csv(self.silver_path["QBO"]["Time"]/"users.csv", index=False)
        self.log.write(f"Total users processed - {len(temp_new)}\n")
        print(f"Total users processed - {len(temp_new)}")

        # jobcodes
        ## load old file
        temp_old = pd.read_csv(self.silver_path["QBO"]["Time"]/"jobcodes.csv")
        ## read new file
        temp_new = self._load_new_file(category="jobcodes")
        ## transform
        for col in [column for column in temp_new.columns if "id" in column]:
            temp_new[col] = temp_new[col].astype(str)
            temp_new[col] = temp_new["corp_short"] + temp_new[col]
        columns = ["id", "corp_short", "corp", "name", "active", "type"]
        renames = {"id": "jobcode_id", "name": "job_name"}
        temp_new = temp_new.loc[:,columns]
        temp_new = temp_new.rename(columns=renames)
        ## dedup and append
        temp_new = pd.concat([temp_new, temp_old])
        temp_new = temp_new.drop_duplicates(subset="jobcode_id", keep='first')
        ## save
        temp_new.to_csv(self.silver_path["QBO"]["Time"]/"jobcodes.csv", index=False)
        self.log.write(f"Total jobcodes processed - {len(temp_new)}\n")
        print(f"Total jobcodes processed - {len(temp_new)}")

        # groups - everytime it's loading the newest extraction, there's no append old records
        ## read new file
        temp_new = self._load_new_file(category="groups")
        ## transform
        temp_new["managers_id"] = temp_new["manager_ids"].apply(lambda x: ",".join(x) if len(x) >=1 else "NA")
        renames = {"id": "group_id", "name":"location_name"}
        temp_new = temp_new.rename(columns=renames)
        temp_new = temp_new.drop(columns=["manager_ids"])
        for col in [column for column in temp_new.columns if "id" in column]:
            temp_new[col] = temp_new[col].astype(str)
            temp_new[col] = temp_new["corp_short"] + temp_new[col]
        ## save
        temp_new[temp_new["active"]==True].to_csv(self.silver_path["QBO"]["Time"]/"group.csv", index=False)
        self.log.write(f"Total groups (inactive & active) processed - {len(temp_new)}\n")
        print(f"Total groups (inactive & active) processed - {len(temp_new)}")
    
    def extract(self, pull_job_source:bool = False) -> None:
        for corp in self.corps_QBOTime:
            print(f"Extracting {corp} ...")
            self.log.write(f"\n\nExtracting {corp} ...\n")
            count = 0
            # timesheets
            querystring = {
                "start_date": self.period_begin.isoformat(),
                "end_date": self.last_sunday.isoformat(),
                "on_the_clock": "both",
                "page": 1
            }
            headers = {
                "Authorization": f"Bearer {self.tokens[corp]["Access Tokens"]}",
            }
            count += self._QBOTime_extraction(category="timesheets",querystring=querystring,headers=headers,corp=corp)
            # groups
            querystring = {
                "page": 1,
                "active": "both"
            }
            count += self._QBOTime_extraction(category="groups",querystring=querystring,headers=headers,corp=corp)
            if pull_job_source:
                # jobcodes_source
                querystring = {
                    "page": 1,
                    "active": "both"
                }
                count += self._QBOTime_extraction(category="jobcodes",querystring=querystring,headers=headers,corp=corp)
            self.log.write(f"\nNumber of API calls - {count}\n")
            print(f"Number of API calls - {count}")
        self.log.write("\nFinished\n\n")
        print("\nFinished\n")
    
    def run(self, force_run:bool=False) -> None:
        if force_run or (self.today.weekday() in [0, 2, 6]): # only run monday night or wednesday night or force_run
            # measure time 
            start = perf_counter()

            # start logging 
            self.create_log(path=self.raw_path["Log"])
            
            # extract
            self.log.write("\n"*4 + "*"*100 + f"\nPeriod: {self.period_begin} - {self.last_sunday}" + "\nStarting QBO Time Pipeline ...\n" +"\nExtracting ...")
            print(f"\nPeriod: {self.period_begin} - {self.last_sunday}" + "\nStarting QBO Time Pipeline ...\n" +"\nExtracting ...")
            self.extract()

            # transforming
            self.log.write("\n\nTransforming ...\n\n")
            print("\nTransforming ...\n")
            self.transform()

            end = perf_counter()
            self.log.write(f"\n\nQBO Time Pipeline Took {(end-start)/60:.3f} minutes\n\n" + "*****"*20 + "\n"*4)
            print(f"\nQBO Time Pipeline Finished with {(end-start)/60:.3f} minutes\n")
            self.close_log()
        else:
            print(f"\n\nQBO Time Pipeline not scheduled to run today on - {self.today}\n\n")

class HPETL(Job):
    """ 
        For extract and transform Harvest Profit Data
    """

    def __init__(self):
        super().__init__()
        self.json_download_path = Path("c:/Users/ZheRao/OneDrive - Monette Farms/Desktop/Work Files/Projects/5 - HP Data")
        # extract credentials locally 
        with open(self.raw_path["Auth"]["Harvest Profit"]/"info.json", "r") as f:
            self.credentials = json.load(f)
        self.hp_locations = list(self.credentials["SwitchAcc"].keys())
        self.BASE = "https://www.harvestprofit.com"

    def _purge_folder(self) -> None:
        """ 
            This function will clear all files in the folder that is used to store emailed links
                called before pipeline and after pipeline to ensure no duplicated download or errors in sending download requests
        """
        for json_path in os.listdir(self.json_download_path):
            Path.unlink(self.json_download_path/json_path, missing_ok=True)

    def _login(self) -> None:
        """ 
            this function performs login action for accessing the HP server
        """
        LOGIN_GET = f"{self.BASE}/users/sign_in"
        def _extract_csrf_and_action(html: str):
            """ 
                this function extract information used for sign in from the sign in web page
            """
            soup = BeautifulSoup(html, "html.parser")
            # auth token can appear as a hidden input or in a meta tag
            token = None 
            # find correct form in the login page to extract tokens and action for sending the login request
            for form in soup.find_all("form"):
                action = (form.get("action") or "")
                if form.find("input", {"type": "password"}) or "/users/sign_in" in action:
                    login_form = form
                    break
            if not login_form:
                raise RuntimeError("Couldn't find the login form.")
            token = (login_form.find("input", {"name": "authenticity_token"}) or {}).get("value")
            if not token:
                meta = soup.find("meta", {"name": "csrf-token"})
                token = meta.get("content") if meta else None
            if not token:
                raise RuntimeError("Missing CSRF token.")
            return token, action
        # extract info from login page
        login = self.s.get(LOGIN_GET)
        login.raise_for_status()
        # retrieve token and action for login request
        token, action = _extract_csrf_and_action(login.text)
        # conpose precise endpoint for login request
        post_url = urljoin(self.BASE, action)
        # gather information for log in
        email = self.credentials["Credentials"]["email"]
        passwrd = self.credentials["Credentials"]["password"]
        # send login request
        payload = {
            "authenticity_token": token,
            "user[email]": email,
            "user[password]": passwrd
        }
        headers = {
            "Origin": self.BASE,
            "Referer": LOGIN_GET
        }
        login = self.s.post(post_url, data=payload, headers=headers, allow_redirects=True)
        login.raise_for_status()
        if login.status_code in ["200", 200]:
            print("\nLogin Successful\n")
    
    def _send_request_for_email(self) -> None:
        """ 
            this function switches accounts to different locations, and send a GET request to HP server that triggers email containing actual data extraction link to be sent
        """
        for l in self.hp_locations:
            # switch account 
            print(f"activating {l}")
            ACTIVATE = self.credentials["SwitchAcc"][l]
            r = self.s.get(ACTIVATE, headers={"Referer": f"{self.BASE}/accounts"}, allow_redirects=True)
            r.raise_for_status()
            # get relevant information from updated load page for sending the request for email
            soup = BeautifulSoup(r.text, features="lxml")
            script = soup.find("script", {"class": "js-react-on-rails-component", "data-component-name": "Application", "type": "application/json"})
            data = json.loads(script.text)
            access_jwt, refresh_jwt = data.get("token"), data.get("refresh_token")
            print(f"Page loaded for {data["entity"]["name"]}")
            # sending the request for email containing data extraction link
            url = f"{self.BASE}/api/v3/grain_inventory/loads/export_all"
            year_ids = [y['id'] for y in data["years"]]
            params = [("year_ids[]", str(y)) for y in year_ids]
            LOADS_URL = "https://www.harvestprofit.com/78691/grain_inventory/loads"
            headers = {
                "Accept": "text/csv, application/json, text/plain, */*",
                "Referer": LOADS_URL,
                "authorization": access_jwt
            }
            resp = self.s.get(url,params = params, headers=headers)
            resp.raise_for_status()
    
    def _extract_HP_data(self) -> None:
        """ 
            this function send GET request for actual data, extract and store the raw data, raw data is csv format
        """
        df_all = pd.DataFrame()
        for json_path in os.listdir(self.json_download_path):
            # open link file
            with open(self.json_download_path/json_path, "r") as f:
                link = json.load(f)
            # send download request
            r = self.s.get(link['url'])
            # extract data from request results
            df = pd.read_csv(
                StringIO(r.text),       # treat the string like a file
                skipinitialspace=True,  # trims the leading space in " Lbs"
                parse_dates=['date'],   # pase the datetime column
                date_format="%m/%d/%Y %I:%M %P",    # speeds up parsing for this format
                dtype={'harvest_profit_id': 'Int64', 'crop_year': 'Int64'},
                na_values=['']          # turn empty quotes into NaN
            )
            print(f"Processed - {df.loc[0,"entity_share"]}")
            # append results 
            df_all = pd.concat([df_all,df], ignore_index=True)
        print(f"\nProcessed {len(df_all)} rows of data, saving ...")
        df_all.to_csv(self.silver_path["Delivery"]["HP"]/f"Loads_{self.today.year}_{self.today.month}.csv", index=False)

    def run(self):
        start = perf_counter()

        print("\nStarting Harvest Profit Data Extraction\n")
        self._purge_folder()
        self.s = requests.Session()
        # login
        self._login()
        # trigger email
        self._send_request_for_email()
        # wait for the Power Automate to process all emails and extract links
        time_waited = 60
        time.sleep(60)
        while len(os.listdir(self.json_download_path)) < 11:
            time_waited += 10
            time.sleep(10)
        print(f"\nall links received and ready to send data extraction request, wait time - {np.round(time_waited/60,2)} minutes\n")
        self._extract_HP_data()
        self._purge_folder()
        self.s.close()

        end = perf_counter()
        print(f"\nHP Extraction Finished with {(end-start)/60:.3f} minutes\n")


# end of file
        
        
