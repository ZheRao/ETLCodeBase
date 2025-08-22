import pandas as pd
import numpy as np
from pathlib import Path 
import os
import datetime as dt
import requests
import json
from intuitlib.client import AuthClient
import urllib.parse
from time import perf_counter
import re


class Job:
    def __init__(self):
        base_dir = Path("c:/Users/ZheRao/OneDrive - Monette Farms/Monette Farms Team Site - Innovation Projects/Production/Database")
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
                "Time":base_dir/"Bronze"/"QBOTime"
            },
            "Delivery": {"Traction":base_dir/"Bronze"/"Traction", "HP":base_dir/"Bronze"/"HarvestProfit"},
            "Auth": {"QBO":base_dir/"Bronze"/"Authentication"/"QBO", "QBOTime": base_dir/"Bronze"/"Authentication"/"QBOTime"},
            "Log": base_dir/"Load_History"/f"{self.today.year}"/month_format
        }
        self.silver_path = {
            "QBO": {
                "Dimension_time": base_dir/"Silver"/"QBO"/"Dimension"/f"{self.today.year}_{self.today.month}",
                "Dimension": base_dir/"Silver"/"QBO"/"Dimension",
                "Raw": base_dir/"Silver"/"QBO"/"Fact"/"Raw",
                "PL": base_dir/"Silver"/"QBO"/"Fact"/"ProfitAndLoss",
                "GL": base_dir/"Silver"/"QBO"/"Fact"/"GeneralLedger",
                "Time": base_dir/"Silver"/"QBOTime"
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

    def __init__(self):
        super().__init__()
        self.names = ["Deposit","CreditMemo", "VendorCredit", "Invoice", "SalesReceipt", "Purchase", "Bill", "JournalEntry"]
        self.other_names = ["Account", "Customer", "Department", "Item", "Term", "Vendor", "Class", "Employee"]
        self.PL_raw_cols = ["TransactionDate", "TransactionType", "DocNumber", "Name", "Location", "Class", "Memo", "SplitAcc", "Amount", "Balance"]
        self.GL_raw_cols = ["TransactionDate", "TransactionType", "DocNumber", "IsAdjust", "Name", "Memo", "SplitAcc", "Amount", "Balance"]
        self.acctype_QBO_expense = ["Expense","Cost of Goods Sold", "Other Expense"]
        self.profitem_cube_expense = ["Cost of Goods Sold", "Direct Operating Expenses", "Operating Overheads", "Other Expense"]
        self.get_fx()
        
    
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
            json.dump(tokens, f)
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
                with open(file_path, "w") as f:
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
            year = self.today.year - 1 
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
            name_list = full_name.split(":")
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
            df.loc[i,"Profitem"] = name_list[1]
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
            df = df.drop(columns=["Id", "Line_Id"])
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
            account = pd.merge(account,account_cube.loc[:,["AccNum", "AccountingType", "Profitem" ,"Category","Subcategory"]],on="AccNum",how="left")
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
                "APAccountRef_value","MetaData_CreateTime", "MetaData_LastUpdatedTime","DocNumber","LinkedTxn"]
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
            if entry["Amount"] == "120000 Accounts Receivable":
                print(entry)
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
            # determine the columns in raw report for the current company - PL only
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
        records_all["AmountAdj"] = records_all.apply(lambda x: self._report_adjust_sign(x, target_col="AmountAdj"), axis=1)
        records_all["AmountCAD"] = records_all.apply(lambda x: self._report_adjust_sign(x, target_col="AmountCAD"), axis=1)
        records_all["FXRate"] = self.fx
        records_all["Country"] = records_all["Corp"].apply(lambda x: "USA" if x in self.us_companies else "Canada")
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

    # flow
    def extract(self, load_raw: bool=True, load_pl: bool=True, light_load: bool=True, extract_only:list[str,None]=[]) -> None:
        
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
            # extract GL - always for Weekly Banking Project
            self._pull_reports(auth_client=auth_client,company=company,light_load=light_load,report_type="GL")
            # extract PL
            if load_pl:
                self._pull_reports(auth_client=auth_client,company=company,light_load=light_load,report_type="PL")
            # extract raw
            if load_raw:
                self._pull_raw(table_names=self.names+self.other_names, auth_client=auth_client, company=company)
        print("Finished QBO Extraction ...")
        self.log.write("\nFinished QBO Extraction\n\n")
        
    def transform(self, light_load:bool=True) -> None:
        print("\nStarting QBO Transformation ...")
        self.log.write("\nStart QBO Transformation\n")
        # QBO_Raw_Processing.py
        self._raw_transform()
        # read account table for PL, GL transformation
        self.account = pd.read_csv(self.silver_path["QBO"]["Dimension_time"]/"Account.csv")
        self.account_QBO_expense = self.account[self.account["AccountType"].isin(self.acctype_QBO_expense)].AccID.unique()
        # transform GL & PL 
        for mode in ["PL", "GL"]:
            path_old = (self.silver_path["QBO"]["PL"]/"ProfitAndLoss.csv") if mode == "PL" else (self.silver_path["QBO"]["GL"]/"GeneralLedger.csv")
            df_new = self._report_transform(report_type=mode,light_load=light_load)
            # if light load mode, load and merge with old records
            if light_load:
                df_new = self._report_merge(mode=mode, df_new=df_new, path_old=path_old)
            df_new.to_csv(path_old, index=False)
        print("Finished QBO Transformation ...")
        self.log.write("\nFinished QBO Transformation\n\n")
            
    def run(self, QBO_light:bool=True, extract:bool=True, extract_only:list[str,None]=[]) -> None:
        # measure time 
        start = perf_counter()

        # start logging
        self.create_log(path=self.raw_path["Log"])
        self.log.write("\n"*4 + "*"*100 + "\n\nStart of QBO Pipeline\n\n")
        print("\nStart QBO Pipeline ...")
        
        # QBO Extracting
        if extract: self.extract(light_load=QBO_light, extract_only=extract_only)

        # QBO Transformation
        self.transform(light_load=QBO_light)

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
        if force_run or (self.today.weekday()==0 or self.today.weekday() == 2): # only run monday night or wednesday night or force_run
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

class Projects(Job):
    """ 
        for project specific data transformations
    """
    
    def __init__(self, focus_last_FY:bool = False):
        super().__init__()
        self.gold_path = {
            "weekly_banking": self.base_dir / "Gold" / "FinanceProject" / "WeeklyBanking",
            "inventory": self.base_dir / "Gold" / "InventoryProject",
            "payroll": self.base_dir / "Gold" / "HRProject" /"PayrollProject",
            "finance_operational": self.base_dir / "Gold" / "FinanceOperationalProject",
            "budget": self.base_dir / "Gold" / "BudgetProject",
            "QBOTime": self.base_dir / "Gold" / "HRProject" / "QBOTimeProject",
            "hr_combined": self.base_dir / "Gold" / "HRProject" / "CombinedSummary"
        }
        self.silver_acc = pd.read_csv(self.silver_path["QBO"]["Dimension_time"]/"Account.csv")
        self.commodities = {
            "Produce": ["Strawberry", "Watermelon", "Cantaloupe", "Market Garden", "Broccoli", "Pumpkin", "Sweet Corn", "Cauliflower", "Squash", "Honeydew Melon", "Potato", "Carrot", "Cabbage",
                        "Lettuce", "Brussel Sprouts", "Prairie Pathways", "Beet", "Corn Maze", "CSA"],
            "Grain": ["Blackeye Pea", "Winter Wheat", "Durum", "Cotton", "Chickpea", "Barley", "Green Lentil", "Red Lentil", "Canola", 
                        "Wheat","Field Pea", "Corn", "Oat", "Soybean", "Bean"],
            "Cattle": ["Weaned Calves", "Cull Bull", "Cull Cow", "Bred Heifer", "Purebred Yealing Bull", "Purebred Heifer", 
                        "Purebred Cow", "Purebred Bull", "Cow", "Bull", "Steer", "Heifer", "Yearling", "Calf"]
        }
        self.locations = {
            "Produce": ["BritishColumbia (produce)", "Outlook", "Arizona (produce)"],
            "Cattle": ["Airdrie", "Eddystone (cattle)", "Ashcroft", "Home Ranch", "Diamond S", "Wolf Ranch", "Fraser River Ranch", "Moon Ranch", "Waldeck", "Calderbank"],
            "Grain": ["Eddystone (grain)", "Arizona (grain)", "Colorado", "Swift Current", "Regina", "Raymore", "Prince Albert", "The Pas",
                      "Kamsack", "Hafford", "Yorkton", "Fly Creek", "Camp 4", "Havre", "Billings"],
            "Seed": ["NexGen", "Seeds", "Seeds USA"],
            "Others": ["Eddystone (corporate)", "Arizona (corporate)", "Legacy", "BritishColumbia (corporate)", "-Corporate"]
        }
        self.bc_ranches = ["Ashcroft", "Fraser River Ranch", "Moon Ranch", "Wolf Ranch", "Home", "Diamond S", "BritishColumbia (corporate)","Home Ranch"]
        self.pl_exist = False # determines whether _financial_operational has run and gold_pl is stored in self, if not, any subsequent downstream projects will run _financial_operational first
        self.currentFY = self.today.year if self.today.month<=10 else self.today.year + 1
        if focus_last_FY: self.currentFY -= 1

    def _pillar_classification(self, entry: pd.Series) -> str:
        """ 
            this function classifies pillar of a transaction based on location
        """
        location = entry["Location"]
        if not isinstance(location, str):
            return "Missing"
        if "produce" in location:
            return "Produce"
        elif "grain" in location:
            return "Grain"
        elif "cattle" in location:
            return "Cattle"
        elif "corporate" in location:
            return "Unclassified"
        match location.lower():
            case "hafford"|"kamsack"|"prince albert"|"raymore"|"regina"|"swift current"|"the pas"|"camp 4"|"fly creek"|"havre"|"yorkton"|"colorado"|"billings":
                return "Grain"
            case "outlook"|"seeds usa":
                return "Produce"
            case "ashcroft"|"diamond s"|"fraser river ranch"|"home ranch"|"moon ranch"|"wolf ranch"|"waldeck"|"calderbank"|"airdrie":
                return "Cattle"
            case "seeds"|"nexgen":
                return "Seed"
            case _:
                return "Unclassified"
    
    def _identify_product(self, entry: pd.Series, for_budget:bool=False) -> str:
        """ 
            this function identifies commodity from account names, except for seed, 
                if this function is called from budget project, it combines MG & CSA and take CM into consideration, and name forage differently
        """
        if not for_budget:
            if entry["AccPillar"] == "Seed":
                return "SeedProduct"
        accname = entry["AccountName"].lower() if not for_budget else entry["AccFull"].lower()
        if "float" in accname:
            return "Others"
        for x in self.commodities["Produce"] + self.commodities["Grain"] + self.commodities["Cattle"]:
            if x.lower() in accname:
                if for_budget:
                    match x:
                        case "Market Garden"|"CSA":
                            return "Market Garden / CSA"
                        case "Corn Maze":
                            return "Prairie Pathways"
                    return x 
        if "straw" in accname or "forage" in accname or "hay bale" in accname:
            if for_budget: 
                return "Hay/Silage" 
            else: 
                return "Forage"
        return "Others"
    
    def _weekly_banking(self) -> None:
        """ 
            weekly banking project: match latest GL bank transactions with raw activities - extract accounts for those activities
                assumptions: a raw entry (e.g., invoice) can have multiple lines - multiple associated accounts, only considering the first one 
        """
        print("\nStarting Weekly Banking Project Transformation\n")
        # determine minal date to keep for GL
        if self.today.month > 6:
            year = self.today.year 
            month = self.today.month - 6 
        else:
            year = self.today.year - 1
            month = self.today.month + 12 - 6
        # load and prepare data
        ## account
        account = self.silver_acc.copy(deep=True)
        ## change some accounts to Transfer category
        acc_list = ["MFL264", "MSL250"]
        account.loc[account["AccID"].isin(acc_list), "Profitem"] = "Asset"
        account.loc[account["AccID"].isin(acc_list), "Category"] = "Transfer"
        account_bank = account[account["AccountType"]=="Bank"]
        ## LinkedTxn for invoice and bill
        invoice_linked = pd.read_csv(self.silver_path["QBO"]["Raw"] / "LinkedTxn"/ "LinkedTxn_Mapping_Invoice.csv")
        bill_linked = pd.read_csv(self.silver_path["QBO"]["Raw"] / "LinkedTxn"/ "LinkedTxn_Mapping_Bill.csv")
        mapping = pd.concat([invoice_linked, bill_linked])
        mapping = mapping.drop(columns=["Corp"])
        # define customized function for processing other raw table
        def _process_facts(df_type:str) -> pd.DataFrame:
            """ 
                function for processing raw tables for mapping table - TransactionID_partial to AccID
            """
            df = pd.read_csv(self.silver_path["QBO"]["Raw"]/(df_type+".csv"), usecols = ["TransactionID", "AccID"])
            df["TransactionID"] = df["TransactionID"].apply(lambda x: x.split("-")[1])
            df = df.drop_duplicates()
            df = df.rename(columns={"TransactionID":"TxnId"})
            return df
        ## purchase table for expense transactions
        purchase = _process_facts("Purchase")
        purchase["TxnType"] = "Expense"
        mapping = pd.concat([mapping,purchase])
        ## journal entries - exclude most entries related to bank
        journal = _process_facts("JournalEntry")
        journal["TxnType"] = "Journal Entry"
        # for journal entries, exclude most of entires where the activity account ID is a bank ID
        exclude_list = list(account_bank.AccID.unique())
        # mylist = ["MFL51", "MFBC470", "MFBC471", "MFL28", "MFL27", "MFL1150040024"]
        mylist = ["MFBC470", "MFBC471"] # should include these accounts
        for acc in mylist:
            exclude_list.remove(acc)
        journal = journal[~journal["AccID"].isin(exclude_list)]
        mapping = pd.concat([mapping,journal])
        ## deposit
        deposit = _process_facts("Deposit")
        deposit["TxnType"] = "Deposit"
        mapping = pd.concat([mapping,deposit])
        ## salesreceipts
        sales = _process_facts("SalesReceipt")
        sales["TxnType"] = "Sales Receipt"
        mapping = pd.concat([mapping,sales])
        # process mapping table - dedup
        mapping = mapping.drop_duplicates(subset=["TxnId"],keep="first")
        ## load GL transacitons
        cols = ["TransactionType","TransactionID_partial","AccID","AccNum","AccName", "TransactionDate", "Amount", "SplitAcc", "SplitAccID", "Memo", "Corp", "Balance"]
        transactions = pd.read_csv(self.silver_path["QBO"]["GL"]/"GeneralLedger.csv",dtype={"TransactionID_partial":str}, usecols=cols)
        transactions = transactions[transactions["AccID"].isin(account_bank.AccID.unique())]
        transactions["TransactionDate"] = pd.to_datetime(transactions["TransactionDate"])
        transactions = transactions[transactions["TransactionDate"]>=dt.datetime(year, month, 1)]
        transactions = transactions.rename(columns={"TransactionType":"TxnType","TransactionID_partial":"TxnId",
                                                    "AccID":"BankAccID","AccNum":"BankAccNum","AccName":"BankAccName",
                                                    "TransactionDate":"BankActivityDate","Amount":"BankAmount"})
        # merge to get CurrencyID for bank_acc
        transactions = pd.merge(transactions, account_bank.loc[:,["AccID","CurrencyID"]], left_on=["BankAccID"], right_on=["AccID"], how="left")
        transactions = transactions.drop(columns=["AccID"])
        # separating transfers - don't merge with mapping table
        transfers = transactions[transactions["TxnType"] == "Transfer"].copy(deep=True)
        transactions = transactions[transactions["TxnType"]!="Transfer"]
        transactions = transactions.drop(columns=["SplitAcc", "SplitAccID"])
        transactions["BankActivityDate"] = pd.to_datetime(transactions["BankActivityDate"])
        transactions["TxnType"] = transactions["TxnType"].replace({"Cheque Expense":"Expense", "Check": "Expense"})
        # merge with mapping table
        transactions_mapped = pd.merge(transactions,mapping,on=["TxnId","TxnType"],how="left")
        non_match = transactions_mapped[transactions_mapped["AccID"].isna()]
        print("None Match Transaction Types")
        print(non_match.TxnType.value_counts())
        print(f"Non matches - {len(non_match)}")
        # function to determine transfer type
        def _determine_transfer_type(entry:str) -> str:
            """ 
                determine whether the transfer is for visa, bank, or other transfer
            """
            if "visa" in entry.lower():
                return "Visa Payment"
            elif "due" in entry.lower():
                return "Bank Transfer"
            else:
                return "Other Transfer"
        # allocate transfer type 
        transfers["TransferType"] = transfers["SplitAcc"].apply(lambda x: _determine_transfer_type(x))
        transfers = transfers.rename(columns={"SplitAccID":"AccID"})
        transfers = transfers.drop(columns=["SplitAcc"])
        transactions_mapped = pd.concat([transactions_mapped,transfers], ignore_index=True)
        # clean up the dataframe
        transactions_mapped = transactions_mapped.rename(columns={"CurrencyID":"BankCurrencyID"})
        transactions_mapped = pd.merge(transactions_mapped, account.loc[:,["AccID","AccName","AccNum","Category","Profitem","CurrencyID"]], on="AccID", how="left")
        transactions_mapped.loc[transactions_mapped["TransferType"]=="Bank Transfer","Category"] = "Bank Transfer"
        transactions_mapped.loc[((transactions_mapped["BankAccNum"].str.startswith("MSL"))&(transactions_mapped["AccNum"]=="MSL120001")), "Category"] = "Seed Processing Revenue"
        transactions_mapped = transactions_mapped.rename(columns={"AccNum":"ActivityAccNum", "AccName":"ActivityAccName"})
        # csv from sharepoint is unstable, and produced unpredictable readings from Power BI
        self.check_file(self.gold_path["weekly_banking"])
        transactions_mapped.to_excel(self.gold_path["weekly_banking"]/"BankingActivity.xlsx", sheet_name="transactions", index=False)

    def _finance_operational(self) -> None:
        """ 
            transform PL data into operational-ready
                1. reclassify accounts
                2. standardize location, classify pillar
                3. revising signs
        """
        print("\nStarting Finance Operational Project Transformation\n")
        # load data from silver space
        data = pd.read_csv(self.silver_path["QBO"]["PL"]/"ProfitAndLoss.csv")
        assert len(data.FXRate.value_counts()) == 1, "different FXRate detected"
        self.fx = data.loc[0,"FXRate"]
        data["TransactionDate"] = pd.to_datetime(data["TransactionDate"])
        data["FiscalYear"] = data.TransactionDate.apply(lambda x: x.year + 1 if x.month >= 11 else x.year)
        # add month to the PL
        data["Month"] = data["TransactionDate"].dt.month_name()
        ## add location for seed operation
        data.loc[data["Corp"]=="MSL","Location"] = "Seeds"
        data.loc[data["Corp"]=="NexGen","Location"] = "NexGen"
        data.loc[data["Corp"]=="MSUSA","Location"] = "Seeds USA"
        # clean location
        data = data.rename(columns={"Location":"LocationRaw"})
        data["Location"] = data["LocationRaw"]
        # switch seeds usa to AZ produce
        data.loc[data["Corp"]=="MSUSA","Location"] = "Arizona (produce)"
        ## clean location
        clean_location = {"Airdrie - Grain":"Airdrie", "Airdrie - Cattle":"Airdrie", "Airdrie - General":"Airdrie", "Airdrie":"Airdrie", 
                        "Eddystone - Grain": "Eddystone (grain)", "Eddystone - Cattle": "Eddystone (cattle)", "Eddystone - General":"Eddystone (corporate)",
                        "Outlook (JV)":"Outlook", "AZ Produce":"Arizona (produce)", "Corporate":"Arizona (corporate)", "BC Produce":"BritishColumbia (produce)",
                        "Grain":"Arizona (grain)", "Cache/Fischer/Loon - DNU":"Legacy", "Ashcroft (CC, Fischer, Loon)":"Ashcroft", 
                        "Outlook (Capital)":"Outlook", "Colorado (MF)":"Colorado", "Colorado (JV)":"Colorado", "Cattle - General":"BritishColumbia (corporate)",
                        "Home (70 M, LF/W, 105 M)":"Home Ranch", "Diamond S (BR)":"Diamond S", "North Farm (deleted)":"Legacy"}
        data["Location"] = data["Location"].replace(clean_location)
        locations = self.locations["Produce"] + self.locations["Grain"] + self.locations["Cattle"] + self.locations["Others"] + self.locations["Seed"]
        unaccounted_location = list(set(data["Location"].unique()) - set(locations))
        print(f"location unaccounted for - {unaccounted_location}")
        # classify pillar
        data["Pillar"] = data.apply(lambda x: self._pillar_classification(x),axis=1)
        # reorganize corp
        ## MPUSA missing location = Arizona (produce)
        data.loc[((data["Corp"] == "MPUSA")&(data["Location"].isna())), "Location"] = "Arizona (produce)"
        data.loc[((data["Corp"] == "MPUSA")&(data["Location"] == "Arizona (produce)")), "Pillar"] = "Produce"
        ## AZ Produce --> MPUSA
        data.loc[data["Location"] == "Arizona (produce)", "Corp"] = "MPUSA"
        ## move everything for AZ in 2025 to produce
        data.loc[((data["FiscalYear"] >= 2025) & (data["Location"].str.contains("Arizona",case=False))),"Pillar"] = "Produce"
        data.loc[((data["FiscalYear"] >= 2025) & (data["Location"].str.contains("Arizona",case=False))),"Location"] = "Arizona (produce)"
        ## BC Produce --> MPL
        data.loc[data["Location"] == "BritishColumbia (produce)", "Corp"] = "MPL"
        ## Outlook --> MPL
        data.loc[data["Location"]=="Outlook", "Corp"] = "MPL"
        # Reclassify accounts for Operational Purpose
        ## read & process operational classification
        acc_operation = pd.read_csv(self.silver_path["QBO"]["Dimension"]/"Accounts Classification - Operation.csv", dtype={"Pillar": str, "IsGenric": str}, keep_default_na=False)
        acc_operation = acc_operation.rename(columns={"Pillar":"AccPillar", "OperationProfiType":"OperationProfType"})
        acc_operation["AccNum"] = acc_operation["AccountName"].apply(lambda x: x.split(" ")[0])
        acc_operation["IsIntercompany"] = acc_operation["AccountName"].apply(lambda x: "Yes" if "intercompany" in x.lower() else "No")
        ## classify commodity
        acc_operation["Commodity"] = acc_operation.apply(lambda x: self._identify_product(x), axis=1)
        self.operation_acc = acc_operation
        acc_operation.to_csv(self.gold_path["finance_operational"]/"accounts_classified_operation.csv", index=False)
        ## read accounts table and apply new classification
        accounts = self.silver_acc
        accounts1 = accounts[accounts["AccNum"].isna()].copy(deep=True)
        accounts = accounts[accounts["AccNum"].notna()]
        accounts = pd.merge(accounts, acc_operation.loc[:,["AccNum","OperationProfType","OperationCategory","OperationSubCategory","AccPillar","Commodity","IsGeneric","IsIntercompany"]],
                            on = "AccNum", how = "left")
        accounts = pd.concat([accounts,accounts1],ignore_index=True)
        # Revising Signs according to Operational Classification
        print("Revising Signs ...")
        expense_accounts = accounts[(accounts["OperationCategory"] == "Expense") | (accounts["OperationCategory"] =="Inventory Consumption")]
        data["AmountDisplay"] = data.apply(lambda x: -x["AmountCAD"] if x["AccID"] in expense_accounts.AccID.unique() else x["AmountCAD"], axis=1)
        self.gold_pl = data
        self.gold_acc = accounts
        # save files
        print("Saving ...")
        self.check_file(self.gold_path["finance_operational"])
        data.to_csv(self.gold_path["finance_operational"]/"PL.csv", index=False)
        accounts.to_excel(self.gold_path["finance_operational"]/"Account_table.xlsx", sheet_name = "Account", index=False)
        data.to_excel(self.gold_path["finance_operational"]/"PL.xlsx", sheet_name="Transactions", index=False)
        self.pl_exist = True

    def _process_pp(self, data:pd.DataFrame) -> pd.DataFrame:
        """ 
            This function takes original dataframe, apply the payperiod number classification based on transactions date, process payperiod columns, and return the new dataframe,
                save the pp table for consolidated tables
        """
        # load payperiods
        payperiods = pd.read_csv(self.gold_path["payroll"]/"Payperiods.csv")
        payperiods["START"] = pd.to_datetime(payperiods["START"])
        payperiods["END"] = pd.to_datetime(payperiods["END"])
        payperiods = payperiods.loc[:,["PP","START","END","Cycle","FiscalYear"]]
        def _determine_pp(entry:pd.Series, date_col:str = "TransactionDate") -> str:
            """ 
                This function determined which payperiod a transaction should be classified into, 
                    starting from most recent payperiod, the algorithm uses period start date + drift to determine which payperiod a transaction should fall into
                Assumptions:
                    1. for outlook and az, each payperiod is shifted by 5 + 7 days forward
                    2. for other location, each payperiod is shifted by 5 days forward
            """
            date = entry[date_col] 
            if isinstance(entry["Location"],str):
                location = entry["Location"].lower()
            else:
                location = "None"
            if "outlook" in location or "az" in location or "arizona" in location:
                date_diff = dt.timedelta(days=5+7)
            else:
                date_diff = dt.timedelta(days=5)
            year = date.year 
            month = date.month
            # push back the most recent payperiod dates for older transactions to save compute
            if month >= 11:
                payperiods_subset = payperiods[payperiods["END"] <= dt.datetime(year+1,2,1)] 
            else:  
                payperiods_subset = payperiods[payperiods["END"] <= dt.datetime(year,month+2,1)]
            for i in range(len(payperiods_subset)-1,-1,-1):
                if date > (payperiods_subset.loc[i,"END"]+date_diff):
                    return "Exceed Max PayPeriod"
                if date >= (payperiods_subset.loc[i,"START"]+date_diff):
                    return str(payperiods_subset.loc[i,"PP"]) + "-" + str(payperiods_subset.loc[i,"Cycle"]) + "-" + str(payperiods_subset.loc[i,"FiscalYear"])
            return "Earlier than Min PayPeriod"
        print("Allocating PPNum for transactions ...")
        date_col = "TransactionDate" if "TransactionDate" in data.columns else "date"
        data["PPNum"] = data.apply(lambda x: _determine_pp(x,date_col),axis=1)
        data = data[data["PPNum"] != "Earlier than Min PayPeriod"].copy(deep=True).reset_index(drop=True) # eliminate earlier than min payperiod in the csv, note dates are shifted in the csv
        data["Cycle"] = data["PPNum"].apply(lambda x: x.split("-")[1])
        data["FiscalYear"] = data["PPNum"].apply(lambda x: int(x.split("-")[2]))
        data["PPNum"] = data["PPNum"].apply(lambda x: x.split("-")[0])
        data["PPName"] = data["PPNum"].apply(lambda x: "PP0" + x if int(x) < 10 else "PP" + x)
        data["PPName"] = data["Cycle"].str.slice(2,) + "-" + data["PPName"]
        data.loc[:,["PPName", "PPNum", "Cycle", "FiscalYear"]].drop_duplicates().to_csv(self.gold_path["payroll"].parent/ "OtherTables" / "PayPeriods.csv", index=False)
        return data

    def _process_units(self) -> None:
        """ 
            this function read and process Unit files that contains unit numbers for each location
        """
        acres = pd.read_csv(self.gold_path["payroll"]/"Unit.csv",dtype={"Location":str, "Unit":float})
        acres["Location"] = acres["Location"].str.strip()
        doc_rename = {"Airdrie Grain": "Airdrie (grain)", "Aridrie Cattle (head days 365)":"Airdrie", "Arizona All":"Arizona (produce)",
                    "BC Cattle (head days 365)":"BritishColumbia (cattle)", "BC Produce":"BritishColumbia (produce)", 
                    "Box Elder":"Havre", "Eddystone Cattle (head days 365)":"Eddystone (cattle)", "Eddystone Grain":"Eddystone (grain)",
                    "Monette Seeds CDN (avg met. ton)":"Seeds", "Monette Seeds USA":"Seeds USA", "NexGen (avg met. ton)":"NexGen",
                    "Waldeck (head days 365)":"Waldeck", "Calderbank  (head days 365)":"Calderbank"}
        acres["Location"] = acres["Location"].replace(doc_rename)
        acres["Pillar"] = acres.apply(lambda x: self._pillar_classification(x),axis=1)
        acres.to_csv(self.gold_path["payroll"].parent/ "OtherTables" /"Unit_PowerBI.csv",index=False)

    def _payroll_project(self) -> None: 
        """ 
            will run _finance_operational() first
            output: details + cost per unit (units per location input sheet) + average cost per unit for FY
        """
        self.check_file(self.gold_path["payroll"].parent/ "OtherTables")
        if not self.pl_exist:
            self._finance_operational()
        print("\nStarting Payroll Project Transformation\n")

        # load and filter accounts for wages and contract labor
        account = self.silver_acc[(self.silver_acc["Category"].isin(["Wages and benefits - direct","Wages and benefits - overhead"]) | (self.silver_acc["AccNum"].isin(["MFAZ595001","MFBC536030"])))] 
        # load only with transaction date later than 2021-12-20, and without "Accrual" in the memo
        data = self.gold_pl.copy(deep=True)
        data = data[data["AccID"].isin(account.AccID.unique())]
        data["TransactionDate"] = pd.to_datetime(data["TransactionDate"])
        data = data[data["TransactionDate"]>=dt.datetime(2021,12,20)].reset_index(drop=True)
        data = data[~data["Memo"].str.contains("Accrual",case=False,na=False)]
        # allocating payperiods
        data = self._process_pp(data=data)
        # standardizing location
        # data.loc[data["Location"]=="Airdrie (corporate)", "Pillar"] = "Cattle"                # deprecated
        # data.loc[data["Location"]=="Airdrie (corporate)", "Location"] = "Airdrie (cattle)"    # deprecated
        data.loc[data["Location"]=="Eddystone (corporate)", "Pillar"] = "Unclassified"
        data.loc[data["Location"]=="Eddystone (corporate)", "Location"] = "Unassigned"
        data.loc[data["Location"]=="Legacy", "Location"] = "Unassigned"
        data.loc[(data["Location"].str.contains("corporate",case=False,na=False)&(data["Location"]!="BritishColumbia (corporate)")),"Location"] = "Corporate"
        ## move BC ranches into BC Cattle
        data.loc[(data["Location"].isin(self.bc_ranches)), "Location"] = "BritishColumbia (cattle)"
        data.loc[data["Location"] == "BritishColumbia (cattle)", "Pillar"] = "Cattle"
        # summarizing data
        ## by Location per PP
        data_summarized = pd.DataFrame(data.groupby(["Location","PPName","Pillar","FiscalYear","Cycle","PPNum"]).agg({"AmountDisplay":"sum"}).reset_index(drop=False))
        assert len(data_summarized) == len(data.groupby(["Location","PPName"]).agg({"AmountDisplay":"sum"}).reset_index(drop=False)), "Duplicated value detected for per Location per PP calculation"
        ## join acres data for CostPerUnit compute
        print("Summarizing ...")
        self._process_units()
        acres = pd.read_csv(self.gold_path["payroll"].parent/ "OtherTables" /"Unit_PowerBI.csv",dtype={"Location":str, "Unit":float})
        acres = acres.loc[:,["Location", "Unit"]]
        print(f"Unaccounted location for Acres Doc: {set(acres.Location.unique()) - set(data_summarized.Location.unique())}")
        data_summarized = pd.merge(data_summarized, acres, on="Location", how="left")
        data_summarized["CostPerUnit"] = data_summarized["AmountDisplay"] / data_summarized["Unit"] * 26
        data_summarized["Count"] = 1
        ## by Location
        data_summarized2 = data_summarized.groupby(by=["Location","FiscalYear","Pillar"]).agg({"CostPerUnit":"mean", "Count":"sum"}).reset_index(drop=False)
        data_summarized2 = data_summarized2.rename(columns={"CostPerUnit":"Avg CostPerUnit"})
        assert len(data_summarized2) == len(data_summarized.groupby(by=["Location","FiscalYear"]).agg({"CostPerUnit":"mean"})), "Duplicated value detected for per Location calculation"
        ## by pillar
        data_summarized3 = data_summarized2.groupby(by=["FiscalYear","Pillar"]).agg({"Avg CostPerUnit":"mean", "Count":"sum"}).reset_index(drop=False)
        assert len(data_summarized3) == len(data_summarized.groupby(by=["Pillar","FiscalYear"]).agg({"CostPerUnit":"mean"})), "Duplicated value detected for per Pillar calculation"
        # saving
        print("Saving ...")
        self.check_file(self.gold_path["payroll"])
        data.to_excel(self.gold_path["payroll"]/"Payroll.xlsx", sheet_name="Payroll", index=False)
        self.check_file(self.gold_path["hr_combined"] / "CSV")
        data_summarized.to_csv(self.gold_path["hr_combined"]/ "CSV" / "payroll_summarized1.csv", index=False)
        data_summarized2.to_csv(self.gold_path["hr_combined"]/ "CSV" / "payroll_summarized2.csv", index=False)
        data_summarized3.to_csv(self.gold_path["hr_combined"]/ "CSV" / "payroll_summarized3.csv", index=False)

    def _QBOTime_project(self) -> None:
        """ 
            apply PP allocation to QBO Time data, clean locaiton, and join relevant info into one table
        """
        print("\nStarting QBO Time Project Transformation\n")
        # read files
        timesheets = pd.read_csv(self.silver_path["QBO"]["Time"]/"timesheets.csv")
        jobcode = pd.read_csv(self.silver_path["QBO"]["Time"]/"jobcodes.csv")
        users = pd.read_csv(self.silver_path["QBO"]["Time"]/"users.csv")
        group = pd.read_csv(self.silver_path["QBO"]["Time"]/"group.csv")
        print(f"Read {len(timesheets)} timesheet records, {len(jobcode)} jobcodes, {len(users)} users, {len(group)} groups")
        timesheets_len, users_len = len(timesheets), len(users)
        # clean up location in group table
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
        # create another location column for general location where bc ranches are merged into one
        group = group.rename(columns={"Location": "Location (detail)"})
        group["Location"] = group["Location (detail)"]
        group.loc[(group["Location (detail)"].isin(self.bc_ranches)), "Location"] = "BritishColumbia (cattle)"
        # merge tables into one table
        ## merge location into users
        users = pd.merge(users, group.loc[:,["group_id", "location_name", "Location", "Location (detail)"]].drop_duplicates(), on="group_id", how="left")
        ## merge users into timesheets
        timesheets = pd.merge(timesheets,users.loc[:,["user_id", "group_id", "username", "full_name", "location_name","Location","Location (detail)"]], on="user_id", how="left")
        ## merge job into timesheets
        timesheets = pd.merge(timesheets, jobcode.loc[:,["jobcode_id","job_name","type"]].rename(columns={"type":"job_type"}), on="jobcode_id", how="left")
        assert (len(users) == users_len) and (len(timesheets) == timesheets_len), f"duplicated records found, timesheets - {timesheets_len} vs {len(timesheets)}; users - {users_len} vs {len(users)}"
        # classify payperiods
        timesheets["date"] = pd.to_datetime(timesheets["date"])
        timesheets = self._process_pp(data=timesheets)
        # modify location for BC0
        timesheets.loc[timesheets["user_id"] == "BC6107856", "Location"] = "Unassigned"
        # classify pillars
        timesheets["Pillar"] = timesheets.apply(lambda x: self._pillar_classification(x), axis=1)
        timesheets.loc[timesheets["Pillar"] == "Missing", "Pillar"] = "Unclassified"
        # summarizing data
        ## by Location per PP 
        summarized = timesheets.groupby(["Location","PPName","FiscalYear","Cycle","PPNum", "Pillar"]).agg({"duration":"sum"}).reset_index(drop=False)
        assert len(summarized) == len(timesheets.groupby(["Location","PPName"]).agg({"duration":"sum"})), "duplicated value detected for timsheet per Location per PP summarization"
        ## read units file
        acres = pd.read_csv(self.gold_path["payroll"].parent/ "OtherTables" /"Unit_PowerBI.csv",dtype={"Location":str, "Unit":float})
        acres = acres.loc[:,["Location", "Unit"]]
        addition = pd.DataFrame(data={"Location":["Billings"], "Unit":[acres[acres["Location"].isin(['Fly Creek', 'Camp 4'])].Unit.sum()]})
        acres = pd.concat([acres,addition],ignore_index=True)
        print(f"Unaccounted location for Acres Doc: {set(acres.Location.unique()) - set(summarized.Location.unique())}")
        print(f"Unaccounted location for timesheets: {set(summarized.Location.unique()) - set(acres.Location.unique())}")
        ## merge with units file
        summarized = pd.merge(summarized, acres, on="Location", how="left")
        ## calculate hours per unit
        summarized["HoursPerUnit"] = summarized["duration"] / summarized["Unit"] * 26
        summarized["Count"] = 1
        # summarize per location
        summarized2 = summarized.groupby(by=["Location","FiscalYear", "Pillar"]).agg({"HoursPerUnit":"mean", "Count":"sum"}).reset_index(drop=False)
        summarized2 = summarized2.rename(columns={"HoursPerUnit":"Avg HoursPerUnit"})
        assert len(summarized2) == len(timesheets.groupby(["Location","FiscalYear"]).agg({"duration":"sum"})), "duplicated value detected for timsheet per Location summarization"
        # summarize per pillar
        summarized3 = summarized2.groupby(by=["FiscalYear", "Pillar"]).agg({"Avg HoursPerUnit":"mean", "Count":"sum"}).reset_index(drop=False)
        assert len(summarized3) == len(timesheets[timesheets["Pillar"]!="Missing"].groupby(["Pillar","FiscalYear"]).agg({"duration":"sum"})), "duplicated value detected for timsheet per Pillar summarization"

        # saving
        print("Saving ...\n")
        self.check_file(self.gold_path["QBOTime"])
        timesheets.to_excel(self.gold_path["QBOTime"]/"QBOTime.xlsx", sheet_name = "QBOTime", index=False)
        self.check_file(self.gold_path["hr_combined"]/ "CSV")
        summarized.to_csv(self.gold_path["hr_combined"]/ "CSV" / "time_summarized1.csv", index=False)
        summarized2.to_csv(self.gold_path["hr_combined"]/ "CSV" / "time_summarized2.csv", index=False)
        summarized3.to_csv(self.gold_path["hr_combined"]/ "CSV" / "time_summarized3.csv", index=False)

    def _hr_summary(self) -> None:
        """ 
            This function consolidate payroll and QBO time summaries into one table for consolidated insights
        """
        final_df = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]
        for i in [1, 2, 3]:
            payroll = pd.read_csv(self.gold_path["hr_combined"] / "CSV" / f"payroll_summarized{i}.csv")
            payroll_rename = {"AmountDisplay": "TotalAmount", "CostPerUnit": "AmountPerUnit", "Avg CostPerUnit": "Avg AmountPerUnit"}
            payroll = payroll.rename(columns=payroll_rename)
            payroll["Mode"] = "Payroll"
            time = pd.read_csv(self.gold_path["hr_combined"] / "CSV" / f"time_summarized{i}.csv")
            time_rename = {"duration": "TotalAmount", "HoursPerUnit": "AmountPerUnit", "Avg HoursPerUnit": "Avg AmountPerUnit"}
            time = time.rename(columns=time_rename)
            time["Mode"] = "Hours"
            final_df[i-1] = pd.concat([payroll, time], ignore_index=True)
        final_df[0].to_excel(self.gold_path["hr_combined"]/"Summarized.xlsx", sheet_name="Summarized", index=False)
        final_df[1].to_excel(self.gold_path["hr_combined"]/"Summarized2.xlsx", sheet_name="Summarized2", index=False)
        final_df[2].to_excel(self.gold_path["hr_combined"]/"Summarized3.xlsx", sheet_name="Summarized3", index=False)

    def _temp_get_product(self, entry:str) -> str:
        """ 
            temporary function for aligning product classification with Traction for QBO accounts, will change for HP
        """
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

    def _raw_inventory(self) -> None:
        """ 
            prepare the data from raw QBO table for inventory project: only extracting partial Invoice, SalesReceipt, and Journal Entry
        """
        print("\nStarting Inventory Project Transformation ...\n")
        corps = ["MFL", "MFUSA"]
        cols = ["TransactionDate", "TransactionType", "TransactionID", "Corp", "Qty", "AccID", "FarmID", "CustomerID",
                "DocNumber", "TransactionEntered", "Amount"]
        journal_cols = [col for col in cols if col != "Qty"]
        # read tables
        print("Loading raw tables ...")
        account = self.silver_acc.copy(deep=True)
        account = account[account["Corp"].isin(corps)]
        account = account[account["AccountType"] == "Income"]
        farm = pd.read_csv(self.silver_path["QBO"]["Dimension_time"]/"Farm.csv")
        farm = farm[farm["Corp"].isin(corps)]
        customer = pd.read_csv(self.silver_path["QBO"]["Dimension_time"]/"Customer.csv")
        customer = customer[customer["Corp"].isin(corps)]
        first_date = dt.datetime(2023,11,1)
        invoice = pd.read_csv(self.silver_path["QBO"]["Raw"]/"Invoice.csv")
        invoice = invoice[invoice["Corp"].isin(corps)]
        invoice["TransactionDate"] = pd.to_datetime(invoice["TransactionDate"])
        invoice = invoice[invoice["TransactionDate"]>=first_date]
        invoice = invoice[invoice["AccID"].isin(account.AccID.unique())]
        sales = pd.read_csv(self.silver_path["QBO"]["Raw"]/"SalesReceipt.csv")
        sales = sales[sales["Corp"].isin(corps)]
        sales["TransactionDate"] = pd.to_datetime(sales["TransactionDate"])
        sales = sales[sales["TransactionDate"]>=first_date]
        sales = sales[sales["AccID"].isin(account.AccID.unique())]
        journal = pd.read_csv(self.silver_path["QBO"]["Raw"]/"JournalEntry.csv",usecols=journal_cols)
        journal = journal[journal["AccID"].isin(account.AccID.unique())]
        journal["TransactionDate"] = pd.to_datetime(journal["TransactionDate"])
        journal = journal[journal["TransactionDate"]>=first_date]
        journal = journal[~journal["TransactionEntered"].str.contains("Delivered and not settled", na=False)]
        journal = journal[~journal["TransactionEntered"].str.contains("Grain Inventory Receivable Adjustment", na=False)]
        # combining tables
        print("Combining Fact Tables ...")
        invoice = invoice.loc[:,[col for col in cols if col in invoice.columns]]
        sales = sales.loc[:,[col for col in cols if col in sales.columns]]
        journal = journal.loc[:,[col for col in cols if col in journal.columns]]
        facts = pd.concat([invoice, sales, journal], ignore_index=True)
        del invoice, sales, journal
        # join facts with dimension tables
        facts = pd.merge(facts, account.loc[:,["AccID","AccNum","AccName","Category","Subcategory"]], on=["AccID"], how="left")
        facts = pd.merge(facts, farm.loc[:,["FarmID","FarmName"]], on=["FarmID"], how="left")
        facts = pd.merge(facts, customer.loc[:,["CustomerID","CustomerName"]], on=["CustomerID"], how="left")
        facts = facts[facts["Subcategory"]=="Grain - cash settlements"]
        print(f"Total Fact Entries - {len(facts)}")
        # product column
        facts["Product"] = facts["AccName"].apply(lambda x: self._temp_get_product(x))
        # saving file
        print("Saving Files ...")
        self.check_file(self.gold_path["inventory"])
        facts.to_excel(self.gold_path["inventory"]/"Excel"/"QBO_Grain_Settlements.xlsx", sheet_name="settlement", index=False)
        print("Finished\n")

    def _buget_process_input(self, inputdata_path:Path, processed_path:Path) -> None:
        """ 
            this function processes and saves budget totals for production, input (chem/fert/seed), produce budgets, and JD Lease
        """
        ## commodity prices - everything is CAD except Winter Wheat is converted to USD
        pricing = pd.read_csv(inputdata_path/"25-Grain-Pricing.csv")
        pricing.loc[pricing["Commodity"]=="WW", "ForecastPrice"] *= self.fx
        ## production budget
        budget_production = pd.read_csv(inputdata_path/"25-Grain-Revenue.csv")
        budget_production = budget_production.melt(
            id_vars=["Location", "Currency", "Type"],
            var_name="Commodity",
            value_name = "Amount"
        )
        budget_production = budget_production.fillna(value = {"Amount": 0})
        budget_production["Commodity"] = budget_production["Commodity"].replace({"Hay/Silage":"Hay"})
        budget_production.loc[((budget_production["Location"]=="Airdrie")&(budget_production["Commodity"]=="Hay")), "Commodity"] = "Silage" # only Airdrie has silage, others have hay
        budget_production_summary = pd.DataFrame(budget_production.groupby(["Location","Currency","Commodity"]).agg({"Amount": "prod"})).reset_index(drop=False)
        budget_production_summary = budget_production_summary.rename(columns={"Amount":"TotalYield"})
        ### merge yield with commodity price to calculate forecast production value of commodities
        budget_production_summary = pd.merge(budget_production_summary,pricing,on=["Commodity"], how="left")
        ### manual adjustments to prices
        budget_production_summary.loc[((budget_production_summary["Location"] == "Airdrie") & (budget_production_summary["Commodity"] == "Hay")), "ForecastPrice"] = 85
        budget_production_summary.loc[((budget_production_summary["Location"] == "Colorado (Genoa)") & (budget_production_summary["Commodity"] == "WW")), "ForecastPrice"] = 13.75
        budget_production_summary.loc[budget_production_summary["Location"] == "Yorkton", "ForecastPrice"] *= 2/3
        budget_production_summary["ForecastProductionCAD"] = budget_production_summary["TotalYield"] * budget_production_summary["ForecastPrice"]
        budget_production_summary = budget_production_summary[budget_production_summary["ForecastProductionCAD"].notna()]
        budget_production_summary = budget_production_summary[budget_production_summary["ForecastProductionCAD"]!=0]
        ### convert prices back to USD for a adjusted column
        budget_production_summary["ForecastProductionAdj"] = budget_production_summary.apply(lambda x: x["ForecastProductionCAD"] / self.fx if x["Currency"] == "USD" else x["ForecastProductionCAD"],axis=1)
        ### save production budget
        budget_production_summary.to_csv(processed_path/"budget_production.csv",index=False)
        ## input budget
        input_budget = pd.read_csv(inputdata_path/"25-Input-Budget.csv")
        input_budget = input_budget.drop(columns=["Total acres"])
        input_budget = input_budget.melt(
            id_vars = ["Location", "Type"],
            var_name = "Commodity",
            value_name = "Amount"
        )
        input_budget = input_budget.fillna(value = {"Amount": 0})
        input_budget.loc[((input_budget["Location"]=="Yorkton")&(input_budget["Type"].isin(["Fertilizer","Chemical","Seed"]))), "Amount"] *= 2/3
        input_budget.to_csv(processed_path/"input_budget.csv",index=False)
        ## labour budget
        labour_budget = pd.read_csv(inputdata_path/"25-Labour-Budget.csv")
        labour_budget = labour_budget.melt(
            id_vars = ["Location","Currency"],
            var_name = "Month",
            value_name = "LabourBudgetCAD"
        )
        labour_budget["LabourBudgetAdj"] = labour_budget.apply(lambda x: x["LabourBudgetCAD"]/self.fx if x["Currency"]=="USD" else x["LabourBudgetCAD"], axis=1)
        labour_budget.to_csv(processed_path/"labour_budget.csv",index=False)
        ## outlook budget
        outlook = pd.read_csv(inputdata_path/"25-Outlook-Detail.csv")
        outlook = outlook.melt(
            id_vars=["Type", "ProfitType"],
            var_name="Commodity",
            value_name="Amount"
        )
        outlook = outlook.fillna(value={"Amount": 0})
        outlook.to_csv(processed_path/"outlook_budget.csv", index=False)
        ## AZ budget
        az = pd.read_csv(inputdata_path / "25-AZ-Detail.csv")
        az = az.melt(
            id_vars=["Type", "ProfitType"],
            var_name="CommodityRaw",
            value_name="AmountCAD"
        )
        az = az.fillna(value={"AmountCAD": 0})
        az.to_csv(processed_path/"az_budget.csv", index=False)
        ## BC produce details
        bc = pd.read_csv(inputdata_path / "25-BC-Detail.csv")
        bc = bc.melt(
            id_vars=["Type", "ProfitType"],
            var_name="CommodityRaw",
            value_name="AmountCAD"
        )
        bc = bc.fillna(value={"AmountCAD": 0})
        bc.to_csv(processed_path/"bc_budget.csv", index=False)
        ## JD lease
        jdlease = pd.read_csv(inputdata_path/"25-JD-Lease-Summary.csv")
        jdlease = jdlease[jdlease["AllocatedCost25"] != 0]
        jdlease.to_csv(processed_path/"JD_lease.csv", index=False)

    def _budget_read_outsidedata(self,processed_path:Path) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
        """ 
            this function reads all the processed outside data and standardize the commodity and location naming
        """
        production_budget = pd.read_csv(processed_path/"budget_production.csv")
        input_budget = pd.read_csv(processed_path/"input_budget.csv")
        labour_budget = pd.read_csv(processed_path/"labour_budget.csv")
        outlook_budget = pd.read_csv(processed_path/"outlook_budget.csv")
        jdlease = pd.read_csv(processed_path/"JD_lease.csv")
        az_budget = pd.read_csv(processed_path/"az_budget.csv")
        bc_budget = pd.read_csv(processed_path/"bc_budget.csv")
        ## standardizing commodity naming
        production_rename_commodity = {"R Lentils":"Red Lentil", "G Lentils":"Green Lentil","Chickpeas":"Chickpea","Peas":"Field Pea", "WW": "Winter Wheat"}
        input_rename_commodity = {"R Lentils":"Red Lentil", "G Lentils":"Green Lentil","Chickpeas":"Chickpea", "WW": "Winter Wheat"}
        outlook_rename_commodity = {"Broccoli-cases/ac":"Broccoli", "Cabbage-lbs/ac":"Cabbage", "Carrots-lbs":"Carrot", "Cauliflower-cases/ac":"Cauliflower",
                                    "Table Potato-lbs":"Potato", "Seed Potato-lbs":"Potato", "Commercial Pumpkins-Bins/ac":"Pumpkin", "Strawberry Upick-lbs":"Strawberry",
                                    "Pumpkin Upick-pieces/ac":"Pumpkin", "Corn Maze-lbs":"Prairie Pathways", "WW": "Winter Wheat", "Corn (Sweet) Cobs":"Sweet Corn"}
        az_rename_commodity = {"Broccoli-cases/ac":"Broccoli", "Cabbage-lbs/ac":"Cabbage", "Pumpkins-Bins/ac":"Pumpkin", "WatermelonLG-bins/ac": "Watermelon",
                            "WatermelonMini-cases/ac": "Watermelon"}
        bc_rename_commodity = {"Broccoli-cases/ac":"Broccoli", "WatermelonLG-bins/ac": "Watermelon", "WatermelonMini-cases/ac": "Watermelon", "Pumpkins-Bins/ac":"Pumpkin",
                            "Squash-lbs": "Squash"}
        outlook_budget["CommodityRaw"] = outlook_budget["Commodity"]
        production_budget["Commodity"] = production_budget["Commodity"].replace(production_rename_commodity)
        input_budget["Commodity"] = input_budget["Commodity"].replace(input_rename_commodity)
        outlook_budget["Commodity"] = outlook_budget["Commodity"].replace(outlook_rename_commodity)
        az_budget["Commodity"] = az_budget["CommodityRaw"].replace(az_rename_commodity)
        bc_budget["Commodity"] = bc_budget["CommodityRaw"].replace(bc_rename_commodity)
        ## standardizing location naming - merge calderbank grain with Swift Current
        jdlease_rename_location = {"Swift Current Total":"Swift Current", "Regina Farm":"Regina", "Calderbank":"Swift Current",
                                "Airdrie":"Airdrie (grain)", "Eddystone":"Eddystone (grain)"}
        labour_rename_location = {"NexGen (avg met. ton)":"NexGen", "Cache/Fisher/Look":"Aschroft", "MF AZ":"Arizona (produce)", "Box Elder":"Havre", 
                                "BC Veg":"BritishColumbia (produce)","Monette Seeds CDN (avg met. ton)":"Monette Seeds", 
                                "BC Cattle (avg head)":"BritishColumbia (cattle)", "Eddystone Cattle (avg head)":"Eddystone (cattle)",
                                "Swift Current Cattle (avg head)":"Waldeck", "Aridrie Cattle (avg head)":"Airdrie (cattle)",
                                "Airdrie Farm":"Airdrie (grain)", "Eddystone Farm":"Eddystone (grain)","Calderbank":"Calderbank (cattle)"}
        input_rename_location =  {"Fly Creek/Camp 1":"Fly Creek", "Regina Farm":"Regina","Swift Current Total":"Swift Current", "Box Elder":"Havre", "Regina Farm":"Regina",
                                "Calderbank":"Calderbank (grain)","Airdrie":"Airdrie (grain)", "Eddystone":"Eddystone (grain)"}
        production_rename_location = {"Fly Creek/Camp 1":"Fly Creek", "Regina Farm":"Regina","Swift Current Total":"Swift Current", "Box Elder":"Havre", "Regina Farm":"Regina",
                                    "Colorado (Genoa)":"Colorado", "Calderbank":"Swift Current","Airdrie":"Airdrie (grain)", "Eddystone":"Eddystone (grain)"}
        input_budget["Location"] = input_budget["Location"].replace(input_rename_location)
        production_budget["Location"] = production_budget["Location"].replace(production_rename_location)
        labour_budget["Location"] = labour_budget["Location"].replace(labour_rename_location)
        jdlease["Location"] = jdlease["Location"].replace(jdlease_rename_location)
        ## put input budget (chem/fert/seed) into aggregated totals
        input_budget2 = input_budget.groupby(["Location","Type"]).agg({"Amount":"sum"}).reset_index(drop=False)
        ## aggregated totals for production budget and JD Lease
        production_budget = pd.DataFrame(production_budget.groupby(["Location","Currency","Commodity","ForecastPrice"]).agg({"TotalYield":"sum", "ForecastProductionCAD":"sum", "ForecastProductionAdj":"sum"}).reset_index(drop=False))
        jdlease = pd.DataFrame(jdlease.groupby(["Location","Country","Currency","TotalCost25"]).agg({"Acres25":"sum","AllocatedCost25":"sum"}).reset_index(drop=False))
        return input_budget2, production_budget, labour_budget, jdlease, az_budget, bc_budget, outlook_budget

    def _budget_process_produce(self, budget_rules:pd.DataFrame,budget:pd.DataFrame,sheetname:str) -> pd.DataFrame:
        """ 
            this function provides a standardized way to process produce budgets
        """
        budget_rules = budget_rules[budget_rules["SheetRef"] == sheetname].copy(deep=True)
        budget_rules["Commodity"] = budget_rules.apply(lambda x: self._identify_product(x,for_budget=True), axis=1)
        budget["Type"] = budget["Type"].str.strip()
        # gross income - by commodity
        reference = budget[budget["Type"].isin(["Acres","Unit Price","YieldPerAc"])]
        reference = reference.groupby(["Commodity","ProfitType","CommodityRaw"]).agg({"AmountCAD":"prod"}).reset_index(drop=False)
        reference = reference.groupby(["Commodity"]).agg({"AmountCAD":"sum"}).reset_index(drop=False)
        reference = reference.rename(columns={"AmountCAD":"TotalAmountCAD"})
        reference["Category"] = "Produce - production"
        if "outlook" in sheetname.lower():
            for item in ["Prairie Pathways", "Market Garden / CSA"]:
                reference.loc[reference["Commodity"] == item, "Category"] = "Produce - cash settlements"
        # seed expense - by commodity
        expense = budget[budget["Type"] == "Seed"].copy(deep=True)
        expense = expense.drop(columns="CommodityRaw")
        expense = expense.groupby(["Commodity"]).agg({"AmountCAD":"sum"}).reset_index(drop=False).rename(columns={"AmountCAD":"TotalAmountCAD"})
        expense["Category"] = "Seed"
        # other expense - Fertilizer/Chemical - not by commodity
        expense2 = budget[budget["Type"].isin(["Fertilizer","Chemical"])]
        expense2 = expense2.groupby(["Type"]).agg({"AmountCAD":"sum"}).reset_index(drop=False).rename(columns={"AmountCAD":"TotalAmountCAD"})
        expense2["Commodity"] = "Others"
        expense2 = expense2.rename(columns={"Type":"Category"})
        # combine
        budget_produce = pd.merge(budget_rules, pd.concat([reference,expense, expense2]), on=["Commodity","Category"], how="left")
        budget_produce = budget_produce.fillna(value={"TotalAmountCAD":0})
        budget_produce["AmountCAD"] = budget_produce.apply(lambda x: eval(f"{x["TotalAmountCAD"]}{x["Formula"]}"),axis=1)
        return budget_produce

    def _budget_get_transactions(self) -> pd.DataFrame:
        """ 
            get actuals
        """
        transactions = self.gold_pl.copy(deep=True)
        transactions = transactions[transactions["FiscalYear"] >= 2024]
        transactions["AccName"] = transactions["AccName"].str.strip()
        # transactions_location_rename = {"Calderbank":"Calderbank (cattle)"}
        # transactions["Location"] = transactions["Location"].replace(transactions_location_rename)
        return transactions

    def _create_budget(self, process_input:bool = False) -> None:
        """ 
            In Progress: this function generates budgets
        """
        print("\nCreating Budget\n")
        if not self.pl_exist:
            self._finance_operational()
        # self.gold_pl = pd.read_csv(self.gold_path["finance_operational"]/"PL.csv")
        # self.fx = 1.3807
        inputdata_path = self.gold_path["budget"] / "Outside Data"
        processed_path = self.gold_path["budget"] / "Processed Data"
        rule_path = self.gold_path["budget"] / "Budget Rules"
        copied_path = self.gold_path["budget"]/"Copied Data"

        # load actuals
        transactions = self._budget_get_transactions()
        
        # process outside data
        if process_input:
            self._buget_process_input(inputdata_path=inputdata_path, processed_path=processed_path)
        
        # read outside data
        input_budget2, production_budget, labour_budget, jdlease, az_budget, bc_budget, outlook_budget = self._budget_read_outsidedata(processed_path=processed_path)

        # calculate Budgets
        ## outside data
        ### read rules
        budget_rules = pd.read_csv(rule_path/"OutsideData.csv")
        budget_rename_category = {"Seed - farm":"Seed"}
        budget_rules["Category"] = budget_rules["Category"].replace(budget_rename_category)
        ### separate locations into individual rows when they are separated with + in the rules df
        budget_rules["Location"] = budget_rules["Location"].str.split("+")
        budget_rules = budget_rules.explode("Location").reset_index(drop=True)
        ### extract formula
        budget_rules = budget_rules.melt(
            id_vars=["Location","Category","AccFull","SheetRef"],
            var_name="Month",
            value_name="Formula"
        )
        budget_rules = budget_rules.fillna(value={"Formula":"0"})
        budget_rules["Formula"] = budget_rules["Formula"].astype(str)
        budget_rules["Formula"] = budget_rules["Formula"].replace({"0": "*0"})
        ### calculating input budget for accounts per location
        budget_rules_input = budget_rules[budget_rules["SheetRef"] == "Input Budget"].copy(deep=True)
        #### workaround input budget for Airdrie grain 
        input_budget2.loc[((input_budget2["Location"]=="Airdrie (grain)")&(input_budget2["Type"]=="Acres")),"Type"] = "Custom work"
        ### merge budget rules with budget total per location
        budget_input = pd.merge(budget_rules_input,input_budget2.rename(columns={"Type":"Category","Amount":"TotalAmountCAD"}),on=["Location","Category"],how="left")
        #### revert back from workaround
        input_budget2.loc[((input_budget2["Location"]=="Airdrie (grain)")&(input_budget2["Type"]=="Custom work")),"Type"] = "Acres"
        ### apply the formula to compute per month
        budget_input["AmountCAD"] = budget_input.apply(lambda x: eval(f"{x["TotalAmountCAD"]}{x["Formula"]}"), axis=1)

        ## production budget
        ### combine Hay and Silage
        production_budget["Commodity"] = production_budget["Commodity"].replace({"Hay":"Hay/Silage", "Silage":"Hay/Silage"})
        ### add commodity column to budget rules
        budget_rules_production = budget_rules[budget_rules["SheetRef"] == "Production Budget"].copy(deep=True)
        budget_rules_production["Commodity"] = budget_rules_production.apply(lambda x: self._identify_product(x, for_budget=True), axis=1)
        ### merge budget rules with budget totals
        budget_production = pd.merge(budget_rules_production,production_budget.loc[:,["Location","Commodity","ForecastProductionCAD"]].rename(columns={"ForecastProductionCAD":"TotalAmountCAD"}),
                                         on = ["Location", "Commodity"], how="left")
        budget_production = budget_production.fillna(value={"TotalAmountCAD":0})
        ### compute budget
        budget_production["AmountCAD"] = budget_production.apply(lambda x: eval(f"{x["TotalAmountCAD"]}{x["Formula"]}"),axis=1)

        ## labour budget
        budget_rules_labour = budget_rules[budget_rules["SheetRef"] == "Labour Budget"].copy(deep=True)
        budget_labour = pd.merge(budget_rules_labour, labour_budget.loc[:,["Location","Month","LabourBudgetCAD"]].rename(columns={"LabourBudgetCAD":"AmountCAD"}),
                                    on=["Location","Month"],how="left")
        
        ## produce budgets
        ### BC
        budget_bc_produce = self._budget_process_produce(budget_rules=budget_rules,budget=bc_budget,sheetname="BC Produce Details")
        ### AZ
        budget_az_produce = self._budget_process_produce(budget_rules=budget_rules,budget=az_budget,sheetname="AZ Details")
        ### outlook
        budget_outlook = self._budget_process_produce(budget_rules=budget_rules,budget=outlook_budget.rename(columns={"Amount":"AmountCAD"}),sheetname="Outlook Details")

        ## JD lease
        budget_rules_jd = budget_rules[budget_rules["SheetRef"]=="JD Lease"].copy(deep=True)
        budget_equipment = pd.merge(budget_rules_jd, jdlease.loc[:,["Location","AllocatedCost25"]].rename(columns={"AllocatedCost25":"TotalAmountCAD"}),
                                        on = "Location", how = "left")
        budget_equipment = budget_equipment.fillna(value={"TotalAmountCAD":0})
        budget_equipment["AmountCAD"] = budget_equipment.apply(lambda x: eval(f"{x["TotalAmountCAD"]}{x["Formula"]}"),axis=1)

        ## adjustment for Swift Current
        months = ["April", "July"]
        for month in months:
            budget_input.loc[((budget_input["Location"]=="Swift Current")&(budget_input["Category"]=="Fertilizer")&(budget_input["Month"]==month)),"TotalAmountCAD"] += \
            budget_input.loc[((budget_input["Location"]=="Calderbank (grain)")&(budget_input["Category"]=="Fertilizer")&(budget_input["Month"]==month)),"TotalAmountCAD"].item()
            budget_input.loc[((budget_input["Location"]=="Swift Current")&(budget_input["Category"]=="Fertilizer")&(budget_input["Month"]==month)),"AmountCAD"] += \
            budget_input.loc[((budget_input["Location"]=="Calderbank (grain)")&(budget_input["Category"]=="Fertilizer")&(budget_input["Month"]==month)),"AmountCAD"].item()
        months = ["June", "September"]
        for month in months:
            budget_input.loc[((budget_input["Location"]=="Swift Current")&(budget_input["Category"]=="Chemical")&(budget_input["Month"]==month)),"TotalAmountCAD"] += \
            budget_input.loc[((budget_input["Location"]=="Calderbank (grain)")&(budget_input["Category"]=="Chemical")&(budget_input["Month"]==month)),"TotalAmountCAD"].item()
            budget_input.loc[((budget_input["Location"]=="Swift Current")&(budget_input["Category"]=="Chemical")&(budget_input["Month"]==month)),"AmountCAD"] += \
            budget_input.loc[((budget_input["Location"]=="Calderbank (grain)")&(budget_input["Category"]=="Chemical")&(budget_input["Month"]==month)),"AmountCAD"].item()
        months = ["May", "June", "September"]
        for month in months:
            budget_input.loc[((budget_input["Location"]=="Swift Current")&(budget_input["Category"]=="Seed")&(budget_input["Month"]==month)),"TotalAmountCAD"] += \
            budget_input.loc[((budget_input["Location"]=="Calderbank (grain)")&(budget_input["Category"]=="Seed")&(budget_input["Month"]==month)),"TotalAmountCAD"].item()
            budget_input.loc[((budget_input["Location"]=="Swift Current")&(budget_input["Category"]=="Seed")&(budget_input["Month"]==month)),"AmountCAD"] += \
            budget_input.loc[((budget_input["Location"]=="Calderbank (grain)")&(budget_input["Category"]=="Seed")&(budget_input["Month"]==month)),"AmountCAD"].item()
        
        # arithmetic rules
        arithmetic = pd.read_csv(rule_path/"Arithmetic.csv")
        ## faltten location
        arithmetic["Location"] = arithmetic["Location"].str.split("+")
        arithmetic = arithmetic.explode("Location").reset_index(drop=True)
        arithmetic_rules = arithmetic.melt(
            id_vars=["Location","Category","AccFull", "AccRef", "FixedRef"],
            var_name="Month",
            value_name="FormulaFull"
        )
        ## housekeeping
        arithmetic_rules = arithmetic_rules.fillna(value={"FormulaFull":"FY-1*0"})
        arithmetic_rules["FormulaFull"] = arithmetic_rules["FormulaFull"].astype(str)
        arithmetic_rules["FormulaFull"] = arithmetic_rules["FormulaFull"].replace({"0":"FY-1*0"})
        arithmetic_rules["ReferenceYear"] = arithmetic_rules["FormulaFull"].str.slice(0,4)
        arithmetic_rules["Formula"] = arithmetic_rules["FormulaFull"].str.slice(4)
        arithmetic_rules = arithmetic_rules.fillna(value={"Formula": "0"})
        arithmetic_rules["Formula"] = arithmetic_rules["Formula"].astype(str)
        arithmetic_rules["Formula"] = arithmetic_rules["Formula"].replace({"0":"*0"})
        ## separating Fixed records
        arithmetic_rules_fixed = arithmetic_rules[arithmetic_rules["AccRef"] == "Fixed"].copy(deep=True)
        arithmetic_rules = arithmetic_rules[arithmetic_rules["AccRef"]!="Fixed"].copy(deep=True)

        ## process fixed records
        arithmetic_rules_fixed = arithmetic_rules_fixed.drop(columns=["FormulaFull","ReferenceYear"]).rename(columns={"FixedRef":"TotalAmountCAD"})
        arithmetic_rules_fixed["AmountCAD"] = arithmetic_rules_fixed.apply(lambda x: eval(f"{x["TotalAmountCAD"]}{x["Formula"]}"),axis=1)

        ## Extract Account Info
        arithmetic_rules["AccNum"] = arithmetic_rules["AccRef"].apply(lambda x: "".join(x.split(" ")[0:2]))
        arithmetic_rules["AccName"] = arithmetic_rules["AccRef"].apply(lambda x: (" ".join(x.split(" ")[2:]).strip()))
        assert "Fixd" not in arithmetic_rules.ReferenceYear.unique(), "Fixd records incorrectly classified"
        ## separate FY-1 & FY+1
        arithmetic_rules_prior = arithmetic_rules[arithmetic_rules["ReferenceYear"] == "FY-1"].copy(deep=True)
        arithmetic_rules = arithmetic_rules[arithmetic_rules["ReferenceYear"] == "FY+1"].copy(deep=True)

        ## process FY-1 with actuals
        actuals = transactions.groupby(["Location", "AccNum", "AccName", "FiscalYear"]).agg({"AmountDisplay":"sum"}).reset_index(drop=False)
        arithmetic_rules_prior["FiscalYear"] = self.currentFY - 1
        assert len(actuals[actuals.duplicated(subset=["AccNum","FiscalYear","Location"],keep=False)]) == 0, "Duplicated AccNum detected for FY-1 Actuals"
        budget_prior = pd.merge(arithmetic_rules_prior,actuals.rename(columns={"AmountDisplay":"TotalAmountCAD"}),
                                on = ["Location","AccNum","FiscalYear"], how="left")
        budget_prior = budget_prior.fillna(value={"TotalAmountCAD": 0})
        budget_prior["AmountCAD"] = budget_prior.apply(lambda x: eval(f"{x["TotalAmountCAD"]}{x["Formula"]}"), axis=1)

        ## processing FY+1 with current budget
        ### budget sales that is based on production budget input sheet
        arithmetic_rules_sales = arithmetic_rules[arithmetic_rules["Category"].str.contains("cash settlements")].copy(deep=True)
        production_reference = pd.concat([budget_production.copy(deep=True), budget_outlook.copy(deep=True), budget_az_produce.copy(deep=True),budget_bc_produce.copy(deep=True)])
        production_reference = production_reference.groupby(["Location","AccFull"]).agg({"AmountCAD":"sum"}).reset_index(drop=False)
        budget_sales = pd.merge(arithmetic_rules_sales,production_reference.rename(columns={"AccFull":"AccRef"}), on=["Location","AccRef"], how="left")
        budget_sales = budget_sales.rename(columns={"AmountCAD":"TotalAmountCAD"})
        budget_sales["AmountCAD"] = budget_sales.apply(lambda x: eval(f"{x["TotalAmountCAD"]}{x["Formula"]}"),axis=1)
        budget_prior = pd.concat([budget_prior, budget_sales],ignore_index=True)

        ### budget inventory adjustment 
        arithmetic_rules_inventory = arithmetic_rules[arithmetic_rules["Category"].str.contains("inventory adjustment",case=False)].copy(deep=True)
        budget_inventory = pd.merge(arithmetic_rules_inventory, budget_prior.loc[:,["Location","AccFull","Month","AmountCAD"]].rename(columns={"AccFull":"AccRef"}),
                                on = ["Location","AccRef", "Month"], how = "left")
        budget_inventory["AmountCAD"] = -budget_inventory["AmountCAD"]
        budget_prior = pd.concat([budget_prior, budget_inventory], ignore_index=True)

        ## combine with fixed budgets
        budget_prior = pd.concat([budget_prior,arithmetic_rules_fixed],ignore_index=True)

        # copied data
        budget_copy = pd.read_csv(copied_path/"Copied Data.csv")
        budget_copy = budget_copy.melt(
            id_vars=["Location","Category","AccFull"],
            var_name = "Month",
            value_name = "AmountCAD"
        )
        budget_copy = budget_copy.fillna(value={"AmountCAD":0})
        budget_copy["AmountCAD"] = budget_copy["AmountCAD"].astype(float)
        budget_copy["FiscalYear"] = self.currentFY
        budget_copy["AccRef"] = "Copy"
        budget_copy["ReferenceYear"] = "NA"
        budget_copy["Formula"] = "NA"
        budget_copy["TotalAmountCAD"] = budget_copy["AmountCAD"]
        budget_copy.loc[budget_copy["Location"]=="Seeds USA", "AmountCAD"] *= self.fx

        # combining all budgets
        budget_outside = pd.concat([budget_input,budget_production,budget_labour,budget_equipment, budget_outlook, budget_az_produce, budget_bc_produce],ignore_index=True)
        budget_outside = budget_outside.drop(columns=["Commodity"])
        budget_prior = budget_prior.drop(columns=["FormulaFull","AccNum","AccName_x", "AccName_y", "AccName", "FixedRef"])
        budget_all = pd.concat([budget_outside,budget_prior,budget_copy],ignore_index=True)
        budget_all["AccNum"] = budget_all["AccFull"].apply(lambda x: "".join(x.split(" ")[0:2]))
        budget_all["AccName"] = budget_all["AccFull"].apply(lambda x: " ".join(x.split(" ")[2:]))
        budget_all["FiscalYear"] = self.currentFY 
        budget_all["DataType"] = "Budget"
        budget_all.loc[budget_all["Category"].str.contains("inventory adjustment",case=False), "AmountCAD"] *= -1 # turn the sign positive for inventory adjustments

        # save
        self.check_file(self.gold_path["budget"]/"OutputFile")
        budget_all.to_csv(self.gold_path["budget"]/"OutputFile"/"budget_all.csv", index=False)

    def _budget_update(self, force_create:bool=False, force_process_input:bool=False) -> None:
        """ 
            generate/update the actuals from the budget system
        """
        print("\nGenerating/Updating Actuals for budget system\n")
        if not self.pl_exist:
            self._finance_operational()
        # self.gold_pl = pd.read_csv(self.gold_path["finance_operational"]/"PL.csv")
        # self.fx = 1.3807
        budget_path = self.gold_path["budget"]/"OutputFile"/"budget_all.csv"
        if not Path.exists(budget_path) or force_create:
            self._create_budget(process_input=force_process_input)
        budget = pd.read_csv(budget_path)
        budget = budget.loc[:,["Location", "SheetRef", "Month", "Formula", "TotalAmountCAD", "AmountCAD", "AccRef", "ReferenceYear","FiscalYear", "AccNum", "DataType", "Category"]]
        budget_location_rename = {"Airdrie (grain)": "Airdrie", "Airdrie (cattle)": "Airdrie", "Calderbank (cattle)": "Calderbank",
                                  "Airdrie (corporate)": "Airdrie", "Seeds USA":"Arizona (produce)"}
        budget["Location"] = budget["Location"].replace(budget_location_rename)
        category_mapping = budget.loc[:,["AccNum", "Category"]].drop_duplicates()
        # organize Actuals
        transactions = self._budget_get_transactions()
        actuals_all = transactions.groupby(["Location","AccNum", "FiscalYear", "Month"]).agg({"AmountDisplay":"sum"}).reset_index(drop=False)
        actuals_all = actuals_all[actuals_all["FiscalYear"] == self.currentFY]
        actuals_all["DataType"] = "Actual"
        actuals_all = actuals_all.rename(columns={"AmountDisplay": "AmountCAD"})
        actuals_all = pd.merge(actuals_all,category_mapping,on="AccNum",how="left")
        actuals_all.to_csv(self.gold_path["budget"]/"OutputFile"/"actuals_all.csv", index=False)
        print(f"Location Unaccounted for in budget: {(set(budget.Location.unique()) - set(actuals_all.Location.unique()))}")
        # combine everything
        all_all = pd.concat([budget,actuals_all],ignore_index=True)
        all_all["FXRate"] = self.fx
        # operational classification
        assert len(self.operation_acc[self.operation_acc.duplicated(subset=["AccNum"],keep=False)]) == 0, "Duplicated AccNum Detected - Operational Accounts Classification"
        all_all = pd.merge(all_all, self.operation_acc, on="AccNum", how="left")
        # classify pillars
        all_all["Pillar"] = all_all.apply(lambda x: self._pillar_classification(x), axis=1)
        # save
        self.check_file(self.gold_path["budget"]/"OutputPowerBI")
        all_all.to_excel(self.gold_path["budget"]/"OutputPowerBI"/"BudgetActual.xlsx", sheet_name="Budget", index=False)

    def run(self, force_run_time:bool=False, force_create_budget:bool=False, force_process_budget_input:bool=False) -> None:
        start = perf_counter()

        self._weekly_banking()
        self._finance_operational()
        self._payroll_project()
        self._budget_update(force_create=force_create_budget, force_process_input=force_process_budget_input)
        if force_run_time or (self.today.weekday()==0 or self.today.weekday() == 2): self._QBOTime_project()
        self._hr_summary()
        self._raw_inventory()

        end = perf_counter()
        print(f"\nProjects Transformation Finished with {(end-start)/60:.3f} minutes\n")



# end of file
        
        
