"""
Docstring for ETLCodeBase.gold.weekly_banking

Purpose:
    - prepare weekly banking report for finance
    - link activity accounts (e.g., Durum Sales Account) to GL bank accounts, for explaining money movements

Exposed API:
    - `create_weekly_bank()`
"""

import pandas as pd 
from pathlib import Path 
import datetime as dt

from ETLCodeBase.utils.filesystem import read_configs


def _determine_min_date(today: dt.date|None = None) -> tuple[int, int]:
    """
    Purpose:
        - return the year, month for 6-month ago
    """
    if not today: today = dt.date.today()
    criteria = today.month < 6
    return today.year - criteria, today.month - 6 + 12 * criteria 


def _change_acc_category_transfer(acc: pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
        - change accounts that finance team specified into Asset Transfer category
    """
    acc_list = ["MFL264", "MSL250"]
    acc.loc[acc["AccID"].isin(acc_list), "ProfitType"] = "Asset"
    acc.loc[acc["AccID"].isin(acc_list), "Category"] = "Transfer"
    return acc

def _read_linked_tables(path_config: dict) -> pd.DataFrame:
    """
    Purpose:
        - read invoice and bill linked mapping table: map `AccID` to `TransactionID_partial`
    """
    linked_path = Path(path_config["root"]) / Path(path_config["silver"]["Raw"])  / "LinkedTxn"
    invoice_linked = pd.read_csv(linked_path / "LinkedTxn_Mapping_Invoice.csv")
    bill_linked = pd.read_csv(linked_path / "LinkedTxn_Mapping_Bill.csv")
    mapping = pd.concat([invoice_linked, bill_linked], ignore_index=True)
    mapping = mapping.drop(columns=["Corp"])
    return mapping

def _process_facts(fact_type:str, path_config:dict) -> pd.DataFrame:
    """
    Purpose:
        - create mapping table for fact tables other than invoice or bill, map `AccID` to `TransactionID_partial`
    """
    path = Path(path_config["root"]) / Path(path_config["silver"]["Raw"]) / (fact_type + ".csv")
    if not path.exists(): raise FileNotFoundError(f"file location <{path}> doesn't exist for creating {fact_type} mapping table for weekly banking transformation")
    df = pd.read_csv(path, usecols = ["TransactionID", "AccID"])
    df["TransactionID"] = df["TransactionID"].str.split("-").str[1]
    df = df.drop_duplicates()
    df = df.rename(columns={"TransactionID":"TxnId"})
    return df

def _consolidate_mapping(fact_type:str, txn_type:str, old_map:pd.DataFrame, path_config:dict, direct_merge:bool=True) -> pd.DataFrame:
    """
    Purpose:
        - read raw table for the `fact_type`, extract mapping `TransactionID_partial` -> `AccID`, then stack the new mapping into old mapping
    """
    df = _process_facts(fact_type=fact_type, path_config=path_config)
    df["TxnType"] = txn_type
    if direct_merge:
        return pd.concat([old_map, df], ignore_index=True)
    else:
        return df
    
def _create_mapping(path_config: dict, exclude_list:list[str]) -> pd.DataFrame:
    """
    Purpose:
        - create the `TxnId` & `TxnType` -> 'AccID' mapping table
        - 'AccID' = `ActivityID`
        - handles journal entry accounts exclusion
    """
    mapping = _read_linked_tables(path_config=path_config)

    # add purchase, deposit, salesreceipt into mapping
    mapping = _consolidate_mapping(fact_type="Purchase", txn_type="Expense", old_map = mapping, path_config=path_config)
    mapping = _consolidate_mapping(fact_type="Deposit", txn_type="Deposit", old_map = mapping, path_config=path_config)
    mapping = _consolidate_mapping(fact_type="SalesReceipt", txn_type="Sales Receipt", old_map = mapping, path_config=path_config)

    # special cases for journal entry
    jr_mapping = _consolidate_mapping(fact_type="JournalEntry", txn_type="Journal Entry", old_map = mapping, path_config=path_config, direct_merge=False)
    should_include = ["MFBC470", "MFBC471"]
    for acc in should_include:
        exclude_list.remove(acc)
    jr_mapping = jr_mapping[~jr_mapping["AccID"].isin(exclude_list)]
    mapping = pd.concat([mapping, jr_mapping], ignore_index=True)

    # drop duplicates
    mapping = mapping.drop_duplicates(subset=["TxnId"],keep="first")
    return mapping

def _process_gl(path_config: dict, bank_acc:pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
        - load and process GL transactions (focus only bank accounts' activities)
    """
    cols = ["TransactionType","TransactionID_partial","AccID","AccNum","AccName", "TransactionDate", "Amount", "SplitAcc", "SplitAccID", "Memo", "Corp", "Balance"]
    path = Path(path_config["root"]) / Path(path_config["silver"]["GL"])
    df = pd.read_csv(path / "GeneralLedger.csv", dtype={"TransactionID_partial":str}, usecols=cols)
    year, month = _determine_min_date()
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])
    df = df[df["TransactionDate"]>=dt.datetime(year, month, 1)]
    df = df[df["AccID"].isin(bank_acc.AccID.unique())]
    df = df.rename(columns={"TransactionType":"TxnType",
                            "TransactionID_partial":"TxnId",
                            "AccID":"BankAccID",
                            "AccNum":"BankAccNum",
                            "AccName":"BankAccName",
                            "TransactionDate":"BankActivityDate",
                            "Amount":"BankAmount"})
    df["Sign"] = df["BankAmount"].apply(lambda x: "Positive" if x>=0 else "Negative")

    # merge to get CurrencyID for bank_acc
    df = pd.merge(df, bank_acc.loc[:,["AccID","CurrencyID"]], left_on=["BankAccID"], right_on=["AccID"], how="left")
    df = df.drop(columns=["AccID"])
    return df

def _process_transfer(transfers:pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
        - determine transfer type
        - light processing of the df to align with transactions mapped with mapping table
    """
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
    return transfers

def _final_cleanup(transactions_mapped:pd.DataFrame, acc:pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
        - final clean up for Power BI ingestion
    """
    transactions_mapped = transactions_mapped.rename(columns={"CurrencyID":"BankCurrencyID"})
    transactions_mapped = pd.merge(transactions_mapped, acc.loc[:,["AccID","AccName","AccNum","Category","ProfitType","CurrencyID"]], on="AccID", how="left")
    transactions_mapped.loc[transactions_mapped["TransferType"]=="Bank Transfer","Category"] = "Bank Transfer"
    transactions_mapped.loc[((transactions_mapped["BankAccNum"].str.startswith("MSL"))&(transactions_mapped["AccNum"]=="MSL120001")), "Category"] = "Seed Processing Revenue"
    transactions_mapped = transactions_mapped.rename(columns={"AccNum":"ActivityAccNum", "AccName":"ActivityAccName"})
    transactions_mapped.loc[((transactions_mapped["TxnType"]=="Sales Tax Payment")&(transactions_mapped["Sign"]=="Positive")), "ProfitType"] = "Other Operating Revenue"
    transactions_mapped.loc[((transactions_mapped["TxnType"]=="Sales Tax Payment")&(transactions_mapped["Sign"]=="Positive")), "Category"] = "Miscellaneous income"
    transactions_mapped.loc[((transactions_mapped["TxnType"]=="Sales Tax Payment")&(transactions_mapped["Sign"]=="Negative")), "ProfitType"] = "Operating Overheads"
    transactions_mapped.loc[((transactions_mapped["TxnType"]=="Sales Tax Payment")&(transactions_mapped["Sign"]=="Negative")), "Category"] = "Office and miscellaneous"
    transactions_mapped.loc[((transactions_mapped["TxnType"]=="Sales Tax Payment")), "ActivityAccNum"] = "Manual Adjustment"
    transactions_mapped.loc[((transactions_mapped["TxnType"]=="Sales Tax Payment")), "ActivityAccName"] = "Manual Adjustment"
    return transactions_mapped


def create_weekly_bank(write_out:bool=True) -> pd.DataFrame:
    """
    Input:
        - write_out: write to disk
    
    Output:
        - df: bank activity df

    Note:
        - match latest GL bank transactions with raw activities & extract accounts for those activities
        - assumptions: a raw entry (e.g., invoice) can have multiple lines - multiple associated accounts, only considering the first one
    """
    print("\nStarting Weekly Banking Project Transformation\n")

    path_config = read_configs(config_type="io", name="path.json")
    acc_path_silver = Path(path_config["root"]) / Path(path_config["silver"]["Dimension"]) / "Account.csv"
    acc = pd.read_csv(acc_path_silver)
    acc = _change_acc_category_transfer(acc=acc)
    acc_bank = acc[acc["AccountType"] == "Bank"].copy()

    mapping = _create_mapping(path_config=path_config, exclude_list=list(acc_bank.AccID.unique()))

    transactions = _process_gl(path_config=path_config,bank_acc=acc_bank)

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

    # process transfers
    transfers = _process_transfer(transfers=transfers)

    # merge transactions and transfers
    transactions_mapped = pd.concat([transactions_mapped,transfers], ignore_index=True)

    # final clean up
    transactions_mapped = _final_cleanup(transactions_mapped=transactions_mapped, acc=acc)

    if write_out:
        path_out = Path(path_config["root"]) / Path(path_config["gold"]["weekly_banking"])
        transactions_mapped.to_excel(path_out/"BankingActivity.xlsx", sheet_name="transactions", index=False)
    
    return transactions_mapped


