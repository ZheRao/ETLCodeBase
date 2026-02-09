"""
Docstring for ETLCodeBase.gold.finance_summary

Purpose:
    Produce summary tables that contains all financial lines, including entries like Gross Margin, EBITDA, ...

Exposed API:
    - `create_financial_summary()`
"""

import pandas as pd
from pathlib import Path

from ETLCodeBase.utils.filesystem import read_configs



def _create_accid_prof_mapping(acc_root:Path) -> dict:
    """
    Purpose:
        - based on account table, map `ProfitType` to `AccID` and return the mapping dictionary
    """
    acc = pd.read_csv(acc_root/"Account_table.csv")
    acc = acc[acc["AccountingType"] == "Income Statement"]
    id_prof_map = acc.set_index("AccID")["ProfitType"]
    return id_prof_map


def _create_additional_financial_lines(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
        - create `Gross Margin`, `Contribution Margin`, `EBITDA`, `Net Income` that don't exist naturally from QBO data
    """
    complement_data = summary.head(0).copy(deep=True)
    for y in summary.FiscalYear.unique():
        subset_year = summary[summary["FiscalYear"] == y]
        for m in subset_year.Month.unique():
            subset_month = subset_year[subset_year["Month"] == m]
            for l in subset_month.Location.unique():
                subset_location = subset_month[subset_month["Location"] == l]
                if len(subset_location) == 0:
                    continue
                for data_type in subset_location.DataType.unique():
                    subset_datatype = subset_location[subset_location["DataType"]==data_type]
                    items = ["Sales Revenue", "Cost of Goods Sold", "Direct Operating Expenses", "Other Operating Revenue", "Operating Overheads", "Other Income", "Other Expense"]
                    values = dict.fromkeys(items, 0)
                    for i in items:
                        if i in subset_datatype.ProfitType.unique():
                            values[i] = subset_datatype.loc[subset_datatype["ProfitType"]==i, "AmountCAD"].item()
                    gross_margin = values["Sales Revenue"] - values["Cost of Goods Sold"]
                    contribution_margin = gross_margin - values["Direct Operating Expenses"] + values["Other Operating Revenue"]
                    ebitda = contribution_margin - values["Operating Overheads"]
                    net_income = ebitda + values["Other Income"] - values["Other Expense"]
                    pillar = subset_datatype.Pillar.unique().item()
                    row = {"FiscalYear": y, "Month": m, "Location": l, "Pillar":pillar, "DataType": data_type}
                    row_GM = row | {"ProfitType": "Gross Margin", "AmountCAD": gross_margin}
                    row_CM = row | {"ProfitType": "Contribution Margin", "AmountCAD": contribution_margin}
                    row_ebitda = row | {"ProfitType": "EBITDA", "AmountCAD": ebitda}
                    row_NI = row | {"ProfitType": "Net Income", "AmountCAD": net_income}
                    complement_data.loc[len(complement_data)] = row_GM
                    complement_data.loc[len(complement_data)] = row_CM 
                    complement_data.loc[len(complement_data)] = row_ebitda 
                    complement_data.loc[len(complement_data)] = row_NI 
    return pd.concat([complement_data, summary], ignore_index=True)


def create_financial_summary(write_out:bool=True) -> pd.DataFrame:
    """
    Input:
        - write_out: write to disk?
    Output:
        - df: financial summary df
    Purpose:
        - assemble summary tables to assemble financial income statement style, including Gross Margin, EBITDA, Net Income
    """
    path_config = read_configs(config_type="io", name="path.json")
    budget_root = Path(path_config["root"]) / Path(path_config["gold"]["budget"])

    finance_root = Path(path_config["root"]) / Path(path_config["gold"]["finance_operational"])
    prof_mapping = _create_accid_prof_mapping(finance_root)

    df = pd.read_csv(budget_root / "PowerBI" / "BudgetActual.csv")
    df["ProfitType"] = df["AccID"].map(prof_mapping)
    summary = df.groupby(["FiscalYear", "Month", "Location", "Pillar", "ProfitType", "DataType"]).agg({"AmountCAD":"sum"}).reset_index(drop=False)
    summary_no_pillar = df.groupby(["FiscalYear", "Month", "Location", "ProfitType", "DataType"]).agg({"AmountCAD":"sum"})
    if len(summary) != len(summary_no_pillar):
        raise ValueError(f"Duplicated location-pillar summarization detected, summary with Pillar had {len(summary)} rows and without Pillar had {len(summary_no_pillar)} rows")
    del summary_no_pillar

    summary = _create_additional_financial_lines(summary=summary)

    summary = summary.rename(columns={"AmountCAD": "AmountDisplay"})

    if write_out:
        summary.to_csv(finance_root/"ProfitTypeSummary.csv", index=False)
        summary.to_excel(finance_root/"ProfitTypeSummary.xlsx", sheet_name="ProfitTypeSummary", index=False)
        pillar_root = Path(path_config["root"]) / Path(path_config["gold"]["pillar_dashboard"])
        for pillar in ["Grain", "Cattle", "Seed", "Produce"]:
            summary[summary["Pillar"]==pillar].to_excel(pillar_root/pillar/"ProfitTypeSummary.xlsx", sheet_name="ProfitTypeSummary", index=False)
    
    return summary



    
    
