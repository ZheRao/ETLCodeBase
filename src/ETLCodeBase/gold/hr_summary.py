"""
Docstring for ETLCodeBase.gold.hr_summary

Purpose:
    produce summary tables from QBO and QBO Time for HR overview dashboard

Exposed API
    - `hr_summary`
"""

import pandas as pd
from pathlib import Path

from ETLCodeBase.utils.filesystem import read_configs


def hr_summary() -> None:
    """ 
        This function consolidate payroll and QBO time summaries into one table for consolidated insights
    """
    # construct path
    path_config = read_configs(config_type="io", name="path.json")
    path_hr = Path(path_config["root"]) / Path(path_config["gold"]["hr_combined"])
    final_df = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]
    for i in [1, 2, 3]:
        payroll = pd.read_csv(path_hr / "CSV" / f"payroll_summarized{i}.csv")
        payroll_rename = {"AmountDisplay": "TotalAmount", "CostPerUnit": "AmountPerUnit", "Avg CostPerUnit": "Avg AmountPerUnit"}
        payroll = payroll.rename(columns=payroll_rename)
        payroll["Mode"] = "Payroll"
        time = pd.read_csv(path_hr / "CSV" / f"time_summarized{i}.csv")
        time_rename = {"duration": "TotalAmount", "HoursPerUnit": "AmountPerUnit", "Avg HoursPerUnit": "Avg AmountPerUnit"}
        time = time.rename(columns=time_rename)
        time["Mode"] = "Hours"
        final_df[i-1] = pd.concat([payroll, time], ignore_index=True)
    final_df[0].to_excel(path_hr/"Summarized.xlsx", sheet_name="Summarized", index=False)
    final_df[1].to_excel(path_hr/"Summarized2.xlsx", sheet_name="Summarized2", index=False)
    final_df[2].to_excel(path_hr/"Summarized3.xlsx", sheet_name="Summarized3", index=False)