"""
Docstring for ETLCodeBase.gold.finance_budget

Purpose:
    Read formatted budget input csv, consolidate with QBO PL actuals, output Power BI ready Excel file

Exposed API:
    - ``
"""

import pandas as pd
from pathlib import Path 

def _extract_actuals(root:Path) -> pd.DataFrame:
    """
    Purpose:
        read and return the df for actuals with `Location`, `AccNum`, `FiscalYear`, `Month`, `AmountCAD` (not to confused with `AmountCAD` from QBO PL, this corresponds to `AmountDisplay`)
    """
    path = root / "Actuals" / "actuals.csv"
    return pd.read_csv(path)

def _extract_budget_25(root:Path) -> pd.DataFrame:
    """
    Purpose:
        read computed 2025 budget file without reprocessing everything from raw Excel file, budget generation engine is in a different script
    """
    path = root / "Budgets" / "2025" / "budget.csv"


def _extract_budget(root:Path, year:list[int]=[2026]) -> pd.DataFrame:
    """
    Purpose:
        read all targeted budget file, standardize column and add `FiscalYear` column if not available

    Note:
        Place holder for actual 2026 budget code
    """
    return 1

