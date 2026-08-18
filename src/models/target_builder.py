import pandas as pd
import numpy as np
from typing import List

def create_forecasting_targets(df: pd.DataFrame,
                               target_col: str = "price_rs_per_qtl",
                               horizons: List[int] = [7, 15, 30],
                               group_cols: List[str] = ["state", "district", "market"]) -> pd.DataFrame:
    """
    Create direct multi-step forecasting targets for 7D, 15D, and 30D horizons.
    Shift is negative (-h) so that date T row contains future price Price(T+h).
    Strict anti-leakage: Date T feature vector X(T) remains invariant to future shifts.
    """
    df = df.sort_values(by=group_cols + ["date"]).reset_index(drop=True)
    
    for h in horizons:
        col_name = f"price_target_{h}d"
        df[col_name] = df.groupby(group_cols)[target_col].shift(-h)
        
    return df
