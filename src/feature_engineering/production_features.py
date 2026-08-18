import pandas as pd
import numpy as np

def add_production_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Year-over-Year (YoY) production, area, and yield percentage changes.
    """
    df = df.sort_values(by=["state", "season", "financial_year"]).reset_index(drop=True)
    
    group_cols = ["state", "season"]
    
    df["production_lag_1yr"] = df.groupby(group_cols)["production_mt"].shift(1)
    df["area_lag_1yr"] = df.groupby(group_cols)["area_ha"].shift(1)
    df["yield_lag_1yr"] = df.groupby(group_cols)["yield_kg_ha"].shift(1)
    
    df["production_yoy_change"] = ((df["production_mt"] - df["production_lag_1yr"]) / df["production_lag_1yr"].replace(0, np.nan)).fillna(0.0)
    df["area_yoy_change"] = ((df["area_ha"] - df["area_lag_1yr"]) / df["area_lag_1yr"].replace(0, np.nan)).fillna(0.0)
    df["yield_yoy_change"] = ((df["yield_kg_ha"] - df["yield_lag_1yr"]) / df["yield_lag_1yr"].replace(0, np.nan)).fillna(0.0)
    
    return df
