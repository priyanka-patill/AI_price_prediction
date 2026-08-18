import pandas as pd
import numpy as np

def add_arrival_features(df: pd.DataFrame, arrival_col: str = "arrival_mt", group_cols: list = ["state", "district", "market"]) -> pd.DataFrame:
    """
    Compute market arrival lags, 7-day rolling mean, and percentage change metrics (7D and 30D).
    Enforces NO FUTURE LEAKAGE with shift(1).
    """
    df = df.sort_values(by=group_cols + ["date"]).reset_index(drop=True)
    
    grouped = df.groupby(group_cols)[arrival_col]
    
    # 1. Arrival Lags
    df["arrival_lag_1"] = grouped.shift(1)
    df["arrival_lag_7"] = grouped.shift(7)
    
    # 2. Rolling Arrival Mean
    df["arrival_rolling_mean_7"] = df.groupby(group_cols)[arrival_col].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    
    # 3. Arrival Percentage Changes (7D and 30D)
    # (Arrival_t-1 - Arrival_t-8) / Arrival_t-8
    lag_1 = grouped.shift(1)
    lag_8 = grouped.shift(8)
    lag_31 = grouped.shift(31)
    
    df["arrival_change_7d"] = ((lag_1 - lag_8) / lag_8.replace(0, np.nan)).fillna(0.0)
    df["arrival_change_30d"] = ((lag_1 - lag_31) / lag_31.replace(0, np.nan)).fillna(0.0)
    
    return df
