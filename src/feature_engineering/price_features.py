import pandas as pd
import numpy as np

def add_price_features(df: pd.DataFrame, price_col: str = "price_rs_per_qtl", group_cols: list = ["state", "district", "market"]) -> pd.DataFrame:
    """
    Compute price lags, rolling means, rolling standard deviations, and coefficient of variation.
    Enforces NO FUTURE LEAKAGE by using shift(1) prior to any rolling window computation.
    """
    df = df.sort_values(by=group_cols + ["date"]).reset_index(drop=True)
    
    # Define group object
    grouped = df.groupby(group_cols)[price_col]
    
    # 1. Price Lags (Strictly past values)
    df["price_lag_1"] = grouped.shift(1)
    df["price_lag_7"] = grouped.shift(7)
    df["price_lag_14"] = grouped.shift(14)
    df["price_lag_30"] = grouped.shift(30)
    
    # 2. Rolling Price Features (Shifted by 1 to prevent leakage of current date T price)
    df["price_rolling_mean_7"] = df.groupby(group_cols)[price_col].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    df["price_rolling_mean_14"] = df.groupby(group_cols)[price_col].transform(lambda s: s.shift(1).rolling(14, min_periods=1).mean())
    df["price_rolling_mean_30"] = df.groupby(group_cols)[price_col].transform(lambda s: s.shift(1).rolling(30, min_periods=1).mean())
    
    df["price_rolling_std_7"] = df.groupby(group_cols)[price_col].transform(lambda s: s.shift(1).rolling(7, min_periods=2).std())
    df["price_rolling_std_30"] = df.groupby(group_cols)[price_col].transform(lambda s: s.shift(1).rolling(30, min_periods=2).std())
    
    # 3. Price Volatility - Coefficient of Variation (Std / Mean)
    df["price_rolling_cv_7"] = (df["price_rolling_std_7"] / df["price_rolling_mean_7"]).fillna(0.0)
    df["price_rolling_cv_30"] = (df["price_rolling_std_30"] / df["price_rolling_mean_30"]).fillna(0.0)
    
    return df
