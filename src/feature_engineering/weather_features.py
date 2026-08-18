import pandas as pd
import numpy as np

def add_weather_features(df: pd.DataFrame, group_cols: list = ["state", "district"]) -> pd.DataFrame:
    """
    Compute cumulative rainfall aggregations (7D, 30D, 90D), temperature rolling means (7D, 30D),
    soil moisture (7D, 30D), and evapotranspiration (7D, 30D).
    Enforces NO FUTURE LEAKAGE using shift(1).
    """
    df = df.sort_values(by=group_cols + ["date"]).reset_index(drop=True)
    
    # 1. Cumulative Rainfall Lags & Rolling Sums
    df["rainfall_7d"] = df.groupby(group_cols)["rainfall_mm"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).sum())
    df["rainfall_30d"] = df.groupby(group_cols)["rainfall_mm"].transform(lambda s: s.shift(1).rolling(30, min_periods=1).sum())
    df["rainfall_90d"] = df.groupby(group_cols)["rainfall_mm"].transform(lambda s: s.shift(1).rolling(90, min_periods=1).sum())
    
    # 2. Temperature Rolling Means (7D, 30D)
    if "temperature_mean_c" in df.columns:
        df["temperature_mean_7d"] = df.groupby(group_cols)["temperature_mean_c"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
        df["temperature_mean_30d"] = df.groupby(group_cols)["temperature_mean_c"].transform(lambda s: s.shift(1).rolling(30, min_periods=1).mean())
        
    # 3. Soil Moisture Rolling Means (7D, 30D)
    if "soil_moisture" in df.columns:
        df["soil_moisture_7d"] = df.groupby(group_cols)["soil_moisture"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
        df["soil_moisture_30d"] = df.groupby(group_cols)["soil_moisture"].transform(lambda s: s.shift(1).rolling(30, min_periods=1).mean())
        
    # 4. Evapotranspiration Rolling Means (7D, 30D)
    if "evapotranspiration" in df.columns:
        df["evapotranspiration_7d"] = df.groupby(group_cols)["evapotranspiration"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
        df["evapotranspiration_30d"] = df.groupby(group_cols)["evapotranspiration"].transform(lambda s: s.shift(1).rolling(30, min_periods=1).mean())

    # 5. 30-day Mean Rainfall Deviation
    if "rainfall_deviation_percent" in df.columns:
        df["rainfall_deviation"] = df.groupby(group_cols)["rainfall_deviation_percent"].transform(lambda s: s.shift(1).rolling(30, min_periods=1).mean()).fillna(0.0)
    else:
        df["rainfall_deviation"] = 0.0
        
    return df
