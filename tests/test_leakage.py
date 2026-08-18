import pytest
import pandas as pd
import numpy as np
from src.feature_engineering.price_features import add_price_features
from src.feature_engineering.arrival_features import add_arrival_features
from src.feature_engineering.weather_features import add_weather_features

def test_price_rolling_mean_no_future_leakage():
    """
    Test that modifying price on date T+1 does NOT change price_rolling_mean_7 or lag features on date T.
    """
    dates = pd.date_range("2025-06-01", "2025-06-20", freq="D").strftime("%Y-%m-%d")
    prices_orig = [3000 + i * 20 for i in range(20)]
    
    df1 = pd.DataFrame({
        "date": list(dates),
        "state": ["Punjab"] * 20,
        "district": ["Ludhiana"] * 20,
        "market": ["Ludhiana Mandi"] * 20,
        "price_rs_per_qtl": prices_orig.copy()
    })
    df1_fe = add_price_features(df1, price_col="price_rs_per_qtl", group_cols=["state", "district", "market"])
    
    # Store features at target date T = 2025-06-10 (index 9)
    target_idx = 9
    orig_rolling_mean_at_T = df1_fe.loc[target_idx, "price_rolling_mean_7"]
    orig_lag1_at_T = df1_fe.loc[target_idx, "price_lag_1"]
    
    # Now drastically mutate future price at date T+1 (2025-06-11, index 10)
    prices_mutated = prices_orig.copy()
    prices_mutated[10] = 99999.0
    
    df2 = pd.DataFrame({
        "date": list(dates),
        "state": ["Punjab"] * 20,
        "district": ["Ludhiana"] * 20,
        "market": ["Ludhiana Mandi"] * 20,
        "price_rs_per_qtl": prices_mutated
    })
    df2_fe = add_price_features(df2, price_col="price_rs_per_qtl", group_cols=["state", "district", "market"])
    
    mutated_rolling_mean_at_T = df2_fe.loc[target_idx, "price_rolling_mean_7"]
    mutated_lag1_at_T = df2_fe.loc[target_idx, "price_lag_1"]
    
    # Strict assertions: Features at date T must be completely invariant to future changes at date T+1
    assert orig_rolling_mean_at_T == mutated_rolling_mean_at_T, "Data Leakage Detected: Rolling mean at T changed when future price at T+1 was modified!"
    assert orig_lag1_at_T == mutated_lag1_at_T, "Data Leakage Detected: Lag 1 at T changed when future price at T+1 was modified!"

def test_arrival_features_no_future_leakage():
    """
    Test that modifying arrival on date T+1 does NOT change arrival_rolling_mean_7 or arrival_lag_1 on date T.
    """
    dates = pd.date_range("2025-06-01", "2025-06-20", freq="D").strftime("%Y-%m-%d")
    arrivals_orig = [100.0 + i * 5 for i in range(20)]
    
    df1 = pd.DataFrame({
        "date": list(dates),
        "state": ["Punjab"] * 20,
        "district": ["Ludhiana"] * 20,
        "market": ["Ludhiana Mandi"] * 20,
        "arrival_mt": arrivals_orig.copy()
    })
    df1_fe = add_arrival_features(df1, arrival_col="arrival_mt", group_cols=["state", "district", "market"])
    
    target_idx = 10
    orig_rolling_at_T = df1_fe.loc[target_idx, "arrival_rolling_mean_7"]
    
    arrivals_mutated = arrivals_orig.copy()
    arrivals_mutated[11] = 55555.0 # mutate T+1
    
    df2 = pd.DataFrame({
        "date": list(dates),
        "state": ["Punjab"] * 20,
        "district": ["Ludhiana"] * 20,
        "market": ["Ludhiana Mandi"] * 20,
        "arrival_mt": arrivals_mutated
    })
    df2_fe = add_arrival_features(df2, arrival_col="arrival_mt", group_cols=["state", "district", "market"])
    
    mutated_rolling_at_T = df2_fe.loc[target_idx, "arrival_rolling_mean_7"]
    assert orig_rolling_at_T == mutated_rolling_at_T, "Data Leakage Detected in Arrival Rolling Features!"

def test_weather_features_no_future_leakage():
    """
    Test that modifying rainfall on date T+1 does NOT change rainfall_7d on date T.
    """
    dates = pd.date_range("2025-06-01", "2025-06-20", freq="D").strftime("%Y-%m-%d")
    rains_orig = [15.0] * 20
    
    df1 = pd.DataFrame({
        "date": list(dates),
        "state": ["Punjab"] * 20,
        "district": ["Ludhiana"] * 20,
        "rainfall_mm": rains_orig.copy(),
        "rainfall_deviation_percent": [0.0] * 20
    })
    df1_fe = add_weather_features(df1, group_cols=["state", "district"])
    
    target_idx = 10
    orig_rain7d_at_T = df1_fe.loc[target_idx, "rainfall_7d"]
    
    rains_mutated = rains_orig.copy()
    rains_mutated[11] = 999.0
    
    df2 = pd.DataFrame({
        "date": list(dates),
        "state": ["Punjab"] * 20,
        "district": ["Ludhiana"] * 20,
        "rainfall_mm": rains_mutated,
        "rainfall_deviation_percent": [0.0] * 20
    })
    df2_fe = add_weather_features(df2, group_cols=["state", "district"])
    
    mutated_rain7d_at_T = df2_fe.loc[target_idx, "rainfall_7d"]
    assert orig_rain7d_at_T == mutated_rain7d_at_T, "Data Leakage Detected in Weather Features!"
