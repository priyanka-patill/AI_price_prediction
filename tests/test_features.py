import pytest
import pandas as pd
import numpy as np
from src.feature_engineering.price_features import add_price_features
from src.feature_engineering.arrival_features import add_arrival_features
from src.feature_engineering.weather_features import add_weather_features

def test_price_features_calculation():
    dates = pd.date_range("2024-01-01", "2024-01-20", freq="D").strftime("%Y-%m-%d")
    data = {
        "date": list(dates),
        "state": ["Punjab"] * 20,
        "district": ["Ludhiana"] * 20,
        "market": ["Ludhiana Mandi"] * 20,
        "price_rs_per_qtl": [3000 + i * 10 for i in range(20)]
    }
    df = pd.DataFrame(data)
    df = add_price_features(df, price_col="price_rs_per_qtl", group_cols=["state", "district", "market"])
    
    # Lag 1 at index 1 should be price at index 0 (3000)
    assert df.loc[1, "price_lag_1"] == 3000.0
    # Lag 7 at index 7 should be price at index 0 (3000)
    assert df.loc[7, "price_lag_7"] == 3000.0
    # Rolling mean 7 at index 7 should be mean of prices from index 0..6: (3000 + 3060)/2 = 3030
    assert df.loc[7, "price_rolling_mean_7"] == pytest.approx(3030.0)

def test_arrival_features_calculation():
    dates = pd.date_range("2024-01-01", "2024-01-15", freq="D").strftime("%Y-%m-%d")
    data = {
        "date": list(dates),
        "state": ["Punjab"] * 15,
        "district": ["Ludhiana"] * 15,
        "market": ["Ludhiana Mandi"] * 15,
        "arrival_mt": [100.0] * 15
    }
    df = pd.DataFrame(data)
    df = add_arrival_features(df, arrival_col="arrival_mt", group_cols=["state", "district", "market"])
    
    assert df.loc[1, "arrival_lag_1"] == 100.0
    assert df.loc[7, "arrival_rolling_mean_7"] == 100.0

def test_weather_features_calculation():
    dates = pd.date_range("2024-01-01", "2024-01-10", freq="D").strftime("%Y-%m-%d")
    data = {
        "date": list(dates),
        "state": ["Punjab"] * 10,
        "district": ["Ludhiana"] * 10,
        "rainfall_mm": [10.0] * 10,
        "rainfall_deviation_percent": [5.0] * 10
    }
    df = pd.DataFrame(data)
    df = add_weather_features(df, group_cols=["state", "district"])
    
    # 7D rainfall sum at index 7 should be sum of 7 past days = 70.0
    assert df.loc[7, "rainfall_7d"] == 70.0
