import pytest
import os
import pandas as pd
import numpy as np
from src.ingestion.weather_api import WeatherApiClient
from src.cleaning.clean_weather import clean_weather_dataset
from src.feature_engineering.weather_features import add_weather_features

def test_openmeteo_geocoding():
    client = WeatherApiClient(cache_csv="data/metadata/location_coordinates.csv")
    lat, lon, gname, status = client.geocode_district("Punjab", "Ludhiana")
    
    assert status == "Success"
    assert lat is not None and lon is not None
    assert pytest.approx(lat, abs=1.0) == 30.9
    assert pytest.approx(lon, abs=1.0) == 75.8
    
    # Verify cache file exists
    assert os.path.exists("data/metadata/location_coordinates.csv")
    cache_df = pd.read_csv("data/metadata/location_coordinates.csv")
    assert not cache_df.empty
    assert "District" in cache_df.columns
    assert "Geocoding_Status" in cache_df.columns

def test_openmeteo_weather_fetch_and_clean():
    client = WeatherApiClient(cache_csv="data/metadata/location_coordinates.csv")
    raw = client.fetch(locations=[("Punjab", "Ludhiana")], start_date="2024-01-01", end_date="2024-01-05")
    assert client.validate_response(raw)
    
    client.save_raw(raw, "data/raw/weather/openmeteo_weather_raw.json")
    df = clean_weather_dataset("data/raw/weather/openmeteo_weather_raw.json", "data/processed/weather.parquet")
    
    assert not df.empty
    assert "rainfall_mm" in df.columns
    assert "temperature_mean_c" in df.columns
    assert "temperature_max_c" in df.columns
    assert "temperature_min_c" in df.columns
    assert "soil_moisture" in df.columns
    assert "evapotranspiration" in df.columns
    assert (df["primary_source"] == "Open-Meteo Historical Weather API").all()
    assert (df["data_status"] == "Official_API").all()

def test_weather_features_leakage_prevention():
    dates = pd.date_range("2025-06-01", "2025-06-20", freq="D").strftime("%Y-%m-%d")
    data_orig = {
        "date": list(dates),
        "state": ["Punjab"] * 20,
        "district": ["Ludhiana"] * 20,
        "rainfall_mm": [10.0] * 20,
        "temperature_mean_c": [28.0] * 20,
        "soil_moisture": [0.25] * 20,
        "evapotranspiration": [4.0] * 20
    }
    df1 = pd.DataFrame(data_orig)
    df1_fe = add_weather_features(df1, group_cols=["state", "district"])
    
    target_idx = 10
    orig_temp7d = df1_fe.loc[target_idx, "temperature_mean_7d"]
    orig_soil7d = df1_fe.loc[target_idx, "soil_moisture_7d"]
    
    # Mutate T+1 future temperature and soil moisture
    data_mutated = data_orig.copy()
    data_mutated["temperature_mean_c"] = data_orig["temperature_mean_c"].copy()
    data_mutated["temperature_mean_c"][11] = 99.0 # mutate date T+1
    
    df2 = pd.DataFrame(data_mutated)
    df2_fe = add_weather_features(df2, group_cols=["state", "district"])
    
    mutated_temp7d = df2_fe.loc[target_idx, "temperature_mean_7d"]
    mutated_soil7d = df2_fe.loc[target_idx, "soil_moisture_7d"]
    
    # Verify strict feature invariance at T
    assert orig_temp7d == mutated_temp7d, "Future temperature data leaked into T features!"
    assert orig_soil7d == mutated_soil7d, "Future soil moisture data leaked into T features!"
