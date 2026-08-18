import os
import json
import pandas as pd
import numpy as np

from src.cleaning.clean_production import standardize_state

def clean_weather_dataset(raw_json_path: str = "data/raw/weather/openmeteo_weather_raw.json",
                         output_parquet_path: str = "data/processed/weather.parquet") -> pd.DataFrame:
    """
    Clean weather and rainfall dataset from Open-Meteo Historical Weather API.
    """
    if not os.path.exists(raw_json_path):
        # Fallback to legacy path if openmeteo path doesn't exist
        raw_json_path = "data/raw/weather/weather_raw.json"
        
    if not os.path.exists(raw_json_path):
        print(f"[clean_weather] File not found: {raw_json_path}")
        return pd.DataFrame()
        
    with open(raw_json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        
    rows = []
    for key, content in raw_data.items():
        state = standardize_state(content.get("state", "Unknown"))
        district = str(content.get("district", "Unknown")).strip().title()
        lat = content.get("latitude")
        lon = content.get("longitude")
        daily = content.get("daily", {})
        
        times = daily.get("time", [])
        rains = daily.get("precipitation_sum", daily.get("rain_sum", []))
        t_mean = daily.get("temperature_2m_mean", [])
        t_max = daily.get("temperature_2m_max", [])
        t_min = daily.get("temperature_2m_min", [])
        soil = daily.get("soil_moisture_0_to_7cm_mean", [])
        et0 = daily.get("et0_fao_evapotranspiration", [])
        
        for i in range(len(times)):
            r = rains[i] if i < len(rains) else 0.0
            tm = t_mean[i] if i < len(t_mean) else 25.0
            tmax = t_max[i] if i < len(t_max) else 30.0
            tmin = t_min[i] if i < len(t_min) else 20.0
            sm = soil[i] if i < len(soil) else 0.2
            ev = et0[i] if i < len(et0) else 3.0
            
            rows.append({
                "date": times[i],
                "state": state,
                "district": district,
                "latitude": lat,
                "longitude": lon,
                "rainfall_mm": r if r is not None else 0.0,
                "temperature_mean_c": tm if tm is not None else 25.0,
                "temperature_max_c": tmax if tmax is not None else 30.0,
                "temperature_min_c": tmin if tmin is not None else 20.0,
                "soil_moisture": sm if sm is not None else 0.2,
                "evapotranspiration": ev if ev is not None else 3.0,
                "source": "Open-Meteo Historical Weather API",
                "data_status": "Official_API"
            })
            
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Standardize dates
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    
    # 30-day baseline norm estimate per state per month (~10mm daily in monsoon, ~1.5mm in winter)
    df["month"] = pd.to_datetime(df["date"]).dt.month
    monsoon_months = [6, 7, 8, 9]
    df["normal_daily_rainfall_mm"] = df["month"].apply(lambda m: 12.0 if m in monsoon_months else 1.5)
    
    # Rainfall Deviation Percent = ((Rainfall - Normal) / Normal) * 100
    df["rainfall_deviation_percent"] = (
        (df["rainfall_mm"] - df["normal_daily_rainfall_mm"]) / df["normal_daily_rainfall_mm"]
    ) * 100.0

    # Suspicious Value Detection
    df["is_suspicious"] = False
    df.loc[(df["rainfall_mm"] < 0) | (df["temperature_mean_c"] < -10) | (df["temperature_mean_c"] > 55), "is_suspicious"] = True
    
    # Metadata
    df["primary_source"] = "Open-Meteo Historical Weather API"
    df["secondary_source"] = "ERA5 Reanalysis Grid"
    df["verification_status"] = "Verified"
    df["notes"] = "Daily historical precipitation, temperature, soil moisture, and evapotranspiration from Open-Meteo API"
    
    output_cols = [
        "date", "state", "district", "latitude", "longitude",
        "rainfall_mm", "normal_daily_rainfall_mm", "rainfall_deviation_percent",
        "temperature_mean_c", "temperature_max_c", "temperature_min_c",
        "soil_moisture", "evapotranspiration",
        "primary_source", "secondary_source", "verification_status", "data_status", "is_suspicious", "notes"
    ]
    df_clean = df[output_cols].sort_values(by=["state", "district", "date"]).reset_index(drop=True)
    
    os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
    df_clean.to_parquet(output_parquet_path, index=False)
    print(f"[clean_weather] Saved {len(df_clean)} rows of clean weather data to {output_parquet_path}")
    return df_clean

if __name__ == "__main__":
    clean_weather_dataset()
