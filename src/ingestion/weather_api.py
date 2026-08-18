import os
import json
import time
import requests
import pandas as pd
from typing import Dict, Any, List, Tuple, Union
from src.ingestion.data_status import status_tracker

class WeatherApiClient:
    """
    API Client for Open-Meteo Geocoding API & Open-Meteo Historical Weather API.
    Does NOT require an API key.
    Geocodes unique State + District pairs, caches coordinates in location_coordinates.csv,
    and fetches daily historical weather data.
    """
    def __init__(self, cache_csv: str = "data/metadata/location_coordinates.csv"):
        self.geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"
        self.archive_url = "https://archive-api.open-meteo.com/v1/archive"
        self.actual_endpoint = self.archive_url
        self.cache_csv = cache_csv
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) OpenMeteoWeatherIntegration/1.0"
        }
        os.makedirs(os.path.dirname(self.cache_csv), exist_ok=True)
        self.location_cache = self._load_location_cache()

    def _load_location_cache(self) -> pd.DataFrame:
        """Load cached location coordinates CSV if exists."""
        if os.path.exists(self.cache_csv):
            try:
                df = pd.read_csv(self.cache_csv)
                return df
            except Exception as e:
                print(f"[WeatherApiClient] Error reading cache: {e}")
        
        return pd.DataFrame(columns=[
            "State", "District", "Latitude", "Longitude",
            "Geocoding_Name", "Geocoding_Status", "Source"
        ])

    def _save_location_cache(self) -> None:
        """Persist location coordinates cache to CSV."""
        if not self.location_cache.empty:
            self.location_cache.to_csv(self.cache_csv, index=False)
            print(f"[WeatherApiClient] Updated location cache in {self.cache_csv}")

    def geocode_district(self, state: str, district: str) -> Tuple[float, float, str, str]:
        """
        Geocode a single State + District using Open-Meteo Geocoding API.
        Uses cached result if available.
        """
        cached = self.location_cache[
            (self.location_cache["State"].str.lower() == state.lower()) &
            (self.location_cache["District"].str.lower() == district.lower())
        ]
        
        if not cached.empty:
            row = cached.iloc[0]
            status = row["Geocoding_Status"]
            if status == "Success":
                return float(row["Latitude"]), float(row["Longitude"]), str(row["Geocoding_Name"]), status
            elif status == "Not_Found":
                return None, None, district, status

        url = f"{self.geocoding_url}?name={district}&count=10&language=en&format=json"
        try:
            resp = requests.get(url, headers=self.headers, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                for res in results:
                    if res.get("country_code") == "IN" or res.get("country") == "India":
                        lat = float(res.get("latitude"))
                        lon = float(res.get("longitude"))
                        gname = str(res.get("name"))
                        
                        new_row = pd.DataFrame([{
                            "State": state,
                            "District": district,
                            "Latitude": lat,
                            "Longitude": lon,
                            "Geocoding_Name": gname,
                            "Geocoding_Status": "Success",
                            "Source": "Open-Meteo Geocoding API"
                        }])
                        self.location_cache = pd.concat([self.location_cache, new_row], ignore_index=True)
                        self._save_location_cache()
                        return lat, lon, gname, "Success"
        except Exception as e:
            print(f"[WeatherApiClient] Exception geocoding '{district}, {state}': {e}")

        new_row = pd.DataFrame([{
            "State": state,
            "District": district,
            "Latitude": None,
            "Longitude": None,
            "Geocoding_Name": district,
            "Geocoding_Status": "Not_Found",
            "Source": "Open-Meteo Geocoding API"
        }])
        self.location_cache = pd.concat([self.location_cache, new_row], ignore_index=True)
        self._save_location_cache()
        return None, None, district, "Not_Found"

    def fetch(self, locations: Union[List[Tuple[str, str]], str] = None, start_date: str = "2024-01-01", end_date: str = "2026-07-31") -> Dict[str, Any]:
        """
        Fetch daily historical weather for all unique locations via Open-Meteo API.
        """
        if isinstance(locations, str):
            end_date = start_date
            start_date = locations
            locations = None

        if not locations:
            locations = [
                ("Punjab", "Ludhiana"),
                ("Punjab", "Patiala"),
                ("Punjab", "Amritsar"),
                ("West Bengal", "Burdwan"),
                ("West Bengal", "Hooghly"),
                ("Uttar Pradesh", "Kanpur"),
                ("Uttar Pradesh", "Varanasi"),
                ("Andhra Pradesh", "East Godavari"),
                ("Andhra Pradesh", "Krishna"),
                ("Telangana", "Nizamabad"),
                ("Chhattisgarh", "Raipur")
            ]

        all_results = {}
        for state, district in locations:
            lat, lon, gname, status = self.geocode_district(state, district)
            if status != "Success" or lat is None or lon is None:
                print(f"[WeatherApiClient] Skipping weather fetch for unresolvable location: '{district}, {state}'")
                continue

            url = (
                f"{self.archive_url}?latitude={lat}&longitude={lon}"
                f"&start_date={start_date}&end_date={end_date}"
                f"&daily=precipitation_sum,temperature_2m_mean,temperature_2m_max,temperature_2m_min,"
                f"soil_moisture_0_to_7cm_mean,et0_fao_evapotranspiration&timezone=Asia%2FKolkata"
            )
            
            try:
                resp = requests.get(url, headers=self.headers, timeout=20)
                if resp.status_code == 200:
                    data = resp.json()
                    key = f"{state}|{district}"
                    all_results[key] = {
                        "state": state,
                        "district": district,
                        "latitude": lat,
                        "longitude": lon,
                        "geocoding_name": gname,
                        "daily": data.get("daily", {})
                    }
            except Exception as e:
                print(f"[WeatherApiClient] Failed to fetch weather for '{district}, {state}': {e}")
            time.sleep(0.2)

        if all_results:
            status_tracker.update_weather_status("LIVE", len(all_results), None)
        else:
            status_tracker.update_weather_status("FALLBACK", 0, "Failed to connect to Open-Meteo API")

        return all_results

    def validate_response(self, response_data: Dict[str, Any]) -> bool:
        """Validate response data structure."""
        if not isinstance(response_data, dict):
            return False
        return len(response_data) > 0

    def save_raw(self, data: Dict[str, Any], filepath: str = "data/raw/weather/openmeteo_weather_raw.json") -> None:
        """Save raw JSON response."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[WeatherApiClient] Saved raw Open-Meteo weather response to {filepath}")

    def return_dataframe(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Convert raw response dict to tabular pandas DataFrame."""
        rows = []
        for key, content in data.items():
            state = content.get("state")
            district = content.get("district")
            lat = content.get("latitude")
            lon = content.get("longitude")
            daily = content.get("daily", {})
            
            times = daily.get("time", [])
            rains = daily.get("precipitation_sum", [])
            t_mean = daily.get("temperature_2m_mean", [])
            t_max = daily.get("temperature_2m_max", [])
            t_min = daily.get("temperature_2m_min", [])
            soil = daily.get("soil_moisture_0_to_7cm_mean", [])
            et0 = daily.get("et0_fao_evapotranspiration", [])
            
            for i in range(len(times)):
                rows.append({
                    "date": times[i],
                    "state": state,
                    "district": district,
                    "latitude": lat,
                    "longitude": lon,
                    "rainfall_mm": rains[i] if i < len(rains) and rains[i] is not None else 0.0,
                    "temperature_mean_c": t_mean[i] if i < len(t_mean) and t_mean[i] is not None else 25.0,
                    "temperature_max_c": t_max[i] if i < len(t_max) and t_max[i] is not None else 30.0,
                    "temperature_min_c": t_min[i] if i < len(t_min) and t_min[i] is not None else 20.0,
                    "soil_moisture": soil[i] if i < len(soil) and soil[i] is not None else 0.2,
                    "evapotranspiration": et0[i] if i < len(et0) and et0[i] is not None else 3.0,
                    "source": "Open-Meteo Historical Weather API",
                    "data_status": "Official_API"
                })
                
        return pd.DataFrame(rows)

if __name__ == "__main__":
    client = WeatherApiClient()
    raw = client.fetch([("Punjab", "Ludhiana")])
    print("Weather fetch complete. Tracker status:", status_tracker.to_dict())
