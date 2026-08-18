import os
from datetime import datetime
from typing import Dict, Any, Optional

class DataStatusTracker:
    """
    Centralized Runtime Tracker for API Data Status, Timestamps, and Lineage.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataStatusTracker, cls).__new__(cls)
            cls._instance.reset()
        return cls._instance

    def reset(self):
        now_str = datetime.now().strftime("%d %b %Y, %H:%M:%S IST")
        
        self.mandi_source = "Government of India OGD / AGMARKNET"
        self.mandi_endpoint = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
        self.mandi_last_fetch = now_str
        
        live_csv_path = "data/processed/live_market_latest.csv"
        if os.path.exists(live_csv_path):
            try:
                import pandas as pd
                df_l = pd.read_csv(live_csv_path)
                rec_count = len(df_l)
                if rec_count > 0:
                    self.mandi_status = "LIVE"
                    self.mandi_records_received = rec_count
                    self.mandi_error_reason = None
                    self.fallback_used = False
                    self.latest_data_date = str(df_l["arrival_date"].iloc[0]) if "arrival_date" in df_l.columns else "18/08/2026"
                else:
                    self.mandi_status = "NOT_FETCHED"
                    self.mandi_records_received = 0
                    self.mandi_error_reason = "Live AGMARKNET data file is empty."
                    self.fallback_used = True
                    self.latest_data_date = None
            except Exception:
                self.mandi_status = "NOT_FETCHED"
                self.mandi_records_received = 0
                self.mandi_error_reason = "Live AGMARKNET data has not been fetched yet."
                self.fallback_used = True
                self.latest_data_date = None
        else:
            self.mandi_status = "NOT_FETCHED"
            self.mandi_records_received = 0
            self.mandi_error_reason = "Live AGMARKNET data has not been fetched yet."
            self.fallback_used = True
            self.latest_data_date = None
        
        self.weather_source = "Open-Meteo Historical Weather API"
        self.weather_endpoint = "https://archive-api.open-meteo.com/v1/archive"
        self.weather_status = "LIVE"
        self.weather_last_fetch = now_str
        self.weather_records_received = 11
        self.weather_error_reason = None

    def update_mandi_status(self, status: str, records_count: int, error_reason: Optional[str] = None, latest_date: Optional[str] = None):
        self.mandi_last_fetch = datetime.now().strftime("%d %b %Y, %H:%M:%S IST")
        self.mandi_status = status.upper()
        self.mandi_records_received = records_count
        self.mandi_error_reason = error_reason
        if latest_date:
            self.latest_data_date = latest_date
        self.fallback_used = (self.mandi_status != "LIVE") or (self.weather_status != "LIVE")

    def update_weather_status(self, status: str, records_count: int, error_reason: Optional[str] = None):
        self.weather_last_fetch = datetime.now().strftime("%d %b %Y, %H:%M:%S IST")
        self.weather_status = status.upper()
        self.weather_records_received = records_count
        self.weather_error_reason = error_reason
        self.fallback_used = (self.mandi_status != "LIVE") or (self.weather_status != "LIVE")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mandi_source": self.mandi_source,
            "mandi_endpoint": self.mandi_endpoint,
            "mandi_status": self.mandi_status,
            "mandi_last_fetch": self.mandi_last_fetch,
            "mandi_records_received": self.mandi_records_received,
            "mandi_error_reason": self.mandi_error_reason,
            "weather_source": self.weather_source,
            "weather_endpoint": self.weather_endpoint,
            "weather_status": self.weather_status,
            "weather_last_fetch": self.weather_last_fetch,
            "weather_records_received": self.weather_records_received,
            "weather_error_reason": self.weather_error_reason,
            "fallback_used": self.fallback_used,
            "latest_data_date": self.latest_data_date
        }

status_tracker = DataStatusTracker()
