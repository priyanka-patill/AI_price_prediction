from typing import Optional
from fastapi import APIRouter, Query, Body
from pydantic import BaseModel

from src.ingestion.market_api import AgmarknetClient
from src.ingestion.weather_api import WeatherApiClient
from src.ingestion.data_status import status_tracker
from src.config import get_api_key

router = APIRouter(prefix="/api", tags=["Real-time Data Sync"])

class SyncRequest(BaseModel):
    commodity: str = "Rice"
    limit: int = 100

class SyncResponse(BaseModel):
    status: str
    commodity: str
    mandi_status: str
    weather_status: str
    records_fetched: int
    weather_locations_synced: int
    source_market: str
    source_weather: str
    error_reason: Optional[str] = None
    message: str

@router.post("/sync-realtime-data", response_model=SyncResponse, summary="Sync Real-Time Live Mandi Prices & Weather Data")
def sync_realtime_data_post(request: SyncRequest = Body(...)):
    return _execute_sync(request.commodity, request.limit)

@router.get("/sync-realtime-data", response_model=SyncResponse, summary="Sync Real-Time Live Mandi Prices & Weather Data (GET)")
def sync_realtime_data_get(
    commodity: str = Query("Rice", description="Commodity to sync"),
    limit: int = Query(100, description="Number of live market records to fetch")
):
    return _execute_sync(commodity, limit)

def _execute_sync(commodity: str, limit: int) -> SyncResponse:
    market_client = AgmarknetClient()
    weather_client = WeatherApiClient()
    
    # 1. Fetch live AGMARKNET mandi data (persists live dataset internally when LIVE)
    raw_market = market_client.fetch(commodity=commodity, limit=limit)
    records = raw_market.get("records", []) if isinstance(raw_market, dict) else []
    
    # 2. Fetch live Open-Meteo weather data
    locations = [
        ("Punjab", "Ludhiana"),
        ("West Bengal", "Burdwan"),
        ("Andhra Pradesh", "East Godavari"),
        ("Uttar Pradesh", "Kanpur")
    ]
    raw_weather = weather_client.fetch(locations=locations, start_date="2026-01-01", end_date="2026-07-31")
    
    st_dict = status_tracker.to_dict()
    mandi_st = st_dict.get("mandi_status", "FALLBACK")
    weather_st = st_dict.get("weather_status", "LIVE")
    err_reason = st_dict.get("mandi_error_reason")
    
    source_market = f"AGMARKNET Live API (api.data.gov.in) [Status: {mandi_st}]"
    source_weather = f"Open-Meteo Live API (archive-api.open-meteo.com) [Status: {weather_st}]"
    
    msg = f"Synced {len(records)} market records ({mandi_st}) and {len(raw_weather)} weather locations ({weather_st})."
    
    return SyncResponse(
        status="success" if mandi_st == "LIVE" else "fallback",
        commodity=commodity,
        mandi_status=mandi_st,
        weather_status=weather_st,
        records_fetched=len(records),
        weather_locations_synced=len(raw_weather),
        source_market=source_market,
        source_weather=source_weather,
        error_reason=err_reason,
        message=msg
    )
