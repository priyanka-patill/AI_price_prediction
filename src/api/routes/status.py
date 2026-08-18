from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from src.ingestion.data_status import status_tracker

router = APIRouter(prefix="/api", tags=["Data Lineage & Status"])

class DataStatusResponse(BaseModel):
    mandi_source: str
    mandi_endpoint: str
    mandi_status: str
    mandi_last_fetch: str
    mandi_records_received: int
    mandi_error_reason: Optional[str] = None
    weather_source: str
    weather_endpoint: str
    weather_status: str
    weather_last_fetch: str
    weather_records_received: int
    weather_error_reason: Optional[str] = None
    fallback_used: bool
    latest_data_date: Optional[str] = None

@router.get("/data-status", response_model=DataStatusResponse, summary="Get Runtime API Data Source & Live/Fallback Lineage Status")
def get_data_status():
    return DataStatusResponse(**status_tracker.to_dict())
