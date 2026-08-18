import os
import pandas as pd
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["Early Warning & Risk Map"])

class RiskAlertItem(BaseModel):
    date: str
    state: str
    district: str
    market: str
    current_price: float
    forecast_7d: float
    expected_change_percent: float
    volatility: float
    spike_score: float
    warning_level: str
    warning_reason: str

class RiskMapItem(BaseModel):
    state: str
    district: str
    market: str
    latitude: float
    longitude: float
    risk_level: str
    price_pressure_score: float
    forecast_price: float

@router.get("/risk", response_model=List[RiskAlertItem], summary="Get Mandi Risk Alerts & Volatility Signals")
def get_risk_alerts(
    state: Optional[str] = Query(None),
    warning_level: Optional[str] = Query(None)
):
    ew_path = "data/processed/early_warning.csv"
    if not os.path.exists(ew_path):
        raise HTTPException(status_code=404, detail="Early warning alerts dataset not found.")
        
    df_ew = pd.read_csv(ew_path)
    
    if state:
        df_ew = df_ew[df_ew["state"].str.lower() == state.lower()]
    if warning_level:
        df_ew = df_ew[df_ew["warning_level"].str.upper() == warning_level.upper()]

    results = []
    for _, row in df_ew.tail(100).iterrows():
        results.append(RiskAlertItem(
            date=str(row["date"]),
            state=str(row["state"]),
            district=str(row["district"]),
            market=str(row["market"]),
            current_price=round(float(row["current_price"]), 2),
            forecast_7d=round(float(row["forecast_7d"]), 2),
            expected_change_percent=round(float(row["expected_change_percent"]), 2),
            volatility=round(float(row["rolling_volatility"]), 4),
            spike_score=round(float(row["spike_score"]), 2),
            warning_level=str(row["warning_level"]),
            warning_reason=str(row["warning_reason"])
        ))
    return results

@router.get("/risk-map", response_model=List[RiskMapItem], summary="Get Geographic Risk Map Points")
def get_risk_map():
    ew_path = "data/processed/early_warning.csv"
    coords_path = "data/metadata/location_coordinates.csv"
    
    if not (os.path.exists(ew_path) and os.path.exists(coords_path)):
        return []
        
    df_ew = pd.read_csv(ew_path)
    df_coords = pd.read_csv(coords_path)
    df_coords.columns = [c.lower() for c in df_coords.columns]
    
    # Get latest alert per market
    latest_ew = df_ew.sort_values(by="date").groupby(["state", "district", "market"]).last().reset_index()
    
    merged = pd.merge(latest_ew, df_coords, on=["state", "district"], how="inner")
    
    results = []
    for _, row in merged.iterrows():
        score = abs(float(row["spike_score"])) * 1.5 + abs(float(row["expected_change_percent"])) * 0.5
        results.append(RiskMapItem(
            state=str(row["state"]),
            district=str(row["district"]),
            market=str(row["market"]),
            latitude=round(float(row["latitude"]), 4),
            longitude=round(float(row["longitude"]), 4),
            risk_level=str(row["warning_level"]),
            price_pressure_score=round(score, 2),
            forecast_price=round(float(row["forecast_7d"]), 2)
        ))
    return results
