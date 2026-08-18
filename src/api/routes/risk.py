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

from src.utils.geo import standardize_state, get_state_center, STATE_COORDINATES

@router.get("/risk-map", response_model=List[RiskMapItem], summary="Get Geographic Risk Map Points")
def get_risk_map(
    state: Optional[str] = Query(None),
    district: Optional[str] = Query(None),
    market: Optional[str] = Query(None)
):
    live_csv_path = "data/processed/live_market_latest.csv"
    ew_path = "data/processed/early_warning.csv"
    coords_path = "data/metadata/location_coordinates.csv"

    norm_state = standardize_state(state) if state and state != "All States" else None

    # Load coordinates lookup table
    coords_dict = {}
    if os.path.exists(coords_path):
        try:
            df_coords = pd.read_csv(coords_path)
            df_coords.columns = [c.lower() for c in df_coords.columns]
            for _, r in df_coords.iterrows():
                st_c = standardize_state(str(r.get("state")))
                dist_c = str(r.get("district")).strip().lower()
                lat_v = r.get("latitude")
                lon_v = r.get("longitude")
                if pd.notna(lat_v) and pd.notna(lon_v):
                    coords_dict[(st_c.lower(), dist_c)] = (float(lat_v), float(lon_v))
        except Exception:
            pass

    points_df = pd.DataFrame()

    # 1. Try Live AGMARKNET data
    if os.path.exists(live_csv_path):
        try:
            df_live = pd.read_csv(live_csv_path)
            if not df_live.empty:
                df_live["state_norm"] = df_live["state"].apply(lambda x: standardize_state(str(x)))
                sub = df_live.copy()
                if norm_state:
                    sub = sub[sub["state_norm"].str.lower() == norm_state.lower()]
                if district and district != "All Districts":
                    sub = sub[sub["district"].astype(str).str.lower() == district.lower()]
                if market and market != "All Markets":
                    sub = sub[sub["market"].astype(str).str.lower() == market.lower()]
                if not sub.empty:
                    points_df = sub
        except Exception:
            pass

    # 2. Fallback to Early Warning dataset
    if points_df.empty and os.path.exists(ew_path):
        try:
            df_ew = pd.read_csv(ew_path)
            if not df_ew.empty:
                df_ew["state_norm"] = df_ew["state"].apply(lambda x: standardize_state(str(x)))
                sub = df_ew.copy()
                if norm_state:
                    sub = sub[sub["state_norm"].str.lower() == norm_state.lower()]
                if district and district != "All Districts":
                    sub = sub[sub["district"].astype(str).str.lower() == district.lower()]
                if market and market != "All Markets":
                    sub = sub[sub["market"].astype(str).str.lower() == market.lower()]
                points_df = sub
        except Exception:
            pass

    if points_df.empty and norm_state:
        st_c_info = get_state_center(norm_state)
        return [
            RiskMapItem(
                state=norm_state,
                district="State Center",
                market=f"{norm_state} Mandi Hub",
                latitude=round(st_c_info["lat"], 4),
                longitude=round(st_c_info["lon"], 4),
                risk_level="NORMAL",
                price_pressure_score=1.5,
                forecast_price=3450.0
            )
        ]

    if points_df.empty:
        return []

    results = []
    for _, row in points_df.head(150).iterrows():
        st_val = str(row.get("state_norm", row.get("state", "Unknown")))
        dist_val = str(row.get("district", "Unknown")).strip()
        mkt_val = str(row.get("market", "Unknown")).strip()

        # Coordinate resolution: District lookup -> State Center fallback
        lat, lon = coords_dict.get((st_val.lower(), dist_val.lower()), (None, None))
        if lat is None or lon is None:
            st_c_info = get_state_center(st_val)
            lat = st_c_info["lat"]
            lon = st_c_info["lon"]

        modal_p = float(row.get("modal_price", row.get("current_price", 3450.0)))
        fc_p = float(row.get("forecast_7d", modal_p * 1.01))
        risk_l = str(row.get("warning_level", row.get("risk_level", "NORMAL")))
        press_score = round(abs(fc_p - modal_p) / max(modal_p, 1.0) * 100.0, 2)

        results.append(RiskMapItem(
            state=st_val,
            district=dist_val,
            market=mkt_val,
            latitude=round(lat, 4),
            longitude=round(lon, 4),
            risk_level=risk_l,
            price_pressure_score=press_score,
            forecast_price=round(fc_p, 2)
        ))
    return results
