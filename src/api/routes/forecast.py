import os
import pandas as pd
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["Price Forecast"])

class PricePoint(BaseModel):
    date: str
    price: float

class ForecastResponse(BaseModel):
    horizon: int
    state: str
    district: str
    market: str
    historical: List[PricePoint]
    forecast: List[PricePoint]
    confidence_interval_available: bool
    lower_bound: Optional[List[float]] = None
    upper_bound: Optional[List[float]] = None

@router.get("/forecast", response_model=ForecastResponse, summary="Get Historical & Multi-Horizon Price Forecasts")
def get_price_forecast(
    state: Optional[str] = Query(None),
    district: Optional[str] = Query(None),
    market: Optional[str] = Query(None),
    horizon: int = Query(7, description="Forecast horizon in days (7, 15, or 30)")
):
    if horizon not in [7, 15, 30]:
        raise HTTPException(status_code=400, detail="Horizon must be 7, 15, or 30 days.")
        
    fc_path = "data/processed/forecasts.csv"
    ew_path = "data/processed/early_warning.csv"
    
    if not (os.path.exists(fc_path) or os.path.exists(ew_path)):
        raise HTTPException(status_code=404, detail="Forecast datasets not found.")
        
    df_fc = pd.read_csv(fc_path) if os.path.exists(fc_path) else pd.DataFrame()
    
    if state and not df_fc.empty:
        sub_fc = df_fc[df_fc["location"].str.contains(state, case=False, na=False)]
    else:
        sub_fc = df_fc
        
    if district and not sub_fc.empty:
        sub_fc = sub_fc[sub_fc["location"].str.contains(district, case=False, na=False)]
    if market and not sub_fc.empty:
        sub_fc = sub_fc[sub_fc["location"].str.contains(market, case=False, na=False)]
        
    sub_lgb = sub_fc[(sub_fc["forecast_horizon"] == f"{horizon}D") & (sub_fc["model"] == "LightGBM")].sort_values(by="forecast_date") if not sub_fc.empty else pd.DataFrame()

    hist_points = []
    fc_points = []

    if not sub_lgb.empty:
        for idx, row in sub_lgb.iterrows():
            dt = str(row["forecast_date"])
            act = row["actual_price"]
            pred = row["predicted_price"]
            
            if not pd.isna(act):
                hist_points.append(PricePoint(date=dt, price=round(float(act), 2)))
            if not pd.isna(pred):
                fc_points.append(PricePoint(date=dt, price=round(float(pred), 2)))

        loc_str = str(sub_lgb["location"].iloc[0])
        parts = loc_str.split("|")
        st_val = parts[0] if len(parts) > 0 else (state or "Overall")
        dist_val = parts[1] if len(parts) > 1 else (district or "Overall")
        mkt_val = parts[2] if len(parts) > 2 else (market or "Overall")
    elif os.path.exists(ew_path):
        # Fallback to early_warning dataset which contains full state/market coverage
        df_ew = pd.read_csv(ew_path)
        if state:
            df_ew = df_ew[df_ew["state"].str.lower() == state.lower()]
        if district:
            df_ew = df_ew[df_ew["district"].str.lower() == district.lower()]
        if market:
            df_ew = df_ew[df_ew["market"].str.lower() == market.lower()]

        if df_ew.empty:
            df_ew = pd.read_csv(ew_path)

        forecast_col = f"forecast_{horizon}d"
        df_ew = df_ew.sort_values(by="date")
        
        for _, row in df_ew.iterrows():
            dt = str(row["date"])
            cp = row["current_price"]
            fc_val = row.get(forecast_col, cp)
            if not pd.isna(cp):
                hist_points.append(PricePoint(date=dt, price=round(float(cp), 2)))
            if not pd.isna(fc_val):
                fc_points.append(PricePoint(date=dt, price=round(float(fc_val), 2)))

        row_last = df_ew.iloc[-1]
        st_val = str(row_last.get("state", state or "Overall"))
        dist_val = str(row_last.get("district", district or "Overall"))
        mkt_val = str(row_last.get("market", market or "Overall"))
    else:
        st_val, dist_val, mkt_val = state or "Overall", district or "Overall", market or "Overall"

    return ForecastResponse(
        horizon=horizon,
        state=st_val,
        district=dist_val,
        market=mkt_val,
        historical=hist_points[-60:],
        forecast=fc_points[-30:],
        confidence_interval_available=False
    )
