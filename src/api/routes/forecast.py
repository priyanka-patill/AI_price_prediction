import os
import pandas as pd
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from src.models.lightgbm_model import LightGBMForecaster

from src.utils.geo import standardize_state

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

    fe_path = "data/processed/feature_engineered_modelling_dataset.parquet"
    live_csv_path = "data/processed/live_market_latest.csv"
    ew_path = "data/processed/early_warning.csv"

    norm_state = standardize_state(state) if state and state != "All States" else None
    st_val = norm_state or "All States"
    dist_val = district if district and district != "All Districts" else "All Districts"
    mkt_val = market if market and market != "All Markets" else "All Markets"

    hist_points = []
    fc_points = []

    # 1. Check feature engineered historical dataset for location series
    if os.path.exists(fe_path):
        try:
            df_fe = pd.read_parquet(fe_path)
            if "state" in df_fe.columns:
                df_fe["state_norm"] = df_fe["state"].apply(lambda x: standardize_state(str(x)))
            sub_fe = df_fe.copy()
            if norm_state and "state_norm" in sub_fe.columns:
                sub_fe = sub_fe[sub_fe["state_norm"].str.lower() == norm_state.lower()]
            if district and district != "All Districts":
                sub_fe = sub_fe[sub_fe["district"].astype(str).str.lower() == district.lower()]
            if market and market != "All Markets":
                sub_fe = sub_fe[sub_fe["market"].astype(str).str.lower() == market.lower()]

            if not sub_fe.empty:
                sub_fe = sub_fe.sort_values(by="date")
                for _, row in sub_fe.iterrows():
                    dt = str(row["date"])
                    p = row.get("price_rs_per_qtl")
                    if not pd.isna(p):
                        hist_points.append(PricePoint(date=dt, price=round(float(p), 2)))

                # Run real LightGBM prediction for future points
                forecaster = LightGBMForecaster()
                lgb_res = forecaster.predict_location_prices(state=state, district=district, market=market)
                p_val = lgb_res.get(f"predicted_price_{horizon}d")
                if p_val is not None:
                    last_dt = pd.to_datetime(sub_fe["date"].max())
                    future_dt = (last_dt + pd.Timedelta(days=horizon)).strftime("%Y-%m-%d")
                    fc_points.append(PricePoint(date=future_dt, price=p_val))

                return ForecastResponse(
                    horizon=horizon,
                    state=st_val,
                    district=dist_val,
                    market=mkt_val,
                    historical=hist_points[-60:],
                    forecast=fc_points,
                    confidence_interval_available=False
                )
        except Exception as e:
            print(f"[ForecastRoute] Notice querying feature engineered dataset: {e}")

    # 2. Check live market dataset for recent prices if historical dataset was not matched
    if os.path.exists(live_csv_path):
        try:
            df_live = pd.read_csv(live_csv_path)
            sub_live = df_live.copy()
            if state and state != "All States":
                sub_live = sub_live[sub_live["state"].astype(str).str.lower() == state.lower()]
            if district and district != "All Districts":
                sub_live = sub_live[sub_live["district"].astype(str).str.lower() == district.lower()]
            if market and market != "All Markets":
                sub_live = sub_live[sub_live["market"].astype(str).str.lower() == market.lower()]

            if not sub_live.empty:
                sub_live["modal_price"] = pd.to_numeric(sub_live["modal_price"], errors="coerce")
                for _, row in sub_live.iterrows():
                    dt = str(row.get("arrival_date", "18/08/2026"))
                    p = row.get("modal_price")
                    if not pd.isna(p):
                        hist_points.append(PricePoint(date=dt, price=round(float(p), 2)))

                return ForecastResponse(
                    horizon=horizon,
                    state=st_val,
                    district=dist_val,
                    market=mkt_val,
                    historical=hist_points[-30:],
                    forecast=[],
                    confidence_interval_available=False
                )
        except Exception as e:
            print(f"[ForecastRoute] Notice querying live market dataset: {e}")

    # 3. STRICT LOCATION RESPONSE: If requested location has no data, return empty points (NO Punjab cross-fallback)
    return ForecastResponse(
        horizon=horizon,
        state=st_val,
        district=dist_val,
        market=mkt_val,
        historical=[],
        forecast=[],
        confidence_interval_available=False
    )
