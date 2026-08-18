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

    # 1. Query feature_engineered_modelling_dataset with fallback hierarchy
    if os.path.exists(fe_path):
        try:
            df_fe = pd.read_parquet(fe_path)
            if "state" in df_fe.columns:
                df_fe["state_norm"] = df_fe["state"].apply(lambda x: standardize_state(str(x)))
            
            sub_fe = df_fe.copy()
            if norm_state and "state_norm" in sub_fe.columns:
                sub_st = sub_fe[sub_fe["state_norm"].str.lower() == norm_state.lower()]
                if not sub_st.empty:
                    sub_fe = sub_st
            
            if district and district != "All Districts":
                sub_dist = sub_fe[sub_fe["district"].astype(str).str.lower() == district.lower()]
                if not sub_dist.empty:
                    sub_fe = sub_dist
            
            if market and market != "All Markets":
                sub_mkt = sub_fe[sub_fe["market"].astype(str).str.lower() == market.lower()]
                if not sub_mkt.empty:
                    sub_fe = sub_mkt

            if not sub_fe.empty:
                # Group by date to compute daily price trajectory
                grouped = sub_fe.groupby("date")["price_rs_per_qtl"].mean().reset_index().sort_values(by="date")
                for _, row in grouped.iterrows():
                    dt = str(row["date"])
                    p = row["price_rs_per_qtl"]
                    if not pd.isna(p):
                        hist_points.append(PricePoint(date=dt, price=round(float(p), 2)))

                forecaster = LightGBMForecaster()
                lgb_res = forecaster.predict_location_prices(state=state, district=district, market=market)
                p_val = lgb_res.get(f"predicted_price_{horizon}d")
                if p_val is not None:
                    last_dt = pd.to_datetime(grouped["date"].max())
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

    # 2. Check early warning dataset if feature engineered dataset was missing
    if os.path.exists(ew_path):
        try:
            df_ew = pd.read_csv(ew_path)
            if "state" in df_ew.columns:
                df_ew["state_norm"] = df_ew["state"].apply(lambda x: standardize_state(str(x)))
            sub_ew = df_ew.copy()
            if norm_state and "state_norm" in sub_ew.columns:
                sub_st = sub_ew[sub_ew["state_norm"].str.lower() == norm_state.lower()]
                if not sub_st.empty:
                    sub_ew = sub_st
            
            if not sub_ew.empty:
                grouped = sub_ew.groupby("date")["current_price"].mean().reset_index().sort_values(by="date")
                for _, row in grouped.iterrows():
                    dt = str(row["date"])
                    p = row["current_price"]
                    if not pd.isna(p):
                        hist_points.append(PricePoint(date=dt, price=round(float(p), 2)))

                forecaster = LightGBMForecaster()
                lgb_res = forecaster.predict_location_prices(state=state, district=district, market=market)
                p_val = lgb_res.get(f"predicted_price_{horizon}d")
                if p_val is not None:
                    last_dt = pd.to_datetime(grouped["date"].max())
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
            print(f"[ForecastRoute] Notice querying early warning dataset: {e}")

    return ForecastResponse(
        horizon=horizon,
        state=st_val,
        district=dist_val,
        market=mkt_val,
        historical=[],
        forecast=[],
        confidence_interval_available=False
    )
