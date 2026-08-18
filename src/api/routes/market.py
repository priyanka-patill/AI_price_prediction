import os
import pandas as pd
from typing import Optional
from fastapi import APIRouter, Query
from pydantic import BaseModel
from src.ingestion.data_status import status_tracker
from src.models.lightgbm_model import LightGBMForecaster
from src.utils.geo import standardize_state

router = APIRouter(prefix="/api", tags=["Market Overview"])

class MarketOverviewResponse(BaseModel):
    state: str
    district: str
    market: str
    date: str
    current_price: Optional[float] = None
    predicted_price_7d: Optional[float] = None
    predicted_price_15d: Optional[float] = None
    predicted_price_30d: Optional[float] = None
    market_arrival_mt: Optional[float] = None
    production_mt: Optional[float] = None
    government_stock_mt: Optional[float] = None
    risk_level: str
    price_aggregation_method: str
    data_source_status: str
    prediction_status: str = "AVAILABLE"
    prediction_message: Optional[str] = None

@router.get("/market-overview", response_model=MarketOverviewResponse, summary="Get Mandi & Supply Overview Metrics")
def get_market_overview(
    state: Optional[str] = Query(None, description="State Name"),
    district: Optional[str] = Query(None, description="District Name"),
    market: Optional[str] = Query(None, description="Market Mandi Name"),
    date: Optional[str] = Query(None, description="Date YYYY-MM-DD")
):
    live_csv_path = "data/processed/live_market_latest.csv"
    ew_path = "data/processed/early_warning.csv"
    data_path = "data/processed/feature_engineered_modelling_dataset.parquet"

    norm_state = standardize_state(state) if state and state != "All States" else None

    # Determine live price first
    is_live_ready = os.path.exists(live_csv_path)
    curr_p = None
    agg_desc = "Live National Mean AGMARKNET Price across All Mandis"
    sub_live = pd.DataFrame()
    data_src_status = "FALLBACK"

    if is_live_ready:
        try:
            df_live = pd.read_csv(live_csv_path)
            if not df_live.empty and "modal_price" in df_live.columns:
                df_live["state_norm"] = df_live["state"].apply(lambda x: standardize_state(str(x)))
                sub_live = df_live.copy()

                # Step 1: Filter by State
                if norm_state:
                    sub_st = sub_live[sub_live["state_norm"].str.lower() == norm_state.lower()]
                    if not sub_st.empty:
                        sub_live = sub_st
                        agg_desc = f"Live AGMARKNET Rice Price across {norm_state} Mandis"

                # Step 2: Filter by District
                if district and district != "All Districts":
                    sub_dist = sub_live[sub_live["district"].astype(str).str.lower() == district.lower()]
                    if not sub_dist.empty:
                        sub_live = sub_dist
                        agg_desc = f"Live AGMARKNET Rice Price across {district} District Mandis"

                # Step 3: Filter by Market
                if market and market != "All Markets":
                    sub_mkt = sub_live[sub_live["market"].astype(str).str.lower() == market.lower()]
                    if not sub_mkt.empty:
                        sub_live = sub_mkt
                        agg_desc = f"Live AGMARKNET Rice Price at {market} Mandi"

                if not sub_live.empty:
                    sub_live["modal_price"] = pd.to_numeric(sub_live["modal_price"], errors="coerce")
                    curr_p = float(sub_live["modal_price"].dropna().mean()) if not sub_live["modal_price"].dropna().empty else None
                    data_src_status = "LIVE"

                print(f"[MarketOverviewRoute] Filter Request: State='{norm_state}', District='{district}', Market='{market}'")
                print(f"[MarketOverviewRoute] Live Records Matched: {len(sub_live)} | Computed Price: Rs.{curr_p} | Desc: {agg_desc}")
        except Exception as e:
            print(f"[MarketOverviewRoute] Notice querying live market dataset: {e}")

    # Fallback to early warning dataset if live price was not found
    if curr_p is None and os.path.exists(ew_path):
        try:
            df_ew = pd.read_csv(ew_path)
            if "state" in df_ew.columns:
                df_ew["state_norm"] = df_ew["state"].apply(lambda x: standardize_state(str(x)))
            sub_ew = df_ew.copy()
            if norm_state:
                sub_st = sub_ew[sub_ew["state_norm"].str.lower() == norm_state.lower()]
                if not sub_st.empty:
                    sub_ew = sub_st
            if not sub_ew.empty and "current_price" in sub_ew.columns:
                curr_p = float(pd.to_numeric(sub_ew["current_price"], errors="coerce").dropna().mean())
                agg_desc = f"Historical Mean Rice Price for {norm_state or 'All States'}"
        except Exception:
            pass

    # Call ML forecaster with effective current_price for state-specific inference
    forecaster = LightGBMForecaster()
    lgb_pred_res = forecaster.predict_location_prices(
        state=norm_state,
        district=district,
        market=market,
        current_price=curr_p
    )

    pred_status = "AVAILABLE" if lgb_pred_res.get("is_available") else "UNAVAILABLE"
    pred_msg = lgb_pred_res.get("prediction_message") or lgb_pred_res.get("reason", "Prediction unavailable for this location.")

    p7d = lgb_pred_res.get("predicted_price_7d")
    p15d = lgb_pred_res.get("predicted_price_15d")
    p30d = lgb_pred_res.get("predicted_price_30d")

    st_val = norm_state or (str(sub_live["state"].iloc[0]) if not sub_live.empty and "state" in sub_live.columns else "All States")
    dist_val = district if district and district != "All Districts" else (str(sub_live["district"].iloc[0]) if not sub_live.empty and "district" in sub_live.columns else "All Districts")
    mkt_val = market if market and market != "All Markets" else (str(sub_live["market"].iloc[0]) if not sub_live.empty and "market" in sub_live.columns else "All Markets")
    latest_dt = str(sub_live["arrival_date"].iloc[0]) if not sub_live.empty and "arrival_date" in sub_live.columns else (status_tracker.latest_data_date or "18/08/2026")

    return MarketOverviewResponse(
        state=st_val,
        district=dist_val,
        market=mkt_val,
        date=latest_dt,
        current_price=round(curr_p, 2) if curr_p is not None else None,
        predicted_price_7d=p7d,
        predicted_price_15d=p15d,
        predicted_price_30d=p30d,
        market_arrival_mt=150.0,
        production_mt=0.0,
        government_stock_mt=135000.0,
        risk_level="NORMAL",
        price_aggregation_method=agg_desc if curr_p is not None else f"No current records available for {st_val}",
        data_source_status=data_src_status if curr_p is not None else "NO_LIVE_DATA_FOR_LOCATION",
        prediction_status=pred_status,
        prediction_message=pred_msg
    )
