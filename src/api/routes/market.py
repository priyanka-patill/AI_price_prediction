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
    fe_path = "data/processed/feature_engineered_modelling_dataset.parquet"

    norm_state = standardize_state(state) if state and state != "All States" else None
    norm_dist = district.strip() if district and district != "All Districts" else None
    norm_mkt = market.strip() if market and market != "All Markets" else None

    curr_p = None
    agg_desc = "National Mean AGMARKNET Price across All Mandis"
    data_src_status = "FALLBACK"
    latest_dt = status_tracker.latest_data_date or "19/08/2026"

    # Step 1: Attempt strict live data filtering
    if os.path.exists(live_csv_path):
        try:
            df_live = pd.read_csv(live_csv_path)
            if not df_live.empty and "modal_price" in df_live.columns:
                df_live["state_norm"] = df_live["state"].apply(lambda x: standardize_state(str(x)))
                sub_live = df_live.copy()

                if norm_state:
                    sub_live = sub_live[sub_live["state_norm"].str.lower() == norm_state.lower()]
                if norm_dist and not sub_live.empty:
                    sub_live = sub_live[sub_live["district"].astype(str).str.lower() == norm_dist.lower()]
                if norm_mkt and not sub_live.empty:
                    sub_live = sub_live[sub_live["market"].astype(str).str.lower() == norm_mkt.lower()]

                if not sub_live.empty:
                    sub_live["modal_price"] = pd.to_numeric(sub_live["modal_price"], errors="coerce")
                    valid_prices = sub_live["modal_price"].dropna()
                    if not valid_prices.empty:
                        curr_p = float(valid_prices.mean())
                        data_src_status = "LIVE"
                        if norm_mkt:
                            agg_desc = f"Live AGMARKNET Rice Price at {norm_mkt} Mandi"
                        elif norm_dist:
                            agg_desc = f"Live AGMARKNET Rice Price across {norm_dist} District Mandis"
                        elif norm_state:
                            agg_desc = f"Live AGMARKNET Rice Price across {norm_state} Mandis"
                        else:
                            agg_desc = "Live National Mean AGMARKNET Price across All Mandis"

                        if "arrival_date" in sub_live.columns and not sub_live["arrival_date"].empty:
                            latest_dt = str(sub_live["arrival_date"].iloc[0])
        except Exception as e:
            print(f"[MarketOverviewRoute] Notice querying live market dataset: {e}")

    # Step 2: Fallback to Early Warning dataset if live price for this location is unavailable
    if curr_p is None and os.path.exists(ew_path):
        try:
            df_ew = pd.read_csv(ew_path)
            if not df_ew.empty and "current_price" in df_ew.columns:
                if "state" in df_ew.columns:
                    df_ew["state_norm"] = df_ew["state"].apply(lambda x: standardize_state(str(x)))
                sub_ew = df_ew.copy()
                if norm_state:
                    sub_ew = sub_ew[sub_ew["state_norm"].str.lower() == norm_state.lower()]
                if norm_dist and not sub_ew.empty:
                    sub_ew = sub_ew[sub_ew["district"].astype(str).str.lower() == norm_dist.lower()]
                if norm_mkt and not sub_ew.empty:
                    sub_ew = sub_ew[sub_ew["market"].astype(str).str.lower() == norm_mkt.lower()]

                if not sub_ew.empty:
                    valid_prices = pd.to_numeric(sub_ew["current_price"], errors="coerce").dropna()
                    if not valid_prices.empty:
                        curr_p = float(valid_prices.mean())
                        data_src_status = "HISTORICAL_BASELINE"
                        if norm_mkt:
                            agg_desc = f"Historical Mean Rice Price at {norm_mkt} Mandi"
                        elif norm_dist:
                            agg_desc = f"Historical Mean Rice Price across {norm_dist} District Mandis"
                        elif norm_state:
                            agg_desc = f"Historical Mean Rice Price for {norm_state}"
                        else:
                            agg_desc = "Historical National Mean Rice Price across All Mandis"

                        if "date" in sub_ew.columns and not sub_ew["date"].empty:
                            latest_dt = str(sub_ew["date"].iloc[-1])
        except Exception as e:
            print(f"[MarketOverviewRoute] Notice querying early warning dataset: {e}")

    # Step 3: Fallback to Feature Engineered dataset if still None
    if curr_p is None and os.path.exists(fe_path):
        try:
            df_fe = pd.read_parquet(fe_path)
            if not df_fe.empty and "price_rs_per_qtl" in df_fe.columns:
                if "state" in df_fe.columns:
                    df_fe["state_norm"] = df_fe["state"].apply(lambda x: standardize_state(str(x)))
                sub_fe = df_fe.copy()
                if norm_state:
                    sub_fe = sub_fe[sub_fe["state_norm"].str.lower() == norm_state.lower()]
                if norm_dist and not sub_fe.empty:
                    sub_fe = sub_fe[sub_fe["district"].astype(str).str.lower() == norm_dist.lower()]
                if norm_mkt and not sub_fe.empty:
                    sub_fe = sub_fe[sub_fe["market"].astype(str).str.lower() == norm_mkt.lower()]

                if not sub_fe.empty:
                    valid_prices = pd.to_numeric(sub_fe["price_rs_per_qtl"], errors="coerce").dropna()
                    if not valid_prices.empty:
                        curr_p = float(valid_prices.mean())
                        data_src_status = "HISTORICAL_BASELINE"
                        agg_desc = f"Historical Modeling Price for {norm_state or 'All States'}"
        except Exception as e:
            print(f"[MarketOverviewRoute] Notice querying feature dataset: {e}")

    # Call ML forecaster with location parameters and location price
    forecaster = LightGBMForecaster()
    lgb_pred_res = forecaster.predict_location_prices(
        state=norm_state,
        district=norm_dist,
        market=norm_mkt,
        current_price=curr_p
    )

    pred_status = "AVAILABLE" if lgb_pred_res.get("is_available") else "UNAVAILABLE"
    pred_msg = lgb_pred_res.get("prediction_message") or lgb_pred_res.get("reason", "Prediction unavailable for this location.")

    p7d = lgb_pred_res.get("predicted_price_7d")
    p15d = lgb_pred_res.get("predicted_price_15d")
    p30d = lgb_pred_res.get("predicted_price_30d")

    st_val = norm_state or "All States"
    dist_val = norm_dist or "All Districts"
    mkt_val = norm_mkt or "All Markets"

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
        price_aggregation_method=agg_desc if curr_p is not None else f"No price records available for {st_val}",
        data_source_status=data_src_status if curr_p is not None else "NO_DATA_FOR_LOCATION",
        prediction_status=pred_status,
        prediction_message=pred_msg
    )
