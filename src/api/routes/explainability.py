import os
import pandas as pd
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["Explainability"])

class FeatureShapItem(BaseModel):
    feature: str
    feature_value: str
    shap_value: float
    contribution_direction: str
    relative_shap_contribution_percent: float
    interpretation: str

class ShapExplainResponse(BaseModel):
    state: str
    district: str
    market: str
    forecast_horizon: str
    features: List[FeatureShapItem]
    disclaimer: str

@router.get("/explain", response_model=ShapExplainResponse, summary="Get SHAP Feature Attribution for Price Forecasts")
def get_shap_explanation(
    state: Optional[str] = Query(None),
    district: Optional[str] = Query(None),
    market: Optional[str] = Query(None),
    horizon: int = Query(7, description="Horizon in days (7, 15, or 30)")
):
    shap_path = "data/processed/shap_local_explanations.csv"
    if not os.path.exists(shap_path):
        raise HTTPException(status_code=404, detail="SHAP local explanations dataset not found.")
        
    df_shap = pd.read_csv(shap_path)
    df_shap = df_shap[df_shap["forecast_horizon"] == f"{horizon}D"]
    
    if state:
        df_shap = df_shap[df_shap["state"].str.lower() == state.lower()]
    if district:
        df_shap = df_shap[df_shap["district"].str.lower() == district.lower()]
    if market:
        df_shap = df_shap[df_shap["market"].str.lower() == market.lower()]

    if df_shap.empty:
        df_shap = pd.read_csv(shap_path)
        df_shap = df_shap[df_shap["forecast_horizon"] == f"{horizon}D"]
        
    latest_dt = df_shap["date"].max()
    sub_latest = df_shap[df_shap["date"] == latest_dt].head(10)
    
    total_abs_shap = sum(abs(float(r["shap_value_rs"])) for _, r in sub_latest.iterrows())
    total_abs_shap = total_abs_shap if total_abs_shap > 0 else 1.0
    
    items = []
    for _, row in sub_latest.iterrows():
        s_val = float(row["shap_value_rs"])
        rel_pct = (abs(s_val) / total_abs_shap) * 100.0
        
        items.append(FeatureShapItem(
            feature=str(row["feature"]),
            feature_value=str(row["feature_value"]),
            shap_value=round(s_val, 2),
            contribution_direction=str(row["contribution_direction"]),
            relative_shap_contribution_percent=round(rel_pct, 2),
            interpretation=str(row["interpretation"])
        ))
        
    st_val = str(sub_latest["state"].iloc[0]) if not sub_latest.empty else "Overall"
    dist_val = str(sub_latest["district"].iloc[0]) if not sub_latest.empty else "Overall"
    mkt_val = str(sub_latest["market"].iloc[0]) if not sub_latest.empty else "Overall"

    return ShapExplainResponse(
        state=st_val,
        district=dist_val,
        market=mkt_val,
        forecast_horizon=f"{horizon}D",
        features=items,
        disclaimer="SHAP values quantify relative feature attribution toward the model's prediction and should not be interpreted as physical proof of causal impact."
    )
