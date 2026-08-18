import os
import pandas as pd
from typing import Optional, List
from fastapi import APIRouter, Query
from src.utils.geo import standardize_state, STATE_CANONICAL_MAP

router = APIRouter(prefix="/api", tags=["State & Location Discovery"])

def get_all_normalized_states() -> List[str]:
    """Dynamically discover and normalize all states from live, fallback, and historical datasets."""
    states_set = set()
    
    live_csv_path = "data/processed/live_market_latest.csv"
    if os.path.exists(live_csv_path):
        try:
            df_live = pd.read_csv(live_csv_path)
            if "state" in df_live.columns:
                for s in df_live["state"].dropna().unique():
                    states_set.add(standardize_state(str(s)))
        except Exception:
            pass

    ew_path = "data/processed/early_warning.csv"
    if os.path.exists(ew_path):
        try:
            df_ew = pd.read_csv(ew_path)
            if "state" in df_ew.columns:
                for s in df_ew["state"].dropna().unique():
                    states_set.add(standardize_state(str(s)))
        except Exception:
            pass

    fe_path = "data/processed/feature_engineered_modelling_dataset.parquet"
    if os.path.exists(fe_path):
        try:
            df_fe = pd.read_parquet(fe_path)
            if "state" in df_fe.columns:
                for s in df_fe["state"].dropna().unique():
                    states_set.add(standardize_state(str(s)))
        except Exception:
            pass

    # Ensure baseline states are always included if dataset is small
    default_states = [
        "Andhra Pradesh", "Assam", "Bihar", "Chhattisgarh", "Gujarat", "Haryana",
        "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Meghalaya", "Odisha",
        "Punjab", "Rajasthan", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh",
        "Uttarakhand", "West Bengal"
    ]
    for s in default_states:
        states_set.add(standardize_state(s))

    return sorted(list(states_set))

@router.get("/states", response_model=List[str], summary="Get List of All Available States")
def get_states():
    return get_all_normalized_states()

@router.get("/districts", response_model=List[str], summary="Get Unique Districts for Selected State")
def get_districts(state: Optional[str] = Query(None)):
    if not state or state == "All States":
        return []
    
    norm_state = standardize_state(state)
    dist_set = set()

    live_csv_path = "data/processed/live_market_latest.csv"
    if os.path.exists(live_csv_path):
        try:
            df_live = pd.read_csv(live_csv_path)
            if "state" in df_live.columns and "district" in df_live.columns:
                df_live["state_norm"] = df_live["state"].apply(lambda x: standardize_state(str(x)))
                sub = df_live[df_live["state_norm"].str.lower() == norm_state.lower()]
                for d in sub["district"].dropna().unique():
                    dist_set.add(str(d).strip())
        except Exception:
            pass

    ew_path = "data/processed/early_warning.csv"
    if os.path.exists(ew_path):
        try:
            df_ew = pd.read_csv(ew_path)
            if "state" in df_ew.columns and "district" in df_ew.columns:
                df_ew["state_norm"] = df_ew["state"].apply(lambda x: standardize_state(str(x)))
                sub = df_ew[df_ew["state_norm"].str.lower() == norm_state.lower()]
                for d in sub["district"].dropna().unique():
                    dist_set.add(str(d).strip())
        except Exception:
            pass

    return sorted(list(dist_set))

@router.get("/markets", response_model=List[str], summary="Get Unique Markets for Selected State and District")
def get_markets(
    state: Optional[str] = Query(None),
    district: Optional[str] = Query(None)
):
    if not state or state == "All States":
        return []

    norm_state = standardize_state(state)
    mkt_set = set()

    live_csv_path = "data/processed/live_market_latest.csv"
    if os.path.exists(live_csv_path):
        try:
            df_live = pd.read_csv(live_csv_path)
            if "state" in df_live.columns and "market" in df_live.columns:
                df_live["state_norm"] = df_live["state"].apply(lambda x: standardize_state(str(x)))
                sub = df_live[df_live["state_norm"].str.lower() == norm_state.lower()]
                if district and district != "All Districts":
                    sub = sub[sub["district"].astype(str).str.lower() == district.lower()]
                for m in sub["market"].dropna().unique():
                    mkt_set.add(str(m).strip())
        except Exception:
            pass

    ew_path = "data/processed/early_warning.csv"
    if os.path.exists(ew_path):
        try:
            df_ew = pd.read_csv(ew_path)
            if "state" in df_ew.columns and "market" in df_ew.columns:
                df_ew["state_norm"] = df_ew["state"].apply(lambda x: standardize_state(str(x)))
                sub = df_ew[df_ew["state_norm"].str.lower() == norm_state.lower()]
                if district and district != "All Districts":
                    sub = sub[sub["district"].astype(str).str.lower() == district.lower()]
                for m in sub["market"].dropna().unique():
                    mkt_set.add(str(m).strip())
        except Exception:
            pass

    return sorted(list(mkt_set))
