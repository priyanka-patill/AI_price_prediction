import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List

def compute_state_price_pressure(forecasts_df: pd.DataFrame,
                                early_warning_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate State-Level Price Pressure Score and rank destination states for buffer stock intervention.
    Combines 7D/15D/30D price percentage growth, spike score, and warning level weights.
    Generates data/processed/state_priority.csv.
    """
    ew_df = early_warning_df.copy()
    
    # State-level aggregation
    ew_df["warning_weight"] = ew_df["warning_level"].map({"HIGH RISK": 5.0, "WARNING": 2.0, "NORMAL": 0.0}).fillna(0.0)
    
    state_summary = ew_df.groupby("state").agg(
        forecast_7d=("forecast_7d", "mean"),
        forecast_15d=("forecast_15d", "mean"),
        forecast_30d=("forecast_30d", "mean"),
        current_price=("current_price", "mean"),
        expected_change_pct=("expected_change_percent", "mean"),
        avg_spike_score=("spike_score", "mean"),
        avg_warning_weight=("warning_weight", "mean")
    ).reset_index()
    
    # Price Pressure Score = 0.5*(Expected Change %) + 1.5*(Spike Score) + 2.0*(Warning Weight)
    state_summary["price_pressure_score"] = (
        0.5 * state_summary["expected_change_pct"].clip(lower=0) +
        1.5 * state_summary["avg_spike_score"].clip(lower=0) +
        2.0 * state_summary["avg_warning_weight"]
    ).round(2)
    
    # Assign warning level label to state
    def get_state_warning(w):
        if w >= 3.5:
            return "HIGH RISK"
        elif w >= 1.0:
            return "WARNING"
        return "NORMAL"
        
    state_summary["warning_level"] = state_summary["avg_warning_weight"].apply(get_state_warning)
    
    # Priority ranking
    state_summary = state_summary.sort_values(by="price_pressure_score", ascending=False).reset_index(drop=True)
    state_summary["priority_rank"] = state_summary.index + 1
    
    # Estimated monthly need (MT) based on state allocation/pressure
    state_summary["estimated_need_mt"] = (state_summary["price_pressure_score"] * 2500.0).round(0)
    state_summary["stock_available_mt"] = 15000.0 # Configurable state buffer stock
    
    output_cols = [
        "priority_rank", "state", "price_pressure_score",
        "forecast_7d", "forecast_15d", "forecast_30d",
        "warning_level", "stock_available_mt", "estimated_need_mt"
    ]
    state_priority_df = state_summary[output_cols]
    
    os.makedirs("data/processed", exist_ok=True)
    state_priority_df.to_csv("data/processed/state_priority.csv", index=False)
    print(f"[price_pressure] Saved state priority ranking to data/processed/state_priority.csv")
    
    return state_priority_df
