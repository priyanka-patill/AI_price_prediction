import pytest
import pandas as pd
import numpy as np
from src.models.target_builder import create_forecasting_targets

def test_target_creation_shifts():
    dates = pd.date_range("2024-01-01", "2024-02-15", freq="D").strftime("%Y-%m-%d")
    prices = [3000 + i * 10 for i in range(len(dates))]
    
    df = pd.DataFrame({
        "date": list(dates),
        "state": ["Punjab"] * len(dates),
        "district": ["Ludhiana"] * len(dates),
        "market": ["Ludhiana Mandi"] * len(dates),
        "price_rs_per_qtl": prices
    })
    
    df_t = create_forecasting_targets(df, target_col="price_rs_per_qtl", horizons=[7, 15, 30])
    
    assert "price_target_7d" in df_t.columns
    assert "price_target_15d" in df_t.columns
    assert "price_target_30d" in df_t.columns
    
    # Check shift correctness: target_7d at index 0 should equal price at index 7
    assert df_t.loc[0, "price_target_7d"] == prices[7]
    assert df_t.loc[0, "price_target_15d"] == prices[15]
    assert df_t.loc[0, "price_target_30d"] == prices[30]
