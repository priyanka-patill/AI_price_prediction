import pytest
import pandas as pd
import numpy as np
from src.models.target_builder import create_forecasting_targets

def test_phase2_target_shift_no_feature_leakage():
    """
    Test that shift(-h) target creation at date T does not mutate or alter feature vector X(T).
    """
    dates = pd.date_range("2025-06-01", "2025-06-20", freq="D").strftime("%Y-%m-%d")
    prices_orig = [3000 + i * 15 for i in range(len(dates))]
    
    df1 = pd.DataFrame({
        "date": list(dates),
        "state": ["Punjab"] * len(dates),
        "district": ["Ludhiana"] * len(dates),
        "market": ["Ludhiana Mandi"] * len(dates),
        "price_rs_per_qtl": prices_orig.copy(),
        "price_lag_1": pd.Series(prices_orig).shift(1).fillna(3000)
    })
    
    # Store features at target date T = 2025-06-10 (index 9)
    target_idx = 9
    orig_lag1_at_T = df1.loc[target_idx, "price_lag_1"]
    
    # Mutate future target price at T+7 (2025-06-17, index 16)
    prices_mutated = prices_orig.copy()
    prices_mutated[16] = 99999.0
    
    df2 = pd.DataFrame({
        "date": list(dates),
        "state": ["Punjab"] * len(dates),
        "district": ["Ludhiana"] * len(dates),
        "market": ["Ludhiana Mandi"] * len(dates),
        "price_rs_per_qtl": prices_mutated,
        "price_lag_1": pd.Series(prices_mutated).shift(1).fillna(3000)
    })
    
    df2_targets = create_forecasting_targets(df2, target_col="price_rs_per_qtl", horizons=[7])
    
    # Verify strict feature invariance at T
    mutated_lag1_at_T = df2_targets.loc[target_idx, "price_lag_1"]
    assert orig_lag1_at_T == mutated_lag1_at_T, "Data Leakage: Feature vector at T altered by future target shift!"
