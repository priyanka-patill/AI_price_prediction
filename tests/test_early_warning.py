import pytest
import pandas as pd
import numpy as np
from src.models.early_warning import EarlyWarningSystem

def test_early_warning_threshold_learning_and_generation():
    dates = pd.date_range("2024-01-01", "2024-03-01", freq="D").strftime("%Y-%m-%d")
    np.random.seed(42)
    prices = 3000 + np.cumsum(np.random.randn(len(dates)) * 10)
    
    df = pd.DataFrame({
        "date": list(dates),
        "state": ["Punjab"] * len(dates),
        "district": ["Ludhiana"] * len(dates),
        "market": ["Ludhiana Mandi"] * len(dates),
        "price_rs_per_qtl": prices,
        "price_rolling_mean_30": pd.Series(prices).rolling(30, min_periods=1).mean(),
        "price_rolling_std_30": pd.Series(prices).rolling(30, min_periods=1).std().fillna(10.0)
    })
    
    ew = EarlyWarningSystem()
    thresholds = ew.fit_thresholds(df)
    
    assert "zscore_warning" in thresholds
    assert "zscore_high_risk" in thresholds
    assert "volatility_threshold_high" in thresholds
    
    # Mock forecasts for testing classification
    lgb_7d = prices + 150 # +5% increase -> triggers WARNING or HIGH RISK
    lgb_15d = prices + 160
    lgb_30d = prices + 170
    
    ew_df = ew.generate_warnings(df, lgb_7d, lgb_15d, lgb_30d)
    assert "warning_level" in ew_df.columns
    assert "spike_score" in ew_df.columns
    assert set(ew_df["warning_level"].unique()).issubset({"NORMAL", "WARNING", "HIGH RISK"})
    
    metrics = ew.evaluate_warnings(ew_df)
    assert "Precision" in metrics
    assert "Recall" in metrics
    assert "F1_Score" in metrics
