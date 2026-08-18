import pytest
import pandas as pd
import numpy as np
import os
from src.models.target_builder import create_forecasting_targets
from src.models.lightgbm_model import LightGBMForecaster

def test_lightgbm_training_and_prediction(tmp_path):
    dates = pd.date_range("2024-01-01", "2025-12-31", freq="D").strftime("%Y-%m-%d")
    np.random.seed(42)
    prices = 3000 + np.cumsum(np.random.randn(len(dates)) * 5)
    
    df = pd.DataFrame({
        "date": list(dates),
        "state": ["Punjab"] * len(dates),
        "district": ["Ludhiana"] * len(dates),
        "market": ["Ludhiana Mandi"] * len(dates),
        "price_rs_per_qtl": prices,
        "price_lag_1": pd.Series(prices).shift(1).fillna(3000),
        "price_rolling_mean_7": pd.Series(prices).shift(1).rolling(7, min_periods=1).mean().fillna(3000),
        "rainfall_mm": np.random.rand(len(dates)) * 10
    })
    
    df_t = create_forecasting_targets(df, target_col="price_rs_per_qtl", horizons=[7])
    
    forecaster = LightGBMForecaster(model_dir=str(tmp_path))
    models = forecaster.train_all_horizons(df_t, horizons=[7])
    
    assert 7 in models
    preds = forecaster.predict(df_t, horizon=7)
    assert len(preds) == len(df_t)
    assert not np.isnan(preds).all()
