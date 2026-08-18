import pytest
import pandas as pd
import numpy as np
from src.models.arima_model import ArimaBaselineModel

def test_arima_fit_predict():
    np.random.seed(42)
    prices = pd.Series(3000 + np.cumsum(np.random.randn(30) * 5))
    
    arima = ArimaBaselineModel(order=(1, 1, 0))
    preds, lower, upper = arima.fit_predict_market(prices, horizon=7)
    
    assert len(preds) == 7
    assert len(lower) == 7
    assert len(upper) == 7
    assert not np.isnan(preds).any()
    assert (upper >= preds).all()
