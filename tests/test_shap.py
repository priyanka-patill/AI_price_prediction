import pytest
import pandas as pd
import numpy as np
import os
from src.explainability.shap_engine import ShapExplainerEngine

def test_shap_explainer_global_and_local():
    model_dir = "models"
    if not os.path.exists(os.path.join(model_dir, "lightgbm_7d.pkl")):
        pytest.skip("LightGBM model artifacts not found, skipping SHAP test.")
        
    engine = ShapExplainerEngine(model_dir=model_dir)
    assert len(engine.feature_cols) > 0
    
    data_path = "data/processed/feature_engineered_modelling_dataset.parquet"
    if not os.path.exists(data_path):
        pytest.skip("Phase 1 dataset not found, skipping SHAP test.")
        
    df = pd.read_parquet(data_path).head(50)
    
    global_df = engine.compute_global_importance(df, horizons=[7])
    assert not global_df.empty
    assert "mean_abs_shap" in global_df.columns
    assert "rank" in global_df.columns
    
    local_df = engine.compute_local_explanations(df, horizons=[7])
    assert not local_df.empty
    assert "shap_value_rs" in local_df.columns
    assert "contribution_direction" in local_df.columns
