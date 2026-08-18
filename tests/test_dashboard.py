import pytest
import os
import pandas as pd

def test_dashboard_files_exist():
    assert os.path.exists("src/dashboard/app.py")
    assert os.path.exists("src/api/main.py")

def test_dashboard_dataset_loading():
    ew_path = "data/processed/early_warning.csv"
    fc_path = "data/processed/forecasts.csv"
    
    if os.path.exists(ew_path):
        df_ew = pd.read_csv(ew_path)
        assert not df_ew.empty
        assert "warning_level" in df_ew.columns
        
    if os.path.exists(fc_path):
        df_fc = pd.read_csv(fc_path)
        assert not df_fc.empty
        assert "predicted_price" in df_fc.columns
