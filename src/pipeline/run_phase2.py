import os
import json
import pandas as pd
import numpy as np

# Phase 2 Model Modules
from src.models.target_builder import create_forecasting_targets
from src.models.naive_baseline import NaiveBaselineModel
from src.models.arima_model import ArimaBaselineModel
from src.models.lightgbm_model import LightGBMForecaster
from src.models.evaluator import evaluate_and_export_forecasts
from src.models.early_warning import EarlyWarningSystem
from src.visualization.evaluate_plots import generate_evaluation_plots

def run_phase2_pipeline():
    """
    Execute end-to-end Phase 2 pipeline:
    Target Creation -> Model Training -> Evaluation -> Early Warning System -> Plots -> Metadata.
    """
    print("=== STARTING PHASE 2: PRICE FORECASTING & EARLY WARNING SYSTEM ===")
    
    # Step 1: Load existing Phase 1 feature engineered dataset
    data_path = "data/processed/feature_engineered_modelling_dataset.parquet"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Phase 1 dataset not found at {data_path}! Please run Phase 1 first.")
        
    df_raw = pd.read_parquet(data_path)
    print(f"Loaded Phase 1 Modelling Dataset: {len(df_raw)} rows, {len(df_raw.columns)} columns.")
    
    # Step 2: Create direct forecasting targets (7D, 15D, 30D)
    df_targets = create_forecasting_targets(df_raw, target_col="price_rs_per_qtl", horizons=[7, 15, 30])
    
    # Step 3: Train LightGBM Primary Models
    print("\n--- Training LightGBM Models (7D, 15D, 30D) ---")
    lgb_forecaster = LightGBMForecaster(model_dir="models")
    lgb_forecaster.train_all_horizons(df_targets, horizons=[7, 15, 30])
    
    # Generate LightGBM predictions for full dataset
    lgb_preds = {}
    for h in [7, 15, 30]:
        lgb_preds[h] = lgb_forecaster.predict(df_targets, horizon=h)
        
    # Step 4: Train ARIMA Baseline Models
    print("\n--- Running ARIMA / SARIMAX Baseline ---")
    arima_model = ArimaBaselineModel(order=(1, 1, 1), seasonal_order=(1, 0, 0, 7))
    arima_res = arima_model.batch_predict(df_targets, horizons=[7, 15, 30])
    
    arima_preds = {
        7: arima_res[7]["arima_pred_7d"].values,
        15: arima_res[15]["arima_pred_15d"].values,
        30: arima_res[30]["arima_pred_30d"].values
    }
    
    # Step 5: Evaluate Models & Export Forecasts
    print("\n--- Evaluating Models & Exporting Forecasts ---")
    forecasts_df, metrics_table = evaluate_and_export_forecasts(
        df_targets,
        lightgbm_preds=lgb_preds,
        arima_preds=arima_preds,
        horizons=[7, 15, 30]
    )
    
    # Step 6: Early Warning System
    print("\n--- Running Early Warning System ---")
    ew_system = EarlyWarningSystem(config_path="config/model_config.yaml")
    
    # Learn thresholds strictly from 2024 historical training data
    df_targets["year"] = pd.to_datetime(df_targets["date"]).dt.year
    train_df = df_targets[df_targets["year"] == 2024]
    ew_system.fit_thresholds(train_df)
    
    # Generate warnings for full dataset
    early_warning_df = ew_system.generate_warnings(
        df_targets,
        lgb_7d=lgb_preds[7],
        lgb_15d=lgb_preds[15],
        lgb_30d=lgb_preds[30]
    )
    
    # Evaluate warnings on unseen test data (2026)
    test_ew_df = early_warning_df[pd.to_datetime(early_warning_df["date"]).dt.year == 2026]
    ew_metrics = ew_system.evaluate_warnings(test_ew_df)
    
    # Save warning metrics
    with open("models/early_warning_metrics.json", "w", encoding="utf-8") as f:
        json.dump(ew_metrics, f, indent=2)

    # Step 7: Generate Evaluation Figures
    print("\n--- Generating Analysis Figures ---")
    generate_evaluation_plots(forecasts_df, early_warning_df, metrics_table, output_dir="reports/figures")
    
    print("\n=== PHASE 2 PIPELINE EXECUTION SUCCESSFULLY COMPLETE ===")

if __name__ == "__main__":
    run_phase2_pipeline()
