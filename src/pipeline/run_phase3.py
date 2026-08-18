import os
import json
import pandas as pd
import numpy as np

# Phase 3 Modules
from src.explainability.shap_engine import ShapExplainerEngine
from src.optimization.price_pressure import compute_state_price_pressure
from src.optimization.optimizer import BufferStockOptimizer
from src.optimization.scenario_analysis import run_scenario_analysis
from src.optimization.sensitivity_analysis import run_sensitivity_analysis
from src.visualization.evaluate_phase3_plots import generate_phase3_plots

def run_phase3_pipeline():
    """
    Execute end-to-end Phase 3 pipeline:
    SHAP Global/Local Explanations -> Price Pressure Ranking -> PuLP Buffer Stock Optimization -> Scenarios -> Sensitivity -> Plots.
    """
    print("=== STARTING PHASE 3: EXPLAINABLE AI & BUFFER STOCK OPTIMIZATION ===")
    
    # Load Phase 1 & 2 Datasets
    data_path = "data/processed/feature_engineered_modelling_dataset.parquet"
    forecast_path = "data/processed/forecasts.csv"
    warning_path = "data/processed/early_warning.csv"
    
    if not (os.path.exists(data_path) and os.path.exists(forecast_path) and os.path.exists(warning_path)):
        raise FileNotFoundError("Required Phase 1 or Phase 2 datasets missing! Please verify previous phases.")
        
    df_raw = pd.read_parquet(data_path)
    forecasts_df = pd.read_csv(forecast_path)
    early_warning_df = pd.read_csv(warning_path)
    
    print(f"Loaded Phase 1 Modelling Dataset: {len(df_raw)} rows.")
    print(f"Loaded Forecasts: {len(forecasts_df)} rows.")
    print(f"Loaded Early Warning Alerts: {len(early_warning_df)} rows.")
    
    # -------------------------------------------------------------
    # PART A — SHAP EXPLAINABLE AI
    # -------------------------------------------------------------
    print("\n--- Running PART A: SHAP Explainable AI ---")
    shap_engine = ShapExplainerEngine(model_dir="models", config_path="config/shap_config.yaml")
    
    global_shap_df = shap_engine.compute_global_importance(df_raw, horizons=[7, 15, 30])
    local_shap_df = shap_engine.compute_local_explanations(df_raw, horizons=[7, 15, 30])
    
    # -------------------------------------------------------------
    # PART B — BUFFER STOCK OPTIMIZATION
    # -------------------------------------------------------------
    print("\n--- Running PART B: Buffer Stock Optimization (PuLP MILP) ---")
    
    # Step 1: Compute State Price Pressure & Priority
    state_priority_df = compute_state_price_pressure(forecasts_df, early_warning_df)
    
    # Step 2: PuLP MILP Optimization Solver
    optimizer = BufferStockOptimizer(config_path="config/optimization_config.yaml")
    recommendations_df, summary_stats = optimizer.solve_optimization(state_priority_df, total_central_stock_mt=135000.0, scenario_name="Optimized")
    
    # Step 3: 3-Scenario Analysis
    scenario_df = run_scenario_analysis(state_priority_df)
    
    # Step 4: Robustness / Sensitivity Analysis
    sensitivity_df = run_sensitivity_analysis(state_priority_df)
    
    # -------------------------------------------------------------
    # VISUALIZATIONS
    # -------------------------------------------------------------
    print("\n--- Generating Phase 3 Analysis Figures ---")
    generate_phase3_plots(
        global_shap_df=global_shap_df,
        local_shap_df=local_shap_df,
        state_priority_df=state_priority_df,
        recommendations_df=recommendations_df,
        scenario_df=scenario_df,
        sensitivity_df=sensitivity_df,
        output_dir="reports/figures"
    )
    
    print("\n=== PHASE 3 PIPELINE EXECUTION SUCCESSFULLY COMPLETE ===")

if __name__ == "__main__":
    run_phase3_pipeline()
