import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from src.optimization.optimizer import BufferStockOptimizer

def run_scenario_analysis(state_priority_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run Scenario Analysis comparing 3 operational scenarios:
    1. Scenario 1: No Intervention (0 MT released)
    2. Scenario 2: Moderate Release (Fixed 50% release)
    3. Scenario 3: PuLP Optimized Release
    """
    optimizer = BufferStockOptimizer()
    
    # 1. Scenario 1: No Intervention
    df_s1, stats_s1 = optimizer.solve_optimization(state_priority_df, scenario_name="No_Intervention")
    df_s1["recommended_release_mt"] = 0.0
    df_s1["transportation_cost_rs"] = 0.0
    df_s1["scenario"] = "Scenario 1: No Intervention"
    
    # 2. Scenario 2: Moderate Release
    df_s2, stats_s2 = optimizer.solve_optimization(state_priority_df, scenario_name="Moderate_Release")
    df_s2["recommended_release_mt"] = (df_s2["recommended_release_mt"] * 0.5).round(2)
    df_s2["transportation_cost_rs"] = (df_s2["transportation_cost_rs"] * 0.5).round(2)
    df_s2["scenario"] = "Scenario 2: Moderate Release"
    
    # 3. Scenario 3: PuLP Optimized Release
    df_s3, stats_s3 = optimizer.solve_optimization(state_priority_df, scenario_name="Optimized_Release")
    df_s3["scenario"] = "Scenario 3: PuLP Optimized"
    
    scenario_comparison = pd.DataFrame([
        {
            "Scenario": "Scenario 1: No Intervention",
            "Total_Released_MT": 0.0,
            "Total_Transport_Cost_Rs": 0.0,
            "Remaining_Stock_MT": 135000.0,
            "Price_Pressure_Mitigation": "NONE (High Risk)",
            "Risk_Level": "HIGH"
        },
        {
            "Scenario": "Scenario 2: Moderate Release",
            "Total_Released_MT": float(df_s2["recommended_release_mt"].sum()),
            "Total_Transport_Cost_Rs": float(df_s2["transportation_cost_rs"].sum()),
            "Remaining_Stock_MT": 135000.0 - float(df_s2["recommended_release_mt"].sum()),
            "Price_Pressure_Mitigation": "MODERATE (Partial Risk Reduction)",
            "Risk_Level": "MEDIUM"
        },
        {
            "Scenario": "Scenario 3: PuLP Optimized",
            "Total_Released_MT": float(df_s3["recommended_release_mt"].sum()),
            "Total_Transport_Cost_Rs": float(df_s3["transportation_cost_rs"].sum()),
            "Remaining_Stock_MT": 135000.0 - float(df_s3["recommended_release_mt"].sum()),
            "Price_Pressure_Mitigation": "OPTIMAL (Maximum Risk Reduction)",
            "Risk_Level": "LOW"
        }
    ])
    
    print("[scenario_analysis] Completed 3-scenario evaluation.")
    return scenario_comparison
