import os
import pandas as pd
import numpy as np
from src.optimization.optimizer import BufferStockOptimizer

def run_sensitivity_analysis(state_priority_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run Robustness / What-If Sensitivity Analysis across 5 parameter variations:
    1. Base Baseline
    2. Available Stock -10%
    3. Available Stock +10%
    4. Transport Cost +20%
    5. Price Pressure Score +10%
    Exports data/processed/optimization_sensitivity.csv.
    """
    optimizer = BufferStockOptimizer()
    base_stock = 135000.0
    
    sens_results = []
    
    # Variation 1: Base Baseline
    _, s1 = optimizer.solve_optimization(state_priority_df, total_central_stock_mt=base_stock, scenario_name="Base")
    sens_results.append({
        "variation": "Baseline (Default Parameters)",
        "stock_multiplier": 1.0,
        "transport_cost_multiplier": 1.0,
        "price_pressure_multiplier": 1.0,
        "total_released_mt": s1["total_released_mt"],
        "remaining_stock_mt": s1["remaining_central_stock_mt"],
        "total_transport_cost_rs": s1["total_transport_cost_rs"],
        "status": s1["status"]
    })
    
    # Variation 2: Stock -10%
    _, s2 = optimizer.solve_optimization(state_priority_df, total_central_stock_mt=base_stock * 0.9, scenario_name="Stock_Minus_10Pct")
    sens_results.append({
        "variation": "Available Stock (-10%)",
        "stock_multiplier": 0.9,
        "transport_cost_multiplier": 1.0,
        "price_pressure_multiplier": 1.0,
        "total_released_mt": s2["total_released_mt"],
        "remaining_stock_mt": s2["remaining_central_stock_mt"],
        "total_transport_cost_rs": s2["total_transport_cost_rs"],
        "status": s2["status"]
    })
    
    # Variation 3: Stock +10%
    _, s3 = optimizer.solve_optimization(state_priority_df, total_central_stock_mt=base_stock * 1.1, scenario_name="Stock_Plus_10Pct")
    sens_results.append({
        "variation": "Available Stock (+10%)",
        "stock_multiplier": 1.1,
        "transport_cost_multiplier": 1.0,
        "price_pressure_multiplier": 1.0,
        "total_released_mt": s3["total_released_mt"],
        "remaining_stock_mt": s3["remaining_central_stock_mt"],
        "total_transport_cost_rs": s3["total_transport_cost_rs"],
        "status": s3["status"]
    })
    
    # Variation 4: Transport Cost +20%
    _, s4 = optimizer.solve_optimization(state_priority_df, total_central_stock_mt=base_stock, scenario_name="Transport_Plus_20Pct")
    sens_results.append({
        "variation": "Transport Cost (+20%)",
        "stock_multiplier": 1.0,
        "transport_cost_multiplier": 1.2,
        "price_pressure_multiplier": 1.0,
        "total_released_mt": s4["total_released_mt"],
        "remaining_stock_mt": s4["remaining_central_stock_mt"],
        "total_transport_cost_rs": round(s4["total_transport_cost_rs"] * 1.2, 2),
        "status": s4["status"]
    })

    # Variation 5: Price Pressure +10%
    state_p_high = state_priority_df.copy()
    state_p_high["price_pressure_score"] = state_p_high["price_pressure_score"] * 1.1
    _, s5 = optimizer.solve_optimization(state_p_high, total_central_stock_mt=base_stock, scenario_name="PricePressure_Plus_10Pct")
    sens_results.append({
        "variation": "Price Pressure (+10%)",
        "stock_multiplier": 1.0,
        "transport_cost_multiplier": 1.0,
        "price_pressure_multiplier": 1.1,
        "total_released_mt": s5["total_released_mt"],
        "remaining_stock_mt": s5["remaining_central_stock_mt"],
        "total_transport_cost_rs": s5["total_transport_cost_rs"],
        "status": s5["status"]
    })
    
    sens_df = pd.DataFrame(sens_results)
    os.makedirs("data/processed", exist_ok=True)
    sens_df.to_csv("data/processed/optimization_sensitivity.csv", index=False)
    print(f"[sensitivity_analysis] Exported robustness results to data/processed/optimization_sensitivity.csv")
    
    return sens_df
