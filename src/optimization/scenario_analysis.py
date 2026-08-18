import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

def compute_worst_case_price_trajectory(
    base_price: float,
    predicted_7d: float = None,
    predicted_15d: float = None,
    predicted_30d: float = None,
    shock_factor: float = 0.18
) -> Dict[str, Any]:
    """
    Compute baseline vs worst-case (supply/monsoon shock) multi-horizon price trajectories.
    """
    if predicted_7d is None:
        predicted_7d = base_price * 1.04
    if predicted_15d is None:
        predicted_15d = base_price * 1.08
    if predicted_30d is None:
        predicted_30d = base_price * 1.12

    # Baseline trajectory
    baseline_trajectory = {
        "current": round(base_price, 2),
        "7d": round(predicted_7d, 2),
        "15d": round(predicted_15d, 2),
        "30d": round(predicted_30d, 2)
    }

    # Worst-case trajectory (+18% to +25% supply shock surge)
    worst_case_trajectory = {
        "current": round(base_price * 1.02, 2),
        "7d": round(predicted_7d * (1 + shock_factor * 0.7), 2),
        "15d": round(predicted_15d * (1 + shock_factor * 0.9), 2),
        "30d": round(predicted_30d * (1 + shock_factor * 1.1), 2)
    }

    return {
        "baseline": baseline_trajectory,
        "worst_case": worst_case_trajectory,
        "shock_factor_pct": round(shock_factor * 100, 1)
    }

def compute_required_release_range(
    unmitigated_price: float,
    ceiling_price: float = 3300.0,
    available_stock_mt: float = 135000.0,
    min_reserve_pct: float = 0.25,
    price_sensitivity_per_1000mt: float = 35.0  # ₹35 per 1,000 MT release
) -> Dict[str, Any]:
    """
    Compute required buffer stock release range (Min, Optimal, Max Safe) to maintain price under ceiling.
    """
    price_excess = max(0.0, unmitigated_price - ceiling_price)

    # Minimum release required to bring price down to ceiling
    if price_excess > 0:
        min_release_mt = (price_excess / price_sensitivity_per_1000mt) * 1000.0
    else:
        min_release_mt = 0.0

    # Optimal release to bring price 5% safely below ceiling
    target_safe_price = ceiling_price * 0.95
    optimal_excess = max(0.0, unmitigated_price - target_safe_price)
    optimal_release_mt = (optimal_excess / price_sensitivity_per_1000mt) * 1000.0

    # Maximum safe release respecting pool reserve floor
    max_safe_release_mt = max(0.0, available_stock_mt * (1.0 - min_reserve_pct))

    # Cap releases to max safe
    min_release_mt = round(min(min_release_mt, max_safe_release_mt), 0)
    optimal_release_mt = round(min(max(optimal_release_mt, min_release_mt), max_safe_release_mt), 0)
    max_safe_release_mt = round(max_safe_release_mt, 0)

    is_alert_triggered = unmitigated_price > ceiling_price

    return {
        "unmitigated_price": round(unmitigated_price, 2),
        "ceiling_price": round(ceiling_price, 2),
        "price_excess_rs": round(price_excess, 2),
        "alert_triggered": is_alert_triggered,
        "min_release_mt": min_release_mt,
        "optimal_release_mt": optimal_release_mt,
        "max_safe_release_mt": max_safe_release_mt,
        "available_stock_mt": round(available_stock_mt, 0),
        "remaining_reserve_mt": round(available_stock_mt - optimal_release_mt, 0)
    }

def simulate_price_mitigation(
    unmitigated_trajectory: Dict[str, float],
    release_mt: float,
    price_sensitivity_per_1000mt: float = 35.0
) -> Dict[str, float]:
    """
    Simulate mitigated price trajectory across horizons for a given buffer stock release (MT).
    """
    price_reduction = (release_mt / 1000.0) * price_sensitivity_per_1000mt
    mitigated_trajectory = {}

    for k, price in unmitigated_trajectory.items():
        # Current price has minimal immediate impact, future horizons feel full effect
        horizon_weight = 0.2 if k == "current" else (0.7 if k == "7d" else (0.9 if k == "15d" else 1.0))
        mitigated_p = max(1800.0, price - (price_reduction * horizon_weight))
        mitigated_trajectory[k] = round(mitigated_p, 2)

    return mitigated_trajectory

def calculate_section_wise_release(
    total_release_mt: float,
    state_priority_df: pd.DataFrame = None
) -> List[Dict[str, Any]]:
    """
    Calculate section / region / state wise distribution of buffer stock release.
    """
    sections = [
        {"section_id": "SEC-1", "name": "North Section (Uttar Pradesh & Punjab Hubs)", "weight": 0.35},
        {"section_id": "SEC-2", "name": "East Section (West Bengal & Bihar Hubs)", "weight": 0.28},
        {"section_id": "SEC-3", "name": "West Section (Maharashtra & Gujarat Hubs)", "weight": 0.22},
        {"section_id": "SEC-4", "name": "South Section (Andhra Pradesh & TN Hubs)", "weight": 0.15}
    ]

    result = []
    accumulated = 0.0

    for idx, sec in enumerate(sections):
        if idx == len(sections) - 1:
            sec_release = round(total_release_mt - accumulated, 0)
        else:
            sec_release = round(total_release_mt * sec["weight"], 0)
            accumulated += sec_release

        result.append({
            "section_id": sec["section_id"],
            "section_name": sec["name"],
            "allocated_release_mt": sec_release,
            "share_percent": round((sec_release / max(1.0, total_release_mt)) * 100, 1),
            "estimated_freight_lakhs": round((sec_release * 500 * 2.0) / 1e5, 2)
        })

    return result

def run_scenario_analysis(state_priority_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run Scenario Analysis comparing operational scenarios.
    """
    from src.optimization.optimizer import BufferStockOptimizer
    optimizer = BufferStockOptimizer()

    df_s1, stats_s1 = optimizer.solve_optimization(state_priority_df, scenario_name="No_Intervention")
    df_s1["recommended_release_mt"] = 0.0
    df_s1["transportation_cost_rs"] = 0.0
    df_s1["scenario"] = "Scenario 1: No Intervention"

    df_s2, stats_s2 = optimizer.solve_optimization(state_priority_df, scenario_name="Moderate_Release")
    df_s2["recommended_release_mt"] = (df_s2["recommended_release_mt"] * 0.5).round(2)
    df_s2["transportation_cost_rs"] = (df_s2["transportation_cost_rs"] * 0.5).round(2)
    df_s2["scenario"] = "Scenario 2: Moderate Release"

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

    print("[scenario_analysis] Completed scenario evaluation.")
    return scenario_comparison

