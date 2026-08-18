import os
import pandas as pd
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["Buffer Stock Optimization"])

class OptimizationItem(BaseModel):
    destination_state: str
    destination_market: str
    recommended_release_mt: float
    available_stock_mt: float
    remaining_stock_mt: float
    transportation_cost_rs: float
    price_pressure_score: float
    warning_level: str
    scenario: str
    explanation: str

class ScenarioItem(BaseModel):
    scenario: str
    total_released_mt: float
    transportation_cost_rs: float
    remaining_stock_mt: float
    risk_level: str

@router.get("/optimization", response_model=List[OptimizationItem], summary="Get PuLP Stock Release Decision Recommendations")
def get_optimization_recommendations(
    state: Optional[str] = Query(None)
):
    opt_path = "data/processed/optimization_recommendations.csv"
    if not os.path.exists(opt_path):
        raise HTTPException(status_code=404, detail="Optimization recommendations dataset not found.")
        
    df_opt = pd.read_csv(opt_path)
    
    if state:
        df_opt = df_opt[df_opt["destination_state"].str.lower() == state.lower()]

    results = []
    for _, row in df_opt.iterrows():
        results.append(OptimizationItem(
            destination_state=str(row["destination_state"]),
            destination_market=str(row["destination_market"]),
            recommended_release_mt=round(float(row["recommended_release_mt"]), 2),
            available_stock_mt=round(float(row["available_stock_mt"]), 2),
            remaining_stock_mt=round(float(row["remaining_stock_mt"]), 2),
            transportation_cost_rs=round(float(row["transportation_cost_rs"]), 2),
            price_pressure_score=round(float(row["price_pressure_score"]), 2),
            warning_level=str(row["warning_level"]),
            scenario=str(row["scenario"]),
            explanation=str(row["recommendation_explanation"])
        ))
    return results

@router.get("/scenarios", response_model=List[ScenarioItem], summary="Get Intervention Scenario Comparisons")
def get_scenario_comparison():
    opt_path = "data/processed/optimization_recommendations.csv"
    if not os.path.exists(opt_path):
        return []
        
    df_opt = pd.read_csv(opt_path)
    total_opt_release = float(df_opt["recommended_release_mt"].sum())
    total_opt_cost = float(df_opt["transportation_cost_rs"].sum())
    
    scenarios = [
        ScenarioItem(
            scenario="Scenario 1: No Intervention",
            total_released_mt=0.0,
            transportation_cost_rs=0.0,
            remaining_stock_mt=135000.0,
            risk_level="HIGH"
        ),
        ScenarioItem(
            scenario="Scenario 2: Moderate Release",
            total_released_mt=round(total_opt_release * 0.5, 2),
            transportation_cost_rs=round(total_opt_cost * 0.5, 2),
            remaining_stock_mt=round(135000.0 - total_opt_release * 0.5, 2),
            risk_level="MEDIUM"
        ),
        ScenarioItem(
            scenario="Scenario 3: PuLP Optimized",
            total_released_mt=round(total_opt_release, 2),
            transportation_cost_rs=round(total_opt_cost, 2),
            remaining_stock_mt=round(135000.0 - total_opt_release, 2),
            risk_level="LOW"
        )
    ]
    return scenarios

@router.get("/scenario-simulator", summary="Get Worst-Case Scenario Simulation & Buffer Stock Decision Support")
def get_scenario_simulator(
    base_price: float = Query(2829.23),
    ceiling_price: float = Query(3300.0),
    simulated_release_mt: float = Query(5000.0),
    scenario: str = Query("worst_case")
):
    from src.optimization.scenario_analysis import (
        compute_worst_case_price_trajectory,
        compute_required_release_range,
        simulate_price_mitigation,
        calculate_section_wise_release
    )
    
    trajectories = compute_worst_case_price_trajectory(
        base_price=base_price,
        predicted_7d=base_price * 1.05,
        predicted_15d=base_price * 1.10,
        predicted_30d=base_price * 1.15
    )
    
    selected_trajectory = trajectories["worst_case"] if scenario == "worst_case" else trajectories["baseline"]
    unmitigated_peak_price = selected_trajectory["30d"]
    
    release_range = compute_required_release_range(
        unmitigated_price=unmitigated_peak_price,
        ceiling_price=ceiling_price
    )
    
    mitigated_trajectory = simulate_price_mitigation(
        unmitigated_trajectory=selected_trajectory,
        release_mt=simulated_release_mt
    )
    
    section_breakdown = calculate_section_wise_release(
        total_release_mt=simulated_release_mt
    )
    
    return {
        "scenario": scenario,
        "base_price": base_price,
        "ceiling_price": ceiling_price,
        "simulated_release_mt": simulated_release_mt,
        "unmitigated_trajectory": selected_trajectory,
        "mitigated_trajectory": mitigated_trajectory,
        "release_range": release_range,
        "sections": section_breakdown
    }

