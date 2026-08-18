import pytest
import pandas as pd
import numpy as np
from src.optimization.optimizer import BufferStockOptimizer
from src.optimization.scenario_analysis import run_scenario_analysis
from src.optimization.sensitivity_analysis import run_sensitivity_analysis

def test_pulp_buffer_stock_optimization():
    state_priority_df = pd.DataFrame({
        "priority_rank": [1, 2],
        "state": ["Punjab", "Maharashtra"],
        "price_pressure_score": [2.5, 8.4],
        "forecast_7d": [3200, 3600],
        "forecast_15d": [3250, 3650],
        "forecast_30d": [3300, 3700],
        "warning_level": ["NORMAL", "HIGH RISK"],
        "stock_available_mt": [15000, 15000],
        "estimated_need_mt": [5000, 15000]
    })
    
    optimizer = BufferStockOptimizer()
    rec_df, summary = optimizer.solve_optimization(state_priority_df, total_central_stock_mt=50000.0, buffer_norm_mt=40000.0, scenario_name="Test")
    
    assert summary["status"] in ["Optimal", "1"]
    assert "recommended_release_mt" in rec_df.columns
    assert (rec_df["recommended_release_mt"] >= 0).all()
    assert summary["remaining_central_stock_mt"] >= summary["minimum_reserve_limit_mt"]
    
    # Test Scenario Analysis
    scenario_comp = run_scenario_analysis(state_priority_df)
    assert len(scenario_comp) == 3
    
    # Test Sensitivity Analysis
    sens_df = run_sensitivity_analysis(state_priority_df)
    assert len(sens_df) == 5

def test_worst_case_scenario_helpers():
    from src.optimization.scenario_analysis import (
        compute_worst_case_price_trajectory,
        compute_required_release_range,
        simulate_price_mitigation,
        calculate_section_wise_release
    )
    
    # 1. Trajectories
    traj = compute_worst_case_price_trajectory(base_price=3000.0, shock_factor=0.20)
    assert traj["worst_case"]["30d"] > traj["baseline"]["30d"]
    assert traj["worst_case"]["30d"] >= 3000.0 * 1.20
    
    # 2. Release Range & Alerts
    rr = compute_required_release_range(unmitigated_price=3650.0, ceiling_price=3300.0)
    assert rr["alert_triggered"] is True
    assert rr["price_excess_rs"] == 350.0
    assert rr["min_release_mt"] > 0
    assert rr["optimal_release_mt"] >= rr["min_release_mt"]
    assert rr["max_safe_release_mt"] <= 135000.0 * 0.75
    
    # 3. Mitigation Simulation
    mit = simulate_price_mitigation(unmitigated_trajectory=traj["worst_case"], release_mt=10000.0)
    assert mit["30d"] < traj["worst_case"]["30d"]
    
    # 4. Section Breakdown
    sections = calculate_section_wise_release(total_release_mt=10000.0)
    assert len(sections) == 4
    total_allocated = sum(s["allocated_release_mt"] for s in sections)
    assert abs(total_allocated - 10000.0) < 1.0

