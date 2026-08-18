import os
import yaml
import numpy as np
import pandas as pd
import pulp
from typing import Dict, Any, List, Tuple

class BufferStockOptimizer:
    """
    Part B — Buffer Stock Optimization Engine using PuLP (MILP).
    Solves optimal stock release quantities across destination states to relieve price pressure while respecting
    central pool reserve limits and transportation costs.
    Results are explicitly framed as DECISION-SUPPORT RECOMMENDATIONS for consideration.
    """
    def __init__(self, config_path: str = "config/optimization_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {
            "optimization": {
                "minimum_reserve_percent": 0.25,
                "price_pressure_weight": 100.0,
                "transport_cost_weight": 0.05,
                "shortage_penalty_weight": 500.0
            },
            "transport_matrix": {
                "origin_hub": "Central Warehouse Hub",
                "destinations": {}
            }
        }

    def solve_optimization(self, state_priority_df: pd.DataFrame,
                          total_central_stock_mt: float = 135000.0,
                          buffer_norm_mt: float = 135000.0,
                          scenario_name: str = "Optimized") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Solve PuLP MILP buffer stock release optimization problem.
        Exports data/processed/optimization_recommendations.csv.
        """
        opt_cfg = self.config.get("optimization", {})
        min_reserve_pct = opt_cfg.get("minimum_reserve_percent", 0.25)
        p_weight = opt_cfg.get("price_pressure_weight", 100.0)
        t_weight = opt_cfg.get("transport_cost_weight", 0.05)
        shortage_penalty = opt_cfg.get("shortage_penalty_weight", 500.0)
        
        min_reserve_mt = buffer_norm_mt * min_reserve_pct
        max_allocatable_stock = max(0.0, total_central_stock_mt - min_reserve_mt)
        
        dest_matrix = self.config.get("transport_matrix", {}).get("destinations", {})
        states = state_priority_df["state"].tolist()
        
        # Define PuLP Linear Programming Problem (Minimization)
        prob = pulp.LpProblem(f"Buffer_Stock_Optimization_{scenario_name}", pulp.LpMinimize)
        
        # Decision Variables: Stock release quantity to state s (in MT)
        x = {s: pulp.LpVariable(f"release_{s}", lowBound=0, cat="Continuous") for s in states}
        shortage = {s: pulp.LpVariable(f"shortage_{s}", lowBound=0, cat="Continuous") for s in states}
        
        # Objective Function Components
        objective_terms = []
        for _, row in state_priority_df.iterrows():
            s = row["state"]
            pressure_score = row["price_pressure_score"]
            need_mt = row["estimated_need_mt"]
            
            # Transport cost calculation (Distance x Cost per MT-km)
            t_info = dest_matrix.get(s, {"distance_km": 500, "cost_per_mt_km": 2.0})
            unit_transport_cost = t_info.get("distance_km", 500) * t_info.get("cost_per_mt_km", 2.0)
            
            # Net objective term = Transport Cost - Price Pressure Benefit + Shortage Penalty
            objective_terms.append(t_weight * unit_transport_cost * x[s])
            objective_terms.append(-1.0 * p_weight * pressure_score * x[s])
            objective_terms.append(shortage_penalty * shortage[s])
            
            # Constraint: Shortage_s = Need_s - Release_s
            prob += (shortage[s] >= need_mt - x[s], f"Shortage_Def_{s}")
            # Constraint: Release_s <= Need_s
            prob += (x[s] <= need_mt, f"Max_Demand_{s}")

        # Total objective
        prob += pulp.lpSum(objective_terms), "Total_Cost_Objective"
        
        # Constraint A: Total Release <= Available Allocatable Stock
        prob += (pulp.lpSum([x[s] for s in states]) <= max_allocatable_stock, "Central_Stock_Limit")
        
        # Solve Problem silently
        solver = pulp.PULP_CBC_CMD(msg=False)
        status_code = prob.solve(solver)
        status_str = pulp.LpStatus[status_code]
        
        # Extract Results
        rec_rows = []
        total_released_mt = 0.0
        total_transport_cost_rs = 0.0
        
        for _, row in state_priority_df.iterrows():
            s = row["state"]
            rel_qty = float(pulp.value(x[s])) if status_code == 1 else 0.0
            t_info = dest_matrix.get(s, {"distance_km": 500, "cost_per_mt_km": 2.0})
            unit_cost = t_info.get("distance_km", 500) * t_info.get("cost_per_mt_km", 2.0)
            state_t_cost = rel_qty * unit_cost
            
            total_released_mt += rel_qty
            total_transport_cost_rs += state_t_cost
            
            explanation = (
                f"Recommended release for consideration: {rel_qty:,.0f} MT to {s}. "
                f"Reason: {s} exhibits {row['warning_level']} price warning with Price Pressure Score {row['price_pressure_score']:.1f}. "
                f"Respects central pool reserve stock constraints."
            )
            
            rec_rows.append({
                "date": "2026-08-01",
                "origin": self.config.get("transport_matrix", {}).get("origin_hub", "Central Warehouse Hub"),
                "destination_state": s,
                "destination_market": f"{s} Primary Mandi",
                "recommended_release_mt": round(rel_qty, 2),
                "available_stock_mt": total_central_stock_mt,
                "remaining_stock_mt": round(total_central_stock_mt - total_released_mt, 2),
                "transportation_cost_rs": round(state_t_cost, 2),
                "price_pressure_score": row["price_pressure_score"],
                "warning_level": row["warning_level"],
                "scenario": scenario_name,
                "optimization_status": status_str,
                "recommendation_explanation": explanation
            })

        rec_df = pd.DataFrame(rec_rows)
        
        summary_stats = {
            "scenario": scenario_name,
            "status": status_str,
            "total_released_mt": round(total_released_mt, 2),
            "remaining_central_stock_mt": round(total_central_stock_mt - total_released_mt, 2),
            "minimum_reserve_limit_mt": round(min_reserve_mt, 2),
            "total_transport_cost_rs": round(total_transport_cost_rs, 2)
        }
        
        if scenario_name == "Optimized":
            os.makedirs("data/processed", exist_ok=True)
            rec_df.to_csv("data/processed/optimization_recommendations.csv", index=False)
            print(f"[optimizer] Saved optimization recommendations to data/processed/optimization_recommendations.csv")
            
        return rec_df, summary_stats
