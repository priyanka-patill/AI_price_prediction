import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def generate_phase3_plots(global_shap_df: pd.DataFrame,
                          local_shap_df: pd.DataFrame,
                          state_priority_df: pd.DataFrame,
                          recommendations_df: pd.DataFrame,
                          scenario_df: pd.DataFrame,
                          sensitivity_df: pd.DataFrame,
                          output_dir: str = "reports/figures"):
    """
    Generate 9 Phase 3 visual evaluation figures:
    1. Global SHAP Feature Importance (7D, 15D, 30D)
    2. Local SHAP Explanation Breakdown for selected market
    3. State Price Pressure Ranking
    4. Recommended Stock Release by State
    5. Available vs Remaining Central Pool Stock
    6. Transportation Cost Breakdown by State
    7. Scenario Comparison Chart
    8. Sensitivity Analysis Chart
    9. Integrated Phase 3 Decision Support Overview
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # 1. Global SHAP Feature Importance Bar Chart
    plt.figure(figsize=(10, 5))
    if not global_shap_df.empty:
        top_7d = global_shap_df[global_shap_df["forecast_horizon"] == "7D"].head(10).sort_values(by="mean_abs_shap")
        plt.barh(top_7d["feature"], top_7d["mean_abs_shap"], color="#1f77b4")
        plt.title("Top 10 Global SHAP Feature Importance (7D Forecast)", fontsize=12, fontweight="bold")
        plt.xlabel("Mean Absolute SHAP Value (₹/Qtl)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shap_global_importance_7d.png"), dpi=150)
    plt.close()

    # 2. Local SHAP Explanation Breakdown
    plt.figure(figsize=(10, 5))
    if not local_shap_df.empty:
        sample_local = local_shap_df.head(5).sort_values(by="shap_value_rs")
        colors = ["#d62728" if v < 0 else "#2ca02c" for v in sample_local["shap_value_rs"]]
        plt.barh(sample_local["feature"], sample_local["shap_value_rs"], color=colors)
        plt.axvline(0, color="black", linestyle=":", linewidth=1)
        plt.title(f"Local SHAP Price Contribution Breakdown for Sample Mandi", fontsize=12, fontweight="bold")
        plt.xlabel("SHAP Contribution Value (₹/Qtl)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shap_local_explanation_sample.png"), dpi=150)
    plt.close()

    # 3. State Price Pressure Ranking
    plt.figure(figsize=(10, 5))
    if not state_priority_df.empty:
        sp = state_priority_df.sort_values(by="price_pressure_score", ascending=True)
        colors = ["#d62728" if w == "HIGH RISK" else "#ff7f0e" if w == "WARNING" else "#2ca02c" for w in sp["warning_level"]]
        plt.barh(sp["state"], sp["price_pressure_score"], color=colors)
        plt.title("State Price Pressure Priority Ranking", fontsize=12, fontweight="bold")
        plt.xlabel("Price Pressure Score")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "state_price_pressure_ranking.png"), dpi=150)
    plt.close()

    # 4. Recommended Stock Release by State
    plt.figure(figsize=(10, 5))
    if not recommendations_df.empty:
        rec = recommendations_df.sort_values(by="recommended_release_mt", ascending=True)
        plt.barh(rec["destination_state"], rec["recommended_release_mt"], color="#2ca02c")
        plt.title("Recommended Rice Buffer Stock Release by Destination State (MT)", fontsize=12, fontweight="bold")
        plt.xlabel("Recommended Release (MT)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "recommended_release_by_state.png"), dpi=150)
    plt.close()

    # 5. Available vs Remaining Stock
    plt.figure(figsize=(8, 5))
    if not recommendations_df.empty:
        total_avail = recommendations_df["available_stock_mt"].iloc[0]
        total_rel = recommendations_df["recommended_release_mt"].sum()
        rem_stock = total_avail - total_rel
        
        plt.bar(["Available Central Stock", "Total Released Stock", "Remaining Stock"], [total_avail, total_rel, rem_stock], color=["#1f77b4", "#ff7f0e", "#2ca02c"])
        plt.title("Central Pool Rice Reserve Balance (MT)", fontsize=12, fontweight="bold")
        plt.ylabel("Quantity (MT)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "available_vs_remaining_stock.png"), dpi=150)
    plt.close()

    # 6. Transportation Cost Breakdown
    plt.figure(figsize=(10, 5))
    if not recommendations_df.empty:
        rec_t = recommendations_df.sort_values(by="transportation_cost_rs", ascending=True)
        plt.barh(rec_t["destination_state"], rec_t["transportation_cost_rs"] / 1e5, color="#9467bd")
        plt.title("Estimated Transportation Cost by State (Lakh ₹)", fontsize=12, fontweight="bold")
        plt.xlabel("Cost (Lakh ₹)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "transportation_cost_by_state.png"), dpi=150)
    plt.close()

    # 7. Scenario Comparison Chart
    plt.figure(figsize=(9, 5))
    if not scenario_df.empty:
        plt.bar(scenario_df["Scenario"], scenario_df["Total_Released_MT"], color=["#d62728", "#ff7f0e", "#2ca02c"])
        plt.title("Stock Release Quantity Across Intervention Scenarios (MT)", fontsize=12, fontweight="bold")
        plt.ylabel("Total Release (MT)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "scenario_comparison.png"), dpi=150)
    plt.close()

    # 8. Sensitivity Analysis Chart
    plt.figure(figsize=(10, 5))
    if not sensitivity_df.empty:
        plt.barh(sensitivity_df["variation"], sensitivity_df["total_released_mt"], color="#17becf")
        plt.title("Optimization Robustness / Sensitivity Analysis (Released MT)", fontsize=12, fontweight="bold")
        plt.xlabel("Total Released Stock (MT)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "optimization_sensitivity_chart.png"), dpi=150)
    plt.close()

    # 9. Integrated Phase 3 Overview
    plt.figure(figsize=(10, 5))
    plt.text(0.5, 0.5, "Phase 3: Explainable AI & Buffer Stock Optimization Complete\nSHAP Engine + PuLP MILP Solver + 3 Scenarios + 5 Robustness Tests",
             ha="center", va="center", fontsize=14, fontweight="bold", color="#1f77b4")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "phase3_overview.png"), dpi=150)
    plt.close()

    print(f"[evaluate_phase3_plots] Saved all 9 Phase 3 figures to {output_dir}")
