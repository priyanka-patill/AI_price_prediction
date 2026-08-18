import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, Any, List

def generate_evaluation_plots(forecasts_df: pd.DataFrame,
                             early_warning_df: pd.DataFrame,
                             metrics_table: List[Dict[str, Any]],
                             output_dir: str = "reports/figures"):
    """
    Generate all 8 evaluation plots for Phase 2:
    1. Actual vs Predicted Price — 7D
    2. Actual vs Predicted Price — 15D
    3. Actual vs Predicted Price — 30D
    4. Forecast Error over Time
    5. Rolling Volatility
    6. Detected Price Spikes
    7. Early Warning Periods
    8. Model Comparison Bar Chart
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # 1, 2, 3: Actual vs Predicted Price (7D, 15D, 30D)
    for h in ["7D", "15D", "30D"]:
        plt.figure(figsize=(10, 5))
        sub = forecasts_df[(forecasts_df["forecast_horizon"] == h) & (forecasts_df["model"] == "LightGBM")].dropna(subset=["actual_price", "predicted_price"])
        if not sub.empty:
            sub = sub.sort_values(by="forecast_date")
            plt.plot(pd.to_datetime(sub["forecast_date"]), sub["actual_price"], label="Actual Price (₹/Qtl)", color="#1f77b4", linewidth=2)
            plt.plot(pd.to_datetime(sub["forecast_date"]), sub["predicted_price"], label=f"LightGBM {h} Predicted", color="#ff7f0e", linestyle="--", linewidth=2)
            plt.title(f"Rice Price Forecast vs Actual — {h} Horizon", fontsize=12, fontweight="bold")
            plt.xlabel("Date")
            plt.ylabel("Price (₹/Qtl)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"actual_vs_predicted_{h.lower()}.png"), dpi=150)
        plt.close()

    # 4. Forecast Error over Time
    plt.figure(figsize=(10, 5))
    sub_err = forecasts_df[forecasts_df["model"] == "LightGBM"].dropna(subset=["error"])
    if not sub_err.empty:
        for h, grp in sub_err.groupby("forecast_horizon"):
            grp = grp.sort_values(by="forecast_date")
            plt.plot(pd.to_datetime(grp["forecast_date"]), grp["error"], label=f"Error {h}", alpha=0.7)
        plt.axhline(0, color="black", linestyle=":", linewidth=1)
        plt.title("LightGBM Forecast Error over Time (₹/Qtl)", fontsize=12, fontweight="bold")
        plt.xlabel("Date")
        plt.ylabel("Forecast Error (Predicted - Actual)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "forecast_error_over_time.png"), dpi=150)
    plt.close()

    # 5. Rolling Volatility
    plt.figure(figsize=(10, 5))
    if "rolling_volatility" in early_warning_df.columns:
        sub_vol = early_warning_df.sort_values(by="date")
        plt.plot(pd.to_datetime(sub_vol["date"]), sub_vol["rolling_volatility"], color="#9467bd", linewidth=1.8)
        plt.axhline(0.015, color="red", linestyle="--", label="Elevated Volatility Threshold (0.015)")
        plt.title("30-Day Rolling Price Return Volatility", fontsize=12, fontweight="bold")
        plt.xlabel("Date")
        plt.ylabel("Volatility")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "rolling_volatility.png"), dpi=150)
    plt.close()

    # 6. Detected Price Spikes
    plt.figure(figsize=(10, 5))
    if "spike_score" in early_warning_df.columns:
        sub_spike = early_warning_df.sort_values(by="date")
        plt.plot(pd.to_datetime(sub_spike["date"]), sub_spike["spike_score"], color="#2ca02c", linewidth=1.5)
        plt.axhline(1.2, color="orange", linestyle="--", label="Warning Spike Threshold (1.2)")
        plt.axhline(2.0, color="red", linestyle="--", label="High Risk Spike Threshold (2.0)")
        plt.title("Standardized Price Spike Score (Z-score)", fontsize=12, fontweight="bold")
        plt.xlabel("Date")
        plt.ylabel("Spike Score (Z-score)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "detected_price_spikes.png"), dpi=150)
    plt.close()

    # 7. Early Warning Periods
    plt.figure(figsize=(10, 5))
    if "warning_level" in early_warning_df.columns:
        sub_warn = early_warning_df.sort_values(by="date")
        colors = {"NORMAL": "#2ca02c", "WARNING": "#ff7f0e", "HIGH RISK": "#d62728"}
        for level in ["NORMAL", "WARNING", "HIGH RISK"]:
            subset = sub_warn[sub_warn["warning_level"] == level]
            if not subset.empty:
                plt.scatter(pd.to_datetime(subset["date"]), subset["current_price"], label=level, color=colors[level], s=25, alpha=0.8)
        plt.title("Early Warning Alert Periods mapped to Rice Prices", fontsize=12, fontweight="bold")
        plt.xlabel("Date")
        plt.ylabel("Current Price (₹/Qtl)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "early_warning_periods.png"), dpi=150)
    plt.close()

    # 8. Model Comparison Bar Chart
    plt.figure(figsize=(10, 5))
    if metrics_table:
        m_df = pd.DataFrame(metrics_table)
        pivot_mae = m_df.pivot(index="Horizon", columns="Model", values="MAE")
        pivot_mae.plot(kind="bar", figsize=(10, 5), width=0.7)
        plt.title("Model MAE Comparison across Forecast Horizons", fontsize=12, fontweight="bold")
        plt.xlabel("Forecast Horizon")
        plt.ylabel("Mean Absolute Error (₹/Qtl)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "model_comparison_bar_chart.png"), dpi=150)
    plt.close()

    print(f"[evaluate_plots] Saved all 8 evaluation figures to {output_dir}")
