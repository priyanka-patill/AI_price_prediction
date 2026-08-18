import os
import json
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple

class ShapExplainerEngine:
    """
    Part A — SHAP Explainable AI Engine.
    Uses shap.TreeExplainer to compute global and local explanations for Phase 2 LightGBM price predictions.
    Extracted feature contributions represent model correlation/feature attribution, NOT causal proof.
    """
    def __init__(self, model_dir: str = "models", config_path: str = "config/shap_config.yaml"):
        self.model_dir = model_dir
        self.models = {}
        self.feature_cols = []
        self._load_artifacts()

    def _load_artifacts(self):
        """Load trained LightGBM models and feature columns list."""
        feat_path = os.path.join(self.model_dir, "feature_columns.json")
        if os.path.exists(feat_path):
            with open(feat_path, "r", encoding="utf-8") as f:
                self.feature_cols = json.load(f)
        else:
            raise FileNotFoundError(f"Feature columns metadata not found at {feat_path}!")
            
        for h in [7, 15, 30]:
            m_path = os.path.join(self.model_dir, f"lightgbm_{h}d.pkl")
            if os.path.exists(m_path):
                self.models[h] = joblib.load(m_path)
            else:
                print(f"[ShapExplainerEngine] Warning: LightGBM model for {h}D horizon not found at {m_path}.")

    def compute_global_importance(self, df: pd.DataFrame, horizons: List[int] = [7, 15, 30]) -> pd.DataFrame:
        """
        Compute global mean absolute SHAP values across feature columns.
        Exports data/processed/shap_global_importance.csv and summary plots.
        """
        X = df[self.feature_cols].copy().fillna(0.0)
        if len(X) > 200:
            X_sample = X.sample(n=200, random_state=42)
        else:
            X_sample = X
            
        global_rows = []
        os.makedirs("reports/shap", exist_ok=True)
        
        for h in horizons:
            if h not in self.models:
                continue
                
            model = self.models[h]
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            
            # Create feature ranking dataframe
            feat_imp = pd.DataFrame({
                "feature": self.feature_cols,
                "mean_abs_shap": mean_abs_shap
            }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)
            
            feat_imp["rank"] = feat_imp.index + 1
            feat_imp["forecast_horizon"] = f"{h}D"
            
            global_rows.append(feat_imp)
            
            # Generate SHAP Summary Plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample, show=False)
            plt.title(f"SHAP Feature Importance Summary — {h}-Day Horizon", fontsize=12, fontweight="bold")
            plt.tight_layout()
            plt.savefig(f"reports/shap/shap_summary_{h}d.png", dpi=150)
            plt.close()

        if global_rows:
            global_df = pd.concat(global_rows, ignore_index=True)
            output_cols = ["forecast_horizon", "feature", "mean_abs_shap", "rank"]
            global_df = global_df[output_cols]
            
            os.makedirs("data/processed", exist_ok=True)
            global_df.to_csv("data/processed/shap_global_importance.csv", index=False)
            print(f"[ShapExplainerEngine] Saved global SHAP importance to data/processed/shap_global_importance.csv")
            return global_df
        return pd.DataFrame()

    def compute_local_explanations(self, df: pd.DataFrame, horizons: List[int] = [7, 15, 30]) -> pd.DataFrame:
        """
        Compute local SHAP explanations (+/- ₹ contribution per feature) for individual market observations.
        Exports data/processed/shap_local_explanations.csv.
        """
        valid_df = df.copy()
        if valid_df.empty:
            return pd.DataFrame()
            
        local_rows = []
        
        for h in horizons:
            if h not in self.models:
                continue
                
            model = self.models[h]
            explainer = shap.TreeExplainer(model)
            
            X = valid_df[self.feature_cols].fillna(0.0)
            shap_vals = explainer.shap_values(X)
            
            for idx in range(len(valid_df)):
                row = valid_df.iloc[idx]
                dt = row["date"]
                st = row["state"]
                dist = row["district"]
                mkt = row["market"]
                
                # Get top 5 features for this local prediction
                row_shaps = shap_vals[idx]
                top_indices = np.argsort(np.abs(row_shaps))[::-1][:5]
                
                for f_idx in top_indices:
                    feat_name = self.feature_cols[f_idx]
                    feat_val = row[feat_name]
                    s_val = float(row_shaps[f_idx])
                    
                    direction = "UPWARD PRESSURE (INCREASE)" if s_val > 0 else "DOWNWARD PRESSURE (DECREASE)"
                    
                    local_rows.append({
                        "date": dt,
                        "state": st,
                        "district": dist,
                        "market": mkt,
                        "forecast_horizon": f"{h}D",
                        "feature": feat_name,
                        "feature_value": round(float(feat_val), 2) if isinstance(feat_val, (int, float, np.number)) and not pd.isna(feat_val) else str(feat_val),
                        "shap_value_rs": round(s_val, 2),
                        "contribution_direction": direction,
                        "interpretation": f"{feat_name} ({feat_val}) contributed {s_val:+.2f} ₹/Qtl toward model prediction"
                    })

        local_df = pd.DataFrame(local_rows)
        if not local_df.empty:
            os.makedirs("data/processed", exist_ok=True)
            local_df.to_csv("data/processed/shap_local_explanations.csv", index=False)
            print(f"[ShapExplainerEngine] Saved local SHAP explanations ({len(local_df)} rows) to data/processed/shap_local_explanations.csv")
            
        return local_df
