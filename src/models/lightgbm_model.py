import os
import joblib
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import List, Dict, Any, Tuple

EXCLUDE_COLS = [
    "date", "state", "district", "market", "commodity", "arrival_unit",
    "primary_source", "secondary_source", "verification_status", "data_status",
    "is_suspicious", "notes", "marketing_year", "financial_year",
    "price_target_7d", "price_target_15d", "price_target_30d"
]

class LightGBMForecaster:
    """
    Primary Machine Learning Forecasting Model using LightGBM.
    Trains separate models for 7D, 15D, and 30D horizons.
    """
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.models = {}
        self.feature_cols = []

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Extract valid numeric feature columns excluding identifiers and target columns."""
        features = [c for c in df.columns if c not in EXCLUDE_COLS and not c.startswith("target_")]
        # Keep numeric types
        numeric_features = df[features].select_dtypes(include=[np.number]).columns.tolist()
        return numeric_features

    def train_horizon_model(self, df: pd.DataFrame, horizon: int, feature_cols: List[str]) -> lgb.LGBMRegressor:
        """
        Train a LightGBM model for a specific forecasting horizon (7D, 15D, or 30D)
        using chronological train/validation data.
        """
        target_col = f"price_target_{horizon}d"
        valid_data = df.dropna(subset=[target_col]).copy()
        
        # Chronological train & validation split
        valid_data["year"] = pd.to_datetime(valid_data["date"]).dt.year
        train_df = valid_data[valid_data["year"] == 2024]
        val_df = valid_data[valid_data["year"] == 2025]
        
        X_train, y_train = train_df[feature_cols], train_df[target_col]
        X_val, y_val = val_df[feature_cols], val_df[target_col]
        
        # Controlled hyperparameter optimization grid search
        best_score = float("inf")
        best_params = {}
        
        param_grid = [
            {"n_estimators": 80, "learning_rate": 0.05, "num_leaves": 15, "max_depth": 4},
            {"n_estimators": 120, "learning_rate": 0.03, "num_leaves": 31, "max_depth": 6},
            {"n_estimators": 150, "learning_rate": 0.08, "num_leaves": 20, "max_depth": 5}
        ]
        
        best_model = None
        for params in param_grid:
            model = lgb.LGBMRegressor(
                **params,
                random_state=42,
                verbosity=-1
            )
            model.fit(X_train, y_train)
            val_preds = model.predict(X_val)
            val_rmse = np.sqrt(np.mean((y_val - val_preds) ** 2))
            
            if val_rmse < best_score:
                best_score = val_rmse
                best_model = model
                best_params = params

        print(f"[LightGBMForecaster] Horizon {horizon}D Best Val RMSE: {best_score:.2f} | Params: {best_params}")
        
        # Save model artifact
        model_path = os.path.join(self.model_dir, f"lightgbm_{horizon}d.pkl")
        joblib.dump(best_model, model_path)
        self.models[horizon] = best_model
        return best_model

    def train_all_horizons(self, df: pd.DataFrame, horizons: List[int] = [7, 15, 30]) -> Dict[int, lgb.LGBMRegressor]:
        """Train LightGBM models for all horizons."""
        self.feature_cols = self.get_feature_columns(df)
        
        # Save feature columns metadata
        feat_path = os.path.join(self.model_dir, "feature_columns.json")
        with open(feat_path, "w", encoding="utf-8") as f:
            json.dump(self.feature_cols, f, indent=2)
            
        for h in horizons:
            self.train_horizon_model(df, h, self.feature_cols)
            
        return self.models

    def predict(self, df: pd.DataFrame, horizon: int) -> np.ndarray:
        """Generate predictions for a given horizon."""
        if horizon not in self.models:
            model_path = os.path.join(self.model_dir, f"lightgbm_{horizon}d.pkl")
            if os.path.exists(model_path):
                self.models[horizon] = joblib.load(model_path)
            else:
                raise FileNotFoundError(f"No LightGBM model found for {horizon}D horizon!")
                
        X = df[self.feature_cols]
        return self.models[horizon].predict(X)

    def predict_location_prices(self, state: str = None, district: str = None, market: str = None, dataset_path: str = "data/processed/feature_engineered_modelling_dataset.parquet") -> Dict[str, Any]:
        """
        Run inference on trained LightGBM models (7D, 15D, 30D) for a given location using actual feature vectors.
        Returns dictionary with predicted_7d, predicted_15d, predicted_30d, or is_available: False if historical data is insufficient.
        """
        if not os.path.exists(dataset_path):
            return {"is_available": False, "reason": "Feature engineered dataset not found"}
            
        feat_meta_path = os.path.join(self.model_dir, "feature_columns.json")
        if not os.path.exists(feat_meta_path):
            return {"is_available": False, "reason": "Feature columns metadata not found"}
            
        with open(feat_meta_path, "r", encoding="utf-8") as f:
            feature_cols = json.load(f)
            
        df_fe = pd.read_parquet(dataset_path)
        sub = df_fe.copy()
        
        if state and state != "All States":
            sub = sub[sub["state"].astype(str).str.lower() == state.lower()]
        if district and district != "All Districts":
            sub = sub[sub["district"].astype(str).str.lower() == district.lower()]
        if market and market != "All Markets":
            sub = sub[sub["market"].astype(str).str.lower() == market.lower()]
            
        if sub.empty:
            return {
                "is_available": False,
                "reason": "Prediction unavailable: insufficient historical Rice price data for this location."
            }
            
        latest_row = sub.sort_values(by="date").iloc[-1:]
        X = latest_row[feature_cols]
        
        preds = {}
        for h in [7, 15, 30]:
            model_path = os.path.join(self.model_dir, f"lightgbm_{h}d.pkl")
            if os.path.exists(model_path):
                m = joblib.load(model_path)
                pred_val = float(m.predict(X)[0])
                preds[f"predicted_price_{h}d"] = round(pred_val, 2)
            else:
                return {"is_available": False, "reason": f"Model for horizon {h}D not found"}
                
        return {
            "is_available": True,
            "predicted_price_7d": preds.get("predicted_price_7d"),
            "predicted_price_15d": preds.get("predicted_price_15d"),
            "predicted_price_30d": preds.get("predicted_price_30d")
        }
