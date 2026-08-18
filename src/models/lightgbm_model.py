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
