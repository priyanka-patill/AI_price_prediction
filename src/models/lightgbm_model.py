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

    def predict_location_prices(
        self,
        state: str = None,
        district: str = None,
        market: str = None,
        current_price: float = None,
        dataset_path: str = "data/processed/feature_engineered_modelling_dataset.parquet"
    ) -> Dict[str, Any]:
        """
        Run inference on trained LightGBM models (7D, 15D, 30D) using state-specific
        live/historical feature vectors.
        """
        if not os.path.exists(dataset_path):
            return {"is_available": False, "reason": "Feature engineered dataset not found"}
            
        feat_meta_path = os.path.join(self.model_dir, "feature_columns.json")
        if not os.path.exists(feat_meta_path):
            return {"is_available": False, "reason": "Feature columns metadata not found"}
            
        with open(feat_meta_path, "r", encoding="utf-8") as f:
            feature_cols = json.load(f)
            
        df_fe = pd.read_parquet(dataset_path)
        
        from src.utils.geo import standardize_state, get_state_center
        if "state" in df_fe.columns:
            df_fe["state"] = df_fe["state"].apply(standardize_state)
        
        norm_state = standardize_state(state) if state and state != "All States" else None
        norm_dist = district.strip() if district and district != "All Districts" else None
        norm_mkt = market.strip() if market and market != "All Markets" else None

        pred_level = "NATIONAL"
        pred_msg = "Prediction based on national-level historical data"
        
        # Step 1: Query exact district/market/state in feature engineered dataset
        sub = df_fe.copy()
        if norm_state:
            sub = sub[sub["state"].astype(str).str.lower() == norm_state.lower()]
        if norm_dist and not sub.empty:
            sub_dist = sub[sub["district"].astype(str).str.lower() == norm_dist.lower()]
            if not sub_dist.empty:
                sub = sub_dist
                pred_level = "DISTRICT"
                pred_msg = f"Prediction based on {norm_dist} district historical data"
        if norm_mkt and pred_level == "DISTRICT" and not sub.empty:
            sub_mkt = sub[sub["market"].astype(str).str.lower() == norm_mkt.lower()]
            if not sub_mkt.empty:
                sub = sub_mkt

        # Step 2: Try State filter if District level wasn't matched
        if sub.empty and norm_state:
            sub_st = df_fe[df_fe["state"].astype(str).str.lower() == norm_state.lower()] if "state" in df_fe.columns else pd.DataFrame()
            if not sub_st.empty:
                sub = sub_st
                pred_level = "STATE"
                pred_msg = f"Prediction based on {norm_state} state historical data"

        # Step 3: If state was not present in feature dataset, construct state-specific feature vector using live/historical price
        is_state_derived = False
        if sub.empty:
            sub = df_fe.copy()
            pred_level = "STATE_DERIVED" if norm_state else "NATIONAL"
            pred_msg = f"Prediction derived from {norm_state or 'national'} price intelligence" if norm_state else "Prediction based on national-level historical data"
            is_state_derived = True

        if sub.empty:
            return {
                "is_available": False,
                "reason": "Prediction unavailable: no feature vectors found in historical dataset."
            }

        # Compute baseline mean feature vector from dataset
        latest_dt = sub["date"].max()
        latest_rows = sub[sub["date"] == latest_dt]
        X = latest_rows[feature_cols].mean(axis=0).to_frame().T

        # Determine effective state/location price
        effective_price = current_price
        if effective_price is None or effective_price <= 0:
            # Check live_market_latest.csv or early_warning.csv for state price
            live_path = "data/processed/live_market_latest.csv"
            ew_path = "data/processed/early_warning.csv"
            if norm_state and os.path.exists(live_path):
                try:
                    df_l = pd.read_csv(live_path)
                    df_l["st_norm"] = df_l["state"].apply(standardize_state)
                    sl = df_l[df_l["st_norm"].str.lower() == norm_state.lower()]
                    if not sl.empty and "modal_price" in sl.columns:
                        effective_price = float(pd.to_numeric(sl["modal_price"], errors="coerce").dropna().mean())
                except Exception:
                    pass

            if (effective_price is None or effective_price <= 0) and norm_state and os.path.exists(ew_path):
                try:
                    df_e = pd.read_csv(ew_path)
                    df_e["st_norm"] = df_e["state"].apply(standardize_state)
                    se = df_e[df_e["st_norm"].str.lower() == norm_state.lower()]
                    if not se.empty and "current_price" in se.columns:
                        effective_price = float(pd.to_numeric(se["current_price"], errors="coerce").dropna().mean())
                except Exception:
                    pass

        # Update price and location features in vector X for state-derived inference
        if norm_state and (is_state_derived or effective_price is not None):
            center = get_state_center(norm_state)
            if "latitude" in X.columns:
                X["latitude"] = center["lat"]
            if "longitude" in X.columns:
                X["longitude"] = center["lon"]

            if effective_price is not None and effective_price > 0:
                price_cols = [c for c in X.columns if "price" in c or "lag" in c or "rolling_mean" in c]
                base_p = float(X["price_rs_per_qtl"].iloc[0]) if "price_rs_per_qtl" in X.columns and X["price_rs_per_qtl"].iloc[0] > 0 else 3400.0
                ratio = effective_price / base_p

                for pc in price_cols:
                    X[pc] = X[pc] * ratio

                if "min_price" in X.columns:
                    X["min_price"] = effective_price * 0.95
                if "max_price" in X.columns:
                    X["max_price"] = effective_price * 1.05
                if "price_rs_per_qtl" in X.columns:
                    X["price_rs_per_qtl"] = effective_price

        preds = {}
        for h in [7, 15, 30]:
            model_path = os.path.join(self.model_dir, f"lightgbm_{h}d.pkl")
            if os.path.exists(model_path):
                m = joblib.load(model_path)
                pred_val = float(m.predict(X)[0])
                preds[f"predicted_price_{h}d"] = round(pred_val, 2)
            else:
                return {"is_available": False, "reason": f"Model for horizon {h}D not found"}

        # Log prediction input and output for full audit verification
        state_label = norm_state or "National"
        eff_price_str = f"Rs.{effective_price:.2f}" if effective_price is not None else "Dataset Base"
        print(f"[LightGBM ML Engine] Prediction Input: State='{state_label}', District='{norm_dist}', Market='{norm_mkt}' | Level={pred_level} | Base Price={eff_price_str}")
        print(f"[LightGBM ML Engine] Output Predictions: 7D=Rs.{preds.get('predicted_price_7d')}, 15D=Rs.{preds.get('predicted_price_15d')}, 30D=Rs.{preds.get('predicted_price_30d')}")

        return {
            "is_available": True,
            "prediction_level": pred_level,
            "prediction_message": pred_msg,
            "predicted_price_7d": preds.get("predicted_price_7d"),
            "predicted_price_15d": preds.get("predicted_price_15d"),
            "predicted_price_30d": preds.get("predicted_price_30d")
        }
