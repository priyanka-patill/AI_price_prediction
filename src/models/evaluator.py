import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate MAE, RMSE, MAPE, sMAPE, and R2 regression metrics.
    Handles zero/small price values safely for MAPE.
    """
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    
    if len(yt) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "sMAPE": np.nan, "R2": np.nan}
        
    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    
    # Safe MAPE: handle division by zero
    safe_denom = np.where(np.abs(yt) < 1e-5, np.nan, np.abs(yt))
    mape = float(np.nanmean(np.abs((yt - yp) / safe_denom)) * 100.0)
    
    # Symmetric MAPE (sMAPE)
    smape_denom = (np.abs(yt) + np.abs(yp)) / 2.0
    smape_denom = np.where(smape_denom < 1e-5, np.nan, smape_denom)
    smape = float(np.nanmean(np.abs(yt - yp) / smape_denom) * 100.0)
    
    # R2 Score
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    
    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE": round(mape, 2),
        "sMAPE": round(smape, 2),
        "R2": round(r2, 4)
    }

def evaluate_and_export_forecasts(df: pd.DataFrame,
                                  lightgbm_preds: Dict[int, np.ndarray],
                                  arima_preds: Dict[int, np.ndarray],
                                  horizons: List[int] = [7, 15, 30]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Evaluate Naive, ARIMA, and LightGBM across 7D, 15D, and 30D horizons.
    Export data/processed/forecasts.csv and models/model_metrics.json.
    """
    df["year"] = pd.to_datetime(df["date"]).dt.year
    test_mask = (df["year"] == 2026).values # Final unseen test set evaluation
    test_df = df[test_mask].copy()
    
    metrics_table = []
    forecast_rows = []
    
    for h in horizons:
        target_col = f"price_target_{h}d"
        if target_col not in test_df.columns:
            continue
            
        y_true = test_df[target_col].values
        
        # 1. Naive Baseline (Predict Price(T))
        y_naive = test_df["price_rs_per_qtl"].values
        m_naive = calculate_metrics(y_true, y_naive)
        m_naive.update({"Model": "Naive Baseline", "Horizon": f"{h}D"})
        metrics_table.append(m_naive)
        
        # 2. ARIMA Baseline
        full_arima = arima_preds.get(h, df["price_rs_per_qtl"].values)
        y_arima = full_arima[test_mask]
        m_arima = calculate_metrics(y_true, y_arima)
        m_arima.update({"Model": "ARIMA Baseline", "Horizon": f"{h}D"})
        metrics_table.append(m_arima)
        
        # 3. LightGBM Primary Model
        full_lgb = lightgbm_preds.get(h, df["price_rs_per_qtl"].values)
        y_lgb = full_lgb[test_mask]
        m_lgb = calculate_metrics(y_true, y_lgb)
        m_lgb.update({"Model": "LightGBM", "Horizon": f"{h}D"})
        metrics_table.append(m_lgb)
        
        # Collect detailed predictions for forecasts.csv
        for idx in range(len(test_df)):
            row = test_df.iloc[idx]
            dt = row["date"]
            loc = f"{row['state']}|{row['district']}|{row['market']}"
            actual = y_true[idx]
            
            # Record LightGBM forecast
            forecast_rows.append({
                "forecast_date": dt,
                "location": loc,
                "actual_price": actual if not np.isnan(actual) else None,
                "predicted_price": round(float(y_lgb[idx]), 2) if not np.isnan(y_lgb[idx]) else None,
                "forecast_horizon": f"{h}D",
                "model": "LightGBM",
                "error": round(float(y_lgb[idx] - actual), 2) if not (np.isnan(actual) or np.isnan(y_lgb[idx])) else None
            })
            
            # Record ARIMA forecast
            forecast_rows.append({
                "forecast_date": dt,
                "location": loc,
                "actual_price": actual if not np.isnan(actual) else None,
                "predicted_price": round(float(y_arima[idx]), 2) if not np.isnan(y_arima[idx]) else None,
                "forecast_horizon": f"{h}D",
                "model": "ARIMA Baseline",
                "error": round(float(y_arima[idx] - actual), 2) if not (np.isnan(actual) or np.isnan(y_arima[idx])) else None
            })

    # Save forecasts.csv
    forecasts_df = pd.DataFrame(forecast_rows)
    out_forecast_path = "data/processed/forecasts.csv"
    os.makedirs(os.path.dirname(out_forecast_path), exist_ok=True)
    forecasts_df.to_csv(out_forecast_path, index=False)
    print(f"[evaluator] Exported {len(forecasts_df)} forecast records to {out_forecast_path}")
    
    # Save model_metrics.json
    out_metrics_path = "models/model_metrics.json"
    with open(out_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_table, f, indent=2)
    print(f"[evaluator] Saved model metrics comparison to {out_metrics_path}")
    
    return forecasts_df, metrics_table
