import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List

class EarlyWarningSystem:
    """
    Part B — Early Warning System for detecting abnormal price increases, spikes, and volatility.
    Generates warning levels: NORMAL, WARNING, HIGH RISK.
    Thresholds are learned strictly from historical training data (2024).
    """
    def __init__(self, config_path: str = "config/model_config.yaml"):
        self.source_name = "AI Price Intelligence Early Warning Engine"
        self.thresholds = {}

    def fit_thresholds(self, train_df: pd.DataFrame) -> Dict[str, float]:
        """
        Derive data-driven spike and volatility thresholds exclusively from historical training data (2024).
        """
        df = train_df.copy()
        
        # Daily return and rolling volatility
        df["daily_return"] = df.groupby(["state", "district", "market"])["price_rs_per_qtl"].pct_change()
        df["rolling_volatility_30d"] = df.groupby(["state", "district", "market"])["daily_return"].transform(lambda s: s.rolling(30, min_periods=5).std())
        
        # Z-score price deviation
        p_mean = df["price_rolling_mean_30"] if "price_rolling_mean_30" in df.columns else df["price_rs_per_qtl"]
        p_std = df["price_rolling_std_30"] if "price_rolling_std_30" in df.columns else df["price_rs_per_qtl"].std()
        df["z_score"] = (df["price_rs_per_qtl"] - p_mean) / p_std.replace(0, np.nan)
        
        # Learn percentiles from training data
        z_warning = float(np.nanpercentile(df["z_score"], 80)) if not df["z_score"].dropna().empty else 1.2
        z_high_risk = float(np.nanpercentile(df["z_score"], 95)) if not df["z_score"].dropna().empty else 2.0
        
        vol_high = float(np.nanpercentile(df["rolling_volatility_30d"], 85)) if not df["rolling_volatility_30d"].dropna().empty else 0.015
        
        self.thresholds = {
            "zscore_warning": round(max(z_warning, 1.0), 2),
            "zscore_high_risk": round(max(z_high_risk, 1.8), 2),
            "expected_change_percent_warning": 3.0,
            "expected_change_percent_high_risk": 6.0,
            "volatility_threshold_high": round(max(vol_high, 0.01), 4)
        }
        
        # Save threshold artifacts
        thresh_path = "models/warning_thresholds.json"
        os.makedirs(os.path.dirname(thresh_path), exist_ok=True)
        with open(thresh_path, "w", encoding="utf-8") as f:
            json.dump(self.thresholds, f, indent=2)
            
        print(f"[EarlyWarningSystem] Learned thresholds from training data: {self.thresholds}")
        return self.thresholds

    def generate_warnings(self, df: pd.DataFrame,
                          lgb_7d: np.ndarray,
                          lgb_15d: np.ndarray,
                          lgb_30d: np.ndarray) -> pd.DataFrame:
        """
        Generate early warning levels (NORMAL, WARNING, HIGH RISK) for each market observation.
        """
        if not self.thresholds:
            # Fallback default thresholds
            self.thresholds = {
                "zscore_warning": 1.2,
                "zscore_high_risk": 2.0,
                "expected_change_percent_warning": 3.0,
                "expected_change_percent_high_risk": 6.0,
                "volatility_threshold_high": 0.015
            }
            
        df = df.copy()
        df["daily_return"] = df.groupby(["state", "district", "market"])["price_rs_per_qtl"].pct_change()
        df["rolling_volatility"] = df.groupby(["state", "district", "market"])["daily_return"].transform(lambda s: s.rolling(30, min_periods=1).std()).fillna(0.0)
        
        # Compute Spike Score (Z-score)
        p_mean = df["price_rolling_mean_30"] if "price_rolling_mean_30" in df.columns else df["price_rs_per_qtl"]
        p_std = df["price_rolling_std_30"] if "price_rolling_std_30" in df.columns else 50.0
        df["spike_score"] = ((df["price_rs_per_qtl"] - p_mean) / p_std.replace(0, np.nan)).fillna(0.0)
        
        df["forecast_7d"] = lgb_7d
        df["forecast_15d"] = lgb_15d
        df["forecast_30d"] = lgb_30d
        
        # Expected % change after 7 days
        df["expected_change_percent"] = ((df["forecast_7d"] - df["price_rs_per_qtl"]) / df["price_rs_per_qtl"]) * 100.0
        
        warning_levels = []
        warning_reasons = []
        
        zw = self.thresholds["zscore_warning"]
        zh = self.thresholds["zscore_high_risk"]
        cw = self.thresholds["expected_change_percent_warning"]
        ch = self.thresholds["expected_change_percent_high_risk"]
        vh = self.thresholds["volatility_threshold_high"]
        
        for i in range(len(df)):
            change = df.iloc[i]["expected_change_percent"]
            spike = df.iloc[i]["spike_score"]
            vol = df.iloc[i]["rolling_volatility"]
            
            if change >= ch or spike >= zh:
                warning_levels.append("HIGH RISK")
                reasons = []
                if change >= ch:
                    reasons.append(f"7D Forecast increase (+{change:.1f}%) exceeds high risk limit (+{ch}%)")
                if spike >= zh:
                    reasons.append(f"Price spike Z-score ({spike:.2f}) exceeds critical limit ({zh})")
                warning_reasons.append("; ".join(reasons))
            elif change >= cw or spike >= zw or vol >= vh:
                warning_levels.append("WARNING")
                reasons = []
                if change >= cw:
                    reasons.append(f"7D Forecast increase (+{change:.1f}%) exceeds warning limit (+{cw}%)")
                if spike >= zw:
                    reasons.append(f"Price spike Z-score ({spike:.2f}) elevated")
                if vol >= vh:
                    reasons.append(f"30D Rolling volatility ({vol:.3f}) is elevated")
                warning_reasons.append("; ".join(reasons))
            else:
                warning_levels.append("NORMAL")
                warning_reasons.append("Price forecast and volatility remain within normal bounds")

        df["warning_level"] = warning_levels
        df["warning_reason"] = warning_reasons
        
        # Select & format output columns
        output_cols = [
            "date", "state", "district", "market",
            "price_rs_per_qtl", "forecast_7d", "forecast_15d", "forecast_30d",
            "expected_change_percent", "rolling_volatility", "spike_score",
            "warning_level", "warning_reason"
        ]
        out_df = df[output_cols].rename(columns={"price_rs_per_qtl": "current_price"}).sort_values(by=["state", "district", "market", "date"]).reset_index(drop=True)
        
        os.makedirs("data/processed", exist_ok=True)
        out_df.to_csv("data/processed/early_warning.csv", index=False)
        print(f"[EarlyWarningSystem] Saved early warning alerts ({len(out_df)} rows) to data/processed/early_warning.csv")
        return out_df

    def evaluate_warnings(self, early_warning_df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate classification performance of early warning system against actual price spikes.
        Computes Precision, Recall, F1-Score, False Alarm Rate, and Missed Spike Rate.
        """
        df = early_warning_df.copy()
        
        # Actual spike definition: Actual price change >= 3% in next 7 days
        df["actual_next_price"] = df.groupby(["state", "district", "market"])["current_price"].shift(-7)
        df["actual_change_pct"] = ((df["actual_next_price"] - df["current_price"]) / df["current_price"]) * 100.0
        
        valid_df = df.dropna(subset=["actual_change_pct"]).copy()
        if valid_df.empty:
            return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "false_alarm_rate": 0.0, "missed_spike_rate": 0.0}
            
        actual_spike = valid_df["actual_change_pct"] >= 3.0
        pred_warning = valid_df["warning_level"].isin(["WARNING", "HIGH RISK"])
        
        tp = np.sum(actual_spike & pred_warning)
        fp = np.sum(~actual_spike & pred_warning)
        fn = np.sum(actual_spike & ~pred_warning)
        tn = np.sum(~actual_spike & ~pred_warning)
        
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        false_alarm_rate = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        missed_spike_rate = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
        
        metrics = {
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1_Score": round(f1, 4),
            "False_Alarm_Rate": round(false_alarm_rate, 4),
            "Missed_Spike_Rate": round(missed_spike_rate, 4)
        }
        
        print(f"[EarlyWarningSystem] Warning Classification Metrics: {metrics}")
        return metrics
