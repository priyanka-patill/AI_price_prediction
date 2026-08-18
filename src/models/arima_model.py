import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from typing import Tuple, Dict, Any, List
import warnings

warnings.filterwarnings("ignore")

class ArimaBaselineModel:
    """
    Traditional Statistical Baseline Model using ARIMA / SARIMAX.
    Generates multi-step price forecasts with confidence bounds.
    Optimized for lightweight execution.
    """
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 0), seasonal_order: Tuple[int, int, int, int] = None):
        self.order = order
        self.seasonal_order = seasonal_order

    def fit_predict_market(self, train_series: pd.Series, horizon: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit ARIMA model on training time series and forecast horizon steps ahead.
        """
        vals = train_series.dropna().values
        if len(vals) < 10:
            last_val = vals[-1] if len(vals) > 0 else 3000.0
            preds = np.full(horizon, last_val)
            return preds, preds - 50.0, preds + 50.0

        try:
            model = ARIMA(vals, order=self.order, enforce_stationarity=False, enforce_invertibility=False)
            fitted = model.fit()
            forecast_res = fitted.get_forecast(steps=horizon)
            mean_preds = forecast_res.predicted_mean
            ci = forecast_res.conf_int(alpha=0.05)
            lower_bounds = ci[:, 0]
            upper_bounds = ci[:, 1]
            return mean_preds, lower_bounds, upper_bounds
        except Exception:
            last_val = vals[-1]
            preds = np.full(horizon, last_val)
            return preds, preds - 50.0, preds + 50.0

    def batch_predict(self, df: pd.DataFrame, horizons: List[int] = [7, 15, 30], group_cols: List[str] = ["state", "district", "market"]) -> Dict[int, pd.DataFrame]:
        """
        Run ARIMA predictions across markets for test evaluation.
        """
        results = {}
        for h in horizons:
            preds_list = []
            lowers_list = []
            uppers_list = []
            
            for keys, sub_df in df.groupby(group_cols):
                sub_df = sub_df.sort_values(by="date")
                prices = sub_df["price_rs_per_qtl"].values
                n = len(prices)
                
                # Fit ARIMA per market on full historical train series once to generate test predictions efficiently
                train_prices = prices[:int(n * 0.7)]
                p, l, u = self.fit_predict_market(pd.Series(train_prices), horizon=h)
                last_pred = p[-1]
                last_low = l[-1]
                last_up = u[-1]
                
                for i in range(n):
                    if i >= int(n * 0.7):
                        preds_list.append(last_pred)
                        lowers_list.append(last_low)
                        uppers_list.append(last_up)
                    else:
                        preds_list.append(prices[i])
                        lowers_list.append(prices[i] - 50)
                        uppers_list.append(prices[i] + 50)

            sub_copy = df.copy()
            sub_copy[f"arima_pred_{h}d"] = preds_list
            sub_copy[f"arima_lower_{h}d"] = lowers_list
            sub_copy[f"arima_upper_{h}d"] = uppers_list
            results[h] = sub_copy
            
        return results
