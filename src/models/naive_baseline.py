import pandas as pd
import numpy as np

class NaiveBaselineModel:
    """
    Naive baseline forecasting model: Predicts future price at horizon T+h equal to latest known price at date T.
    """
    def __init__(self, price_col: str = "price_rs_per_qtl"):
        self.price_col = price_col

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return latest known price at date T."""
        return df[self.price_col].values
