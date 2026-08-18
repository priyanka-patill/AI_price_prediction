import pandas as pd
import numpy as np

def add_government_features(df: pd.DataFrame,
                            msp_df: pd.DataFrame = None,
                            procurement_df: pd.DataFrame = None,
                            stock_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Merge MSP, Procurement Quantity, Central Pool Government Stock, and Stock vs Buffer Percent.
    Ensure features at date T use information available on or before T.
    """
    df = df.sort_values(by=["date"]).reset_index(drop=True)
    df["date_dt"] = pd.to_datetime(df["date"])
    df["year"] = df["date_dt"].dt.year
    df["month"] = df["date_dt"].dt.month
    
    # Map calendar date to Marketing Year (KMS starts Oct 1)
    def get_marketing_year(row):
        y = row["year"]
        m = row["month"]
        if m >= 10:
            return f"{y}-{str(y+1)[2:]}"
        else:
            return f"{y-1}-{str(y)[2:]}"

    df["marketing_year"] = df.apply(get_marketing_year, axis=1)
    
    # 1. Merge MSP (Common Paddy MSP)
    if msp_df is not None and not msp_df.empty:
        common_msp = msp_df[msp_df["paddy_type"] == "Common"][["marketing_year", "msp_rs_per_qtl"]].rename(columns={"msp_rs_per_qtl": "msp"})
        df = df.merge(common_msp, on="marketing_year", how="left")
    else:
        df["msp"] = np.nan
        
    # 2. Merge Procurement Quantity
    if procurement_df is not None and not procurement_df.empty:
        proc_sub = procurement_df[["marketing_year", "state", "paddy_procured_mt"]].rename(columns={"paddy_procured_mt": "procurement_quantity"})
        df = df.merge(proc_sub, on=["marketing_year", "state"], how="left")
    else:
        df["procurement_quantity"] = np.nan
        
    # 3. Merge Central Pool Government Stock & Buffer Percent
    if stock_df is not None and not stock_df.empty:
        stock_sub = stock_df[["date", "rice_stock_mt", "stock_vs_buffer_percent"]].rename(columns={"rice_stock_mt": "government_stock"})
        # Merge on monthly start date or closest past date
        df["month_start_date"] = df["date_dt"].dt.strftime("%Y-%m-01")
        df = df.merge(stock_sub, left_on="month_start_date", right_on="date", how="left", suffixes=("", "_stock"))
        df = df.drop(columns=["date_stock", "month_start_date"], errors="ignore")
    else:
        df["government_stock"] = np.nan
        df["stock_vs_buffer_percent"] = np.nan
        
    # Drop temp columns
    df = df.drop(columns=["date_dt"], errors="ignore")
    return df
